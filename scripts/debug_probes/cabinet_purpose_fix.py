#!/usr/bin/env python
"""`impact_drawer` is not drawn on og391 because OmniGibson overwrites its `purpose` to `guide`.

WHAT THE FIRST PROBE (cabinet_render_probe.py) ESTABLISHED, on the live stage, task 8:

    world bound, purpose set `default+render`   EMPTY
    world bound, purpose set `guide`            extent (0.512, 0.669, 0.545)
    subtree purposes                           guide: 56 (== every Mesh), default: 30 (Xforms only)
    ancestor chain visibility                  all `inherited` -- NOT the link8 mechanism
    meshes with no/missing material             0 -- NOT the material fallback
    hide-and-diff, cabinet                     27-182 px   (== the rt noise floor)
    hide-and-diff, breakfast_table (control)   245,182 px
    solo render, cabinet vs all-hidden          0 px on all three cameras

So: every piece of this asset's geometry is `guide`, and guide never reaches the colour pass.

WHY OmniGibson DOES THAT, and why 1.1.1 did not. `RigidPrim._post_load` classifies each geom as
collision-or-visual and sets `purpose = "guide"` on the collision ones (rigid_prim.py:242). Both
stacks do that much. The two things that changed:

  1. TRAVERSAL DEPTH. 1.1.1 scanned exactly two levels below the link:

         for prim in self._prim.GetChildren():
             prims_to_check.append(prim)
             for child in prim.GetChildren():
                 prims_to_check.append(child)

     og391 recurses to unbounded depth (`_find_geom_prims`). This asset's drawer geometry sits FOUR
     levels below its link -- `drawer_blender_cut_0N/ObjectCapture/Geometry/Mesh/Mesh` -- so 1.1.1
     never visited it, never wrapped it, and never wrote its purpose. Its authored `default`
     survived and it rendered. og391 reaches it and guides it.

  2. INHERITED `is_collision`. og391 makes the flag STICKY down the recursion:

         def _find_geom_prims(prim, is_collision=False):
             if prim.HasAPI(UsdPhysics.CollisionAPI):   is_collision = True
             ...
             for child in prim.GetChildren():
                 _find_geom_prims(child, is_collision)      # <-- inherited by every descendant

     1.1.1 read the flag off each geom's OWN API and nothing else. So under og391 a CollisionAPI on
     an ancestor Xform guides every mesh beneath it.

THIS PROBE does two things in one boot, because a boot is ~6 minutes:

  (1) MEASURES WHICH OF THE TWO IT IS. Per mesh: does it carry CollisionAPI itself, or does it only
      inherit from an ancestor? Its depth below its link? Its authored purpose in cabinet.usd
      versus its runtime purpose? That separates cause 1 from cause 2 and tells a fix what to
      target. It also names, exactly, which prims OmniGibson overwrote.

  (2) TESTS THE FIX, which is deliberately the narrowest thing that can work: restore `purpose` to
      its AUTHORED value on exactly those meshes whose authored value OmniGibson overwrote with
      `guide`. Nothing else. The asset's own `collider_guide` / `Cube` helpers authored `guide`
      themselves and stay guide; only the overwritten ones come back. The authored value is read
      out of the property stack's opinion in the cabinet.usd layer, not guessed from prim names.

PHYSICS. `purpose` is a `UsdGeom.Imageable` render attribute. The CollisionAPI, the contact/rest
offsets, the collision approximation and the link centre-of-mass are all left exactly as
OmniGibson set them -- this probe writes NOTHING else. To prove that rather than assert it, joint
positions, joint velocities and every link's pose are fingerprinted before and after the purpose
writes and compared BITWISE.

    ./scripts/clara/interactive/rr python -u scripts/debug_probes/cabinet_purpose_fix.py \
        --task-id 8 --out /logs/cabinet_render --label t8fix
"""

import argparse
import hashlib
import json
import os
import sys
import traceback

import numpy as np

DIFF_THRESH = 24


def luma(a):
    a = np.asarray(a).astype(np.float64)
    return 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]


def changed(a, b, thresh=DIFF_THRESH):
    d = np.abs(a.astype(np.int16) - b.astype(np.int16)).max(axis=-1)
    m = d > thresh
    n = int(m.sum())
    rec = {"n_changed": n, "frac_changed": round(float(n) / float(m.size), 6),
           "max_delta": int(d.max())}
    if n:
        ys, xs = np.nonzero(m)
        rec["box"] = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
        rec["centroid"] = [round(float(xs.mean()), 1), round(float(ys.mean()), 1)]
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--task-id", type=int, default=8)
    ap.add_argument("--pert-id", type=int, default=0)
    ap.add_argument("--robot", default="DROID")
    ap.add_argument("--obj-name", default="drawer")
    ap.add_argument("--asset-layer", default="cabinet.usd",
                    help="substring identifying the asset's own USD layer in a property stack")
    ap.add_argument("--frames", type=int, default=5)
    ap.add_argument("--settle", type=int, default=4)
    ap.add_argument("--no-fix", action="store_true", help="measure only, apply nothing")
    args = ap.parse_args()

    assert not args.out.startswith("/tmp"), "/tmp is node-local and wiped -- artifacts go on Lustre"
    os.makedirs(args.out, exist_ok=True)
    report = {"label": args.label, "argv": sys.argv, "args": vars(args)}
    json_path = os.path.join(args.out, f"{args.label}.json")

    def flush():
        with open(json_path, "w") as f:
            json.dump(report, f, indent=2, sort_keys=False, default=str)

    def say(*a):
        print(*a)
        sys.stdout.flush()

    sys.path.insert(0, "/app")
    import omnigibson as og
    import omnigibson.lazy as lazy

    try:
        from realm.sim_config import set_sim_config
    except ImportError:
        from realm.eval import set_sim_config
    from realm.eval import SUPPORTED_TASKS, SUPPORTED_PERTURBATIONS
    from realm.environments.env_dynamic import RealmEnvironmentDynamic

    set_sim_config(robot=args.robot)
    task = SUPPORTED_TASKS[args.task_id]
    pert = SUPPORTED_PERTURBATIONS[args.pert_id]
    report["scene"] = {"task": task, "task_id": args.task_id, "perturbation": pert}
    say(f"[scene] task={task} ({args.task_id}) pert={pert}")
    flush()

    try:
        env = RealmEnvironmentDynamic(
            config_path="/app/realm/config",
            task_cfg_path=f"REALM_DROID10/{task}/default.yaml",
            perturbations=[pert], multi_view=True, no_rendering=False, robot=args.robot)
    except Exception as e:
        report["env_creation_error"] = {"type": type(e).__name__, "msg": str(e),
                                       "traceback": traceback.format_exc()[-6000:]}
        say(f"[env] FAILED: {type(e).__name__}: {e}")
        flush()
        return 3
    say("[env] created")

    Usd, UsdGeom, UsdPhysics = lazy.pxr.Usd, lazy.pxr.UsdGeom, lazy.pxr.UsdPhysics
    PhysxSchema = lazy.pxr.PhysxSchema
    stage = og.sim.stage
    scene = env.omnigibson_env.scene
    obj = scene.object_registry("name", args.obj_name)
    assert obj is not None, f"{args.obj_name} not in scene"
    root = stage.GetPrimAtPath(obj.prim_path)
    report["target"] = {"name": obj.name, "prim_path": obj.prim_path}
    say(f"[target] {obj.prim_path}")

    # ============================================================================================
    # (1) WHICH MECHANISM: own CollisionAPI vs inherited, depth below link, authored vs runtime
    # ============================================================================================
    link_names = set(getattr(obj, "links", {}) or {})
    report["link_names"] = sorted(link_names)

    def authored_purpose_in_asset(prim):
        """The `purpose` opinion coming from the asset's OWN layer.

        Read off the property stack rather than inferred from the prim's name: OmniGibson's write
        lands in a stronger layer, so `GetPurposeAttr().Get()` returns the OVERWRITTEN value while
        the asset's opinion is still there, lower in the stack. This is what makes "restore exactly
        what was overwritten" a precise operation instead of a guess.
        """
        attr = UsdGeom.Imageable(prim).GetPurposeAttr()
        if not attr:
            return None, []
        stack = []
        try:
            for spec in attr.GetPropertyStack(Usd.TimeCode.Default()):
                lid = spec.layer.identifier if spec.layer else "<none>"
                stack.append({"layer": lid, "value": str(spec.default)})
        except Exception as e:
            return None, [{"error": f"{type(e).__name__}: {e}"}]
        for s in stack:
            if args.asset_layer in s["layer"]:
                return s["value"], stack
        return None, stack

    meshes = []
    for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate)):
        if prim.GetTypeName() not in ("Mesh", "Cube", "Sphere", "Cone", "Cylinder"):
            continue
        path = str(prim.GetPath())
        own_coll = bool(prim.HasAPI(UsdPhysics.CollisionAPI))
        own_physx = bool(prim.HasAPI(PhysxSchema.PhysxCollisionAPI))
        anc_with_api, cur, depth_from_root = [], prim.GetParent(), 0
        link_ancestor, depth_below_link = None, None
        while cur and str(cur.GetPath()) != "/":
            if cur.HasAPI(UsdPhysics.CollisionAPI) or cur.HasAPI(PhysxSchema.PhysxCollisionAPI):
                anc_with_api.append(str(cur.GetPath()))
            if cur.GetName() in link_names and link_ancestor is None:
                link_ancestor, depth_below_link = str(cur.GetPath()), depth_from_root + 1
            cur = cur.GetParent()
            depth_from_root += 1
        auth, stack = authored_purpose_in_asset(prim)
        img = UsdGeom.Imageable(prim)
        rec = {
            "path": path, "type": str(prim.GetTypeName()),
            "own_collision_api": own_coll, "own_physx_collision_api": own_physx,
            "ancestors_with_collision_api": anc_with_api,
            "link": link_ancestor, "depth_below_link": depth_below_link,
            "purpose_runtime": str(img.ComputePurpose()),
            "purpose_authored_in_asset": auth,
            "purpose_layer_stack": stack,
            "n_points": len(UsdGeom.Mesh(prim).GetPointsAttr().Get() or [])
            if prim.GetTypeName() == "Mesh" else None,
        }
        rec["overwritten"] = (rec["purpose_runtime"] == "guide"
                             and auth is not None and auth != "guide")
        rec["classified_collision_by"] = (
            "own API" if own_coll or own_physx else
            ("INHERITED from " + anc_with_api[0]) if anc_with_api else "not a collider at all")
        meshes.append(rec)

    report["meshes"] = meshes
    n_own = sum(1 for r in meshes if r["own_collision_api"] or r["own_physx_collision_api"])
    n_inh = sum(1 for r in meshes if not (r["own_collision_api"] or r["own_physx_collision_api"])
                and r["ancestors_with_collision_api"])
    n_neither = sum(1 for r in meshes if not (r["own_collision_api"] or r["own_physx_collision_api"])
                    and not r["ancestors_with_collision_api"])
    overwritten = [r for r in meshes if r["overwritten"]]
    depths = sorted({r["depth_below_link"] for r in meshes if r["depth_below_link"] is not None})
    report["mechanism"] = {
        "n_geoms": len(meshes),
        "n_own_collision_api": n_own,
        "n_inherited_only": n_inh,
        "n_no_collision_api_anywhere": n_neither,
        "n_purpose_overwritten_by_og": len(overwritten),
        "depths_below_link": depths,
        "deeper_than_111_would_reach": [r["path"] for r in meshes
                                        if (r["depth_below_link"] or 0) > 2],
        "n_deeper_than_111_would_reach": sum(1 for r in meshes
                                            if (r["depth_below_link"] or 0) > 2),
    }
    say("\n" + "=" * 92)
    say("[mechanism] per-geom classification")
    for k, v in report["mechanism"].items():
        if k != "deeper_than_111_would_reach":
            say(f"   {k}: {v}")
    say("=" * 92)
    say("\n[geoms] path | own API | inherited-from | depth below link | authored -> runtime")
    for r in meshes:
        say(f"   {r['path'].split('/World/scene_0/drawer/')[-1]:56s} "
            f"own={str(r['own_collision_api'])[0]}{str(r['own_physx_collision_api'])[0]} "
            f"d={r['depth_below_link']} "
            f"{str(r['purpose_authored_in_asset']):8s} -> {r['purpose_runtime']:8s} "
            f"{'OVERWRITTEN' if r['overwritten'] else ''}  [{r['classified_collision_by']}]")
    flush()

    # ============================================================================================
    # rendering + physics fingerprint
    # ============================================================================================
    def collect_rgb(o):
        out = {}

        def walk(node, path):
            if isinstance(node, dict):
                for k, v in node.items():
                    if k == "rgb":
                        a = v.cpu().numpy() if hasattr(v, "cpu") else v
                        a = np.asarray(a)
                        if a.ndim == 3 and a.shape[-1] >= 3:
                            out[path] = a[..., :3].astype(np.uint8)
                    else:
                        walk(v, f"{path}.{k}" if path else str(k))

        walk(o, "")
        return out

    obs, _ = env.reset()
    obs, _r, _t, _tr, _i = env.warmup(obs)
    hold = np.concatenate((np.asarray(env.reset_qpos)[:7], np.atleast_1d(-1.0)))

    def step_n(n):
        nonlocal obs
        got = []
        for _ in range(n):
            obs, _r2, _t2, _tr2, _i2 = env.step(hold)
            got.append(collect_rgb(obs))
        return got

    step_n(args.settle)
    cams = sorted(collect_rgb(obs).keys())

    def measure(name):
        step_n(args.settle)
        seq = step_n(args.frames)
        out = {}
        for cam in cams:
            imgs = [s[cam] for s in seq if cam in s]
            if not imgs:
                continue
            med = np.median(np.stack(imgs, 0), axis=0).astype(np.uint8)
            out[cam] = med
            try:
                from PIL import Image
                Image.fromarray(med).save(
                    os.path.join(args.out, f"{args.label}__{name}__{cam.replace('.', '-')}.png"))
            except Exception as e:
                say(f"   [png] {type(e).__name__}: {e}")
        say(f"[render] {name}: " + "  ".join(f"mean{luma(out[c]).mean():.2f}" for c in out))
        return out

    def phys_fingerprint():
        """Everything a render change must NOT move. Raw bytes, so the comparison is bitwise."""
        parts = {}

        def add(key, val):
            if val is None:
                return
            a = val.cpu().numpy() if hasattr(val, "cpu") else np.asarray(val)
            a = np.ascontiguousarray(a, dtype=np.float64)
            parts[key] = {"sha256": hashlib.sha256(a.tobytes()).hexdigest(),
                          "shape": list(a.shape),
                          "sum": float(a.sum())}

        for who, o in (("drawer", obj), ("robot", scene.robots[0] if scene.robots else None)):
            if o is None:
                continue
            for attr in ("get_joint_positions", "get_joint_velocities"):
                try:
                    add(f"{who}.{attr}", getattr(o, attr)())
                except Exception:
                    pass
            try:
                p, q = o.get_position_orientation()
                add(f"{who}.pos", p)
                add(f"{who}.orn", q)
            except Exception:
                pass
            for lname, link in (getattr(o, "links", {}) or {}).items():
                try:
                    p, q = link.get_position_orientation()
                    add(f"{who}.link.{lname}.pos", p)
                    add(f"{who}.link.{lname}.orn", q)
                except Exception:
                    pass
        return parts

    before = measure("before_fix")
    fp_before = phys_fingerprint()
    report["physics_fingerprint_keys"] = len(fp_before)

    if args.no_fix:
        report["fix"] = {"applied": False, "reason": "--no-fix"}
        flush()
        return 0

    # ============================================================================================
    # (2) THE FIX: restore the authored purpose on exactly the overwritten geoms.
    # ============================================================================================
    applied, failed = [], {}
    with og.sim.editing_usd():
        for r in overwritten:
            prim = stage.GetPrimAtPath(r["path"])
            try:
                UsdGeom.Imageable(prim).CreatePurposeAttr().Set(
                    r["purpose_authored_in_asset"] or UsdGeom.Tokens.default_)
                applied.append(r["path"])
            except Exception as e:
                failed[r["path"]] = f"{type(e).__name__}: {e}"
    say(f"\n[fix] restored authored purpose on {len(applied)} geom(s); {len(failed)} failed")
    if failed:
        say(f"[fix] failures: {failed}")

    # confirm the writes took, per prim, and recompute the world bound
    now = {}
    for p in applied:
        now[p] = str(UsdGeom.Imageable(stage.GetPrimAtPath(p)).ComputePurpose())
    still_guide = [p for p, v in now.items() if v == "guide"]
    T = UsdGeom.Tokens

    def world_bound(purposes):
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), purposes, useExtentsHint=False)
        rng = cache.ComputeWorldBound(root).ComputeAlignedRange()
        if rng.IsEmpty():
            return {"empty": True}
        mn, mx = rng.GetMin(), rng.GetMax()
        return {"empty": False, "min": [round(float(v), 5) for v in mn],
                "max": [round(float(v), 5) for v in mx],
                "extent": [round(float(mx[i] - mn[i]), 5) for i in range(3)]}

    report["fix"] = {
        "applied": True, "n_applied": len(applied), "paths": applied, "failures": failed,
        "still_guide_after_write": still_guide,
        "world_bound_default_render_after": world_bound([T.default_, T.render]),
        "world_bound_guide_after": world_bound([T.guide]),
    }
    say(f"[fix] world bound default+render AFTER: {report['fix']['world_bound_default_render_after']}")
    flush()

    after = measure("after_fix")
    report["fix"]["per_cam_diff"] = {}
    for cam in before:
        if cam in after:
            report["fix"]["per_cam_diff"][cam] = changed(before[cam], after[cam])
            say(f"   fix-diff {cam}: {report['fix']['per_cam_diff'][cam]}")

    # ============================================================================================
    # PHYSICS PROOF -- bitwise, not asserted
    # ============================================================================================
    fp_after = phys_fingerprint()
    moved = {k: {"before": fp_before[k], "after": fp_after[k]}
             for k in fp_before if k in fp_after
             and fp_before[k]["sha256"] != fp_after[k]["sha256"]}
    report["physics_unchanged"] = {
        "n_compared": len(fp_before), "n_moved": len(moved), "moved": moved,
        "keys_only_before": sorted(set(fp_before) - set(fp_after)),
        "keys_only_after": sorted(set(fp_after) - set(fp_before)),
    }
    say(f"\n[physics] {len(fp_before)} fingerprints compared, {len(moved)} moved")
    if moved:
        for k, v in list(moved.items())[:20]:
            say(f"   MOVED {k}: sum {v['before']['sum']} -> {v['after']['sum']}")
    else:
        say("   all bitwise identical across the purpose writes")

    say("\n" + "=" * 92)
    say(f"  mechanism: purpose overwritten to `guide` on {len(overwritten)}/{len(meshes)} geoms")
    say(f"  own CollisionAPI: {n_own}   inherited-only: {n_inh}   neither: {n_neither}")
    say(f"  deeper than 1.1.1's 2-level scan: "
        f"{report['mechanism']['n_deeper_than_111_would_reach']}")
    say(f"  fix restored {len(applied)} geom(s); pixels changed: "
        f"{ {c: v['n_changed'] for c, v in report['fix']['per_cam_diff'].items()} }")
    say(f"  physics fingerprints moved: {len(moved)}")
    say("=" * 92)
    flush()

    try:
        og.clear()
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
