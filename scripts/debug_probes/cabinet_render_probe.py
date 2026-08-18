#!/usr/bin/env python
"""Why is `impact_drawer`'s cabinet physically present on og391 but absent from the render?

THE QUESTION HAS FOUR CANDIDATE MECHANISMS and they are not separable from frames:

  M1  material resolution fails so badly nothing is drawn
  M2  the prim, or an ANCESTOR, is invisible -- ancestor `invisible` cannot be overridden by any
      descendant, which is exactly how the `panda_link8` adapter pad was invisible on this same port
  M3  wrong render `purpose` (guide/proxy never reach the colour pass)
  M4  drawn correctly but out of frustum, or fully occluded

Arguing between them from a PNG is how the link8 bug got misattributed twice. So this probe runs
measurements whose outcomes point at DIFFERENT mechanisms.

--------------------------------------------------------------------------------------------------
THE DECISIVE PAIR, and why each needs its control

  (A) HIDE-AND-DIFF.  Render baseline; hide the cabinet; render again; count changed pixels.
      Zero changed pixels means the asset contributes nothing to the image.
      *** This is worthless without a POSITIVE CONTROL. *** "0 px changed" is equally consistent
      with "the asset is not drawn" and with "my hide call did nothing". So the same hide-and-diff
      is run on the breakfast_table immutable, which is unambiguously visible in both stacks. The
      pair (cabinet -> 0 px, table -> many px) is decisive; either number alone is not. A RESTORE
      control then re-renders after un-hiding and must return to baseline, proving the hide was
      reversible and the diff was not drift.

  (B) SOLO RENDER.  Hide every object EXCEPT the cabinet and render. This kills M4 outright without
      any argument about occlusion: with nothing else on the stage there is nothing to occlude it.
      If the frame is still empty where the cabinet should be, it is not drawn, full stop.

Both are gated: `rt` re-renders an unchanged scene differently on ~25% of pixels on this stack, so
every measurement is a median over N frames and the changed-pixel count uses a threshold well above
that noise. A single frame is not evidence here.

--------------------------------------------------------------------------------------------------
SUPPORTING READBACKS, each aimed at one mechanism

  * per-prim `ComputeVisibility()` / `ComputePurpose()` over the cabinet subtree      -> M2, M3
  * the ROOT's full ANCESTOR CHAIN visibility, up to `/`                              -> M2
  * Mesh point counts, bound material path, whether that material prim exists         -> M1
  * `UsdGeom.BBoxCache` world bound computed separately for the `render`/`default`
    purpose set and for `guide`, so "the only geometry is guide geometry" is a number  -> M3
  * world AABB corners projected through the sensor's own `intrinsic_matrix`           -> M4
    ...and the projection is VALIDATED, not trusted: the table's projected box is compared against
    the pixels that actually changed when the table was hidden. If those agree, the same maths
    applied to the cabinet is trustworthy.

Instance proxies are traversed explicitly (`Usd.TraverseInstanceProxies`): OmniGibson instances
almost every object and a plain Traverse() silently returns nothing inside one.

PHYSICS IS NEVER TOUCHED. Every write is `visibility`, a render-only USD attribute, and every one is
restored. No joint, mass, collider or transform attribute is written anywhere in this file.

    ./scripts/clara/interactive/rr python -u scripts/debug_probes/cabinet_render_probe.py \
        --task-id 8 --out /logs/cabinet_render --label t8
"""

import argparse
import json
import os
import sys
import traceback

import numpy as np

# Changed-pixel threshold. rt frame-to-frame noise on this stack is a few LSB on a quarter of the
# frame; 24 is far above that and far below any real object edge.
DIFF_THRESH = 24


def luma(a):
    a = np.asarray(a).astype(np.float64)
    return 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]


def changed(a, b, thresh=DIFF_THRESH):
    """Pixels whose max abs channel delta exceeds `thresh`, plus their bounding box.

    Returns the count, the fraction, and the tight screen box of the changed region -- the box is
    what gets compared against the projected AABB, so a "many pixels changed" result can be checked
    for being in the RIGHT PLACE rather than just large.
    """
    d = np.abs(a.astype(np.int16) - b.astype(np.int16)).max(axis=-1)
    m = d > thresh
    n = int(m.sum())
    rec = {"n_changed": n, "frac_changed": round(float(n) / float(m.size), 6),
           "max_delta": int(d.max()), "mean_delta": round(float(d.mean()), 4)}
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
    ap.add_argument("--obj-name", default="drawer", help="scene object to interrogate")
    ap.add_argument("--control-name", default="breakfast_table_support",
                    help="an object known to be visible -- the positive control for hide-and-diff")
    ap.add_argument("--frames", type=int, default=5)
    ap.add_argument("--settle", type=int, default=4)
    ap.add_argument("--skip-solo", action="store_true")
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
    from omnigibson.macros import gm
    import omnigibson.lazy as lazy

    try:
        from realm.sim_config import set_sim_config
    except ImportError:
        from realm.eval import set_sim_config
    from realm.eval import SUPPORTED_TASKS, SUPPORTED_PERTURBATIONS
    from realm.environments.env_dynamic import RealmEnvironmentDynamic

    # NB: `lazy.pxr` is NOT bound until Isaac has booted -- accessing it before env creation raises
    # "module lazy_ has no attribute pxr". Bound after the env exists, below.
    report["identity"] = {"omnigibson_file": og.__file__,
                          "omnigibson_version": getattr(og, "__version__", None),
                          "REALM_LIGHT_FIX": os.environ.get("REALM_LIGHT_FIX", "<unset>"),
                          "gm_FORCE_LIGHT_INTENSITY": getattr(gm, "FORCE_LIGHT_INTENSITY", None)}
    say(f"[identity] og={og.__file__}")

    set_sim_config(robot=args.robot)
    task = SUPPORTED_TASKS[args.task_id]
    pert = SUPPORTED_PERTURBATIONS[args.pert_id]
    report["scene"] = {"task": task, "task_id": args.task_id, "perturbation": pert}
    say(f"[scene] task={task} ({args.task_id}) pert={pert} robot={args.robot}")
    flush()

    try:
        env = RealmEnvironmentDynamic(
            config_path="/app/realm/config",
            task_cfg_path=f"REALM_DROID10/{task}/default.yaml",
            perturbations=[pert], multi_view=True, no_rendering=False, robot=args.robot)
    except Exception as e:
        report["env_creation_error"] = {"type": type(e).__name__, "msg": str(e),
                                        "traceback": traceback.format_exc()[-6000:]}
        say(f"[env] CREATION FAILED: {type(e).__name__}: {e}")
        flush()
        return 3
    say("[env] created")
    flush()

    # Isaac has booted, so the USD python bindings are reachable now (see the note above).
    Usd, UsdGeom, UsdShade, Sdf = lazy.pxr.Usd, lazy.pxr.UsdGeom, lazy.pxr.UsdShade, lazy.pxr.Sdf

    stage = og.sim.stage

    # ============================================================================================
    # locate the object
    # ============================================================================================
    # REALM wraps og.Environment rather than subclassing it, so the scene hangs off
    # `.omnigibson_env` (env_dynamic.py:117). Resolved by search, not assumed, so a rename upstream
    # degrades to a clear error instead of an AttributeError mid-probe.
    scene = None
    for chain in ("omnigibson_env.scene", "scene", "env.scene", "_env.scene"):
        cur = env
        for part in chain.split("."):
            cur = getattr(cur, part, None)
            if cur is None:
                break
        if cur is not None:
            scene, scene_via = cur, chain
            break
    assert scene is not None, "could not locate the OmniGibson scene on the REALM env wrapper"
    report["scene_via"] = scene_via
    say(f"[scene] resolved via env.{scene_via}")
    robots = list(getattr(scene, "robots", None) or getattr(env, "robots", None) or [])

    def find_obj(name):
        try:
            o = scene.object_registry("name", name)
            if o is not None:
                return o
        except Exception:
            pass
        for o in scene.objects:
            if getattr(o, "name", None) == name:
                return o
        return None

    obj = find_obj(args.obj_name)
    ctrl = find_obj(args.control_name)
    report["objects_in_scene"] = [getattr(o, "name", "?") for o in scene.objects]
    say(f"[objects] {report['objects_in_scene']}")
    if obj is None:
        report["error"] = f"object {args.obj_name!r} not in scene"
        say(f"[FATAL] {report['error']}")
        flush()
        return 4
    root_path = obj.prim_path
    report["target"] = {"name": obj.name, "prim_path": root_path,
                        "type": type(obj).__name__,
                        "control_name": getattr(ctrl, "name", None),
                        "control_prim_path": getattr(ctrl, "prim_path", None)}
    say(f"[target] {obj.name} at {root_path}")
    flush()

    # ============================================================================================
    # M2 / M3: per-prim visibility + purpose over the subtree, and the ROOT's ancestor chain
    # ============================================================================================
    def imageable_state(prim):
        rec = {"path": str(prim.GetPath()), "type": str(prim.GetTypeName()),
               "active": bool(prim.IsActive()),
               "instance": bool(prim.IsInstance()),
               "instance_proxy": bool(prim.IsInstanceProxy())}
        img = UsdGeom.Imageable(prim)
        if img:
            va = img.GetVisibilityAttr()
            rec["vis_authored"] = bool(va and va.HasAuthoredValue())
            rec["vis"] = str(va.Get()) if va else None
            try:
                rec["vis_computed"] = str(img.ComputeVisibility())
            except Exception as e:
                rec["vis_computed"] = f"<err {type(e).__name__}>"
            pa = img.GetPurposeAttr()
            rec["purpose_authored"] = bool(pa and pa.HasAuthoredValue())
            try:
                rec["purpose_computed"] = str(img.ComputePurpose())
            except Exception as e:
                rec["purpose_computed"] = f"<err {type(e).__name__}>"
        if prim.GetTypeName() == "Mesh":
            pts = UsdGeom.Mesh(prim).GetPointsAttr().Get()
            rec["n_points"] = len(pts) if pts is not None else 0
            b = UsdShade.MaterialBindingAPI(prim).GetDirectBinding()
            t = b.GetMaterialPath()
            rec["material"] = str(t) if t else None
            if t:
                mp = stage.GetPrimAtPath(t)
                rec["material_exists"] = bool(mp and mp.IsValid())
                rec["material_active"] = bool(mp and mp.IsActive()) if mp else None
        return rec

    # ANCESTOR CHAIN FIRST -- this is the link8 trap. An `invisible` anywhere above the meshes
    # prunes the whole subtree and no descendant can override it.
    chain, p = [], stage.GetPrimAtPath(root_path)
    cur = p
    while cur and cur.GetPath() != Sdf.Path.absoluteRootPath:
        chain.append(imageable_state(cur))
        cur = cur.GetParent()
    chain.reverse()
    report["ancestor_chain"] = chain
    say("\n[ancestors] root -> target (an `invisible` here prunes everything below it)")
    for r in chain:
        say(f"   {r['path']:60s} {r['type']:14s} vis={r.get('vis_computed')} "
            f"purpose={r.get('purpose_computed')} active={r['active']}")
    bad_anc = [r for r in chain if r.get("vis_computed") == "invisible"]
    report["ancestor_invisible"] = [r["path"] for r in bad_anc]

    subtree = [imageable_state(pr) for pr in Usd.PrimRange(
        p, Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate))]
    report["subtree"] = subtree
    meshes = [r for r in subtree if r["type"] == "Mesh"]
    report["subtree_summary"] = {
        "n_prims": len(subtree), "n_meshes": len(meshes),
        "n_meshes_empty": sum(1 for r in meshes if not r.get("n_points")),
        "n_vis_invisible": sum(1 for r in subtree if r.get("vis_computed") == "invisible"),
        "purposes": {k: sum(1 for r in subtree if r.get("purpose_computed") == k)
                     for k in {r.get("purpose_computed") for r in subtree}},
        "n_material_missing": sum(1 for r in meshes if r.get("material")
                                  and not r.get("material_exists")),
        "materials": sorted({r.get("material") for r in meshes if r.get("material")}),
    }
    say(f"\n[subtree] {report['subtree_summary']}")
    inv = [r for r in subtree if r.get("vis_computed") == "invisible"]
    if inv:
        say(f"[subtree] {len(inv)} prim(s) compute INVISIBLE:")
        for r in inv[:30]:
            say(f"   {r['path']}  ({r['type']})  authored={r.get('vis_authored')}")
    flush()

    # ============================================================================================
    # M3: world bound per purpose set. If the only non-empty bound is `guide`, the asset's only
    # geometry is collision-helper geometry and nothing was ever meant to be drawn.
    # ============================================================================================
    def world_bound(prim, purposes):
        try:
            cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), purposes, useExtentsHint=False)
            rng = cache.ComputeWorldBound(prim).ComputeAlignedRange()
            if rng.IsEmpty():
                return {"empty": True}
            mn, mx = rng.GetMin(), rng.GetMax()
            return {"empty": False,
                    "min": [round(float(v), 5) for v in mn],
                    "max": [round(float(v), 5) for v in mx],
                    "extent": [round(float(mx[i] - mn[i]), 5) for i in range(3)],
                    "centre": [round(float((mx[i] + mn[i]) / 2), 5) for i in range(3)]}
        except Exception as e:
            return {"error": f"{type(e).__name__}: {e}"}

    T = UsdGeom.Tokens
    purpose_sets = {
        "default+render": [T.default_, T.render],
        "render_only": [T.render],
        "default_only": [T.default_],
        "guide": [T.guide],
        "proxy": [T.proxy],
    }
    report["world_bound"] = {k: world_bound(p, v) for k, v in purpose_sets.items()}
    say("\n[world bound by purpose set] -- target")
    for k, v in report["world_bound"].items():
        say(f"   {k:16s} {v}")
    if ctrl is not None:
        cp = stage.GetPrimAtPath(ctrl.prim_path)
        report["control_world_bound"] = {k: world_bound(cp, v) for k, v in purpose_sets.items()}
        say("[world bound by purpose set] -- control")
        for k, v in report["control_world_bound"].items():
            say(f"   {k:16s} {v}")
    flush()

    # ============================================================================================
    # rendering helpers
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

    # Same sequence post_tone_sweep.py uses: reset, then REALM's own warmup, then a hold action
    # built from reset_qpos with the gripper term appended. env.action_space is not the right shape
    # here -- REALM's wrapper takes 7 joint targets plus one gripper command.
    obs, _ = env.reset()
    obs, _r, _t, _tr, _i = env.warmup(obs)
    hold = np.concatenate((np.asarray(env.reset_qpos)[:7], np.atleast_1d(-1.0)))

    def step_n(n):
        nonlocal obs
        got = []
        for _ in range(n):
            obs, _r, _t, _tr, _i = env.step(hold)
            got.append(collect_rgb(obs))
        return got

    step_n(args.settle)
    cams = sorted(collect_rgb(obs).keys())
    report["cameras"] = cams
    say(f"\n[cams] {cams}")

    def measure(name):
        """Median over `--frames`, per camera, saved as PNG. rt is stochastic; one frame is noise."""
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
        say(f"[render] {name}: " + "  ".join(
            f"{c.split('.')[-2] if '.' in c else c}=mean{luma(out[c]).mean():.1f}" for c in out))
        return out

    # ============================================================================================
    # (A) HIDE-AND-DIFF, with its positive and restore controls
    # ============================================================================================
    def set_visible(o, vis):
        """Only ever writes `visibility`. Returns what ComputeVisibility says afterwards, so a
        no-op write is caught rather than assumed to have worked."""
        try:
            o.visible = vis
        except Exception as e:
            return f"<err {type(e).__name__}: {e}>"
        try:
            return str(UsdGeom.Imageable(stage.GetPrimAtPath(o.prim_path)).ComputeVisibility())
        except Exception as e:
            return f"<err {type(e).__name__}>"

    base = measure("baseline")

    ab = {}
    for tag, o in (("cabinet", obj), ("control_table", ctrl)):
        if o is None:
            continue
        after = set_visible(o, False)
        say(f"\n[hide] {tag} ({o.prim_path}) -> ComputeVisibility={after}")
        hidden = measure(f"hide_{tag}")
        rec = {"prim_path": o.prim_path, "vis_after_hide": after, "per_cam": {}}
        for cam in base:
            if cam in hidden:
                rec["per_cam"][cam] = changed(base[cam], hidden[cam])
                say(f"   diff {cam}: {rec['per_cam'][cam]}")
        restored = set_visible(o, True)
        rec["vis_after_restore"] = restored
        back = measure(f"restore_{tag}")
        rec["restore_per_cam"] = {c: changed(base[c], back[c]) for c in base if c in back}
        for c, v in rec["restore_per_cam"].items():
            say(f"   restore-diff {c}: {v}")
        ab[tag] = rec
        flush()
    report["hide_and_diff"] = ab

    # ============================================================================================
    # (B) SOLO RENDER -- hide everything except the cabinet. Kills the occlusion candidate.
    # ============================================================================================
    if not args.skip_solo:
        hidden_names, fails = [], {}
        for o in list(scene.objects):
            if o is obj:
                continue
            if "Light" in type(o).__name__:
                continue                       # keep the lights or the frame is black by design
            r = set_visible(o, False)
            hidden_names.append(getattr(o, "name", "?"))
            if "err" in str(r):
                fails[getattr(o, "name", "?")] = r
        for rb in robots:
            r = set_visible(rb, False)
            hidden_names.append(getattr(rb, "name", "robot"))
            if "err" in str(r):
                fails[getattr(rb, "name", "robot")] = r
        say(f"\n[solo] hid {len(hidden_names)} object(s): {hidden_names}")
        if fails:
            say(f"[solo] hide FAILED on: {fails}")
        solo = measure("solo_cabinet_only")
        report["solo"] = {"hidden": hidden_names, "hide_failures": fails, "per_cam": {}}
        for cam in solo:
            # Against the all-hidden frame there is no "background object" left, so the useful
            # number is how much of the frame is NOT the modal colour, plus the diff against the
            # frame with the cabinet ALSO hidden -- computed next.
            report["solo"]["per_cam"][cam] = {
                "mean": round(float(luma(solo[cam]).mean()), 3),
                "std": round(float(luma(solo[cam]).std()), 3),
            }
        # ...and the same scene with the cabinet hidden too: the difference between these two IS
        # the cabinet's contribution, measured with nothing able to occlude it.
        set_visible(obj, False)
        empty = measure("solo_all_hidden")
        for cam in solo:
            if cam in empty:
                report["solo"]["per_cam"][cam]["diff_vs_all_hidden"] = changed(solo[cam], empty[cam])
                say(f"   solo-vs-empty {cam}: {report['solo']['per_cam'][cam]['diff_vs_all_hidden']}")
        set_visible(obj, True)
        for o in list(scene.objects):
            if o is not obj and "Light" not in type(o).__name__:
                set_visible(o, True)
        for rb in robots:
            set_visible(rb, True)
        flush()

    # ============================================================================================
    # M4: project the world AABB through each sensor's own intrinsics.
    # VALIDATED, not trusted -- the control's projected box is compared with the pixels that
    # actually changed when the control was hidden.
    # ============================================================================================
    def sensors():
        out = {}
        for name, s in (getattr(og.sim, "_sensors", None) or {}).items():
            out[name] = s
        for rb in robots:
            for n, s in (getattr(rb, "sensors", {}) or {}).items():
                out[n] = s
        try:
            from omnigibson.sensors import VisionSensor
            for n, s in VisionSensor.SENSORS.items():
                out[n] = s
        except Exception:
            pass
        return out

    def project(bound, sensor):
        if bound.get("empty") or bound.get("error"):
            return {"skipped": bound}
        mn, mx = bound["min"], bound["max"]
        corners = np.array([[x, y, z] for x in (mn[0], mx[0])
                            for y in (mn[1], mx[1]) for z in (mn[2], mx[2])], dtype=np.float64)
        try:
            pos, orn = sensor.get_position_orientation()
            pos = np.asarray(pos.cpu() if hasattr(pos, "cpu") else pos, dtype=np.float64)
            orn = np.asarray(orn.cpu() if hasattr(orn, "cpu") else orn, dtype=np.float64)
            K = sensor.intrinsic_matrix
            K = np.asarray(K.cpu() if hasattr(K, "cpu") else K, dtype=np.float64)
        except Exception as e:
            return {"error": f"pose/intrinsics: {type(e).__name__}: {e}"}
        x, y, z, w = orn                      # OG stores quaternions xyzw
        R = np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
        pc = (corners - pos) @ R              # world -> camera-local (R^T applied on the right)
        # A USD camera looks down its local -Z, +Y up. Depth is therefore -z_cam.
        depth = -pc[:, 2]
        front = depth > 1e-6
        rec = {"n_corners_in_front": int(front.sum()),
               "depth_range": [round(float(depth.min()), 4), round(float(depth.max()), 4)]}
        if not front.any():
            rec["verdict"] = "entirely BEHIND the camera"
            return rec
        u = K[0, 0] * (pc[front, 0] / depth[front]) + K[0, 2]
        v = K[1, 1] * (-pc[front, 1] / depth[front]) + K[1, 2]
        rec["screen_box"] = [round(float(u.min()), 1), round(float(v.min()), 1),
                             round(float(u.max()), 1), round(float(v.max()), 1)]
        rec["screen_centroid"] = [round(float(u.mean()), 1), round(float(v.mean()), 1)]
        return rec

    proj = {}
    for sname, s in sensors().items():
        try:
            res = getattr(s, "image_height", None), getattr(s, "image_width", None)
            proj[sname] = {"image_hw": [res[0], res[1]],
                           "target": project(report["world_bound"]["default+render"], s)}
            if "control_world_bound" in report:
                proj[sname]["control"] = project(
                    report["control_world_bound"]["default+render"], s)
        except Exception as e:
            proj[sname] = {"error": f"{type(e).__name__}: {e}"}
    report["projection"] = proj
    say("\n[projection] world AABB -> screen (control box validates the maths)")
    for sname, v in proj.items():
        say(f"   {sname}: {json.dumps(v, default=str)}")
    flush()

    # ============================================================================================
    # VERDICT -- mechanical, from the numbers above, so the log states a conclusion rather than
    # leaving it to be read off a table.
    # ============================================================================================
    v = {}
    cab = ab.get("cabinet", {}).get("per_cam", {})
    tab = ab.get("control_table", {}).get("per_cam", {})
    cab_max = max([r["n_changed"] for r in cab.values()], default=None)
    tab_max = max([r["n_changed"] for r in tab.values()], default=None)
    v["cabinet_max_changed_px"] = cab_max
    v["control_max_changed_px"] = tab_max
    v["control_valid"] = bool(tab_max and tab_max > 500)
    if not v["control_valid"]:
        v["verdict"] = ("INCONCLUSIVE -- the positive control did not move either, so hide-and-diff "
                        "is not measuring anything on this run")
    elif cab_max is not None and cab_max < 50:
        v["verdict"] = "CONFIRMED NOT DRAWN -- cabinet contributes ~no pixels while the control does"
    else:
        v["verdict"] = "cabinet IS drawn -- the premise that it is absent is wrong"
    if report.get("ancestor_invisible"):
        v["mechanism_hint"] = f"M2 visibility pruning: {report['ancestor_invisible']}"
    elif report["subtree_summary"]["n_vis_invisible"]:
        v["mechanism_hint"] = "M2 visibility, on the subtree itself (not an ancestor)"
    elif report["world_bound"]["default+render"].get("empty") and not \
            report["world_bound"]["guide"].get("empty"):
        v["mechanism_hint"] = "M3 purpose: the only non-empty world bound is `guide`"
    else:
        v["mechanism_hint"] = ("not M2/M3 -- visibility and purpose are clean and render-purpose "
                               "geometry has a non-empty world bound; look at M1 (material) or M4")
    report["verdict"] = v
    say("\n" + "=" * 90)
    for k, val in v.items():
        say(f"  {k}: {val}")
    say("=" * 90)
    flush()

    try:
        og.clear()
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
