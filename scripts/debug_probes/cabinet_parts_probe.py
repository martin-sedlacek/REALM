#!/usr/bin/env python
"""Does each cabinet PART render after the purpose fix, and which way is the asset facing?

CONTEXT. `restore_double_duty_render_purpose` makes `impact_drawer` render again -- measured, task 8:
the `default`+`render` world bound goes from EMPTY to extent (0.512, 0.669, 0.545) and the object
appears in all three cameras. But against the 1.1.1 reference frame
(`logs/scene_sweep/frames_native/og111_ss_t9__baseline__external-external_sensor0.png`) the og391
render is NOT the same picture: 1.1.1 shows the drawer fronts and their handles facing the camera
with the top drawer pulled out, og391 shows a flat side panel. Two different things could produce
that and they have different owners:

  P1  the drawer links do not render at all, and what is visible is only the body carcass
  P2  everything renders, but the asset is ROTATED relative to 1.1.1, so the camera sees its side

So this measures both, per part, instead of inferring one from a picture:

  * PER-PART HIDE-AND-DIFF. Hide the body alone, then each of the five drawer links alone, and count
    the pixels each one owns. A part that renders has a pixel count far above the rt noise floor
    (~200 px here); a part that does not is indistinguishable from it. This settles P1 directly, and
    the per-part screen boxes say where each part actually landed.
  * ORIENTATION. The cabinet root's world quaternion, the config quaternion it is supposed to carry,
    and the angle between them; plus each link's local axes expressed in world, and the drawer's
    open-direction in both world and CAMERA coordinates. If the drawer's travel axis points away
    from the camera rather than toward it, P2 is the answer and the amount is a number.

Neither question is answerable from a frame, and P1 vs P2 decides whether the remaining gap belongs
to the render path at all -- a rotation is a transform defect, already tracked separately on this
asset (`docs/og_deviations/transforms_and_assets.md`, and the xformOp-order assert quoted in
`realm/misc/material_prim_preset_og391.patch`).

Writes nothing. Hides are restored, and `visibility` is the only attribute touched.

    ./scripts/clara/interactive/rr python -u scripts/debug_probes/cabinet_parts_probe.py \
        --task-id 9 --out /logs/cabinet_render --label parts_t9
"""

import argparse
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
    rec = {"n_changed": n, "max_delta": int(d.max())}
    if n:
        ys, xs = np.nonzero(m)
        rec["box"] = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
        rec["centroid"] = [round(float(xs.mean()), 1), round(float(ys.mean()), 1)]
    return rec


def quat_to_mat(q):
    x, y, z, w = [float(v) for v in q]
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--task-id", type=int, default=9)
    ap.add_argument("--pert-id", type=int, default=0)
    ap.add_argument("--robot", default="DROID")
    ap.add_argument("--obj-name", default="drawer")
    ap.add_argument("--cam", default="external_sensor0",
                    help="camera the 1.1.1 reference frame was taken with")
    ap.add_argument("--frames", type=int, default=5)
    ap.add_argument("--settle", type=int, default=4)
    args = ap.parse_args()

    assert not args.out.startswith("/tmp"), "/tmp is node-local -- artifacts go on Lustre"
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
    say(f"[scene] task={task} pert={pert}")
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

    Usd, UsdGeom = lazy.pxr.Usd, lazy.pxr.UsdGeom
    stage = og.sim.stage
    scene = env.omnigibson_env.scene
    obj = scene.object_registry("name", args.obj_name)
    assert obj is not None

    # ============================================================================================
    # ORIENTATION
    # ============================================================================================
    import yaml
    cfg_path = f"/app/realm/config/tasks/REALM_DROID10/{task}/default.yaml"
    cfg_orn = None
    try:
        y = yaml.safe_load(open(cfg_path))
        for mo in y.get("main_objects", []):
            if mo.get("name") == args.obj_name:
                cfg_orn = mo.get("orientation")
    except Exception as e:
        say(f"[cfg] could not read config orientation: {e}")

    pos, orn = obj.get_position_orientation()
    pos = np.asarray(pos.cpu() if hasattr(pos, "cpu") else pos, dtype=np.float64)
    orn = np.asarray(orn.cpu() if hasattr(orn, "cpu") else orn, dtype=np.float64)
    R = quat_to_mat(orn)
    ori = {"world_pos": [round(float(v), 5) for v in pos],
           "world_quat_xyzw": [round(float(v), 6) for v in orn],
           "config_quat": cfg_orn,
           "local_x_in_world": [round(float(v), 4) for v in R[:, 0]],
           "local_y_in_world": [round(float(v), 4) for v in R[:, 1]],
           "local_z_in_world": [round(float(v), 4) for v in R[:, 2]]}
    if cfg_orn:
        qc = np.asarray(cfg_orn, dtype=np.float64)
        qc = qc / np.linalg.norm(qc)
        qo = orn / np.linalg.norm(orn)
        dot = abs(float(np.dot(qc, qo)))
        ori["angle_to_config_quat_deg"] = round(
            float(np.degrees(2 * np.arccos(min(1.0, dot)))), 3)
    report["orientation"] = ori
    say("\n[orientation]")
    for k, v in ori.items():
        say(f"   {k}: {v}")

    # per-link world AABB, so the drawer stack's geometry can be located
    T = UsdGeom.Tokens
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [T.default_, T.render], useExtentsHint=False)

    def link_bound(prim_path):
        p = stage.GetPrimAtPath(prim_path)
        if not p or not p.IsValid():
            return {"invalid": True}
        rng = cache.ComputeWorldBound(p).ComputeAlignedRange()
        if rng.IsEmpty():
            return {"empty": True}
        mn, mx = rng.GetMin(), rng.GetMax()
        return {"centre": [round(float((mx[i] + mn[i]) / 2), 5) for i in range(3)],
                "extent": [round(float(mx[i] - mn[i]), 5) for i in range(3)]}

    parts = {}
    for lname in sorted(getattr(obj, "links", {}) or {}):
        parts[lname] = f"{obj.prim_path}/{lname}"
    report["link_bounds"] = {k: link_bound(v) for k, v in parts.items()}
    say("\n[link world bounds, purpose default+render]")
    for k, v in report["link_bounds"].items():
        say(f"   {k:26s} {v}")
    flush()

    # ============================================================================================
    # camera geometry: where is the camera, and which way does the drawer travel relative to it
    # ============================================================================================
    def find_sensor():
        try:
            from omnigibson.sensors import VisionSensor
            for n, s in VisionSensor.SENSORS.items():
                if args.cam in str(n):
                    return n, s
        except Exception:
            pass
        return None, None

    sname, sensor = find_sensor()
    if sensor is not None:
        sp, sq = sensor.get_position_orientation()
        sp = np.asarray(sp.cpu() if hasattr(sp, "cpu") else sp, dtype=np.float64)
        sq = np.asarray(sq.cpu() if hasattr(sq, "cpu") else sq, dtype=np.float64)
        SR = quat_to_mat(sq)
        fwd = -SR[:, 2]                      # a USD camera looks down its local -Z
        to_obj = pos - sp
        to_obj_n = to_obj / (np.linalg.norm(to_obj) + 1e-12)
        cam = {"sensor": str(sname), "cam_pos": [round(float(v), 5) for v in sp],
               "cam_forward_world": [round(float(v), 4) for v in fwd],
               "cam_to_object_unit": [round(float(v), 4) for v in to_obj_n],
               "distance": round(float(np.linalg.norm(to_obj)), 4)}
        # Which of the cabinet's local axes faces the camera? The drawer face is whichever local
        # axis has the most negative dot with the camera-to-object direction (i.e. points back at
        # the camera). Comparing that against 1.1.1 is what a rotation defect shows up in.
        for i, nm in enumerate("xyz"):
            cam[f"dot_local_{nm}_with_cam_to_obj"] = round(float(np.dot(R[:, i], to_obj_n)), 4)
        report["camera_geometry"] = cam
        say("\n[camera geometry]")
        for k, v in cam.items():
            say(f"   {k}: {v}")
    flush()

    # ============================================================================================
    # PER-PART HIDE-AND-DIFF
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
    target_cam = next((c for c in cams if args.cam in c), cams[0] if cams else None)
    report["cameras"] = cams
    report["target_cam"] = target_cam
    say(f"\n[cams] {cams}\n[target_cam] {target_cam}")

    def measure(name):
        step_n(args.settle)
        seq = step_n(args.frames)
        out = {}
        for cam in cams:
            imgs = [s[cam] for s in seq if cam in s]
            if imgs:
                out[cam] = np.median(np.stack(imgs, 0), axis=0).astype(np.uint8)
        if target_cam in out:
            try:
                from PIL import Image
                Image.fromarray(out[target_cam]).save(
                    os.path.join(args.out, f"{args.label}__{name}.png"))
            except Exception:
                pass
        return out

    def set_prim_visible(prim_path, vis):
        prim = stage.GetPrimAtPath(prim_path)
        with og.sim.editing_usd():
            img = UsdGeom.Imageable(prim)
            if vis:
                img.MakeVisible()
            else:
                img.MakeInvisible()
        return str(UsdGeom.Imageable(stage.GetPrimAtPath(prim_path)).ComputeVisibility())

    base = measure("all_visible")
    report["per_part"] = {}
    say("\n[per-part hide-and-diff]  (rt noise floor here is ~200 px)")
    for lname, lpath in parts.items():
        after = set_prim_visible(lpath, False)
        hid = measure(f"hide_{lname}")
        rec = {"prim_path": lpath, "vis_after_hide": after, "per_cam": {}}
        for cam in base:
            if cam in hid:
                rec["per_cam"][cam] = changed(base[cam], hid[cam])
        set_prim_visible(lpath, True)
        rec["renders"] = bool(
            target_cam in rec["per_cam"] and rec["per_cam"][target_cam]["n_changed"] > 500)
        report["per_part"][lname] = rec
        tc = rec["per_cam"].get(target_cam, {})
        say(f"   {lname:26s} {target_cam.split('.')[-1]:18s} "
            f"n_changed={tc.get('n_changed', 0):7d}  box={tc.get('box')}  "
            f"{'RENDERS' if rec['renders'] else 'no pixels'}")
        flush()

    n_render = sum(1 for v in report["per_part"].values() if v["renders"])
    report["verdict"] = {
        "n_links": len(parts), "n_links_rendering": n_render,
        "links_rendering": [k for k, v in report["per_part"].items() if v["renders"]],
        "links_not_rendering": [k for k, v in report["per_part"].items() if not v["renders"]],
    }
    say("\n" + "=" * 92)
    for k, v in report["verdict"].items():
        say(f"  {k}: {v}")
    if "angle_to_config_quat_deg" in ori:
        say(f"  root deviation from config quaternion: {ori['angle_to_config_quat_deg']} deg")
    say("=" * 92)
    flush()

    try:
        og.clear()
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
