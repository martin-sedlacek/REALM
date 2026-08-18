#!/usr/bin/env python
"""Validate `restore_double_duty_render_purpose` to the standard the link8 adapter fix was held to.

That standard is three things, and the third is the one that matters: BEFORE/AFTER FRAMES, a
READBACK proving the intended attribute actually moved, and a BITWISE-IDENTICAL ROLLOUT proving
physics did not. A render fix that changes physics is not acceptable, and "it only writes a render
attribute" is an argument, not a measurement.

HOW THE A/B IS DONE WITHOUT A PRODUCTION FLAG. `--fix off` monkeypatches
`SceneSetupMixin.restore_double_duty_render_purpose` to a no-op before the env is built. Same
binary, same code path, one call disabled -- so the comparison cannot be contaminated by a config
knob that also does something else, and no kill switch has to exist in shipped code.

WHAT EACH ARM MUST SHOW

    --fix off, task 8   world bound `default+render` EMPTY; cabinet absent from frames
    --fix on,  task 8   world bound `default+render` non-empty; cabinet present in frames
    both                IDENTICAL rollout fingerprint

The rollout fingerprint is the sha256 of the concatenated per-step robot joint positions over
`--steps` deterministic steps under a fixed hold action, plus the same for the cabinet's joints. If
those two hashes match across the arms, the fix did not move physics -- and if they do not, the fix
is wrong no matter how good the frames look.

NO-OP CHECK. Run it on a task whose objects are all BEHAVIOR assets (`--task-id 3`, rotate_mug).
Those have dedicated `visuals/` geometry, so every link is skipped and the fix must report
`restored 0`. Both arms must then also produce identical frames, not merely identical physics --
that is what proves the fix cannot un-hide a real collider on a normal asset.

    ./scripts/clara/interactive/rr python -u scripts/debug_probes/cabinet_fix_verify.py \
        --task-id 8 --fix on  --out /logs/cabinet_render --label v_t8_on
"""

import argparse
import hashlib
import json
import os
import sys
import traceback

import numpy as np


def luma(a):
    a = np.asarray(a).astype(np.float64)
    return 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--task-id", type=int, default=8)
    ap.add_argument("--pert-id", type=int, default=0)
    ap.add_argument("--robot", default="DROID")
    ap.add_argument("--obj-name", default="drawer")
    ap.add_argument("--fix", choices=("on", "off"), default="on")
    ap.add_argument("--frames", type=int, default=5)
    ap.add_argument("--settle", type=int, default=4)
    ap.add_argument("--steps", type=int, default=30, help="deterministic steps in the fingerprint")
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
    from realm.environments.scene_setup import SceneSetupMixin

    # --- the A/B switch: disable the fix by replacing the method, before any env is built ---
    if args.fix == "off":
        def _noop(self):
            print("[render_purpose] DISABLED by cabinet_fix_verify.py --fix off")
            return [], []
        SceneSetupMixin.restore_double_duty_render_purpose = _noop
    report["fix_arm"] = args.fix
    say(f"[fix] arm = {args.fix}")

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

    Usd, UsdGeom = lazy.pxr.Usd, lazy.pxr.UsdGeom
    stage = og.sim.stage
    scene = env.omnigibson_env.scene

    # ---- readback: did `purpose` actually move, and is the render-purpose bound non-empty? ----
    obj = scene.object_registry("name", args.obj_name)
    if obj is not None:
        root = stage.GetPrimAtPath(obj.prim_path)
        T = UsdGeom.Tokens

        def wb(purposes):
            cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), purposes, useExtentsHint=False)
            rng = cache.ComputeWorldBound(root).ComputeAlignedRange()
            if rng.IsEmpty():
                return {"empty": True}
            mn, mx = rng.GetMin(), rng.GetMax()
            return {"empty": False,
                    "extent": [round(float(mx[i] - mn[i]), 5) for i in range(3)],
                    "centre": [round(float((mx[i] + mn[i]) / 2), 5) for i in range(3)]}

        purposes = {}
        for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate)):
            if prim.GetTypeName() not in ("Mesh", "Cube", "Sphere", "Cone", "Cylinder"):
                continue
            k = str(UsdGeom.Imageable(prim).ComputePurpose())
            purposes[k] = purposes.get(k, 0) + 1
        report["target_readback"] = {
            "prim_path": obj.prim_path,
            "world_bound_default_render": wb([T.default_, T.render]),
            "world_bound_guide": wb([T.guide]),
            "geom_purpose_histogram": purposes,
        }
        say(f"[readback] default+render bound: {report['target_readback']['world_bound_default_render']}")
        say(f"[readback] guide bound         : {report['target_readback']['world_bound_guide']}")
        say(f"[readback] geom purposes       : {purposes}")
    flush()

    # ---- frames ----
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
    seq = step_n(args.frames)
    report["frames"] = {}
    for cam in cams:
        imgs = [s[cam] for s in seq if cam in s]
        if not imgs:
            continue
        med = np.median(np.stack(imgs, 0), axis=0).astype(np.uint8)
        png = os.path.join(args.out, f"{args.label}__{cam.replace('.', '-')}.png")
        try:
            from PIL import Image
            Image.fromarray(med).save(png)
        except Exception as e:
            say(f"   [png] {type(e).__name__}: {e}")
        report["frames"][cam] = {"png": png, "mean": round(float(luma(med).mean()), 3),
                                 "sha256": hashlib.sha256(med.tobytes()).hexdigest()}
        say(f"[frame] {cam}: mean={report['frames'][cam]['mean']} "
            f"sha={report['frames'][cam]['sha256'][:16]}")
    flush()

    # ---- the rollout fingerprint: this is what must be identical across the arms ----
    # Fresh reset so the trajectory starts from the same state in both arms, then a fixed action for
    # `--steps` steps, accumulating raw float64 bytes. No rendering enters this -- only physics.
    obs, _ = env.reset()
    obs, _r, _t, _tr, _i = env.warmup(obs)
    robot = scene.robots[0]
    traj_robot, traj_obj = [], []
    for _ in range(args.steps):
        obs, _r2, _t2, _tr2, _i2 = env.step(hold)
        q = robot.get_joint_positions()
        traj_robot.append(np.ascontiguousarray(
            (q.cpu().numpy() if hasattr(q, "cpu") else np.asarray(q)), dtype=np.float64))
        if obj is not None:
            try:
                qo = obj.get_joint_positions()
                traj_obj.append(np.ascontiguousarray(
                    (qo.cpu().numpy() if hasattr(qo, "cpu") else np.asarray(qo)), dtype=np.float64))
            except Exception:
                pass

    def fp(traj):
        if not traj:
            return None
        arr = np.stack(traj, 0)
        return {"sha256": hashlib.sha256(arr.tobytes()).hexdigest(),
                "shape": list(arr.shape),
                "sum": float(arr.sum()),
                "first": [round(float(v), 12) for v in arr[0].ravel()[:8]],
                "last": [round(float(v), 12) for v in arr[-1].ravel()[:8]]}

    report["rollout_fingerprint"] = {"steps": args.steps, "robot_joints": fp(traj_robot),
                                     "object_joints": fp(traj_obj)}
    say("\n" + "=" * 92)
    say(f"  arm={args.fix}  task={task}")
    for k in ("robot_joints", "object_joints"):
        v = report["rollout_fingerprint"][k]
        if v:
            say(f"  {k}: sha256={v['sha256']}  shape={v['shape']}  sum={v['sum']!r}")
    say("=" * 92)
    flush()

    try:
        og.clear()
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
