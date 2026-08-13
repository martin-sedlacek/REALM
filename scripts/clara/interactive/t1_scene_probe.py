"""Thread 1 probe: why do the scene fixes only take in scene 0?

Symptom (docs/vector_env/README.md): with num_envs=4, the breakfast table is pinned and the chair
removed only in scene 0. In scenes 1..N-1 the table is unpinned and the chair still present, so the
task objects end up on the rug.

This dumps, per member, everything needed to discriminate the three standing hypotheses without
guessing:

  H1 object names carry a globally-numbered instance suffix, so 'breakfast_table_uhrsex_0' matches
     only in scene 0                  -> names differ across scenes in the dump
  H2 the batched stop/play changed the fixes' effect
                                      -> names identical AND the fix still fails at num_envs>1,
                                         but succeeds at num_envs=1
  H3 remove_object() mutates scene.objects while apply_scene_fixes_from_cfg iterates it
                                      -> visible as a skipped entry in the per-object trace

It patches apply_scene_fixes_from_cfg so the state is captured immediately before and after the
fix runs for each member, plus once more after full construction. No production code is touched.

    ./scripts/clara/interactive/rr python -u scripts/clara/interactive/t1_scene_probe.py --num_envs 2 --task_id 0
"""
import argparse
import os

import yaml

import omnigibson as og

import realm.environments.env_dynamic as ed
from realm.environments.env_vector import RealmVectorEnvironment
from realm.eval import SUPPORTED_PERTURBATIONS, SUPPORTED_TASKS
from realm.sim_config import set_sim_config

WATCH = ("breakfast_table", "straight_chair")


def snapshot(tag, env, member_idx=None):
    """Print the scene state that decides the hypotheses. Never raises -- a probe must not be the
    thing that kills the run."""
    print(f"\n----- [{tag}] member={member_idx} -----", flush=True)
    try:
        oenv = env.omnigibson_env
        scene = oenv.scene
        names = sorted(o.name for o in scene.objects)
        print(f"  scene.prim_path = {scene.prim_path}")
        print(f"  scene.idx       = {getattr(scene, 'idx', '<none>')}")
        print(f"  id(scene)       = {id(scene)}")
        print(f"  n_objects       = {len(names)}")
        print(f"  scene_model     = {env.scene_model!r}   scene_part = {env.scene_part!r}")

        # What the config says this member should fix/remove, resolved exactly as the real
        # function resolves it.
        spawn_cfg = yaml.load(open(f"{env.config_path}/scenes/scenes.yaml", "r"), Loader=yaml.FullLoader)
        entered = env.scene_model in spawn_cfg and env.scene_part in spawn_cfg.get(env.scene_model, {})
        print(f"  cfg branch taken = {entered}")
        if entered:
            sd = spawn_cfg[env.scene_model][env.scene_part]
            print(f"    to_fix    = {sd.get('to_fix', [])}")
            print(f"    to_remove = {sd.get('to_remove', [])}")

        for pat in WATCH:
            hits = [o for o in scene.objects if pat in o.name]
            print(f"  '{pat}' matches: {[o.name for o in hits] or 'NONE'}")
            for o in hits:
                try:
                    pos = o.get_position_orientation()[0]
                    pos = [round(float(v), 4) for v in pos]
                except Exception as e:  # pose read needs a live view; not always available when stopped
                    pos = f"<unavailable: {type(e).__name__}>"
                has_joint = og.sim.stage.GetPrimAtPath(f"{o.prim_path}/rootJoint").IsValid()
                print(f"      {o.name}: fixed_base={o.fixed_base} rootJoint={has_joint} "
                      f"prim={o.prim_path} pos={pos}")

        # Cross-member name comparison lives or dies on the exact name set, so dump a stable digest
        # plus the head of the list.
        import hashlib
        digest = hashlib.md5("\n".join(names).encode()).hexdigest()[:12]
        print(f"  name_set_md5    = {digest}")
        print(f"  first 8 names   = {names[:8]}")

        # --- z-offset check -------------------------------------------------------------------
        # Scene._load_scene_prim_with_objects parks the scene prim at INITIAL_SCENE_PRIM_Z_OFFSET
        # (-100) for idx != 0, sets every scene-file object's pose in the WORLD frame while it is
        # parked, then moves the prim to z=0 without compensating. That would leave every
        # registered object in scenes 1..N-1 a full 100 m above the scene structure it belongs to.
        # Dump the distribution so the split is visible rather than inferred.
        try:
            sp = scene._scene_prim.get_position_orientation()[0]
            print(f"  scene_prim pos  = {[round(float(v), 4) for v in sp]}")
        except Exception as e:
            print(f"  scene_prim pos  = <unavailable: {type(e).__name__}>")
        zs = []
        for o in scene.objects:
            try:
                zs.append((float(o.get_position_orientation()[0][2]), o.name))
            except Exception:
                pass
        if zs:
            zs.sort()
            hi = [n for z, n in zs if z > 50.0]
            print(f"  object z range  = {zs[0][0]:.3f} .. {zs[-1][0]:.3f}   "
                  f"(n={len(zs)}, above_50m={len(hi)})")
            if hi:
                print(f"    LIFTED objects (z>50): {len(hi)}/{len(zs)}, e.g. {hi[:5]}")

        # Registry removal is not stage removal: scene.remove_object() drops the registry entry and
        # then calls obj.remove() -> delete_or_deactivate_prim(), which may DEACTIVATE rather than
        # delete. A deactivated prim still satisfies IsValid() but does not render, so IsActive() is
        # the test that decides whether the chair is still on screen.
        for gone in ("straight_chair_pmpwwi_0",):
            p = f"{scene.prim_path}/{gone}"
            prim = og.sim.stage.GetPrimAtPath(p)
            valid = prim.IsValid()
            active = prim.IsActive() if valid else False
            renders = valid and active
            print(f"  stage prim {gone}: valid={valid} active={active} "
                  f"-> would_render={renders}")

        # Where REALM's own additions ended up, for comparison with the scene-file objects above.
        for label, objs in (("robot", getattr(env, "robot", None) and [env.robot] or []),
                            ("main_objects", getattr(env, "main_objects", []) or []),
                            ("target_objects", getattr(env, "target_objects", []) or [])):
            for o in objs:
                try:
                    pos = [round(float(v), 4) for v in o.get_position_orientation()[0]]
                    print(f"  {label}: {o.name} pos={pos}")
                except Exception as e:
                    print(f"  {label}: {o.name} pos=<{type(e).__name__}>")
    except Exception as e:
        import traceback
        print(f"  PROBE FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()


def main(num_envs, task_id, perturbation_id, robot, warmup, rendering_mode, frames_dir=None):
    set_sim_config(robot=robot)

    task_cfg_path = f"REALM_DROID10/{SUPPORTED_TASKS[task_id]}/default.yaml"
    perturbation = SUPPORTED_PERTURBATIONS[perturbation_id]

    # Wrap the real fix so we see the scene immediately either side of it, per member, in order.
    original = ed.RealmEnvironmentDynamic.apply_scene_fixes_from_cfg
    state = {"n": 0}

    def traced(self, manage_sim_state=True):
        i = state["n"]
        state["n"] += 1
        print(f"\n===== apply_scene_fixes_from_cfg CALL #{i} "
              f"(manage_sim_state={manage_sim_state}, sim.is_playing={og.sim.is_playing()}) =====",
              flush=True)
        snapshot("BEFORE fixes", self, i)
        result = original(self, manage_sim_state=manage_sim_state)
        snapshot("AFTER fixes", self, i)
        return result

    ed.RealmEnvironmentDynamic.apply_scene_fixes_from_cfg = traced

    vec_env = RealmVectorEnvironment(
        num_envs,
        task_cfg_path=task_cfg_path,
        perturbations=[perturbation],
        robot=robot,
        rendering_mode=rendering_mode,
    )

    print("\n\n########## POST-CONSTRUCTION ##########", flush=True)
    for i, env in enumerate(vec_env.envs):
        snapshot("post-construction", env, i)

    if warmup:
        results = vec_env.warmup()
        print("\n\n########## POST-WARMUP (physics has run) ##########", flush=True)
        for i, env in enumerate(vec_env.envs):
            snapshot("post-warmup", env, i)

        if frames_dir:
            # Same montage as examples/03_vector_first_frames.py, so the output is directly
            # comparable to docs/vector_env/frames/montage_external.png.
            import numpy as np_
            from PIL import Image
            from realm.inference import extract_from_obs
            os.makedirs(frames_dir, exist_ok=True)
            externals, wrists = [], []
            for i, (obs, _tp, _term, _trunc, _info) in enumerate(results):
                base_im, _, _, _, wrist_im, _, _ = extract_from_obs(obs, robot_name=robot)
                for im, bucket in ((base_im, externals), (wrist_im, wrists)):
                    a = np_.asarray(im)
                    if a.dtype.kind == "f":
                        a = (a * 255).clip(0, 255)
                    bucket.append(a.astype(np_.uint8)[..., :3])
                Image.fromarray(externals[-1]).save(os.path.join(frames_dir, f"env{i}_external.png"))
                Image.fromarray(wrists[-1]).save(os.path.join(frames_dir, f"env{i}_wrist.png"))

            def montage(images, cols=2, pad=8, bg=32):
                h, w = images[0].shape[:2]
                rows = int(np_.ceil(len(images) / cols))
                canvas = np_.full((rows * h + (rows + 1) * pad, cols * w + (cols + 1) * pad, 3),
                                  bg, dtype=np_.uint8)
                for i, im in enumerate(images):
                    r, c = divmod(i, cols)
                    y, x = pad + r * (h + pad), pad + c * (w + pad)
                    canvas[y:y + h, x:x + w] = im
                return canvas

            Image.fromarray(montage(externals)).save(os.path.join(frames_dir, "montage_external.png"))
            Image.fromarray(montage(wrists)).save(os.path.join(frames_dir, "montage_wrist.png"))
            print(f"\nWrote frames to {frames_dir}")

    print("\n########## SUMMARY ##########")
    print(f"apply_scene_fixes_from_cfg was called {state['n']} times for {num_envs} members")
    og.shutdown()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--num_envs", type=int, default=2)
    p.add_argument("--task_id", type=int, default=0)
    p.add_argument("--perturbation_id", type=int, default=0)
    p.add_argument("--robot", type=str, default="DROID")
    p.add_argument("--rendering_mode", type=str, default="rt")
    p.add_argument("--frames_dir", type=str, default=None,
                   help="also save per-member frames + montages here (requires --warmup)")
    p.add_argument("--warmup", action="store_true",
                   help="also run the 30-step settle so displacement from physics is visible")
    a = p.parse_args()
    main(a.num_envs, a.task_id, a.perturbation_id, a.robot, a.warmup, a.rendering_mode,
         a.frames_dir)
