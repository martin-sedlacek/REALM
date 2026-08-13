"""Does VB-POSE still work -- and now work VECTORIZED -- without cycling og.sim.stop()/play()?

Background. VB-POSE used to bracket its pose writes in og.sim.stop()/og.sim.play(). That cycle is
GLOBAL while REALM applies perturbations per member inside reset(), so in a vector env member i's
perturbation tore down and rebuilt every other member's scene mid-reset. Measured cost (job 190555,
VB-POSE Vec=4): the main object fell out of the contact view for scenes 1,2,3 -- 18 of 25 rollouts
logged zero environment collisions, never advanced past REACH, and the job still exited 0.

realm/environments/perturbations/vb_pose.py now writes poses on a live sim via _place()
(set_position_orientation + keep_still). This is the cheap confirmation of that change, meant for an
interactive allocation rather than the batch queue: a handful of resets and a few steps each.

Five checks, all of which failed (or were meaningless) before the fix:

  1. NO STOP/PLAY -- og.sim.stop/play are wrapped with counters for the duration of the resets.
     This is the direct assertion that the fix is in force, independent of its consequences.
  2. CONTACT ROWS -- every member's main object must be a ROW of its own scene's contact view.
     This is the exact thing that broke: rows are dynamic bodies only, and once the object is
     missing, get_contact_pairs returns set(), is_grasping is permanently False, and the rollout
     silently scores zero rather than failing.
  3. GRASP CHECK LIVE -- check_grasp_condition() must run without raising for every member.
     Before the dtype fix in OG-lite this raised IndexError from an empty float32 index tensor.
  4. PERTURBATION STILL PERTURBS -- whatever the perturbation is supposed to re-randomise must
     actually vary across resets, and members must differ. A "fix" that quietly stopped moving
     anything would pass 1-3. What counts as "varying" is per-perturbation (see MOVES): the main
     object's pose for VB-POSE, the external sensors for V-VIEW, and the planted distractors'
     categories/positions for V-SC, which by design never moves the main object at all.
  5. KEEP_STILL HOLDS -- objects must stay on the table. Teleporting on a live sim leaves the
     pre-teleport velocity attached; without keep_still() the object launches out of its new pose.
     This is the specific regression the live-write path could introduce.

    MODE=oglite ./scripts/clara/interactive/rr \
        python -u scripts/clara/interactive/t9_vbpose_nostopplay.py --num_envs 4 --resets 3 --steps 15
"""
import argparse

import numpy as np

import omnigibson as og
from omnigibson.utils.usd_utils import RigidContactAPI

from realm.environments.env_vector import RealmVectorEnvironment
from realm.environments.perturbations._helpers import NEEDS_STOPPED_SIM
from realm.eval import SUPPORTED_PERTURBATIONS, SUPPORTED_TASKS
from realm.sim_config import set_sim_config

TABLE_Z_MIN = 0.5   # below this the object has left the table (the z-offset bug parked them at ~0.015)


def obj_link_paths(obj):
    """Prim paths of @obj's rigid links -- the identities the contact view indexes rows by."""
    return [link.prim_path for link in obj.links.values()]


def row_report(env):
    """(scene_idx, n_rows, [(path, is_row)]) for this member's main object."""
    scene_idx = env.main_objects[0].scene.idx
    row_map = RigidContactAPI._PATH_TO_ROW_IDX.get(scene_idx, {})
    paths = obj_link_paths(env.main_objects[0])
    return scene_idx, len(row_map), [(p, p in row_map) for p in paths]


# What each perturbation is supposed to MOVE. Checked so that a "fix" which quietly turns the
# perturbation into a no-op fails loudly instead of sailing through the contact-row checks, which it
# otherwise would -- a perturbation that does nothing keeps every object exactly where the scene
# loader put it, which is a perfectly healthy contact view.
#   "objects" -- main object pose must vary across resets
#   "cameras" -- external sensor pose must vary across resets
#   "nothing" -- control: Default should move neither, so this catches a probe that reports motion
#                where there is none (e.g. reading a pose in the wrong frame)
MOVES = {
    "VB-POSE": "objects",
    "V-VIEW": "cameras",
    # NOT "objects". V-SC re-places and re-models the DISTRACTORS -- README: "Randomly spawn new
    # distractors in the scene" -- and passes the main and target objects to
    # get_non_colliding_positions_for_objects as objects_to_skip, so they are deliberately left
    # exactly where the restore put them. Measured with the working V-SC (2 members x 3 resets):
    # main-object xy spread was 0.0000 for both members, i.e. asserting "objects" here is a
    # guaranteed false failure that says nothing about whether V-SC ran.
    "V-SC": "distractors",
    "Default": "nothing",
}
# Deliberately absent: VSB-NOBJ, VB-MOBJ, SB-VRB. Those swap an object's model or rescale it while
# RESTORING its pose, so "did it move" is the wrong question for them and asserting it would produce
# a confident false failure. They still get every other check; check 4 just reports and asserts
# nothing, which is what an unknown entry does.


def planted_objects(env):
    """(name, category, scene-frame xy) for every object REALM planted in this member's scene.

    Everything REALM puts on the table is in env.cfg["objects"] -- main object, target, declared
    distractors, and the extra distractors V-SC's presence makes env_config sample. Read by NAME out
    of the scene registry rather than from env.distractors, because replace_obj swaps an object by
    removing it and adding a new one under the same name, which leaves env.distractors holding
    references to removed prims.

    Scene frame, so the ~25.3 m tile offset between members does not swamp the comparison.
    """
    scene = env.omnigibson_env.scene
    out = []
    for cfg in env.cfg["objects"]:
        obj = scene.object_registry("name", cfg["name"])
        if obj is None:
            out.append((cfg["name"], "<missing>", 0.0, 0.0))
            continue
        pos = obj.get_position_orientation(frame="scene")[0]
        pos = pos.cpu().numpy() if hasattr(pos, "cpu") else np.asarray(pos)
        out.append((cfg["name"], obj.category, round(float(pos[0]), 4), round(float(pos[1]), 4)))
    return tuple(sorted(out))


def camera_poses(env):
    """World poses of this member's external sensors, as a flat list of floats."""
    out = []
    for sensor in env.omnigibson_env.external_sensors.values():
        pos, ori = sensor.get_position_orientation()
        pos = pos.cpu().numpy() if hasattr(pos, "cpu") else np.asarray(pos)
        ori = ori.cpu().numpy() if hasattr(ori, "cpu") else np.asarray(ori)
        out.extend([*np.asarray(pos, dtype=float), *np.asarray(ori, dtype=float)])
    return np.array(out, dtype=float)


def main(num_envs, resets, steps, task_id, robot, perturbation):
    set_sim_config(robot=robot)
    vec_env = RealmVectorEnvironment(
        num_envs,
        task_cfg_path=f"REALM_DROID10/{SUPPORTED_TASKS[task_id]}/default.yaml",
        perturbations=[perturbation],
        robot=robot,
    )
    vec_env.warmup()

    failures = []
    poses_by_reset = []
    cams_by_reset = []
    planted_by_reset = []

    # Check 1's expectation depends on the perturbation. A pose-only one (VB-POSE, V-VIEW) must not
    # cycle the sim AT ALL. One that adds or removes objects (V-SC and the rest of NEEDS_STOPPED_SIM)
    # genuinely cannot avoid a stopped sim, so the contract there is not "zero" but "EXACTLY ONE for
    # the whole reset": RealmVectorEnvironment.reset() hoists the cycle out of the per-member loop, so
    # a count of N is the bug this script exists to catch and 1 is correct.
    expected_cycles = 1 if perturbation in NEEDS_STOPPED_SIM else 0

    for r in range(resets):
        # ---- check 1: the reset must cycle the sim expected_cycles times, no more ------------
        counts = {"stop": 0, "play": 0}
        real_stop, real_play = og.sim.stop, og.sim.play

        def counting_stop(*a, **k):
            counts["stop"] += 1
            return real_stop(*a, **k)

        def counting_play(*a, **k):
            counts["play"] += 1
            return real_play(*a, **k)

        og.sim.stop, og.sim.play = counting_stop, counting_play
        try:
            vec_env.reset()
        finally:
            og.sim.stop, og.sim.play = real_stop, real_play

        was_playing = og.sim.is_playing()
        print(f"\n===== reset {r + 1}/{resets} =====", flush=True)
        print(f"  [1] stop() calls={counts['stop']}  play() calls={counts['play']}  "
              f"(expected {expected_cycles} of each)  sim playing after reset={was_playing}",
              flush=True)
        if counts["stop"] != expected_cycles or counts["play"] != expected_cycles:
            failures.append(f"reset {r+1}: sim cycled {counts['stop']} stop / {counts['play']} play, "
                            f"expected {expected_cycles} of each")
        if not was_playing:
            failures.append(f"reset {r+1}: sim not playing after reset")

        # ---- check 2: main object must be a contact-view ROW in every member ------------------
        for i, env in enumerate(vec_env.envs):
            scene_idx, n_rows, flags = row_report(env)
            missing = [p for p, ok in flags if not ok]
            print(f"  [2] member {i} scene {scene_idx}: {n_rows} rows, "
                  f"main-object links present={sum(ok for _, ok in flags)}/{len(flags)}", flush=True)
            if missing:
                failures.append(f"reset {r+1}: member {i} (scene {scene_idx}) main-object links "
                                f"NOT rows: {missing}")

        # Snapshot for check 4 BEFORE stepping, so it measures what the perturbation produced rather
        # than where physics and the arm subsequently pushed things.
        planted_by_reset.append([planted_objects(e) for e in vec_env.envs])

        # ---- step a little, then checks 3-5 ---------------------------------------------------
        ee_cmds = [e.warmup_ee_cmd() for e in vec_env.envs]
        results = None
        for t in range(steps):
            actions = [e.warmup_action(t, c) for e, c in zip(vec_env.envs, ee_cmds)]
            results = vec_env.step(actions)

        poses = []
        for i, (env, res) in enumerate(zip(vec_env.envs, results)):
            obs = res[0]

            # check 3: the grasp path must be callable, not raise
            try:
                grasping = env.check_grasp_condition(obs)
            except Exception as e:
                grasping = None
                failures.append(f"reset {r+1}: member {i} check_grasp_condition raised "
                                f"{type(e).__name__}: {e}")

            # check 5: object still on the table
            pos = env.main_objects[0].get_position_orientation()[0]
            pos = pos.cpu().numpy() if hasattr(pos, "cpu") else np.asarray(pos)
            poses.append(pos)
            if pos[2] < TABLE_Z_MIN:
                failures.append(f"reset {r+1}: member {i} main object left the table (z={pos[2]:.3f})")

            print(f"  [3/5] member {i}: grasping={grasping}  "
                  f"main-object xyz=({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})", flush=True)

        poses_by_reset.append(np.array(poses))
        cams_by_reset.append(np.array([camera_poses(e) for e in vec_env.envs]))

    # ---- check 4: the perturbation must actually move what it claims to move -----------------
    expect = MOVES.get(perturbation)
    print(f"\n===== [4] does {perturbation} still perturb? (expects to move: {expect}) =====",
          flush=True)
    if expect is None:
        print(f"  no expectation recorded for {perturbation} -- reporting motion, asserting nothing",
              flush=True)

    obj = np.array(poses_by_reset)                           # (resets, members, 3)
    cam = np.array(cams_by_reset)                            # (resets, members, 7*n_sensors)
    for i in range(num_envs):
        obj_spread = obj[:, i, :2].ptp(axis=0)               # xy range across resets, per member
        cam_spread = cam[:, i, :].ptp(axis=0).max() if cam.size else 0.0

        # How the planted objects differ between consecutive resets: how many swapped CATEGORY (a
        # replace_obj that actually picked a new model) and how many MOVED. A V-SC that silently
        # became a no-op scores 0 on both and fails below, which the main-object pose cannot detect.
        recat, removed_xy = 0, 0
        for r in range(1, resets):
            prev = {n: (c, x, y) for n, c, x, y in planted_by_reset[r - 1][i]}
            for n, c, x, y in planted_by_reset[r][i]:
                if n not in prev:
                    continue
                recat += prev[n][0] != c
                removed_xy += abs(prev[n][1] - x) > 1e-4 or abs(prev[n][2] - y) > 1e-4
        print(f"  member {i}: object xy spread=({obj_spread[0]:.4f}, {obj_spread[1]:.4f})  "
              f"camera max spread={cam_spread:.4f}  planted objects: {recat} category change(s), "
              f"{removed_xy} xy change(s) across {max(resets - 1, 0)} reset transition(s)", flush=True)
        if resets > 1:
            obj_moved = not np.all(obj_spread < 1e-4)
            cam_moved = cam_spread > 1e-4
            if expect == "distractors" and recat == 0 and removed_xy == 0:
                failures.append(f"member {i}: no planted object changed category or position across "
                                f"resets -- {perturbation} is a no-op")
            if expect == "objects" and not obj_moved:
                failures.append(f"member {i}: main object never moved across resets -- "
                                f"{perturbation} is a no-op")
            if expect == "cameras" and not cam_moved:
                failures.append(f"member {i}: external cameras never moved across resets -- "
                                f"{perturbation} is a no-op")
            if expect == "nothing" and (obj_moved or cam_moved):
                failures.append(f"member {i}: {perturbation} moved something it should not "
                                f"(object={obj_moved}, camera={cam_moved})")

    # Members must not share a pose. NOTE: post-frame-fix this is nearly free for objects, since
    # members sit in tiles ~25 m apart and read as distinct no matter what -- keep it for the
    # cameras and as a guard against a regression that re-collapses the tiles, but the per-member
    # spread above is what actually shows the perturbation is re-randomising.
    for r in range(resets):
        uniq = len({tuple(np.round(p[:2], 4)) for p in obj[r]})
        print(f"  reset {r+1}: {uniq}/{num_envs} distinct member xy positions", flush=True)
        if uniq < num_envs:
            failures.append(f"reset {r+1}: members share an xy position ({uniq}/{num_envs} distinct)")

    print("\n" + "=" * 70, flush=True)
    if failures:
        print(f"FAILED -- {len(failures)} problem(s):", flush=True)
        for f in failures:
            print(f"  - {f}", flush=True)
    else:
        print(f"PASSED -- {resets} resets x {num_envs} members: {expected_cycles} sim stop/play per "
              f"reset, main object is a contact row in every scene, grasp check live, "
              f"{expect or 'nothing asserted to'} vary, nothing left the table.", flush=True)
    print("=" * 70, flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    # 2 is the cheapest configuration that can still see the bug this script exists for: the
    # failure is "scene 0 is fine, every OTHER scene is wrong", because scene 0's origin is the
    # world origin, so one offset scene is enough to expose a frame error. The scene build is the
    # dominant cost (~12 min at 4 envs, ~6 at 2), so debug at 2 and only go wider to confirm.
    p.add_argument("--num_envs", type=int, default=2)
    p.add_argument("--resets", type=int, default=3)
    p.add_argument("--steps", type=int, default=15)
    p.add_argument("--task_id", type=int, default=0)
    p.add_argument("--robot", type=str, default="DROID_robolab")
    p.add_argument("--perturbation", type=str, default="VB-POSE",
                   help="perturbation to exercise; see MOVES for what each is asserted to move")
    a = p.parse_args()
    raise SystemExit(main(a.num_envs, a.resets, a.steps, a.task_id, a.robot, a.perturbation))
