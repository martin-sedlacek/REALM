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
  4. PERTURBATION STILL PERTURBS -- the main object pose must actually vary across resets, and
     differ between members. A "fix" that quietly stopped moving anything would pass 1-3.
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
    "V-SC": "objects",
    "VB-MOBJ": "size",
    "VSB-NOBJ": "identity",
    "Default": "nothing",
}
# "size" exists because VB-MOBJ rescales the main object while RESTORING its pose, so the object's
# xy never changes and an "objects" expectation would be a confident false failure -- but leaving it
# out entirely is worse: VB-MOBJ then passes every check while nothing verifies it did ANYTHING.
# That is the exact hole this map exists to close (measured: VB-MOBJ passed with object spread
# 0.0000 and no expectation, which is indistinguishable from the perturbation being a no-op).
# The observable is the object's AABB extent across resets.
#
# "identity" is for perturbations that REPLACE the main object with a different model (VSB-NOBJ):
# pose is restored and size is clamped, so the only honest observable is which object it now is.
# Compared against the baseline captured BEFORE the first reset, and reported either way.
#
# CAVEAT, so a failure here is read correctly: the replacement category is SAMPLED, so there is a
# small chance of drawing the original (or the same one repeatedly) by luck, which would look like a
# no-op. Treat a lone identity failure as "re-run before believing", not as proof of breakage.
#
# Still absent: SB-VRB. It rewrites env.task_type, the task progression AND the instruction, so its
# honest observable is the instruction/task, not the object -- and the harness does not record that
# yet. It gets every other check; an unknown entry reports and asserts nothing.


def obj_identity(env):
    """(category, model) of this member's main object -- the observable for replacement perturbations."""
    o = env.main_objects[0]
    return (getattr(o, "category", "?"), getattr(o, "model", "?"))


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
    sizes_by_reset = []
    ident_by_reset = []
    # Baseline identity, captured BEFORE any reset so a replacement is detectable even if every
    # reset happens to draw the same model as every other reset.
    ident_base = [obj_identity(e) for e in vec_env.envs]
    print(f"\n[baseline] main-object identity per member: {ident_base}", flush=True)

    for r in range(resets):
        # ---- check 1: the reset must not stop the sim ----------------------------------------
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

        # The expected count depends on the perturbation, and getting this wrong reads as a code bug.
        #
        #   pose-only (VB-POSE, V-VIEW, Default, ...)  -> 0 stop / 0 play. Nothing needs a stopped
        #       sim, so any cycling at all is the per-member disruption this work removed.
        #   NEEDS_STOPPED_SIM (V-SC, VB-MOBJ, VSB-NOBJ, SB-VRB) -> EXACTLY 1 stop / 1 play, no
        #       matter how many members. Those perturbations add or remove objects and genuinely
        #       need a stopped sim, so RealmVectorEnvironment.reset() does ONE cycle for all of them.
        #       Asserting 0 here would flag correct batching as a failure -- which it did, for
        #       VB-MOBJ, until this check learned the difference. Asserting "<= 1" rather than "any"
        #       is the point: N cycles for N members is the original bug, and 1 is the fix.
        expect_cycle = 1 if perturbation in NEEDS_STOPPED_SIM else 0
        print(f"  [1] stop() calls={counts['stop']}  play() calls={counts['play']}  "
              f"(expected {expect_cycle} each for {perturbation})  "
              f"sim playing after reset={was_playing}", flush=True)
        if counts["stop"] != expect_cycle or counts["play"] != expect_cycle:
            failures.append(
                f"reset {r+1}: sim cycled {counts['stop']} stop / {counts['play']} play, "
                f"expected {expect_cycle} each -- more than one cycle means it is being done per "
                f"member instead of once for all members"
            )
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

        # ---- step a little, then checks 3-5 ---------------------------------------------------
        ee_cmds = [e.warmup_ee_cmd() for e in vec_env.envs]
        results = None
        for t in range(steps):
            actions = [e.warmup_action(t, c) for e, c in zip(vec_env.envs, ee_cmds)]
            results = vec_env.step(actions)

        poses = []
        sizes = []
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

            # AABB extent is the observable for rescaling perturbations (VB-MOBJ), which restore
            # pose and so are invisible to the xy check above.
            ext = env.main_objects[0].aabb_extent
            ext = ext.cpu().numpy() if hasattr(ext, "cpu") else np.asarray(ext)
            sizes.append(np.asarray(ext, dtype=float))

            print(f"  [3/5] member {i}: grasping={grasping}  "
                  f"main-object xyz=({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})  "
                  f"aabb=({ext[0]:.4f}, {ext[1]:.4f}, {ext[2]:.4f})", flush=True)

        poses_by_reset.append(np.array(poses))
        sizes_by_reset.append(np.array(sizes))
        ident_by_reset.append([obj_identity(e) for e in vec_env.envs])
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
    siz = np.array(sizes_by_reset)                           # (resets, members, 3)
    for i in range(num_envs):
        obj_spread = obj[:, i, :2].ptp(axis=0)               # xy range across resets, per member
        cam_spread = cam[:, i, :].ptp(axis=0).max() if cam.size else 0.0
        siz_spread = siz[:, i, :].ptp(axis=0).max() if siz.size else 0.0
        print(f"  member {i}: object xy spread=({obj_spread[0]:.4f}, {obj_spread[1]:.4f})  "
              f"camera max spread={cam_spread:.4f}  size spread={siz_spread:.4f}  "
              f"identities={sorted({ident_base[i]} | {ident_by_reset[r][i] for r in range(resets)})}",
              flush=True)
        if resets > 1:
            obj_moved = not np.all(obj_spread < 1e-4)
            cam_moved = cam_spread > 1e-4
            siz_changed = siz_spread > 1e-4
            if expect == "size" and not siz_changed:
                failures.append(f"member {i}: main object never changed size across resets -- "
                                f"{perturbation} is a no-op")
            if expect == "objects" and not obj_moved:
                failures.append(f"member {i}: main object never moved across resets -- "
                                f"{perturbation} is a no-op")
            if expect == "cameras" and not cam_moved:
                failures.append(f"member {i}: external cameras never moved across resets -- "
                                f"{perturbation} is a no-op")
            seen = {ident_base[i]} | {ident_by_reset[r][i] for r in range(resets)}
            ident_changed = len(seen) > 1
            if expect == "identity" and not ident_changed:
                failures.append(f"member {i}: main object identity never changed from "
                                f"{ident_base[i]} across {resets} resets -- {perturbation} is a "
                                f"no-op (or the sampler drew the same model every time; re-run)")
            if expect == "nothing" and (obj_moved or cam_moved or siz_changed or ident_changed):
                failures.append(f"member {i}: {perturbation} changed something it should not "
                                f"(object={obj_moved}, camera={cam_moved}, size={siz_changed}, "
                                f"identity={ident_changed})")

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
        # Spell out what was ACTUALLY asserted for this perturbation. The old wording was fixed text
        # claiming "no sim cycling ... poses vary", which is false for a NEEDS_STOPPED_SIM
        # perturbation (one batched cycle is correct) and false for a rescaling one (VB-MOBJ
        # restores pose; its sizes are what vary). A summary that misdescribes the checks is how a
        # green run turns into false confidence.
        cycle_txt = (f"exactly one batched stop/play per reset (not {num_envs})"
                     if perturbation in NEEDS_STOPPED_SIM else "no sim cycling")
        moved_txt = {
            "objects": "object poses vary per member",
            "cameras": "camera poses vary per member",
            "size": "object size varies per member (pose correctly restored)",
            "identity": "object identity changes",
            "nothing": "nothing moved, as expected for a no-op perturbation",
        }.get(MOVES.get(perturbation), "NO effect assertion for this perturbation -- unverified "
                                       "that it does anything")
        print(f"PASSED -- {perturbation}, {resets} resets x {num_envs} members: {cycle_txt}; "
              f"main object is a contact row in every scene; grasp check live; {moved_txt}; "
              f"nothing left the table.", flush=True)
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
