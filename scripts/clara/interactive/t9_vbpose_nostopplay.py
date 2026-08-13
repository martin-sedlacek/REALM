"""Does a perturbation still work -- and now work VECTORIZED -- without cycling og.sim.stop()/play()?

Background. VB-POSE used to bracket its pose writes in og.sim.stop()/og.sim.play(). That cycle is
GLOBAL while REALM applies perturbations per member inside reset(), so in a vector env member i's
perturbation tore down and rebuilt every other member's scene mid-reset. Measured cost (job 190555,
VB-POSE Vec=4): the main object fell out of the contact view for scenes 1,2,3 -- 18 of 25 rollouts
logged zero environment collisions, never advanced past REACH, and the job still exited 0.

realm/environments/perturbations/vb_pose.py now writes poses on a live sim via _place()
(set_position_orientation + keep_still). This is the cheap confirmation of that change, meant for an
interactive allocation rather than the batch queue: a handful of resets and a few steps each.

Checks 1-5 are the original ones, all of which failed (or were meaningless) before that fix:

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

Checks 6-8 were added 2026-08-13 to audit the perturbations that were declared vector-safe FROM
CODE READING ONLY (Default, V-AUG, V-LIGHT, S-PROP, S-LANG, S-MO, S-AFF, S-INT, SB-NOUN). Checks
1-5 are all about object POSES, so a perturbation whose entire effect is a language string or a USD
light attribute passes them while doing nothing at all, or while doing it to the wrong scene:

  6. INSTRUCTION -- env.instruction must change (or must not), per the INSTRUCTION table. The five
     S-* perturbations move nothing physical, so this is the ONLY check that can tell them apart
     from Default.
  7. SCENE FROZEN -- every object in every member's scene must sit exactly where an UNPERTURBED
     reset puts it. Each phase first does two perturbation-free resets: the first is the baseline,
     the second measures how much the pose readback drifts on its own, which calibrates the
     tolerance instead of guessing it. This is the check that would catch a perturbation writing
     into a SIBLING member's tile, since that sibling's own probe sees objects it never touched
     move.
  8. LIGHTS / TARGET IDENTITY -- per-perturbation probes for the two claims checks 1-7 cannot see:
     V-LIGHT must actually change the intensity of light prims IN ITS OWN SCENE (the old code built
     "/World/scene_0" + <link path> by hand, which is a no-op in scene 0 and cross-scene everywhere
     else), and SB-NOUN must re-designate main_objects[0] to one of THIS member's own distractors
     and name it in the instruction.

A comma-separated --perturbation list runs several perturbations as sequential PHASES against one
scene build, which is what makes auditing nine of them affordable -- the build is ~6 min and a
phase is ~1 min. With a list, warmup runs perturbation-free and each phase sets
active_perturbations itself; with a single perturbation the script behaves exactly as before.
Phases are not isolated from each other, so order the list so that anything with a lasting effect
(SB-NOUN re-points main_objects[0]; V-LIGHT leaves the lights it wrote) comes last.

    MODE=oglite ./scripts/clara/interactive/rr \
        python -u scripts/clara/interactive/t9_vbpose_nostopplay.py --num_envs 4 --resets 3 --steps 15
"""
import argparse
import time
import traceback

import numpy as np

import omnigibson as og
import omnigibson.lazy as lazy
from omnigibson.utils.usd_utils import RigidContactAPI

from realm.environments.env_vector import RealmVectorEnvironment
from realm.environments.perturbations._helpers import SETTLE_STEPS
from realm.eval import SUPPORTED_PERTURBATIONS, SUPPORTED_TASKS
from realm.sim_config import set_sim_config

TABLE_Z_MIN = 0.5   # below this the object has left the table (the z-offset bug parked them at ~0.015)
FROZEN_TOL = 1e-3   # metres; check 7's floor, raised to 4x the measured readback drift if that is larger


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
#   "nothing" -- control: must move neither, so this catches a probe that reports motion where
#                there is none (e.g. reading a pose in the wrong frame). For the perturbations that
#                only rewrite a string or a light attribute, "nothing" is the HONEST expectation and
#                check 6 / check 8 is what proves they are not no-ops.
MOVES = {
    "VB-POSE": "objects",
    "V-VIEW": "cameras",
    "V-SC": "objects",
    "Default": "nothing",
    "V-AUG": "nothing",     # distorts the rendered observation only; never touches the scene
    "V-LIGHT": "nothing",   # writes light intensity/colour; no pose write anywhere
    "S-PROP": "nothing",    # all five S-* only reassign env.instruction
    "S-LANG": "nothing",
    "S-MO": "nothing",
    "S-AFF": "nothing",
    "S-INT": "nothing",
}
# Deliberately absent: VSB-NOBJ, VB-MOBJ, SB-VRB. Those swap an object's model or rescale it while
# RESTORING its pose, so "did it move" is the wrong question for them and asserting it would produce
# a confident false failure. They still get every other check; check 4 just reports and asserts
# nothing, which is what an unknown entry does.
#
# Also deliberately absent: SB-NOUN. It moves NOTHING, but it re-points main_objects[0] at a
# different (stationary) object each reset, so the main-object pose read by check 4 jumps between
# objects. "nothing" would be a confident FALSE FAILURE and "objects" would pass for the wrong
# reason. SB-NOUN is covered by check 7 (nothing in any scene moved) plus check 8's identity probe.

# Must env.instruction differ from the task's base instruction after the perturbation?
# "changed" / "unchanged" / absent = report only, assert nothing.
INSTRUCTION = {
    "Default": "unchanged",
    "V-AUG": "unchanged",
    "V-LIGHT": "unchanged",
    "S-PROP": "changed",
    "S-LANG": "changed",
    "S-MO": "changed",
    "S-AFF": "changed",
    "S-INT": "changed",
    # SB-NOUN is deliberately NOT "changed". It re-points main_objects[0] at a distractor and
    # appends the OLD main object to the distractor list, so the original object is back in the
    # pool from the second reset on and can be drawn again -- at which point the instruction is
    # legitimately identical to the task default. Asserting "changed" would fail on that draw
    # roughly a fifth of the time. Check 8 asserts the sound invariant instead: whatever the
    # instruction says, it must NAME the object that is currently the target.
}

# Perturbations that must leave every object in every scene exactly where an unperturbed reset puts
# it. Only listed for perturbations whose whole effect is non-physical -- for anything that moves,
# rescales or replaces an object this check is meaningless.
SCENE_FROZEN = frozenset({
    "Default", "V-AUG", "V-LIGHT", "S-PROP", "S-LANG", "S-MO", "S-AFF", "S-INT", "SB-NOUN",
})

_LIGHT_PRIM_CACHE = {}   # scene prim path -> [Usd.Prim] carrying inputs:intensity


def camera_poses(env):
    """World poses of this member's external sensors, as a flat list of floats."""
    out = []
    for sensor in env.omnigibson_env.external_sensors.values():
        pos, ori = sensor.get_position_orientation()
        pos = pos.cpu().numpy() if hasattr(pos, "cpu") else np.asarray(pos)
        ori = ori.cpu().numpy() if hasattr(ori, "cpu") else np.asarray(ori)
        out.extend([*np.asarray(pos, dtype=float), *np.asarray(ori, dtype=float)])
    return np.array(out, dtype=float)


def _np(x):
    return x.cpu().numpy() if hasattr(x, "cpu") else np.asarray(x)


def probe_poses(env):
    """{object name: world xyz} for EVERY object in this member's own scene.

    Deliberately not restricted to the task objects: the failure mode this exists for is one
    member's perturbation writing into a sibling's tile, and the sibling's own probe is where that
    shows up -- as scenery moving in a scene whose perturbation touched nothing.
    """
    return {obj.name: np.asarray(_np(obj.get_position_orientation()[0]), dtype=float)
            for obj in env.omnigibson_env.scene.objects}


def probe_lights(env):
    """{prim path: intensity} for every light prim under THIS member's scene prim.

    Walks the member's scene subtree directly instead of reusing v_light's own traversal, so the
    probe cannot inherit the bug it is looking for. The walk is cached per scene because these
    perturbations never add or remove prims.
    """
    scene_root = env.omnigibson_env.scene.prim_path
    prims = _LIGHT_PRIM_CACHE.get(scene_root)
    if prims is None:
        t0 = time.perf_counter()
        root_prim = lazy.omni.isaac.core.utils.prims.get_prim_at_path(scene_root)
        prims = []
        if root_prim is not None and root_prim.IsValid():
            prims = [p for p in lazy.pxr.Usd.PrimRange(root_prim) if p.HasAttribute("inputs:intensity")]
        _LIGHT_PRIM_CACHE[scene_root] = prims
        print(f"  [probe] {scene_root}: {len(prims)} light prim(s) found in "
              f"{time.perf_counter() - t0:.2f}s", flush=True)
    return {p.GetPath().pathString: p.GetAttribute("inputs:intensity").Get() for p in prims}


def probe_identity(env):
    """What this member currently believes the task target is."""
    mo = env.main_objects[0]
    return {
        "main": mo.name,
        "category": getattr(mo, "category", None),
        "scene_idx": mo.scene.idx,
        "prim_path": mo.prim_path,
        "distractors": tuple(d.name for d in env.distractors),
    }


def snapshot(vec_env, with_lights):
    """Per-member (poses, lights, instruction, identity), taken right after a reset."""
    return [{
        "poses": probe_poses(env),
        "lights": probe_lights(env) if with_lights else {},
        "instruction": env.instruction,
        "identity": probe_identity(env),
    } for env in vec_env.envs]


def pose_delta(snap_a, snap_b):
    """(max |dx| over shared objects, name of the worst one) between two per-member pose dicts."""
    worst, worst_name = 0.0, None
    for name, pos in snap_a.items():
        if name not in snap_b:
            return float("inf"), f"{name} (present in one snapshot only)"
        d = float(np.max(np.abs(pos - snap_b[name])))
        if d > worst:
            worst, worst_name = d, name
    return worst, worst_name


def hold_still(vec_env, steps):
    """Step every member with the warmup hold-still action, the same stepping a phase does."""
    ee_cmds = [e.warmup_ee_cmd() for e in vec_env.envs]
    results = None
    for t in range(steps):
        results = vec_env.step([e.warmup_action(t, c) for e, c in zip(vec_env.envs, ee_cmds)])
    return results


def run_phase(vec_env, perturbation, resets, steps):
    """Run every check for one perturbation against an already-built vector env."""
    failures = []
    num_envs = len(vec_env.envs)
    print("\n" + "#" * 70, flush=True)
    print(f"# PHASE {perturbation}", flush=True)
    print("#" * 70, flush=True)

    # ---- check 7 setup: two UNPERTURBED resets ------------------------------------------------
    # The first is the reference every perturbed reset is compared against; the second measures how
    # much a pose readback moves on its own between two identical resets, which is the noise floor
    # the tolerance has to clear. Guessing that tolerance is how a check ends up either vacuous or
    # confidently wrong. The same @steps of stepping happen in between as in a real phase, so the
    # floor covers "does reset() fully undo physics", not just "is a readback repeatable".
    want_lights = perturbation == "V-LIGHT" or perturbation in SCENE_FROZEN
    for env in vec_env.envs:
        env.active_perturbations = []
        # Nothing in reset() restores env.instruction -- a perturbation that rewrites it leaves the
        # rewrite in place forever. That is fine in production (one perturbation per process) but
        # inside a multi-phase run it would carry S-LANG's string into the next phase and make the
        # "must not touch the instruction" check fail on the WRONG phase. Restore the task default.
        env.instruction = env.cfg["instruction"]
    vec_env.reset()
    base = snapshot(vec_env, want_lights)
    hold_still(vec_env, steps)
    vec_env.reset()
    base2 = snapshot(vec_env, want_lights)
    hold_still(vec_env, steps)
    noise = max(pose_delta(base[i]["poses"], base2[i]["poses"])[0] for i in range(num_envs))
    tol = max(FROZEN_TOL, 4 * noise)
    print(f"  [7] unperturbed reset-to-reset drift = {noise:.2e} m  -> tolerance {tol:.2e} m",
          flush=True)

    for env in vec_env.envs:
        env.active_perturbations = [perturbation]

    poses_by_reset = []
    cams_by_reset = []
    prev_ident = [b["identity"] for b in base]   # rolls forward; see the SB-NOUN block below

    for r in range(resets):
        # ---- check 1: the reset must not stop, play or over-step the sim ----------------------
        # og.sim.step() is as global as stop()/play(): it advances EVERY scene, and a member that
        # steps it from inside its own reset advances its siblings while feeding them no action.
        # A vector-env reset is allowed exactly one shared settle loop (SETTLE_STEPS, driven by
        # RealmVectorEnvironment._settle for all members at once), so anything beyond that means a
        # per-member step loop got in -- and the count then scales with num_envs.
        counts = {"stop": 0, "play": 0, "step": 0}
        real_stop, real_play, real_step = og.sim.stop, og.sim.play, og.sim.step

        def counting_stop(*a, **k):
            counts["stop"] += 1
            return real_stop(*a, **k)

        def counting_play(*a, **k):
            counts["play"] += 1
            return real_play(*a, **k)

        def counting_step(*a, **k):
            counts["step"] += 1
            return real_step(*a, **k)

        og.sim.stop, og.sim.play, og.sim.step = counting_stop, counting_play, counting_step
        try:
            vec_env.reset()
        finally:
            og.sim.stop, og.sim.play, og.sim.step = real_stop, real_play, real_step

        was_playing = og.sim.is_playing()
        print(f"\n===== {perturbation} reset {r + 1}/{resets} =====", flush=True)
        print(f"  [1] stop() calls={counts['stop']}  play() calls={counts['play']}  "
              f"step() calls={counts['step']} (<= {SETTLE_STEPS} expected)  "
              f"sim playing after reset={was_playing}", flush=True)
        if counts["stop"] or counts["play"]:
            failures.append(f"reset {r+1}: sim was cycled ({counts['stop']} stop / {counts['play']} play)")
        if counts["step"] > SETTLE_STEPS:
            failures.append(f"reset {r+1}: sim was stepped {counts['step']} times during reset, more "
                            f"than the single shared settle loop of {SETTLE_STEPS} -- a per-member "
                            f"step loop is advancing every sibling scene")
        if not was_playing:
            failures.append(f"reset {r+1}: sim not playing after reset")

        # ---- checks 6-8: read the non-physical state before anything steps --------------------
        snap = snapshot(vec_env, want_lights)
        for i, env in enumerate(vec_env.envs):
            base_instruction = env.cfg["instruction"]
            instruction = snap[i]["instruction"]
            want = INSTRUCTION.get(perturbation)
            drawer_task = env.task_type in ("open_drawer", "close_drawer")
            if perturbation == "SB-NOUN" and drawer_task:
                # On a drawer task SB-NOUN swaps the TARGET DRAWER, not the noun: it draws
                # random.choice(["middle", "top"]) and substitutes it into an instruction that
                # already says "top", so half its draws legitimately leave the string identical.
                # Asserting "changed" here would fail on a coin flip.
                want = None
            print(f"  [6] member {i} instruction: {instruction!r}", flush=True)
            if want == "changed" and instruction == base_instruction:
                failures.append(f"reset {r+1}: member {i} instruction unchanged from the task "
                                f"default {base_instruction!r} -- {perturbation} is a no-op")
            if want == "unchanged" and instruction != base_instruction:
                failures.append(f"reset {r+1}: member {i} instruction changed to {instruction!r} "
                                f"but {perturbation} must not touch it")

            if perturbation in SCENE_FROZEN:
                d, name = pose_delta(base[i]["poses"], snap[i]["poses"])
                print(f"  [7] member {i} (scene {env.omnigibson_env.scene.idx}): max object "
                      f"displacement vs unperturbed reset = {d:.2e} m ({name})", flush=True)
                if d > tol:
                    failures.append(f"reset {r+1}: member {i} object {name!r} moved {d:.2e} m vs an "
                                    f"unperturbed reset, but {perturbation} must move nothing")

            if want_lights:
                # Reported for every phase (a perturbation that has no business touching lights
                # showing up here would be a finding), asserted only for V-LIGHT: it is the one
                # that MUST change them, and it must change the ones in its OWN scene.
                lit = snap[i]["lights"]
                changed = {p: v for p, v in lit.items() if base[i]["lights"].get(p) != v}
                vals = sorted({round(float(v), 3) for v in changed.values() if v is not None})
                print(f"  [8] member {i}: {len(changed)}/{len(lit)} light prim(s) in its own scene "
                      f"changed intensity, values={vals[:4]}", flush=True)
                if perturbation == "V-LIGHT" and not changed:
                    failures.append(f"reset {r+1}: member {i} -- no light prim in ITS OWN scene "
                                    f"changed intensity; V-LIGHT either did nothing or wrote into "
                                    f"another member's scene")

            if perturbation == "SB-NOUN" and drawer_task:
                # The drawer branch re-points the joint, not main_objects[0]; the identity checks
                # below would be confidently wrong here, so report and assert nothing.
                print(f"  [8] member {i}: drawer task -- main object stays "
                      f"{snap[i]['identity']['main']!r}, target drawer follows the instruction",
                      flush=True)
            elif perturbation == "SB-NOUN":
                # Compared against the PREVIOUS reset, not against the unperturbed baseline: the
                # old main object is appended back into env.distractors, so from reset 2 on it can
                # be drawn again and "main == the original object" is a legitimate outcome. What is
                # never legitimate is main not changing at all, because sb_noun pops the new target
                # out of the distractor list BEFORE appending the old one, so the new target can
                # never be the object that was just the target.
                ident, b_ident = snap[i]["identity"], prev_ident[i]
                print(f"  [8] member {i}: main object {b_ident['main']!r} -> {ident['main']!r} "
                      f"(category {ident['category']!r}), {len(ident['distractors'])} distractors",
                      flush=True)
                if ident["main"] == b_ident["main"]:
                    failures.append(f"reset {r+1}: member {i} main object still {ident['main']!r} "
                                    f"-- SB-NOUN did not re-designate the target")
                if ident["scene_idx"] != env.omnigibson_env.scene.idx:
                    failures.append(f"reset {r+1}: member {i} new main object lives in scene "
                                    f"{ident['scene_idx']}, not its own scene "
                                    f"{env.omnigibson_env.scene.idx}")
                # The instruction must NAME the object that is now the target, or the policy is
                # being asked for one object while the success checks watch another.
                noun = (ident["category"] or "").replace("_", " ")
                if noun and noun not in snap[i]["instruction"]:
                    failures.append(f"reset {r+1}: member {i} instruction "
                                    f"{snap[i]['instruction']!r} does not name the new target "
                                    f"{noun!r}")
                if len(ident["distractors"]) != len(b_ident["distractors"]):
                    failures.append(f"reset {r+1}: member {i} distractor count changed "
                                    f"{len(b_ident['distractors'])} -> "
                                    f"{len(ident['distractors'])} -- objects are leaking")

        prev_ident = [s["identity"] for s in snap]

        if perturbation == "V-LIGHT" and num_envs > 1:
            # Each member draws its own intensity from U(20000, 750000), so two members landing on
            # the SAME value means one member's write reached the other's scene (or every member is
            # writing every scene and the last one wins). Two independent uniform draws never
            # collide by accident, so this cannot false-fail.
            per_member = [tuple(sorted(round(float(v), 3) for v in snap[i]["lights"].values()
                                       if v is not None)) for i in range(num_envs)]
            if len(set(per_member)) < num_envs:
                failures.append(f"reset {r+1}: members share an identical set of light intensities "
                                f"-- V-LIGHT is not writing per-scene")

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
        results = hold_still(vec_env, steps)

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
            pos = _np(env.main_objects[0].get_position_orientation()[0])
            poses.append(np.asarray(pos, dtype=float))
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
        print(f"  member {i}: object xy spread=({obj_spread[0]:.4f}, {obj_spread[1]:.4f})  "
              f"camera max spread={cam_spread:.4f}", flush=True)
        if resets > 1:
            obj_moved = not np.all(obj_spread < 1e-4)
            cam_moved = cam_spread > 1e-4
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

    return failures


def main(num_envs, resets, steps, task_id, robot, perturbation):
    perturbations = [p.strip() for p in perturbation.split(",") if p.strip()]
    set_sim_config(robot=robot)
    vec_env = RealmVectorEnvironment(
        num_envs,
        task_cfg_path=f"REALM_DROID10/{SUPPORTED_TASKS[task_id]}/default.yaml",
        # Build with every perturbation in the list so each one's construction-time setup runs
        # (bind_scene_handles() special-cases V-AUG and VSB-NOBJ); the phases then own
        # active_perturbations from here on.
        perturbations=list(perturbations),
        robot=robot,
    )
    if len(perturbations) > 1:
        # Warm up perturbation-free, so no phase inherits a scene another phase perturbed.
        for env in vec_env.envs:
            env.active_perturbations = []
    vec_env.warmup()

    failures = {}
    for p in perturbations:
        try:
            failures[p] = run_phase(vec_env, p, resets, steps)
        except Exception as e:
            # A phase that dies is a result, not a reason to lose the phases that already ran --
            # which is the whole point of running the risky ones last.
            traceback.print_exc()
            failures[p] = [f"phase raised {type(e).__name__}: {e}"]
        print(f"\n----- {p}: {'FAILED, ' + str(len(failures[p])) + ' problem(s)' if failures[p] else 'PASSED'} -----",
              flush=True)
        for f in failures[p]:
            print(f"  - {f}", flush=True)

    print("\n" + "=" * 70, flush=True)
    total = sum(len(v) for v in failures.values())
    for p, fs in failures.items():
        print(f"  {p:10s} {'FAILED (' + str(len(fs)) + ')' if fs else 'PASSED'}", flush=True)
    if total:
        print(f"FAILED -- {total} problem(s) across {len(perturbations)} perturbation(s):", flush=True)
        for p, fs in failures.items():
            for f in fs:
                print(f"  - [{p}] {f}", flush=True)
    else:
        print(f"PASSED -- {resets} resets x {num_envs} members for {len(perturbations)} "
              f"perturbation(s): no sim cycling, main object is a contact row in every scene, "
              f"grasp check live, instructions as declared, scenes frozen where declared, nothing "
              f"left the table.", flush=True)
    print("=" * 70, flush=True)
    return 1 if total else 0


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
                   help="perturbation, or a comma-separated list run as sequential phases against "
                        "one scene build; see MOVES/INSTRUCTION/SCENE_FROZEN for what each is "
                        "asserted to do")
    a = p.parse_args()
    for _p in a.perturbation.split(","):
        assert _p.strip() in SUPPORTED_PERTURBATIONS, f"unknown perturbation {_p!r}"
    raise SystemExit(main(a.num_envs, a.resets, a.steps, a.task_id, a.robot, a.perturbation))
