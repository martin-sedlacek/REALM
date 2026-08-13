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

Check 9 was added 2026-08-13 for SB-VRB, the last perturbation to be run vectorized. It is the only
one whose whole effect is on the TASK rather than on the scene or the instruction: it redraws
env.task_type, swaps env.task_progression for the new verb's rubric, and rebuilds the instruction
around it. Checks 1-8 are blind to all three, so without check 9 a totally inert SB-VRB passes:

  9. TASK REWRITTEN -- env.task_type must change to a verb COMPATIBILITY_MATRIX allows, the
     progression must be the new verb's rubric, the instruction must open with the new verb, and
     every member's task_progression must be its OWN dict. That last one is not hypothetical: an
     env_base.py version of it (the module-level rubric assigned rather than deepcopied) is what
     inflated a 25-rollout vectorized eval to SR 0.960 with the block never grasped. On a put/stack
     draw it also checks the target object -- SB-VRB is the only perturbation that ADDS an object
     ("receiver") to a member whose siblings may not have one, so it is the only one where "the
     target is a live, initialized object in THIS member's own scene" can fail.

A comma-separated --perturbation list runs several perturbations as sequential PHASES against one
scene build, which is what makes auditing nine of them affordable -- the build is ~6 min and a
phase is ~1 min. With a list, warmup runs perturbation-free and each phase sets
active_perturbations itself; with a single perturbation the script behaves exactly as before.
Phases are not isolated from each other, so order the list so that anything with a lasting effect
(SB-NOUN re-points main_objects[0]; V-LIGHT leaves the lights it wrote) comes last.

    MODE=oglite ./scripts/clara/interactive/rr \
        python -u scripts/clara/interactive/t9_vbpose_nostopplay.py --num_envs 4 --resets 3 --steps 15

Measured 2026-08-13, Vec=2, task 0, DROID_robolab, 3 resets per perturbation: Default, S-LANG,
S-PROP, S-MO, S-AFF, S-INT, V-LIGHT and SB-NOUN all PASS. V-AUG FAILED with KeyError('DROID')
from inside reset() -- see perturbations/v_aug.py -- and passes after that fix. Confirmed again at
Vec=4 for Default, S-PROP, V-AUG, V-LIGHT and SB-NOUN. Check 7's unperturbed reset-to-reset drift
measured 1e-5..1e-4 m, so the 1e-3 m gate is ~10x the noise and a cross-scene pose write (tiles
are ~25 m apart) is not remotely close to it.

Check 1's step counter reads exactly num_envs on every reset here, and that is not a perturbation:
og.Environment.reset(get_obs=True) does one og.sim.step() plus three og.sim.render() calls, and
reset_pre_perturbation() runs it per member -- so a vector reset issues N global steps and 3N
global renders before any perturbation runs. It is well under SETTLE_STEPS and check 7 shows it
moves nothing (<= 4e-5 m at Vec=4), but it is O(N) global work in the reset path.
"""
import argparse
import copy
import time
import traceback

import numpy as np

import omnigibson as og
import omnigibson.lazy as lazy
from omnigibson.utils.usd_utils import RigidContactAPI

from realm.environments.env_base import TASK_PROGRESS_RUBRICS
from realm.environments.env_vector import RealmVectorEnvironment
from realm.environments.perturbations._helpers import NEEDS_STOPPED_SIM, SETTLE_STEPS
# The module-level objects themselves, not a fresh load_task_progressions() copy: check 9 asserts
# that no member's task_progression IS one of these, and a private copy would make that vacuous.
from realm.environments.perturbations.sb_vrb import COMPATIBILITY_MATRIX, TASK_PROGRESSIONS
from realm.eval import SUPPORTED_PERTURBATIONS, SUPPORTED_TASKS
from realm.sim_config import set_sim_config

# Which verb every task_type must open its instruction with, so that "the instruction and the task
# type agree" can be checked instead of assumed. sb_vrb.py builds the string from the same mapping.
TASK_VERB_PHRASE = {
    "pick": "pick up", "put": "put", "stack": "stack", "rotate": "rotate",
    "push": "push", "open": "open", "close": "close",
}

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
    # NOT "objects". V-SC passes the main and target objects to
    # get_non_colliding_positions_for_objects as objects_to_skip and re-randomises only the
    # DISTRACTORS, so the main object deliberately stays put -- measured main-object xy spread with
    # a WORKING V-SC was exactly 0.0000, i.e. "objects" is a guaranteed false failure here.
    "V-SC": "distractors",
    # VB-MOBJ rescales the main object while restoring its pose, so pose sees nothing; the AABB
    # extent does (measured 0.019-0.031 spread against 0.0000 for Default).
    "VB-MOBJ": "size",
    # VSB-NOBJ REPLACES the object, so neither pose nor size sees it -- identity does. The
    # replacement is sampled, so a lone identity failure means "re-run before believing".
    "VSB-NOBJ": "identity",
    "Default": "nothing",
    "V-AUG": "nothing",     # distorts the rendered observation only; never touches the scene
    "V-LIGHT": "nothing",   # writes light intensity/colour; no pose write anywhere
    "S-PROP": "nothing",    # all five S-* only reassign env.instruction
    "S-LANG": "nothing",
    "S-MO": "nothing",
    "S-AFF": "nothing",
    "S-INT": "nothing",
}
# Deliberately absent: SB-VRB. It rewrites task_type, the task progression and the instruction, so
# its honest observable is the task rather than any object property. Check 9 below is that
# observable; an unknown entry here reports and asserts nothing. Note that the instruction is NOT a
# sound observable for it either: the new verb is drawn from COMPATIBILITY_MATRIX[current verb],
# which excludes the CURRENT verb but not the task's ORIGINAL one, so from reset 2 on the draw can
# come back to it -- on pick_spoon, pick -> rotate -> pick regenerates the task's own instruction
# verbatim. INSTRUCTION["SB-VRB"] = "changed" would therefore fail on a coin flip.
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


def _np(x):
    """Whatever OmniGibson handed back (torch on GPU, torch on CPU, list) as a numpy array."""
    return np.asarray(x.cpu().numpy() if hasattr(x, "cpu") else x, dtype=float)


def obj_identity(env):
    """(category, model) of this member's main object -- the observable for replacement perturbations."""
    o = env.main_objects[0]
    return (getattr(o, "category", "?"), getattr(o, "model", "?"))


def planted_identities(env):
    """(name, category) for every object this member's scene currently holds.

    The observable for V-SC, which re-models the DISTRACTORS and deliberately leaves the main and
    target objects alone (it passes them as objects_to_skip), so main-object pose says nothing.
    """
    return sorted((o.name, getattr(o, "category", "?")) for o in env.omnigibson_env.scene.objects)


def camera_poses(env):
    """World poses of this member's external sensors, as a flat list of floats."""
    out = []
    for sensor in env.omnigibson_env.external_sensors.values():
        pos, ori = sensor.get_position_orientation()
        out.extend([*_np(pos), *_np(ori)])
    return np.array(out, dtype=float)


def probe_poses(env):
    """{object name: world xyz} for EVERY object in this member's own scene.

    Deliberately not restricted to the task objects: the failure mode this exists for is one
    member's perturbation writing into a sibling's tile, and the sibling's own probe is where that
    shows up -- as scenery moving in a scene whose perturbation touched nothing.
    """
    return {obj.name: _np(obj.get_position_orientation()[0])
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


def counting_reset(vec_env):
    """vec_env.reset() with og.sim.stop/play/step counted. Returns {"stop","play","step"}.

    og.sim.step() is as global as stop()/play(): it advances EVERY scene, and a member that steps it
    from inside its own reset advances its siblings while feeding them no action. The counts are
    check 1.
    """
    counts = {"stop": 0, "play": 0, "step": 0}
    real = {k: getattr(og.sim, k) for k in counts}

    def wrap(name):
        def counted(*a, **k):
            counts[name] += 1
            return real[name](*a, **k)
        return counted

    for k in counts:
        setattr(og.sim, k, wrap(k))
    try:
        vec_env.reset()
    finally:
        for k, fn in real.items():
            setattr(og.sim, k, fn)
    return counts


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
        # Same argument for the task itself, which only SB-VRB rewrites: nothing in reset() restores
        # task_type or task_progression either, so without this the "unperturbed" baseline below
        # would run under the PREVIOUS phase's verb, and check 9's first comparison would be against
        # a leftover rather than against the task. deepcopy for the reason sb_vrb.py and env_base.py
        # both deepcopy: recompute_task_progression mutates the rubric dict in place, so handing out
        # the module-level object shares progression between members.
        env.task_type = env.cfg["task_type"]
        env.task_progression = copy.deepcopy(TASK_PROGRESSIONS[env.task_type])
    counting_reset(vec_env)
    base = snapshot(vec_env, want_lights)
    hold_still(vec_env, steps)
    base_counts = counting_reset(vec_env)
    base2 = snapshot(vec_env, want_lights)
    hold_still(vec_env, steps)
    noise = max(pose_delta(base[i]["poses"], base2[i]["poses"])[0] for i in range(num_envs))
    tol = max(FROZEN_TOL, 4 * noise)
    print(f"  [7] unperturbed reset-to-reset drift = {noise:.2e} m  -> tolerance {tol:.2e} m",
          flush=True)

    # Check 1's step budget, MEASURED rather than assumed, for the same reason as check 7's
    # tolerance. An unperturbed reset already issues og.sim.step() once per member, inside
    # og.Environment.reset(get_obs=True) -- that is the floor, and it is what @base_counts reads.
    # On top of it a perturbed reset may add:
    #   + SETTLE_STEPS   the ONE shared settle loop RealmVectorEnvironment._settle runs for all
    #                    members when any of them asked to settle
    #   + num_envs       the single og.sim.step() inside each member's deferred _post_play block
    #                    (V-SC, VB-MOBJ, VSB-NOBJ and SB-VRB each register one)
    # Anything past that is a PER-MEMBER settle loop, which is the regression this check exists to
    # catch and which costs SETTLE_STEPS * num_envs -- 60 at 2 members against a budget of 34, so
    # the bound stays comfortably discriminating. The old fixed `> SETTLE_STEPS` bound predated the
    # add/replace perturbations being run under this harness and would have false-failed all four
    # of them by ~4 steps.
    step_budget = base_counts["step"] + SETTLE_STEPS + num_envs
    # Check 1's stop/play expectation, which is NOT "never" for every perturbation. Adding or
    # removing an object requires a stopped simulator, so RealmVectorEnvironment.reset() gives the
    # perturbations in NEEDS_STOPPED_SIM exactly ONE shared cycle covering all members -- that is
    # the fix, not the bug. The bug was a cycle PER MEMBER (VB-POSE, job 190555), and for a
    # pose-only perturbation any cycle at all. Asserting an exact count catches both, where the
    # earlier `if counts["stop"] or counts["play"]` false-failed all four add/replace perturbations
    # and the earlier "expected 1 of each" printout asserted nothing at all.
    want_cycles = 1 if perturbation in NEEDS_STOPPED_SIM else 0
    print(f"  [1] unperturbed reset issues {base_counts['step']} og.sim.step() call(s) "
          f"-> perturbed-reset budget {step_budget}; expecting {want_cycles} stop/play cycle(s)",
          flush=True)

    for env in vec_env.envs:
        env.active_perturbations = [perturbation]

    poses_by_reset = []
    cams_by_reset = []
    sizes_by_reset = []
    ident_by_reset = []
    planted_by_reset = []
    # Identity baseline BEFORE the first reset, so a replacement is detectable even if every reset
    # happens to draw the same model as every other reset.
    ident_base = [obj_identity(e) for e in vec_env.envs]
    print(f"\n[baseline] main-object identity per member: {ident_base}", flush=True)
    prev_ident = [b["identity"] for b in base]   # rolls forward; see the SB-NOUN block below
    # Same, for check 9: SB-VRB draws its new verb from COMPATIBILITY_MATRIX[the CURRENT verb], so
    # the comparison has to be against the verb this member had going into this reset, not against
    # the task's original one.
    prev_task_type = [e.task_type for e in vec_env.envs]

    for r in range(resets):
        # ---- check 1: the reset must not stop, play or over-step the sim ----------------------
        counts = counting_reset(vec_env)

        was_playing = og.sim.is_playing()
        print(f"\n===== {perturbation} reset {r + 1}/{resets} =====", flush=True)
        print(f"  [1] stop() calls={counts['stop']}  play() calls={counts['play']} "
              f"({want_cycles} of each expected)  "
              f"step() calls={counts['step']} (<= {step_budget} expected)  "
              f"sim playing after reset={was_playing}", flush=True)
        if counts["stop"] != want_cycles or counts["play"] != want_cycles:
            failures.append(f"reset {r+1}: sim was cycled {counts['stop']}x stop / "
                            f"{counts['play']}x play, expected {want_cycles} of each")
        if counts["step"] > step_budget:
            failures.append(f"reset {r+1}: sim was stepped {counts['step']} times during reset, more "
                            f"than the budget of {step_budget} (an unperturbed reset's "
                            f"{base_counts['step']} + one shared settle loop of {SETTLE_STEPS} + one "
                            f"deferred post-play step per member) -- a per-member step loop is "
                            f"advancing every sibling scene")
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

            # ---- check 9: SB-VRB rewrote the TASK, and consistently ------------------------
            if perturbation == "SB-VRB":
                prev_tt, tt = prev_task_type[i], env.task_type
                scene = env.omnigibson_env.scene
                tgt = env.target_objects[0] if env.target_objects else None
                print(f"  [9] member {i} (scene {scene.idx}): task_type {prev_tt!r} -> {tt!r}, "
                      f"progression {list(env.task_progression)}, target="
                      f"{None if tgt is None else (tgt.name, getattr(tgt, 'category', '?'), tgt.scene.idx)}, "
                      f"{len(scene.objects)} objects in scene", flush=True)

                # The draw excludes the current verb (no key of COMPATIBILITY_MATRIX lists itself),
                # so "unchanged" is not a random outcome -- it means sb_vrb never ran for this member.
                if tt == prev_tt:
                    failures.append(f"reset {r+1}: member {i} task_type still {tt!r} -- SB-VRB is a "
                                    f"no-op for this member")
                if tt not in COMPATIBILITY_MATRIX.get(prev_tt, []):
                    failures.append(f"reset {r+1}: member {i} task_type {prev_tt!r} -> {tt!r}, which "
                                    f"COMPATIBILITY_MATRIX does not allow")
                # The progression rubric must follow the verb. If it does not, the rollout is scored
                # against the OLD skill's stages while the policy is asked for the new one.
                if list(env.task_progression) != list(TASK_PROGRESSIONS.get(tt, {})):
                    failures.append(f"reset {r+1}: member {i} task_progression "
                                    f"{list(env.task_progression)} is not the rubric for {tt!r} "
                                    f"({list(TASK_PROGRESSIONS.get(tt, {}))})")
                # The deepcopy in sb_vrb.py. Assigning TASK_PROGRESSIONS[verb] directly would give
                # every member that drew the same verb ONE shared dict, and recompute_task_progression
                # mutates it in place -- the exact defect that inflated a 25-rollout vectorized eval
                # to SR 0.960 with the object never grasped.
                if env.task_progression is TASK_PROGRESSIONS.get(tt) or \
                        env.task_progression is TASK_PROGRESS_RUBRICS.get(tt):
                    failures.append(f"reset {r+1}: member {i} task_progression IS the module-level "
                                    f"rubric object -- it is shared with every other member that "
                                    f"draws {tt!r}")
                # The policy must be asked for the verb the progression scores.
                phrase = TASK_VERB_PHRASE.get(tt)
                if phrase and not snap[i]["instruction"].startswith(phrase):
                    failures.append(f"reset {r+1}: member {i} instruction "
                                    f"{snap[i]['instruction']!r} does not open with the new verb "
                                    f"{phrase!r} (task_type {tt!r})")
                if tt in ("put", "stack"):
                    # The vector-specific half. On a task whose YAML has no target (pick_spoon),
                    # SB-VRB ADDS a "receiver" -- an object present in this member's scene and
                    # absent from a sibling's. Every step of that is where a name-keyed global
                    # lookup can hand back the wrong scene's object: env.target_objects[0] must be
                    # THIS scene's live, initialized object.
                    if tgt is None:
                        failures.append(f"reset {r+1}: member {i} task_type {tt!r} needs a target "
                                        f"object and has none")
                    else:
                        if tgt.scene.idx != scene.idx:
                            failures.append(f"reset {r+1}: member {i} target {tgt.name!r} lives in "
                                            f"scene {tgt.scene.idx}, not its own scene {scene.idx}")
                        if scene.object_registry("name", tgt.name) is not tgt:
                            failures.append(f"reset {r+1}: member {i} target {tgt.name!r} is not the "
                                            f"object its own scene registry holds under that name -- "
                                            f"a stale handle survived replace_obj")
                        if not tgt.initialized:
                            failures.append(f"reset {r+1}: member {i} target {tgt.name!r} was never "
                                            f"initialized -- it was evicted from the sim's init "
                                            f"queue and the repair missed it")
                # ...and it has to be somewhere the robot can reach. Read in SCENE frame and
                # compared against this member's own spawn_bbox, which is the frame sb_vrb places
                # it in: a placement bug that used a WORLD quantity would land member 0 (whose
                # scene origin is the world origin) correctly and put every other member's target
                # metres away, which no other check here would notice.
                if tgt is not None and getattr(env, "spawn_bbox", None) is not None:
                    tp = _np(tgt.get_position_orientation(frame="scene")[0])
                    xmin, xmax, ymin, ymax = env.spawn_bbox[:4]
                    margin = 0.2
                    inside = (xmin - margin <= tp[0] <= xmax + margin
                              and ymin - margin <= tp[1] <= ymax + margin)
                    print(f"  [9] member {i}: target scene-frame xy=({tp[0]:.3f}, {tp[1]:.3f}) "
                          f"z={tp[2]:.3f}, spawn box x[{xmin:.2f},{xmax:.2f}] "
                          f"y[{ymin:.2f},{ymax:.2f}] -> inside={inside}", flush=True)
                    if not inside:
                        failures.append(f"reset {r+1}: member {i} target {tgt.name!r} sits at "
                                        f"scene-frame xy=({tp[0]:.3f}, {tp[1]:.3f}), outside its "
                                        f"own spawn box x[{xmin:.2f},{xmax:.2f}] "
                                        f"y[{ymin:.2f},{ymax:.2f}] (+-{margin} m)")

        prev_ident = [s["identity"] for s in snap]
        prev_task_type = [e.task_type for e in vec_env.envs]

        if perturbation == "SB-VRB" and num_envs > 1:
            # The members' progression dicts must be N DISTINCT objects. Per-member the check above
            # only rules out the module-level one; two members could still share a third dict.
            prog_ids = [id(e.task_progression) for e in vec_env.envs]
            if len(set(prog_ids)) < num_envs:
                failures.append(f"reset {r+1}: members share a task_progression dict "
                                f"({len(set(prog_ids))}/{num_envs} distinct) -- one member's "
                                f"progress would be credited to another")
            tgts = [e.target_objects[0] for e in vec_env.envs if e.target_objects]
            if len({id(t) for t in tgts}) < len(tgts):
                failures.append(f"reset {r+1}: two members hold the SAME target object instance -- "
                                f"a name-keyed lookup crossed scenes")

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
            # Perturbation-independent, and the general form of the bug the whole add/replace family
            # kept hitting: a sibling's remove_object() evicts a freshly added object from the
            # simulator's GLOBAL init queue by NAME, so it stays in the scene and on the stage but
            # is never initialized. RealmVectorEnvironment._repair_init_queue() repairs that,
            # and this is the independent statement that the repair worked -- previously the only
            # evidence was the absence of an unrelated assert from dump_state() much later.
            uninit = [o.name for o in env.omnigibson_env.scene.objects if not o.initialized]
            if uninit:
                failures.append(f"reset {r+1}: member {i} (scene {scene_idx}) has uninitialized "
                                f"objects after reset: {uninit}")

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
            poses.append(pos)
            if pos[2] < TABLE_Z_MIN:
                failures.append(f"reset {r+1}: member {i} main object left the table (z={pos[2]:.3f})")

            print(f"  [3/5] member {i}: grasping={grasping}  "
                  f"main-object xyz=({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})", flush=True)

        poses_by_reset.append(np.array(poses))
        cams_by_reset.append(np.array([camera_poses(e) for e in vec_env.envs]))
        sizes_by_reset.append(np.array([
            (lambda x: x.cpu().numpy() if hasattr(x, "cpu") else np.asarray(x))(
                e.main_objects[0].aabb_extent) for e in vec_env.envs], dtype=float))
        ident_by_reset.append([obj_identity(e) for e in vec_env.envs])
        planted_by_reset.append([planted_identities(e) for e in vec_env.envs])

    # ---- check 4: the perturbation must actually move what it claims to move -----------------
    expect = MOVES.get(perturbation)
    print(f"\n===== [4] does {perturbation} still perturb? (expects to move: {expect}) =====",
          flush=True)
    if expect is None:
        print(f"  no expectation recorded for {perturbation} -- reporting motion, asserting nothing",
              flush=True)

    obj = np.array(poses_by_reset)                           # (resets, members, 3)
    cam = np.array(cams_by_reset)                            # (resets, members, 7*n_sensors)
    siz = np.array(sizes_by_reset) if sizes_by_reset else np.zeros((0,))
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
            # Observables for perturbations whose effect pose cannot see. Without these an entry in
            # MOVES silently asserts nothing, which is the hollow pass this map exists to prevent.
            siz_spread = siz[:, i, :].ptp(axis=0).max() if siz.size else 0.0
            siz_changed = siz_spread > 1e-4
            seen_ident = {ident_base[i]} | {ident_by_reset[k][i] for k in range(resets)}
            ident_changed = len(seen_ident) > 1
            n_planted_changes = sum(
                planted_by_reset[k][i] != planted_by_reset[k - 1][i] for k in range(1, resets))
            print(f"  member {i}: size spread={siz_spread:.4f}  identities={sorted(seen_ident)}  "
                  f"planted-set changes={n_planted_changes}/{max(resets - 1, 0)}", flush=True)
            if expect == "size" and not siz_changed:
                failures.append(f"member {i}: main object never changed size across resets -- "
                                f"{perturbation} is a no-op")
            if expect == "identity" and not ident_changed:
                failures.append(f"member {i}: main object identity never changed from "
                                f"{ident_base[i]} -- {perturbation} is a no-op (or the sampler drew "
                                f"the same model every time; re-run)")
            if expect == "distractors" and n_planted_changes == 0:
                failures.append(f"member {i}: the set of objects in the scene never changed across "
                                f"resets -- {perturbation} is a no-op")
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
