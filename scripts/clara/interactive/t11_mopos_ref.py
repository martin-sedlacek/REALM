"""Does env.mo_pos_orig still describe the object being scored after SB-NOUN swaps the target?

env.mo_pos_orig / env.mo_rot_orig are the START-OF-ROLLOUT reference poses that
RealmEnvironmentBase judges progression against:

    check_lift_and_distance_condition()  ->  LIFT_SLIGHT, LIFT_LARGE, PUSH
        lifted   = mo.pos.z - mo_pos_orig.z > lift_threshold
        traveled = ||mo.pos - mo_pos_orig|| > distance_threshold
    check_rotated()                      ->  ROTATED  (vs mo_rot_orig)

They are captured from the TASK CONFIG in RealmEnvironmentBase.__init__ (mo_cfgs[0]["position"]),
i.e. once, at build time. SB-NOUN (perturbations/sb_noun.py) then re-points main_objects[0] at a
random DISTRACTOR on every reset -- a different object, standing somewhere else on the table --
and VSB-NOBJ / VB-MOBJ REPLACE main_objects[0] outright. So between the swap and the next capture
the reference describes one object while the checks read another, and both a false LIFT (the new
target already sits above/away from the old one's config pose) and a missed LIFT (it sits below)
are possible.

This probe measures the divergence directly, at the three points a rollout passes through, because
they do NOT all behave the same and only measuring one of them gives the wrong answer:

    A  after reset_pre_perturbation()   -- scene restored, perturbation has not run yet
    B  after apply_perturbations()      -- this is where SB-NOUN swaps the target
    C  after warmup(obs)                -- == what realm/eval.py:139-140 actually does per rollout

RealmEnvironmentDynamic.reset() is literally those first two calls in sequence, so driving them
separately loses no coverage and is the only way to read the state BETWEEN them.

For each it prints main_objects[0]'s name/category, its ACTUAL world position, env.mo_pos_orig,
and the gap between them -- plus what the lift/rotate checks would answer AT REST, before any
policy has acted. At the start of a rollout every one of them must be False; a True there is the
bug scoring progression that never happened.

Two failure modes are checked separately, because a fix for one can silently cause the other:

  [STALE]  at B and C the reference must describe the object main_objects[0] currently points at
           (gap <= --tol), and no lift/rotate check may fire at rest.
  [FROZEN] the reference must NOT track the object during stepping. A "fix" that re-captures every
           step would make gap 0 forever and pass [STALE] perfectly while making every lift check
           trivially False -- strictly worse than the staleness. So after warmup the probe holds
           still for --steps steps (mo_pos_orig must not move) and then teleports the main object
           +0.25 m in x and +0.25 m in z, where check_lift_and_distance_condition() MUST become
           True and mo_pos_orig MUST still not have moved.

Single env is enough: the reference is per-member bookkeeping and RealmVectorEnvironment.warmup()
re-captures it exactly the same way, so vectorizing only multiplies the ~4 min scene build. The
vectorized path is covered by t9_vbpose_nostopplay.py --perturbation SB-NOUN.

    ./run python -u scripts/clara/interactive/t11_mopos_ref.py --resets 6
    ./run python -u scripts/clara/interactive/t11_mopos_ref.py --resets 4 --perturbation VSB-NOBJ
"""
import argparse

import numpy as np

import omnigibson as og

from realm.environments.env_dynamic import RealmEnvironmentDynamic
from realm.eval import SUPPORTED_PERTURBATIONS, SUPPORTED_TASKS
from realm.sim_config import set_sim_config

# Metres. The reference is captured at the END of warmup, after the arm has settled, so the object
# may still be creeping by a fraction of a millimetre when C is read. 5 mm is ~50x that and two
# orders of magnitude below the object-to-object distances SB-NOUN produces (tens of centimetres),
# so it cannot be met by accident and cannot false-fail on settle noise.
TOL = 5e-3
# How far to teleport the object for the [FROZEN] probe. Must clear check_lift_large_condition's
# thresholds (0.1 m travel, 0.075 m lift) with room to spare.
TELEPORT = 0.25


def _np(x):
    """Whatever OmniGibson/REALM handed back (torch on GPU, torch on CPU, list) as a numpy array."""
    return np.asarray(x.cpu().numpy() if hasattr(x, "cpu") else x, dtype=float)


def probe(env, obs, label):
    """One (object identity, actual pose, reference pose, at-rest check answers) reading."""
    mo = env.main_objects[0]
    actual = _np(mo.get_position_orientation()[0])
    ref = _np(env.mo_pos_orig)
    row = {
        "label": label,
        "name": mo.name,
        "category": getattr(mo, "category", None),
        "actual": actual,
        "ref": ref,
        "gap": float(np.linalg.norm(actual - ref)),
        "dz": float(actual[2] - ref[2]),
        # Called DIRECTLY rather than through recompute_task_progression, which breaks at the first
        # unmet stage and would hide a LIFT that is judged True but gated behind an unmet GRASP.
        # What is being measured is what the check itself answers, not whether the rubric reaches it.
        "lift_slight": bool(env.check_lift_slight_condition(obs)),
        "lift_large": bool(env.check_lift_large_condition(obs)),
        "rotated": bool(env.check_rotated(obs)),
    }
    print(f"  [{label:>12s}] main={row['name']:<16s} category={str(row['category']):<16s}\n"
          f"                 actual = [{actual[0]: .4f} {actual[1]: .4f} {actual[2]: .4f}]\n"
          f"                 mo_pos_orig = [{ref[0]: .4f} {ref[1]: .4f} {ref[2]: .4f}]\n"
          f"                 gap = {row['gap']:.4f} m   dz = {row['dz']:+.4f} m   "
          f"at-rest checks: LIFT_SLIGHT={row['lift_slight']} LIFT_LARGE={row['lift_large']} "
          f"ROTATED={row['rotated']}", flush=True)
    return row


def frozen_probe(env, obs, steps, failures):
    """Is the reference still a START-of-rollout pose, or has it been made to track the object?

    Two things have to hold at once and a naive fix breaks one of them:
      1. holding still must not move mo_pos_orig (a per-step re-capture would);
      2. moving the object 0.25 m must still make check_lift_and_distance_condition() fire (a
         reference that tracks the object, or one accidentally set to the object's live pose every
         step, would make it permanently False and silently delete the LIFT stage).
    """
    print("\n" + "=" * 78, flush=True)
    print(f"[FROZEN] reference must be start-of-rollout, not live", flush=True)
    ref_before = _np(env.mo_pos_orig)
    rot_before = _np(env.mo_rot_orig)

    ee_cmd = env.warmup_ee_cmd()
    for t in range(steps):
        obs, _, _, _, _ = env.step(env.warmup_action(t, ee_cmd))
    ref_after = _np(env.mo_pos_orig)
    drift = float(np.linalg.norm(ref_after - ref_before))
    print(f"    after {steps} hold-still steps: mo_pos_orig moved {drift:.2e} m", flush=True)
    if drift > 1e-9:
        failures.append(f"FROZEN: mo_pos_orig moved {drift:.2e} m during {steps} steps of stepping "
                        f"-- the reference is being re-captured mid-rollout, so lift/distance is "
                        f"measured against the object's CURRENT pose and can never fire")

    # Teleport well past both lift thresholds. Read after one og.sim.step() so the physics view is
    # refreshed; one step of free fall is ~5 mm, negligible against the 0.25 m displacement.
    mo = env.main_objects[0]
    pos, ori = mo.get_position_orientation()
    target = _np(pos) + np.array([TELEPORT, 0.0, TELEPORT])
    mo.set_position_orientation(position=_np(target).tolist(), orientation=ori)
    mo.keep_still()
    og.sim.step()
    moved = _np(mo.get_position_orientation()[0])
    fired = bool(env.check_lift_and_distance_condition())
    ref_teleport = _np(env.mo_pos_orig)
    print(f"    teleported main object to [{moved[0]: .4f} {moved[1]: .4f} {moved[2]: .4f}] "
          f"({np.linalg.norm(moved - ref_before):.4f} m from the reference)", flush=True)
    print(f"    check_lift_and_distance_condition() = {fired}   "
          f"mo_pos_orig now [{ref_teleport[0]: .4f} {ref_teleport[1]: .4f} {ref_teleport[2]: .4f}]",
          flush=True)
    if not fired:
        failures.append("FROZEN: the main object was moved 0.25 m up and across and "
                        "check_lift_and_distance_condition() still returned False -- the lift check "
                        "has been disabled, not fixed")
    if float(np.linalg.norm(ref_teleport - ref_before)) > 1e-9:
        failures.append("FROZEN: mo_pos_orig followed the teleport -- the reference tracks the "
                        "object instead of recording where it started")
    if float(np.linalg.norm(rot_before - _np(env.mo_rot_orig))) > 1e-9:
        failures.append("FROZEN: mo_rot_orig changed during stepping -- ROTATED is measured against "
                        "a moving reference")


def main(resets, task_id, robot, perturbation, steps, tol):
    set_sim_config(robot=robot)
    env = RealmEnvironmentDynamic(
        config_path="/app/realm/config",
        task_cfg_path=f"REALM_DROID10/{SUPPORTED_TASKS[task_id]}/default.yaml",
        perturbations=[perturbation],
        robot=robot,
    )

    print(f"\n===== build-time state (RealmEnvironmentBase.__init__, from the task config) =====",
          flush=True)
    print(f"  main_objects[0] = {env.main_objects[0].name!r} "
          f"(category {getattr(env.main_objects[0], 'category', None)!r})", flush=True)
    print(f"  distractors     = {[d.name for d in env.distractors]}", flush=True)
    print(f"  mo_pos_orig     = {_np(env.mo_pos_orig)}", flush=True)
    print(f"  instruction     = {env.instruction!r}", flush=True)

    failures = []
    rows = []
    obs = None
    for r in range(resets):
        print(f"\n===== {perturbation} reset {r + 1}/{resets} =====", flush=True)
        # reset() driven as its two documented phases so the reference can be read BETWEEN them --
        # A is before the swap, B is immediately after it. B is exactly env.reset().
        obs, _ = env.reset_pre_perturbation()
        a = probe(env, obs, "A pre-pert")
        obs = env.apply_perturbations(obs)
        b = probe(env, obs, "B post-reset")
        # Exactly what realm/eval.py does for every rollout: reset(), then warmup(obs). obs is not
        # None, so warmup does NOT reset again -- it steps 30 hold-still steps and re-captures.
        obs, _, _, _, _ = env.warmup(obs)
        c = probe(env, obs, "C post-warmup")
        print(f"  instruction: {env.instruction!r}", flush=True)
        print(f"  task_progression after warmup: {env.task_progression}", flush=True)
        rows.append((a, b, c))

        for row in (b, c):
            if row["gap"] > tol:
                failures.append(
                    f"STALE: reset {r + 1} at {row['label']}: main_objects[0] is "
                    f"{row['name']!r} at {np.round(row['actual'], 4).tolist()} but mo_pos_orig is "
                    f"{np.round(row['ref'], 4).tolist()} -- {row['gap']:.4f} m away "
                    f"(dz {row['dz']:+.4f} m). Lift and distance are being scored against a "
                    f"different object's start pose.")
            for stage in ("lift_slight", "lift_large", "rotated"):
                if row[stage]:
                    failures.append(
                        f"STALE: reset {r + 1} at {row['label']}: {stage.upper()} is already True "
                        f"at rest, before any policy acted -- progression that never happened.")
        # A stage latched during the 30 warmup steps is the same bug reaching the score sheet.
        latched = [k for k, v in (env.task_progression or {}).items() if v]
        if latched:
            failures.append(f"STALE: reset {r + 1}: task_progression already has {latched} set "
                            f"after warmup, before the policy acted")

    frozen_probe(env, obs, steps, failures)

    # ---- summary --------------------------------------------------------------------------------
    print("\n" + "=" * 78, flush=True)
    print("per-reset gap between main_objects[0]'s actual pose and mo_pos_orig", flush=True)
    print(f"  {'reset':>5s} {'A pre-pert':>28s} {'B post-reset':>28s} {'C post-warmup':>28s}", flush=True)
    for r, (a, b, c) in enumerate(rows):
        def cell(x):
            return f"{x['name'][:12]:<12s} {x['gap']:7.4f} m"
        print(f"  {r + 1:>5d} {cell(a):>28s} {cell(b):>28s} {cell(c):>28s}", flush=True)
    gaps_b = [b["gap"] for _, b, _ in rows]
    gaps_c = [c["gap"] for _, _, c in rows]
    print(f"  B post-reset  gap: min={min(gaps_b):.4f} max={max(gaps_b):.4f} "
          f"mean={float(np.mean(gaps_b)):.4f} m", flush=True)
    print(f"  C post-warmup gap: min={min(gaps_c):.4f} max={max(gaps_c):.4f} "
          f"mean={float(np.mean(gaps_c)):.4f} m", flush=True)

    print("\n" + "=" * 78, flush=True)
    if failures:
        print(f"FAILED -- {len(failures)} problem(s):", flush=True)
        for f in failures:
            print(f"  - {f}", flush=True)
    else:
        print(f"PASSED -- {resets} resets of {perturbation}: mo_pos_orig describes the object "
              f"main_objects[0] actually points at (within {tol} m) both right after reset() and "
              f"after warmup(), no lift/rotate check fires at rest, and the reference still does "
              f"not follow the object during stepping (a 0.25 m move still trips the lift check).",
              flush=True)
    print("=" * 78, flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--resets", type=int, default=6,
                   help="SB-NOUN draws uniformly from the distractor pool, so a few resets are "
                        "needed before the spread of object-to-object distances is representative")
    p.add_argument("--task_id", type=int, default=0,
                   help="0 = put_green_block_into_bowl, the configuration the pi0.5 evals run")
    p.add_argument("--robot", type=str, default="DROID_robolab")
    p.add_argument("--perturbation", type=str, default="SB-NOUN",
                   help="any perturbation that re-points or replaces main_objects[0]: "
                        "SB-NOUN, VSB-NOBJ, VB-MOBJ")
    p.add_argument("--steps", type=int, default=10,
                   help="hold-still steps for the [FROZEN] probe")
    p.add_argument("--tol", type=float, default=TOL, help="metres")
    a = p.parse_args()
    assert a.perturbation in SUPPORTED_PERTURBATIONS, f"unknown perturbation {a.perturbation!r}"
    raise SystemExit(main(a.resets, a.task_id, a.robot, a.perturbation, a.steps, a.tol))
