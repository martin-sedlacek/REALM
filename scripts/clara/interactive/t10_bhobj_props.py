"""Do B-HOBJ's physical-property perturbations COMPOUND across resets?

B-HOBJ (realm/environments/perturbations/b_hobj.py) rescales the payload's mass, joint stiffness,
joint damping and joint max-effort by a fresh random factor on every reset. Every one of those
writes used to READ THE CURRENT VALUE and multiply it:

    link.mass       = min(link.mass * s, 2.0)
    joint.stiffness = joint.stiffness * s_stif      # and damping, and max_effort

Those getters read the LIVE articulation/rigid-prim view (see OG-lite prims/joint_prim.py and
prims/rigid_dynamic_prim.py), and og.Environment.reset() restores pose/velocity state, not physical
properties. So if the properties are not restored, reset N is perturbed relative to reset N-1 and
the factors MULTIPLY: over a 25-rollout eval the payload ends up orders of magnitude off, and no
single rollout carries the perturbation the perturbation claims to apply.

This probe measures that directly. A single env is enough -- the drift is per-member bookkeeping,
so vectorizing it only multiplies the scene-build cost. reset() is driven as its two documented
phases so the property state can be read BETWEEN them:

    reset_pre_perturbation()   <- og.Environment.reset() + reset_joints();  "pre" snapshot
    apply_perturbations()      <- b_hobj runs here;                         "post" snapshot

which separates the two questions that matter:

  1. DOES RESET RESTORE?  pre[r] vs post[r-1]. Equal => reset restores nothing, so a read-current-
     and-multiply perturbation necessarily compounds. This is the premise being tested, not assumed.
  2. DOES IT COMPOUND?    post[r]/baseline must stay inside the range of ONE draw:
        joints: exp(U(-1, 1))  = [1/e, e]
        mass:   U(0.25, 3), then clipped to 2.0 kg
     A ratio outside that band is proof of compounding, and its size says how bad it is.
  3. IS IT STILL A PERTURBATION?  the ratios must VARY across resets. A "fix" that scales from the
     baseline but always by the same factor -- or that stops writing at all -- would satisfy (2)
     perfectly while silently disabling B-HOBJ, which is worse than the drift it replaced. Checked
     for explicitly; a frozen property fails.

Baselines of exactly 0 are reported but excluded from (2) and (3): 0 * anything is 0, so a passive
joint with no drive gains cannot be perturbed multiplicatively and cannot drift either.

    ./run python -u scripts/clara/interactive/t10_bhobj_props.py --resets 10 --task_id 0
    ./run python -u scripts/clara/interactive/t10_bhobj_props.py --resets 6  --task_id 8

task_id 0 (put_green_block_into_bowl) is the configuration the pi0.5 evals actually run and its
main object is a single-link cube, so it exercises the MASS path only. task_id 8 (open_drawer) has
an articulated USD cabinet as its main object and is the cheapest way to exercise the JOINT path.
"""
import argparse
import math

from realm.environments.env_dynamic import RealmEnvironmentDynamic
from realm.eval import SUPPORTED_TASKS
from realm.sim_config import set_sim_config

# Bounds of a SINGLE b_hobj draw. Kept here rather than imported so the probe still fails if a
# future edit widens the distribution in b_hobj without anyone thinking about the consequences.
JOINT_LO, JOINT_HI = math.exp(-1.0), math.exp(1.0)   # exp(U(-1, 1))
MASS_LO, MASS_HI = 0.25, 3.0                          # U(0.25, 3)
MASS_CLIP = 2.0                                       # b_hobj clips the payload at 2 kg
TOL = 1e-3          # float32 round-trip through the articulation view
VARY_TOL = 1e-6     # below this two ratios are "the same value", i.e. the property is frozen

JOINT_PROPS = ("max_effort", "stiffness", "damping")


def legacy_b_hobj(env):
    """b_hobj EXACTLY as it was before the baseline fix -- read the live value, multiply, write back.

    Kept so `--legacy` can reproduce the bug on demand. Two reasons that is worth the duplication:
    a check that has never been seen to fire is not evidence of anything, and the next person to
    touch b_hobj can re-measure the old behaviour in one command instead of reconstructing it from
    git history. Do not "tidy" this to call the real b_hobj -- the whole point is that it does not.
    """
    import torch
    s = np.random.uniform(0.25, 3)
    s_mass, s_mvel, s_meff, s_stif, s_damp, s_fric = np.exp(np.random.uniform(-1, 1, size=(6,)))
    for obj in env.main_objects:
        for link in obj._links.values():
            link.mass = min(link.mass * s, 2.0)
        for joint in obj.joints.values():
            joint.max_effort = joint.max_effort * float(s_meff)
            joint.stiffness = joint.stiffness * s_stif
            joint.damping = joint.damping * s_damp
            joint._articulation_view.set_max_efforts(
                torch.tensor([[joint.max_effort]], dtype=torch.float32), joint_indices=joint.dof_indices)
            joint._articulation_view.set_gains(
                kps=torch.tensor([[joint.stiffness]]), joint_indices=joint.dof_indices)
            joint._articulation_view.set_gains(
                kds=torch.tensor([[joint.damping]]), joint_indices=joint.dof_indices)


def snapshot(env):
    """Live physical properties of every main object, keyed (obj, prop, link/joint name)."""
    snap = {}
    for obj in env.main_objects:
        for name, link in obj._links.items():
            snap[(obj.name, "mass", name)] = float(link.mass)
        for name, joint in obj.joints.items():
            for prop in JOINT_PROPS:
                try:
                    snap[(obj.name, prop, name)] = float(getattr(joint, prop))
                except Exception as e:  # noqa: BLE001  multi-DOF joints assert in the getter
                    print(f"  (skipping {obj.name}.{name}.{prop}: {type(e).__name__}: {e})",
                          flush=True)
    return snap


def ratio(value, base):
    """value/base, or None when the baseline is 0 and no multiplicative ratio exists."""
    return None if base == 0.0 else value / base


def bounds_for(prop):
    return (JOINT_LO, JOINT_HI) if prop in JOINT_PROPS else (MASS_LO, MASS_HI)


def add_articulated_object(env):
    """Append the first articulated object in the scene to env.main_objects, and say which.

    Why this is needed. b_hobj's joint loop runs over `obj.joints` for every object in
    env.main_objects, and only open_drawer/close_drawer have an articulated main object -- every
    other REALM task's payload is a single rigid body, so the joint half of B-HOBJ is never
    exercised. Both drawer tasks are currently unloadable on the OG 3.9.1 port: their main object is
    custom_assets/impact_drawer/usd/cabinet.usd, and loading it dies in
    omnigibson/prims/material_prim.py get_material with `TypeError: missing a required argument:
    'preset_name'` (probe run 2026-08-13, tmp/probe_t8_prefix.log) -- an asset/OG issue with nothing
    to do with perturbations.

    So instead borrow a real articulated body from the scene (Pomaria_1_int is full of cabinets) and
    put it in main_objects. b_hobj cannot tell the difference: its contract is "every object in
    env.main_objects", and this is one. Nothing else in the probe steps a policy, so nothing cares
    that the payload is now a cupboard.

    CAVEAT, measured 2026-08-13: this destabilises PhysX after a handful of resets and the process
    segfaults -- unsurprising, since B-HOBJ slams a 21 kg fixed-base scene prop down to the 2 kg
    payload cap and then keeps rewriting its drive gains. Read the per-reset table, which is flushed
    as it goes, rather than waiting for the summary; run 3-4 resets, not 10. The crash is an artifact
    of perturbing scene furniture and says nothing about B-HOBJ on a real payload -- the same probe
    without --add_articulated runs 10 resets clean.
    """
    # scene.objects can include the robot, which is articulated and would be picked first. Rescaling
    # the arm's drive gains is not what B-HOBJ does and would make the probe unreadable.
    skip = list(env.main_objects) + list(env.omnigibson_env.robots)
    for obj in env.omnigibson_env.scene.objects:
        if obj in skip:
            continue
        joints = getattr(obj, "joints", None)
        if joints:
            env.main_objects.append(obj)
            print(f"  added articulated object {obj.name!r} to main_objects: "
                  f"{len(joints)} joint(s), {len(obj._links)} link(s)", flush=True)
            return obj
    print("  NO articulated object found in the scene -- the joint path stays untested", flush=True)
    return None


def main(resets, task_id, robot, legacy, add_articulated):
    set_sim_config(robot=robot)
    env = RealmEnvironmentDynamic(
        config_path="/app/realm/config",
        task_cfg_path=f"REALM_DROID10/{SUPPORTED_TASKS[task_id]}/default.yaml",
        perturbations=["B-HOBJ"],
        robot=robot,
    )

    if legacy:
        # Swap in the pre-fix implementation. The env dispatches perturbations through this dict, so
        # overriding the entry is enough -- no import games, and the rest of reset() is untouched.
        env.supported_pertrubations["B-HOBJ"] = lambda: legacy_b_hobj(env)
        print("\n*** --legacy: running the PRE-FIX b_hobj (read live value, multiply, write back) "
              "***", flush=True)

    if add_articulated:
        add_articulated_object(env)

    baseline = snapshot(env)
    print(f"\n===== baseline (as loaded, before any reset) -- {len(baseline)} properties =====",
          flush=True)
    for key in sorted(baseline):
        print(f"  {key[0]}.{key[2]:<28s} {key[1]:<11s} = {baseline[key]:.6g}", flush=True)
    if not baseline:
        raise SystemExit("no properties found on the main objects -- probe cannot say anything")

    pres, posts = [], []
    for r in range(resets):
        obs, _ = env.reset_pre_perturbation()
        pres.append(snapshot(env))
        env.apply_perturbations(obs)
        posts.append(snapshot(env))

        print(f"\n===== reset {r + 1}/{resets} =====", flush=True)
        for key in sorted(baseline):
            base = baseline[key]
            pre, post = pres[-1][key], posts[-1][key]
            rr = ratio(post, base)
            rr_s = "n/a (baseline 0)" if rr is None else f"{rr:.4g}x baseline"
            print(f"  {key[0]}.{key[2]:<28s} {key[1]:<11s} pre={pre:<12.6g} post={post:<12.6g} {rr_s}",
                  flush=True)

    # ---- 1: does reset() restore physical properties? -------------------------------------------
    print("\n" + "=" * 78, flush=True)
    print("[1] does reset() restore physical properties? (pre[r] vs post[r-1])", flush=True)
    restored, carried = 0, 0
    for r in range(1, resets):
        for key in baseline:
            if abs(pres[r][key] - posts[r - 1][key]) <= TOL * max(1.0, abs(posts[r - 1][key])):
                carried += 1
            elif abs(pres[r][key] - baseline[key]) <= TOL * max(1.0, abs(baseline[key])):
                restored += 1
    print(f"    carried over from the previous reset: {carried}", flush=True)
    print(f"    restored to the baseline value:       {restored}", flush=True)
    print(f"    (neither: {max(0, (resets - 1) * len(baseline) - carried - restored)})", flush=True)

    # ---- 2/3: compounding, and still-varying ----------------------------------------------------
    failures = []
    print("\n[2/3] per-property ratio to baseline across resets", flush=True)
    for key in sorted(baseline):
        obj, prop, name = key
        base = baseline[key]
        ratios = [ratio(posts[r][key], base) for r in range(resets)]
        if base == 0.0:
            print(f"    {obj}.{name:<28s} {prop:<11s} baseline 0 -- not perturbable, skipped "
                  f"(values: {[posts[r][key] for r in range(resets)]})", flush=True)
            continue

        lo, hi = bounds_for(prop)
        spread = max(ratios) - min(ratios)
        # b_hobj caps mass at MASS_CLIP kg, and min(base*s, cap) legitimately lands BELOW base*s.
        # So for mass, the draw range only applies while the cap cannot bite, and a link heavy
        # enough that even the smallest draw exceeds the cap is pinned at the cap by construction --
        # constant, but not frozen in the sense that matters, so asserting variation on it would be
        # a confident false failure. Distinguish the two cases explicitly.
        can_clip = (prop == "mass") and (base * hi > MASS_CLIP)
        always_clipped = (prop == "mass") and (base * lo >= MASS_CLIP)
        note = ("  (always at the mass cap)" if always_clipped else
                "  (mass cap can bite)" if can_clip else "")
        worst = max(ratios, key=abs)
        print(f"    {obj}.{name:<28s} {prop:<11s} ratios min={min(ratios):.4g} max={max(ratios):.4g} "
              f"spread={spread:.4g} allowed=[{lo}, {hi}]{note}", flush=True)

        if can_clip:
            for r in range(resets):
                if posts[r][key] > MASS_CLIP + TOL:
                    failures.append(f"CAP BROKEN: {obj}.{name}.mass reset {r + 1} is "
                                    f"{posts[r][key]:.4g} kg, above the {MASS_CLIP} kg cap")
                    break
        if always_clipped:
            continue  # pinned at the cap; neither compounding nor variation is meaningful

        for r, rr in enumerate(ratios):
            if rr > hi + TOL or (rr < lo - TOL and not can_clip):
                failures.append(f"COMPOUNDING: {obj}.{name}.{prop} reset {r + 1} is {rr:.4g}x "
                                f"baseline, outside the single-draw range [{lo}, {hi}]")
                break
        if resets > 1 and spread <= VARY_TOL:
            if can_clip:
                print(f"      (not asserting variation: the cap can bite here)", flush=True)
            else:
                failures.append(f"FROZEN: {obj}.{name}.{prop} never varies across {resets} resets "
                                f"(always {worst:.6g}x baseline) -- B-HOBJ is a no-op for it")

    print("\n" + "=" * 78, flush=True)
    if failures:
        print(f"FAILED -- {len(failures)} problem(s):", flush=True)
        for f in failures:
            print(f"  - {f}", flush=True)
    else:
        print(f"PASSED -- {resets} resets: every property stays within ONE draw of its baseline "
              f"(no compounding) and still varies from reset to reset (not frozen).", flush=True)
    print("=" * 78, flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--resets", type=int, default=10,
                   help="more resets = more compounding; 10 is already ~13x on the drifting path")
    p.add_argument("--task_id", type=int, default=0,
                   help="0 = the eval default; its payload is a single rigid cube (mass path only)")
    p.add_argument("--robot", type=str, default="DROID_robolab_v2")
    p.add_argument("--legacy", action="store_true",
                   help="run the PRE-FIX b_hobj instead, to re-measure the drift (expects to FAIL)")
    p.add_argument("--add_articulated", action="store_true",
                   help="borrow an articulated scene object into main_objects so the joint half of "
                        "b_hobj is exercised; see add_articulated_object() for why that is needed")
    a = p.parse_args()
    raise SystemExit(main(a.resets, a.task_id, a.robot, a.legacy, a.add_articulated))
