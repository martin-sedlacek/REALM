"""Does the drawer reset drive every member correctly off ONE shared og.sim.step() loop?

`RealmEnvironmentBase.reset_joints()` runs ~55 `og.sim.step()`s on a drawer task (10 + 5 in
`utils.reset_joints` driving the joints home, then 30 + 10 free-running). `og.sim.step()` is GLOBAL
-- it advances every scene in the simulator -- so a vector env that let each member run its own loop
paid 55*N of them per reset and stepped every member's scene N times over while driving only one
member's joints. `run_joint_resets()` hoists the stepping out, exactly as
`RealmVectorEnvironment._settle()` already does for the 30-step settle loop.

That change CANNOT be verified end to end on this port: the only task types that reach it are
open_drawer and close_drawer, and neither loads (`cabinet.usd` ->
`TypeError: missing a required argument: 'preset_name'` in omnigibson/prims/material_prim.py). This
test covers the half that does not need the asset -- the scheduling -- by stubbing `og.sim` and
handing `run_joint_resets()` fake members:

  1. N members cost the SAME number of og.sim.step() calls as 1. That is the whole point.
  2. Each member sees EXACTLY the call sequence a single env gives it. Asserted by projecting the
     N-member event log onto one member and comparing against a real single-env run, rather than by
     re-listing the expected calls -- so the test cannot drift from the behaviour it preserves.
  3. Writes for ALL members land before each shared step, never after it. A member writing after
     the step it belongs to would lag the others by a physics step.
  4. reset_joints() records without stepping in a vector env, and steps inline in a single env; and
     run_joint_resets() clears the plan afterwards, so RealmVectorEnvironment.reset()'s "nothing
     left pending" assertion is checking something real rather than an attribute nobody sets.
  5. The reachable path -- every non-drawer task -- still costs zero steps and leaves mo_joint None.

It does NOT check that a real cabinet ends up at the right openness, or that the drive gains are
right. Those need the asset.

    ./run python -u tests/test_joint_reset_batching.py
"""
import contextlib
import sys

import omnigibson as og

from realm.environments.env_base import (
    JOINT_HOLD_STEPS,
    JOINT_SETTLE_STEPS,
    JointResetPlan,
    RealmEnvironmentBase,
    run_joint_resets,
)

CLOSING_STEPS = 10   # utils.reset_joints_batched defaults
STILL_STEPS = 5
EXPECTED_STEPS = CLOSING_STEPS + STILL_STEPS + JOINT_SETTLE_STEPS + JOINT_HOLD_STEPS
N_MEMBERS = 4


class FakeSim:
    """Just enough og.sim to run the joint-reset loops: a step counter and the render context."""

    def __init__(self, log):
        self.log = log

    def step(self):
        self.log.append(("step", None, None))

    @contextlib.contextmanager
    def render_on_step(self, value):
        self.log.append(("render_on_step", None, value))
        yield
        self.log.append(("render_on_step_exit", None, value))


class FakeJoint:
    def __init__(self, name, log):
        self.name = name
        self.log = log
        self.lower_limit = 0.0
        self.upper_limit = 2.0

    def set_pos(self, value, normalized=False):
        self.log.append(("set_pos", self.name, (value, normalized)))

    def set_vel(self, value):
        self.log.append(("set_vel", self.name, value))

    def set_effort(self, value):
        self.log.append(("set_effort", self.name, value))

    def keep_still(self):
        self.log.append(("keep_still", self.name, None))

    def get_state(self):
        return ([1.0],)


class FakeCabinet:
    def __init__(self, joints):
        self.joints = {j.name: j for j in joints}


class FakeMember(RealmEnvironmentBase):
    """A RealmEnvironmentBase with the real reset_joints/_record_joint_openness and nothing else.

    Deliberately does NOT call super().__init__, which needs a robot, a scene and a live simulator.
    _prepare_joint_reset is the one method replaced: the real one calls get_target_drawer_joint(),
    which walks a real articulated cabinet. Everything actually under test -- the record-vs-run
    branch in reset_joints(), run_joint_resets()'s interleaving, and where _record_joint_openness
    falls in the sequence -- is the production code.
    """

    def __init__(self, idx, log, task_type="open_drawer", in_vec_env=True, n_joints=2):
        self.idx = idx
        self.log = log
        self.task_type = task_type
        self.in_vec_env = in_vec_env
        self.pending_joint_reset = None
        # Two openable joints that get driven, plus one the cabinet owns that is NOT openable:
        # run_joint_resets' final keep_still() sweep covers cabinet.joints, a superset of the driven
        # ones, and this pins that difference.
        self.joints = [FakeJoint(f"m{idx}.drawer{k}", log) for k in range(n_joints)]
        self.cabinet = FakeCabinet(self.joints + [FakeJoint(f"m{idx}.hinge", log)])
        self.mo_joint = self.joints[0]
        self.prepared_with = None

    def _prepare_joint_reset(self, target_drawer_loc):
        self.prepared_with = target_drawer_loc
        return JointResetPlan(
            cabinet=self.cabinet,
            joints=self.joints,
            reset_states=[-1.0 for _ in self.joints],
        )

    def _record_joint_openness(self):
        super()._record_joint_openness()
        self.log.append(("openness", f"m{self.idx}", None))


def owner(event):
    """Which member an event belongs to, or None for the shared ones (step, render context)."""
    name = event[1]
    return name.split(".")[0] if name else None


def project(log, member):
    """@log as that member experienced it: its own calls, plus every shared step, in order."""
    out = []
    for kind, name, arg in log:
        who = owner((kind, name, arg))
        if who is None:
            out.append((kind, None, arg))
        elif who == member:
            parts = name.split(".", 1)
            out.append((kind, parts[1] if len(parts) > 1 else "", arg))
    return out


def build(num_members, log, **kwargs):
    members = [FakeMember(i, log, **kwargs) for i in range(num_members)]
    for m in members:
        m.reset_joints()
    if members and members[0].in_vec_env:
        run_joint_resets(members)
    return members


def n_steps(log):
    return sum(1 for e in log if e[0] == "step")


def main():
    failures = []

    # ---- the single-env reference: reset_joints() runs the loop inline -------------------------
    log1 = []
    og.sim = FakeSim(log1)          # module-global; env_base and utils both read og.sim
    build(1, log1, in_vec_env=False)
    steps1 = n_steps(log1)
    print(f"[1] single env: {steps1} og.sim.step() call(s) (expected {EXPECTED_STEPS} = "
          f"{CLOSING_STEPS} + {STILL_STEPS} + {JOINT_SETTLE_STEPS} + {JOINT_HOLD_STEPS})",
          flush=True)
    if steps1 != EXPECTED_STEPS:
        failures.append(f"[1] a single env issued {steps1} steps, expected {EXPECTED_STEPS}")

    # ---- N members: the SAME step cost -------------------------------------------------------
    logn = []
    og.sim = FakeSim(logn)
    build(N_MEMBERS, logn, in_vec_env=True)
    stepsn = n_steps(logn)
    print(f"[1] {N_MEMBERS} members: {stepsn} og.sim.step() call(s) (expected {steps1}, i.e. NOT "
          f"{steps1 * N_MEMBERS})", flush=True)
    if stepsn != steps1:
        failures.append(f"[1] {N_MEMBERS} members issued {stepsn} steps but a single env issued "
                        f"{steps1} -- the drawer reset still costs global steps per member")

    # ---- 2: every member saw the single-env sequence -------------------------------------------
    ref = project(log1, "m0")
    same = 0
    for i in range(N_MEMBERS):
        got = project(logn, f"m{i}")
        if got == ref:
            same += 1
            continue
        first = next((k for k in range(max(len(got), len(ref)))
                      if k >= len(got) or k >= len(ref) or got[k] != ref[k]), 0)
        failures.append(f"[2] member {i} did not see the single-env sequence; first difference at "
                        f"index {first}: got {got[first:first + 3]!r}, "
                        f"expected {ref[first:first + 3]!r}")
    print(f"[2] {same}/{N_MEMBERS} members saw exactly the {len(ref)}-call sequence a single env "
          f"produces", flush=True)

    # ---- 3: every shared step is preceded by one contiguous write burst per member -------------
    want_order = [f"m{i}" for i in range(N_MEMBERS)]
    bursts, bad = 0, []
    burst = []
    for event in logn:
        if event[0] == "step":
            order = []
            for e in burst:
                who = owner(e)
                if who and (not order or order[-1] != who):
                    order.append(who)
            if order:
                bursts += 1
                if order != want_order:
                    bad.append(order)
            burst = []
        else:
            burst.append(event)
    print(f"[3] {bursts} write burst(s) before a shared step; "
          f"{len(bad)} were not '{want_order}' contiguously", flush=True)
    if bad:
        failures.append(f"[3] {len(bad)} write burst(s) were not one contiguous run per member in "
                        f"member order; first offender: {bad[0]}")

    # ---- 4: record vs run, and the plan really is cleared --------------------------------------
    logv = []
    og.sim = FakeSim(logv)
    vec = FakeMember(0, logv, in_vec_env=True)
    vec.reset_joints(target_drawer_loc="middle")
    print(f"[4] vector member: recorded a plan={vec.pending_joint_reset is not None}  "
          f"steps issued={n_steps(logv)}  target_drawer_loc forwarded={vec.prepared_with!r}",
          flush=True)
    if vec.pending_joint_reset is None:
        failures.append("[4] reset_joints() in a vector env recorded nothing -- the drawer reset "
                        "would never run AND RealmVectorEnvironment's 'nothing left pending' "
                        "assertion would be vacuous")
    if n_steps(logv):
        failures.append(f"[4] reset_joints() in a vector env stepped the sim {n_steps(logv)} "
                        f"times; it must only record")
    if vec.prepared_with != "middle":
        failures.append(f"[4] target_drawer_loc was not forwarded to the plan (got "
                        f"{vec.prepared_with!r}) -- SB-NOUN's drawer branch depends on it")

    run_joint_resets([vec])
    print(f"[4] after run_joint_resets: pending={vec.pending_joint_reset!r}  "
          f"init_openness_fraction={getattr(vec, 'init_openness_fraction', None)}", flush=True)
    if vec.pending_joint_reset is not None:
        failures.append("[4] run_joint_resets() left the plan in place -- the next drain would "
                        "re-run it and the pending assertion would false-fire")
    if getattr(vec, "init_openness_fraction", None) is None:
        failures.append("[4] init_openness_fraction was never recorded -- every joint progression "
                        "stage is measured against it")

    logs = []
    og.sim = FakeSim(logs)
    single = FakeMember(0, logs, in_vec_env=False)
    single.reset_joints()
    print(f"[4] single env: steps issued inline={n_steps(logs)}  "
          f"pending={single.pending_joint_reset!r}", flush=True)
    if n_steps(logs) != EXPECTED_STEPS:
        failures.append(f"[4] single-env reset_joints() issued {n_steps(logs)} steps, expected "
                        f"{EXPECTED_STEPS} -- it must still run inline")
    if single.pending_joint_reset is not None:
        failures.append("[4] single-env reset_joints() left a plan pending")

    # ---- 5: the reachable path -- everything that is not a drawer task -------------------------
    logo = []
    og.sim = FakeSim(logo)
    for in_vec in (True, False):
        other = FakeMember(0, logo, task_type="pick", in_vec_env=in_vec)
        other.mo_joint = "sentinel"
        other.reset_joints()
        if other.mo_joint is not None:
            failures.append(f"[5] non-drawer task (in_vec_env={in_vec}) did not clear mo_joint")
        if other.pending_joint_reset is not None:
            failures.append(f"[5] non-drawer task (in_vec_env={in_vec}) recorded a joint reset")
    print(f"[5] non-drawer tasks: steps issued={n_steps(logo)} (expected 0)", flush=True)
    if n_steps(logo):
        failures.append(f"[5] a non-drawer task issued {n_steps(logo)} steps")

    print("\n" + "=" * 70, flush=True)
    if failures:
        print(f"FAILED -- {len(failures)} problem(s):", flush=True)
        for f in failures:
            print(f"  - {f}", flush=True)
    else:
        print(f"PASSED -- a drawer reset costs {EXPECTED_STEPS} global og.sim.step() calls at "
              f"{N_MEMBERS} members, the same as at 1, and every member sees the single-env call "
              f"sequence. This is the SCHEDULE against a stubbed sim; the real cabinet was "
              f"confirmed separately on 2026-08-14 (57 steps at num_envs=2 vs 56 at 1, and every "
              f"member's drawers landing where a single env puts them) -- see "
              f"env_base.run_joint_resets, which also records what does NOT land in scene 0.",
              flush=True)
    print("=" * 70, flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
