"""Vectorized REALM environments: N scenes in one simulator, stepped together.

OmniGibson 3.9.1 loads several scenes into one physics stage, tiled side by side with a margin
(Simulator.import_scene -> Scene.load(idx=..., last_scene_edge=...)). One og.sim.step() advances
every scene, so members must not be stepped individually -- doing so would advance all scenes while
applying only one member's action. That is what this class exists to prevent; it mirrors
OmniGibson's own VectorEnvironment, with REALM's per-member setup and task bookkeeping layered on.

Construction is three-phase because og.sim.play() and og.sim.stop() are global:

    1. build every member with in_vec_env=True   (loads scenes, does not play)
    2. og.sim.play() once, then post_play_load() + bind_scene_handles() per member
    3. one stop/play cycle around apply_scene_fixes_from_cfg() for all members, then rebase every
       member's reset baseline onto the fixed scene, then finalize

Poses stay scene-relative throughout: REALM's external cameras use pose_frame "parent" and are
loaded into their own env's scene, and robots default to frame "scene", so each member's camera
extrinsics and spawn pose land correctly inside its own tile without knowing the tile offset.
"""
import omnigibson as og

from realm.environments.env_dynamic import RealmEnvironmentDynamic, WARMUP_STEPS
from realm.environments.joint_reset import run_joint_resets
from realm.environments.perturbations._helpers import (
    NEEDS_STOPPED_SIM,
    SETTLE_STEPS,
    settle_action,
)
from realm.environments.vec_init_queue import repair_init_queue


class RealmVectorEnvironment:
    """N RealmEnvironmentDynamic members sharing one simulator.

    The members are independent scenes running the same task config; per-member variation comes
    from the sampling done while building each member's config (object placement is drawn per
    member), not from different task files.
    """

    def __init__(self, num_envs, on_first_env_built=None, **env_kwargs):
        """
        Args:
            num_envs (int): number of parallel environments
            on_first_env_built (None or callable): invoked once, right after member 0 exists. Isaac
                is not running until the first og.Environment is created, so carb settings cannot be
                touched before then -- but they must be set BEFORE the remaining scenes load if they
                are to affect how the renderer sizes its pools. Passing a lighter renderer profile
                here is what makes higher member counts reachable; docs/vector_env/SCALING.md has the
                measurement and the settings that raise the ceiling to 16.
            **env_kwargs: forwarded verbatim to each RealmEnvironmentDynamic
        """
        assert num_envs >= 1, f"num_envs must be >= 1, got {num_envs}"
        self.num_envs = num_envs

        # OmniGibson requires a stopped sim to import a scene; a previous run may have left it playing.
        if og.sim is not None:
            og.sim.stop()

        og.log.info(f"Loading {num_envs} parallel environments...")
        self.envs = []
        for i in range(num_envs):
            og.log.info(f"  environment {i + 1}/{num_envs}")
            self.envs.append(RealmEnvironmentDynamic(in_vec_env=True, **env_kwargs))
            if i == 0 and on_first_env_built is not None:
                on_first_env_built()

        # Play once for every scene, then let each member finish the loading that needs a live sim.
        og.sim.play()
        for env in self.envs:
            env.omnigibson_env.post_play_load()
        for env in self.envs:
            env.bind_scene_handles()

        # Scene fixes stop and play the sim, so batch them: one cycle for all members instead of one
        # per member. Ordering within a member is unchanged from the single-env path.
        og.sim.stop()
        for env in self.envs:
            env.apply_scene_fixes_from_cfg(manage_sim_state=False)
        og.sim.play()
        # After the shared play(), because Scene.save() asserts a non-stopped sim. The single-env
        # path makes this call inside apply_scene_fixes_from_cfg, right after its own play(); here
        # the play is hoisted out for all members, so the rebase has to be hoisted with it.
        for env in self.envs:
            env.rebase_initial_file()

        for env in self.envs:
            env.finalize_setup()
        # finalize_setup() -> RealmEnvironmentBase.__init__ -> reset_joints(), which in a vector env
        # only records the plan.
        self._drain_joint_resets()
        og.log.info(f"{num_envs} environments ready.")

    # ============================== [ROLLOUT] ==============================
    def reset(self):
        """Reset every member, doing each GLOBAL simulator operation exactly once.

        A member's reset is not self-contained: perturbations stop, play and step the simulator, and
        every one of those acts on ALL scenes -- so `[env.reset() for env in self.envs]` had each
        member tear down and rebuild its siblings' scenes mid-reset. See
        perturbations/_helpers.py for the measurement that cost.

        The per-member work is therefore split into phases and the global operations hoisted out:

            1. every member restores its own scene                    (no global state touched)
            2. ONE joint-reset loop for every member that asked for one (drawer tasks only)
            3. ONE og.sim.stop(), if any member's perturbation needs it
            4. every member's perturbations run
            5. ONE og.sim.play(), then repair of the sim's object-init queue
            6. work the perturbations deferred because it needs a playing sim
            7. ONE joint-reset loop again, for the perturbations that ask for one
            8. ONE settle loop driving all members together, if any asked for it
            9. every member re-takes its main-object scoring reference

        Mirrors the batching already used for scene fixes in __init__ and for the warmup loop.
        Returns a list of per-member (obs, info).
        """
        results = [env.reset_pre_perturbation() for env in self.envs]
        # Drain HERE, before the stop below: og.sim.step() asserts a playing sim.
        self._drain_joint_resets()

        # Only cycle the sim if a perturbation actually requires a stopped one (adding or removing
        # objects). Pose-only perturbations such as VB-POSE and V-VIEW write on a live sim, and
        # cycling for them would reintroduce the very disruption this method exists to avoid.
        needs_stop = any(
            p in NEEDS_STOPPED_SIM for env in self.envs for p in env.active_perturbations
        )
        if needs_stop:
            og.sim.stop()

        obss = [env.apply_perturbations(res[0]) for env, res in zip(self.envs, results)]

        if needs_stop:
            # BEFORE play(), never after -- see environments/vec_init_queue.py.
            orphans = repair_init_queue(self.envs)
            og.sim.play()
            # Loud rather than silent, kept from when the repair ran after play(): if re-queueing
            # ever stops working, the next symptom is the same opaque "Object must be initialized
            # before dumping state!" raised from an unrelated call site much later.
            failed = [f"scene{o.scene.idx}/{o.name}" for o in orphans if not o.initialized]
            assert not failed, f"objects still uninitialized after re-queue + play(): {failed}"

        for env in self.envs:
            for fn in env.deferred_post_play:
                fn()
            env.deferred_post_play.clear()

        # Perturbations call reset_joints() too: V-VIEW, VB-POSE and SB-NOUN inline during
        # apply_perturbations(), and V-SC, VB-MOBJ, VSB-NOBJ and SB-VRB from the _post_play blocks
        # just drained. Both land here -- after the shared play, so the sim is playing, and before
        # the shared settle, which is the order a single env runs them in too.
        self._drain_joint_resets()

        if any(env.wants_settle for env in self.envs):
            self._settle()
        for env in self.envs:
            env.wants_settle = False

        # LAST, after the shared play and settle: a replaced object is not initialized (and a
        # settling one has not stopped moving) before that. RealmEnvironmentDynamic.reset() makes
        # this call at its own tail; a vector env drives the phases itself, so it makes it itself --
        # same reason the settle and the deferred post-play work are hoisted up here.
        for env in self.envs:
            env.capture_mo_reference()

        # Nothing may still be pending. reset_joints() RECORDS in a vector env, so a call site that
        # lands outside the drain points above would leave a drawer at whatever openness the
        # previous rollout ended on and score the next one against it -- silently, since every other
        # check would still pass. Loud here, exactly as for the init-queue repair above.
        stuck = [i for i, env in enumerate(self.envs) if env.pending_joint_reset is not None]
        assert not stuck, (
            f"members {stuck} recorded a joint reset that was never run -- a reset_joints() call "
            f"site is outside RealmVectorEnvironment's drain points"
        )

        return [(obs, res[1]) for obs, res in zip(obss, results)]

    def _drain_joint_resets(self):
        """Run every member's pending drawer reset on ONE shared step loop.

        Called at every point a member could have recorded a plan: after construction's
        finalize_setup() loop, after phase 1 of reset(), and after the perturbations. reset() then
        asserts nothing is left pending, so a future reset_joints() call site outside those points
        fails loudly instead of silently skipping the drawer reset. See environments/joint_reset.py
        for why the stepping is shared.
        """
        run_joint_resets(self.envs)

    def _settle(self, steps=SETTLE_STEPS):
        """Let every member's scene come to rest, on one shared step loop.

        Deliberately goes through the underlying OmniGibson env rather than this class's step():
        post_step() calls recompute_task_progression(), which MUTATES task_progression, so settling
        through it would credit the rollout with progress made before the policy ever acted. The
        single-env path avoids this the same way, by calling omnigibson_env.step() rather than
        RealmEnvironmentDynamic.step().
        """
        actions = [settle_action(env) for env in self.envs]
        # Nothing reads a camera here, so skip the render pass on each step. gm.HEADLESS does NOT do
        # this -- og.sim.step() renders every call regardless; only this context suppresses it.
        with og.sim.render_on_step(False):
            for _ in range(steps):
                for env, action in zip(self.envs, actions):
                    env.omnigibson_env._pre_step(action)
                og.sim.step()
                for env, action in zip(self.envs, actions):
                    env.omnigibson_env._post_step(action)

    def step(self, actions, n_render_iterations=1):
        """Apply one action per member and advance the shared simulator once.

        Args:
            actions (list): one action per member, same format as RealmEnvironmentDynamic.step
            n_render_iterations (int): render passes before observations are read. Mirrors
                og.Environment.step: one render happens inside og.sim.step() when rendering is
                enabled, and this issues the remaining n-1 as explicit og.sim.render() calls. The
                flush is GLOBAL -- one pass refreshes every scene -- so it is done once here rather
                than per member.

        Returns:
            list: per-member (obs, task_progression, terminated, truncated, info)
        """
        assert len(actions) == self.num_envs, f"expected {self.num_envs} actions, got {len(actions)}"
        for env, action in zip(self.envs, actions):
            env.pre_step(action)
        og.sim.step()
        for _ in range(n_render_iterations - 1):
            og.sim.render()
        return [env.post_step(action) for env, action in zip(self.envs, actions)]

    def warmup(self):
        """Settle every member together: hold the arm still, open then close the gripper.

        Mirrors RealmEnvironmentDynamic.warmup() but drives all members off one shared step loop,
        so the 30 settle steps cost the same wall time for 1 member as for N.
        """
        og.log.info("Starting vector warmup...")
        for _ in range(30):
            og.sim.render()

        results = self.reset()
        ee_cmds = [env.warmup_ee_cmd() for env in self.envs]

        for t in range(WARMUP_STEPS):
            actions = [env.warmup_action(t, ee_cmd) for env, ee_cmd in zip(self.envs, ee_cmds)]
            results = self.step(actions)

        # Refine the reference now the arms have settled, mirroring the single-env warmup. reset()
        # above already took it from the right object; this only updates it to the settled pose.
        for env in self.envs:
            env.capture_mo_reference()
        og.log.info("Vector warmup finished.")
        return results

    def close(self):
        pass

    def __len__(self):
        return self.num_envs

    def __getitem__(self, idx):
        return self.envs[idx]
