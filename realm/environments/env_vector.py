"""Vectorized REALM environments sharing one simulator."""
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
    """N independent REALM scenes advanced by one simulator step."""

    def __init__(self, num_envs, on_first_env_built=None, **env_kwargs):
        """Build ``num_envs`` members from the same environment arguments."""
        assert num_envs >= 1, f"num_envs must be >= 1, got {num_envs}"
        self.num_envs = num_envs

        if og.sim is not None:
            og.sim.stop()

        og.log.info(f"Loading {num_envs} parallel environments...")
        self.envs = []
        for i in range(num_envs):
            og.log.info(f"  environment {i + 1}/{num_envs}")
            self.envs.append(RealmEnvironmentDynamic(in_vec_env=True, **env_kwargs))
            if i == 0 and on_first_env_built is not None:
                on_first_env_built()

        og.sim.play()
        for env in self.envs:
            env.omnigibson_env.post_play_load()
        for env in self.envs:
            env.bind_scene_handles()

        # Simulator play/stop is global, so setup transitions are shared.
        og.sim.stop()
        for env in self.envs:
            env.apply_scene_fixes_from_cfg(manage_sim_state=False)
        og.sim.play()
        for env in self.envs:
            env.rebase_initial_file()

        for env in self.envs:
            env.finalize_setup()
        self._drain_joint_resets()
        og.log.info(f"{num_envs} environments ready.")

    def reset(self):
        """Reset every member while batching global simulator transitions."""
        results = [env.reset_pre_perturbation() for env in self.envs]
        self._drain_joint_resets()

        needs_stop = any(
            p in NEEDS_STOPPED_SIM for env in self.envs for p in env.active_perturbations
        )
        if needs_stop:
            og.sim.stop()

        obss = [env.apply_perturbations(res[0]) for env, res in zip(self.envs, results)]

        if needs_stop:
            orphans = repair_init_queue(self.envs)
            og.sim.play()
            failed = [f"scene{o.scene.idx}/{o.name}" for o in orphans if not o.initialized]
            assert not failed, f"objects still uninitialized after re-queue + play(): {failed}"

        for env in self.envs:
            for fn in env.deferred_post_play:
                fn()
            env.deferred_post_play.clear()

        self._drain_joint_resets()

        if any(env.wants_settle for env in self.envs):
            self._settle()
        for env in self.envs:
            env.wants_settle = False

        for env in self.envs:
            env.capture_mo_reference()

        stuck = [i for i, env in enumerate(self.envs) if env.pending_joint_reset is not None]
        assert not stuck, (
            f"members {stuck} recorded a joint reset that was never run -- a reset_joints() call "
            f"site is outside RealmVectorEnvironment's drain points"
        )

        return [(obs, res[1]) for obs, res in zip(obss, results)]

    def _drain_joint_resets(self):
        """Run pending joint resets on one shared step loop."""
        run_joint_resets(self.envs)

    def _settle(self, steps=SETTLE_STEPS):
        """Settle all scenes without updating task progression."""
        actions = [settle_action(env) for env in self.envs]
        with og.sim.render_on_step(False):
            for _ in range(steps):
                for env, action in zip(self.envs, actions):
                    env.omnigibson_env._pre_step(action)
                og.sim.step()
                for env, action in zip(self.envs, actions):
                    env.omnigibson_env._post_step(action)

    def step(self, actions, n_render_iterations=1):
        """Apply one action per member and advance the simulator once."""
        assert len(actions) == self.num_envs, f"expected {self.num_envs} actions, got {len(actions)}"
        for env, action in zip(self.envs, actions):
            env.pre_step(action)
        og.sim.step()
        for _ in range(n_render_iterations - 1):
            og.sim.render()
        return [env.post_step(action) for env, action in zip(self.envs, actions)]

    def warmup(self):
        """Warm up all members on one shared step loop."""
        og.log.info("Starting vector warmup...")
        for _ in range(30):
            og.sim.render()

        results = self.reset()
        ee_cmds = [env.warmup_ee_cmd() for env in self.envs]

        for t in range(WARMUP_STEPS):
            actions = [env.warmup_action(t, ee_cmd) for env, ee_cmd in zip(self.envs, ee_cmds)]
            results = self.step(actions)

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
