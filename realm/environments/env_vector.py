"""Vectorized REALM environments: N scenes in one simulator, stepped together.

OmniGibson 3.9.1 loads several scenes into one physics stage, tiled side by side with a margin
(Simulator.import_scene -> Scene.load(idx=..., last_scene_edge=...)). One og.sim.step() advances
every scene, so members must not be stepped individually -- doing so would advance all scenes while
applying only one member's action. That is what this class exists to prevent; it mirrors
OmniGibson's own VectorEnvironment, with REALM's per-member setup and task bookkeeping layered on.

Construction is three-phase because og.sim.play() and og.sim.stop() are global:

    1. build every member with in_vec_env=True   (loads scenes, does not play)
    2. og.sim.play() once, then post_play_load() + bind_scene_handles() per member
    3. one stop/play cycle around apply_scene_fixes_from_cfg() for all members, then finalize

Poses stay scene-relative throughout: REALM's external cameras use pose_frame "parent" and are
loaded into their own env's scene, and robots default to frame "scene", so each member's camera
extrinsics and spawn pose land correctly inside its own tile without knowing the tile offset.
"""
import numpy as np

import omnigibson as og

from realm.environments.env_dynamic import RealmEnvironmentDynamic, WARMUP_STEPS


class RealmVectorEnvironment:
    """N RealmEnvironmentDynamic members sharing one simulator.

    The members are independent scenes running the same task config; per-member variation comes
    from the sampling done while building each member's config (object placement is drawn per
    member), not from different task files.
    """

    def __init__(self, num_envs, **env_kwargs):
        """
        Args:
            num_envs (int): number of parallel environments
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

        for env in self.envs:
            env.finalize_setup()
        og.log.info(f"{num_envs} environments ready.")

    # ============================== [ROLLOUT] ==============================
    def reset(self):
        """Reset every member. Returns a list of per-member (obs, info)."""
        return [env.reset() for env in self.envs]

    def step(self, actions):
        """Apply one action per member and advance the shared simulator once.

        Args:
            actions (list): one action per member, same format as RealmEnvironmentDynamic.step

        Returns:
            list: per-member (obs, task_progression, terminated, truncated, info)
        """
        assert len(actions) == self.num_envs, f"expected {self.num_envs} actions, got {len(actions)}"
        for env, action in zip(self.envs, actions):
            env.pre_step(action)
        og.sim.step()
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

        for env in self.envs:
            env.mo_pos_orig, env.mo_rot_orig = env.main_objects[0].get_position_orientation()
        og.log.info("Vector warmup finished.")
        return results

    def close(self):
        pass

    def __len__(self):
        return self.num_envs

    def __getitem__(self, idx):
        return self.envs[idx]
