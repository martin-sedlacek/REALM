import contextlib

import omnigibson as og

from realm import frame_mode
from realm.environments.env_dynamic import RealmEnvironmentDynamic, WARMUP_STEPS
from realm.environments.foam_ball_reset import run_foam_ball_placements
from realm.environments.joint_reset import run_joint_resets
from realm.environments.perturbations._helpers import (
    NEEDS_STOPPED_SIM,
    SETTLE_STEPS,
    settle_action,
)
from realm.environments.vec_init_queue import repair_init_queue

def real_scene_bbox(idx):
    """Real bbox (min, max) of /World/scene_<idx> right now, or None if it is empty."""
    import omni.usd
    bmin, bmax = omni.usd.get_context().compute_path_world_bounding_box(f"/World/scene_{idx}")
    bmin, bmax = [float(v) for v in bmin], [float(v) for v in bmax]
    return (bmin, bmax) if bmin[0] <= bmax[0] else None


@contextlib.contextmanager
def use_template_bbox(template):
    """While active: when OmniGibson measures a scene that is still empty during loading
    (plain Scene, objects added later), answer with `template` (scene 0's real bbox).
    Non-empty boxes (e.g. houses) are returned unchanged."""
    import omni.usd
    real_get_context = omni.usd.get_context
    tmin, tmax = template

    class _Ctx:
        def __init__(self, ctx):
            self._ctx = ctx
        def __getattr__(self, name):
            return getattr(self._ctx, name)
        def compute_path_world_bounding_box(self, path):
            bmin, bmax = self._ctx.compute_path_world_bounding_box(path)
            if bmin[0] > bmax[0]:                       # empty box
                return tuple(tmin), tuple(tmax)
            return bmin, bmax

    omni.usd.get_context = lambda *a, **k: _Ctx(real_get_context(*a, **k)) # -- monkey patching
    try:
        yield
    finally:
        omni.usd.get_context = real_get_context


class RealmVectorEnvironment:


    def __init__(self, num_envs, on_first_env_built=None, wait_idle=False, hdri_background=False, **env_kwargs):

        assert num_envs >= 1, f"num_envs must be >= 1, got {num_envs}"
        self.num_envs = num_envs

        if og.sim is not None:
            og.sim.stop()

        og.log.info(f"Loading {num_envs} parallel environments...")
        self.envs = []

        # Isaac Sim launches with the first member (realm.frame_mode).
        with frame_mode.launch_settings():
            for i in range(num_envs):
                og.log.info(f"  environment {i + 1}/{num_envs}")
                if i == 0:
                    self.envs.append(RealmEnvironmentDynamic(in_vec_env=True, hdri_background=hdri_background, **env_kwargs))
                    size = real_scene_bbox(0)
                    og.sim._last_scene_edge = size[1][0]
                else:
                    with use_template_bbox(size):   # -- empty scene i -> measured as scene 0's size
                        self.envs.append(RealmEnvironmentDynamic(in_vec_env=True, hdri_background=hdri_background, **env_kwargs))
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
        # After the joint drain and after rebase_initial_file above: this ends by re-snapshotting
        # each member's initial file with the balls inside its settled source, which has to be the
        # last word on what reset() restores. One batched call for all members -- see
        # foam_ball_reset for why a per-member loop would be wrong here.
        run_foam_ball_placements(self.envs)
        frame_mode.apply(wait_idle)
        og.log.info(f"{num_envs} environments ready.")

    def reset(self):

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
            env.capture_foam_ball_reference()

        stuck = [i for i, env in enumerate(self.envs) if env.pending_joint_reset is not None]
        assert not stuck, (
            f"members {stuck} recorded a joint reset that was never run -- a reset_joints() call "
            f"site is outside RealmVectorEnvironment's drain points"
        )

        return [(obs, res[1]) for obs, res in zip(obss, results)]

    def _drain_joint_resets(self):

        run_joint_resets(self.envs)

    def _settle(self, steps=SETTLE_STEPS):

        actions = [settle_action(env) for env in self.envs]
        with og.sim.render_on_step(False):
            for _ in range(steps):
                for env, action in zip(self.envs, actions):
                    env.omnigibson_env._pre_step(action)
                og.sim.step()
                for env, action in zip(self.envs, actions):
                    env.omnigibson_env._post_step(action)

    def step(self, actions, n_render_iterations=1):

        assert len(actions) == self.num_envs, f"expected {self.num_envs} actions, got {len(actions)}"
        for env, action in zip(self.envs, actions):
            env.pre_step(action)
        og.sim.step()
        for _ in range(n_render_iterations - 1):
            og.sim.render()
        return [env.post_step(action) for env, action in zip(self.envs, actions)]

    def current_obs(self):
        """One waited render pass for the whole batch, then every member's `read_obs()`."""
        frame_mode.render_current()
        return [env.read_obs() for env in self.envs]

    def warmup(self):

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
            env.capture_foam_ball_reference()
        og.log.info("Vector warmup finished.")
        return results

    def close(self):
        pass

    def __len__(self):
        return self.num_envs

    def __getitem__(self, idx):
        return self.envs[idx]
