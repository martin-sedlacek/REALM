
from __future__ import annotations
from realm.config.shared import WARMUP_STEPS, SETTLE_STEPS

from typing import TYPE_CHECKING

import numpy as np

import omnigibson as og

if TYPE_CHECKING:
    from realm.environments.env_dynamic import RealmEnvironmentDynamic



# OmniGibson requires a stopped simulator when objects are added or removed.
NEEDS_STOPPED_SIM = frozenset({"V-SC", "VB-MOBJ", "VSB-NOBJ", "SB-VRB"})


def sim_stop(env: "RealmEnvironmentDynamic") -> None:
    if not env.in_vec_env:
        og.sim.stop()


def sim_play(env: "RealmEnvironmentDynamic") -> None:
    if not env.in_vec_env:
        og.sim.play()


def sim_step(env: "RealmEnvironmentDynamic") -> None:
    if not env.in_vec_env:
        og.sim.step()


def after_play(env: "RealmEnvironmentDynamic", fn) -> None:

    if env.in_vec_env:
        env.deferred_post_play.append(fn)
    else:
        fn()


def settle_action(env: "RealmEnvironmentDynamic"):
    """Hold the reset pose with the gripper closed while a perturbation's objects settle.

    DROID keeps the historical vector verbatim (7 reset joints + a closed gripper). Every other robot
    gets `env.warmup_action` at its closed-gripper phase, which knows the robot's action layout -- the
    hardcoded `[:7] + 1` was 8 wide for the 7-DOF YAM action and for the 14-DOF bimanual one, so
    VB-POSE (and anything else that settles) crashed in `robot.apply_action` for both (job 204600).
    """
    if env.use_droid_with_base:
        return np.concatenate((env.reset_qpos[:7], np.atleast_1d(np.array([-1]))))
    return env.warmup_action(WARMUP_STEPS, env.warmup_ee_cmd())


def settle(env: "RealmEnvironmentDynamic", steps: int = SETTLE_STEPS) -> None:

    if env.in_vec_env:
        env.wants_settle = True
        return
    # HEADLESS still renders; this context is required to skip camera work.
    with og.sim.render_on_step(False):
        for _ in range(steps):
            env.omnigibson_env.step(settle_action(env))


def backfill_object_cfgs(scene_objects, cfgs) -> None:

    for scene_obj in scene_objects:
        for cfg in cfgs:
            if cfg["name"] == scene_obj.name:
                if "position" not in cfg:
                    cfg["position"] = scene_obj.get_position_orientation(frame="scene")[0].tolist()
                if "bounding_box" not in cfg:
                    cfg["bounding_box"] = scene_obj.aabb_extent.tolist()


def set_scene_positions(env: "RealmEnvironmentDynamic", cfgs) -> None:
    for cfg in cfgs:
        env.omnigibson_env.scene.object_registry("name", cfg["name"]).set_position_orientation(
            position=cfg["position"], frame="scene")


def rebase_after_play(env: "RealmEnvironmentDynamic", vec_only_rebase: bool, extra=None) -> None:

    def _post_play():
        og.sim.step()
        if not vec_only_rebase or env.in_vec_env:
            env.omnigibson_env.scene.update_initial_file()
        env.reset_joints()
        if extra is not None:
            extra()

    after_play(env, _post_play)
