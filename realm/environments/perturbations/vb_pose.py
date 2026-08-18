"""VB-POSE: perturb the pose of the task objects at reset.

Push tasks get a switch-specific nudge (`_perturb_switch`); everything else gets a full
re-placement of the movable objects plus rotation noise on the main objects (`_perturb_tabletop`).

What one call mutates on @env: ``cfg["objects"]`` positions (tabletop branch -- the placement pass
writes them in place), the live objects' poses, and -- push branch only, see the KNOWN ISSUE
inside -- ``init_poses``.
"""
from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING

from realm.environments.perturbations._helpers import backfill_object_cfgs, settle
from realm.geometry import add_rotation_noise
from realm.placement import place_within

if TYPE_CHECKING:
    from realm.environments.env_dynamic import RealmEnvironmentDynamic


#: Push-task switch nudge (metres): the switch slides across the wall plane.
SWITCH_DZ_RANGE = 0.15
SWITCH_DXY_RANGE = 0.075

#: Drawer rotation noise (radians): a yaw-only draw from N(mean 0.25, std 0.12), clipped to
#: [0, 0.57]. The roll/pitch entries have std 0, so their +/-3.14 clips are inert.
DRAWER_YAW_NOISE_STD = (0, 0, 0.12)
DRAWER_YAW_NOISE_MEAN = (0, 0, 0.25)
DRAWER_YAW_CLIP_MIN = [-3.14, -3.14, 0]
DRAWER_YAW_CLIP_MAX = [3.14, 3.14, 0.57]

#: Free yaw noise (radians, std of a normal draw) for every non-drawer main object.
TABLETOP_YAW_NOISE_STD = (0, 0, 3.14)

#: Drawers are re-placed this far below the sampled tabletop height (metres).
DRAWER_Z_OFFSET = 0.3


def _place(obj, position=None, orientation=None, frame="scene"):
    """Teleport @obj on a LIVE sim, leaving no residual motion.

    FRAME. Defaults to "scene", NOT OmniGibson's "world" default, because the coordinates this
    perturbation works in are scene-relative: spawn_bbox comes from authored constants in
    scenes.yaml, and OmniGibson's own loader offsets authored object poses by scene_position when
    it builds a scene (scene_base.py, "local then works out to exactly the authored pose").
    Writing them as world coordinates is a no-op in scene 0 -- whose origin IS the world origin --
    and silently wrong in every other scene, which is why this only ever showed up vectorized.
    Measured (job 190555 and the t9 smoke test): all four members' cubes landed within 0.2 m of
    each other in WORLD space, i.e. every member's object was teleported into scene 0's tile.
    Scenes are tiled along +x, so those objects ended up many metres from their own robot, well
    outside gm.PROXIMITY_GATE_RADIUS (1.5 m); the proximity gate then dropped them from the
    contact view, and a body that is not a contact ROW can never register a grasp. Result: scenes
    1..3 sat at 40 contact rows against scene 0's 49-51, TP pinned at 0.00, zero environment
    collisions, every rollout stuck at REACH -- while the job still exited 0.

    Callers working in world coordinates (env.init_poses is captured with the default world frame)
    must pass frame="world" explicitly.

    This exists so VB-POSE does not have to cycle og.sim.stop()/play(). That cycle is GLOBAL, and
    REALM applies perturbations per member inside reset(), so in a vector env one member's
    perturbation tears down and rebuilds every other member's scene mid-reset. Measured cost of that
    (job 190555, VB-POSE Vec=4): the main object dropped out of the contact view for scenes 1, 2 and
    3 -- 18 of 25 rollouts recorded zero environment collisions and never advanced past REACH, i.e.
    three quarters of the run measured nothing while still exiting 0.

    Stopping was never needed for a pose write in the first place: only add/remove needs a stopped
    sim (see the note in realm/placement.py), and set_position_orientation works on a playing sim --
    placement.py itself calls it both ways. What a live write does need is keep_still(): teleporting
    a body leaves its pre-teleport linear/angular velocity attached, which stop()/play() used to
    discard for us. Without it the object launches itself out of the new pose on the next step.
    """
    obj.set_position_orientation(position=position, orientation=orientation, frame=frame)
    obj.keep_still()


def vb_pose(env: "RealmEnvironmentDynamic") -> None:
    if env.task_type == "push":
        _perturb_switch(env)
    else:
        _perturb_tabletop(env)

    # Settle the teleported objects onto the surface. In a vector env this is a no-op and
    # RealmVectorEnvironment.reset() runs the equivalent loop once for every member -- og.sim.step()
    # is global, so doing it per member would step the shared sim 30*N times while feeding N-1
    # members no action.
    settle(env)


def _perturb_switch(env: "RealmEnvironmentDynamic") -> None:
    """Nudge the light switch across its wall plane."""
    delta_z = np.random.uniform(-SWITCH_DZ_RANGE, SWITCH_DZ_RANGE)
    delta_xy = np.random.uniform(-SWITCH_DXY_RANGE, SWITCH_DXY_RANGE)
    for obj_cfg in env.cfg["objects"]:
        if obj_cfg["name"] == "electric_switch":
            obj = env.omnigibson_env.scene.object_registry("name", obj_cfg["name"])
            # KNOWN ISSUE, deliberately not fixed in the behaviour-preserving cleanup pass:
            # init_pos ALIASES env.init_poses[...]["pos"], so the += below mutates the stored
            # reference and the offsets COMPOUND across resets -- the switch drifts further every
            # reset. Fixing it changes what VB-POSE has historically measured on push tasks, so it
            # is gated with the other number-moving fixes rather than slipped into a refactor.
            init_pos = env.init_poses[obj._relative_prim_path]["pos"]
            init_pos[2] += delta_z
            init_pos[0] += delta_xy # TODO: this is only for pomaria light switch, elsewhere it might be y axis on the wall...
            # world frame here: env.init_poses is captured with get_position_orientation()'s
            # default (world), so this branch reads and writes the same frame throughout.
            _place(obj, position=init_pos, frame="world")


def _perturb_tabletop(env: "RealmEnvironmentDynamic") -> None:
    """Re-place the movable objects collision-free, then add rotation noise to the main objects."""
    # --------------- Translation ---------------
    backfill_object_cfgs(env.main_objects + env.distractors + env.target_objects,
                         env.cfg["objects"])

    env.cfg["objects"] = place_within(
        env.spawn_bbox,
        env.cfg["objects"],
        objects_to_skip=[obj.name for obj in env.distractors + env.target_objects],
        main_object_names=[],
        max_attempts_per_object=25000 # TODO: this must be successful, careful what we do here...
    )

    for obj_cfg in env.cfg["objects"]:
        if env.task_type in ["open_drawer", "close_drawer"] and obj_cfg["name"] == "drawer":
            obj_cfg["position"][-1] -= DRAWER_Z_OFFSET
        _place(env.omnigibson_env.scene.object_registry("name", obj_cfg["name"]),
               position=obj_cfg["position"])

    # --------------- Rotation ---------------
    # Reads the orientation set by the loop above, which wrote position only -- so this still
    # sees the pre-perturbation orientation, exactly as it did when both loops ran stopped.
    for o in env.main_objects:
        if env.task_type in ["open_drawer", "close_drawer"]:
            drawer_cfg = next((c for c in env.cfg["objects"] if c["name"] == "drawer"), None)
            if drawer_cfg is None:
                raise RuntimeError(
                    "VB-POSE on a drawer task found no 'drawer' entry in cfg['objects']")
            current = drawer_cfg["orientation"] if "orientation" in drawer_cfg else [0, 0, 0, 1]
            new_rot = add_rotation_noise(current, DRAWER_YAW_NOISE_STD,
                                         DRAWER_YAW_CLIP_MIN, DRAWER_YAW_CLIP_MAX,
                                         DRAWER_YAW_NOISE_MEAN)
            _place(o, orientation=new_rot)
        else:
            # Scene frame to match the write below. Scene prims are placed with an identity
            # orientation, so world and scene orientations coincide today and this is not a
            # behaviour change -- it is written explicitly so the read and the write cannot
            # drift apart if a scene is ever placed rotated.
            current = o.get_position_orientation(frame="scene")[1] # TODO: also from orig rot?
            _place(o, orientation=add_rotation_noise(current, TABLETOP_YAW_NOISE_STD))

    # Kept from the stop/play version: reset() already ran this before the perturbation, and the
    # robot is no longer disturbed now that nothing stops the sim, so it is a no-op here. Left in
    # so this function's post-state is identical to what it was.
    env.reset_joints()
