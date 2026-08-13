from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING

import omnigibson as og
from realm.environments.perturbations._helpers import settle
from realm.geometry import add_rotation_noise
from realm.placement import get_non_colliding_positions_for_objects

if TYPE_CHECKING:
    from realm.environments.env_dynamic import RealmEnvironmentDynamic


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
    # --------------- Translation ---------------
    if env.task_type == "push":
        delta_z = np.random.uniform(-0.15, 0.15)
        delta_xy = np.random.uniform(-0.075, 0.075)
        for obj_cfg in env.cfg["objects"]:
            if obj_cfg["name"] == "electric_switch":
                obj = env.omnigibson_env.scene.object_registry("name", obj_cfg["name"])
                init_pos = env.init_poses[obj._relative_prim_path]["pos"]
                init_pos[2] += delta_z
                init_pos[0] += delta_xy # TODO: this is only for pomaria light switch, elsewhere it might be y axis on the wall...
                # world frame here: env.init_poses is captured with get_position_orientation()'s
                # default (world), so this branch reads and writes the same frame throughout.
                _place(obj, position=init_pos, frame="world")
    else:
        for scene_obj in env.main_objects + env.distractors + env.target_objects:
            for cfg in env.cfg["objects"]:
                if cfg["name"] == scene_obj.name:
                    if "position" not in cfg:
                        # Scene frame: this backfills the same cfg["position"] field that
                        # get_non_colliding_positions_for_objects rewrites from the scene-relative
                        # spawn_bbox, so it has to be in that frame too. Reading it in world frame
                        # mixed the two and only agreed for scene 0.
                        cfg["position"] = scene_obj.get_position_orientation(frame="scene")[0].tolist()
                    if "bounding_box" not in cfg:
                        cfg["bounding_box"] = scene_obj.aabb_extent.tolist()

        env.cfg["objects"] = get_non_colliding_positions_for_objects(
            xmin=env.spawn_bbox[0],
            xmax=env.spawn_bbox[1],
            ymin=env.spawn_bbox[2],
            ymax=env.spawn_bbox[3],
            z=env.spawn_bbox[4],
            obj_cfg=env.cfg["objects"],
            objects_to_skip=[obj.name for obj in env.distractors + env.target_objects],
            main_object_names=[],
            max_attempts_per_object=25000 # TODO: this must be successful, careful what we do here...
        )

        for obj_cfg in env.cfg["objects"]:
            if env.task_type in ["open_drawer", "close_drawer"] and obj_cfg["name"] == "drawer":
                obj_cfg["position"][-1] -= 0.3
            _place(env.omnigibson_env.scene.object_registry("name", obj_cfg["name"]),
                   position=obj_cfg["position"])

        # --------------- Rotation ---------------
        # Reads the orientation set by the loop above, which wrote position only -- so this still
        # sees the pre-perturbation orientation, exactly as it did when both loops ran stopped.
        for o in env.main_objects:
            if env.task_type in ["open_drawer", "close_drawer"]:
                for obj_cfg in env.cfg["objects"]:
                    if obj_cfg["name"] == "drawer":
                        tmp_obj_cfg = obj_cfg
                tmp = tmp_obj_cfg["orientation"] if "orientation" in tmp_obj_cfg else [0, 0, 0, 1]
                new_rot = add_rotation_noise(tmp, (0, 0, 0.12), [-3.14, -3.14, 0], [3.14, 3.14, 0.57], (0, 0, 0.25))
                _place(o, orientation=new_rot)
            else:
                # Scene frame to match the write below. Scene prims are placed with an identity
                # orientation, so world and scene orientations coincide today and this is not a
                # behaviour change -- it is written explicitly so the read and the write cannot
                # drift apart if a scene is ever placed rotated.
                tmp = o.get_position_orientation(frame="scene")[1] # TODO: also from orig rot?
                _place(o, orientation=add_rotation_noise(tmp, (0, 0, 3.14)))
        # Kept from the stop/play version: reset() already ran this before the perturbation, and the
        # robot is no longer disturbed now that nothing stops the sim, so it is a no-op here. Left in
        # so this function's post-state is identical to what it was.
        env.reset_joints()

    # Settle the teleported objects onto the surface. This used to also drive the robot back to its
    # pose after og.sim.stop() reset it; with the stop gone the robot never moves, so these steps now
    # do only the settling. In a vector env this is a no-op and RealmVectorEnvironment.reset() runs
    # the equivalent loop once for every member -- og.sim.step() is global, so doing it per member
    # stepped the shared sim 30*N times while feeding N-1 members no action.
    settle(env)
