from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING

import omnigibson as og
from realm.helpers import get_non_colliding_positions_for_objects, add_rotation_noise

if TYPE_CHECKING:
    from realm.environments.env_dynamic import RealmEnvironmentDynamic


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
                og.sim.stop()
                obj.set_position_orientation(init_pos)
                og.sim.play()
    else:
        for scene_obj in env.main_objects + env.distractors + env.target_objects:
            for cfg in env.cfg["objects"]:
                if cfg["name"] == scene_obj.name:
                    if "position" not in cfg:
                        cfg["position"] = scene_obj.get_position_orientation()[0].tolist()
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

        og.sim.stop()
        for obj_cfg in env.cfg["objects"]:
            if env.task_type in ["open_drawer", "close_drawer"] and obj_cfg["name"] == "drawer":
                obj_cfg["position"][-1] -= 0.3
            env.omnigibson_env.scene.object_registry("name", obj_cfg["name"]).set_position_orientation(obj_cfg["position"])

        # --------------- Rotation ---------------
        for o in env.main_objects:
            if env.task_type in ["open_drawer", "close_drawer"]:
                for obj_cfg in env.cfg["objects"]:
                    if obj_cfg["name"] == "drawer":
                        tmp_obj_cfg = obj_cfg
                tmp = tmp_obj_cfg["orientation"] if "orientation" in tmp_obj_cfg else [0, 0, 0, 1]
                new_rot = add_rotation_noise(tmp, (0, 0, 0.12), [-3.14, -3.14, 0], [3.14, 3.14, 0.57], (0, 0, 0.25))
                o.set_orientation(new_rot)
            else:
                tmp = o.get_position_orientation()[1] # TODO: also from orig rot?
                o.set_orientation(add_rotation_noise(tmp, (0, 0, 3.14)))
        og.sim.play()
        env.reset_joints()

    # fake rest to get to original pose after stopping sim
    for _ in range(30):
        env.omnigibson_env.step(np.concatenate((env.reset_qpos[:7], np.atleast_1d(np.array([-1])))))
