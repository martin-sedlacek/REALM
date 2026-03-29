from queue import Queue
import datetime
import time
import os
import random
import csv
import numpy as np
import torch
from scipy.spatial.transform import Rotation as Rot

import omnigibson as og
from omnigibson.macros import gm

from realm.environments.env_dynamic import RealmEnvironmentDynamic
from realm.inference import InferenceClient, extract_from_obs
from realm.realm_logging import VideoRecorder, save_results, append_trajectory, append_video



SUPPORTED_TASKS = [
    "put_green_block_into_bowl", #0
    "put_banana_into_box", #1
    "rotate_marker", #2
    "rotate_mug", #3
    "pick_spoon", #4
    "pick_water_bottle", #5
    "stack_cubes", #6
    "push_switch", #7
    "open_drawer", #8
    "close_drawer", #9
]

SUPPORTED_PERTURBATIONS = [
    'Default', #0
    'V-AUG', # 1
    'V-VIEW',  # 2
    'V-SC', # 1
    'V-LIGHT', # 4
    'S-PROP', # 5
    'S-LANG', # 6
    'S-MO', # 7
    'S-AFF', # 8
    'S-INT', # 9
    'B-HOBJ', # 10
    'SB-NOUN', # 11
    'SB-VRB', # 12
    'VB-POSE',  # 13
    'VB-MOBJ',  # 14
    'VSB-NOBJ' # 15
]


def set_sim_config(rendering_mode=None, robot="DROID"):
    if robot == "WidowX": # TODO: just read this from the yamls...
        gm.DEFAULT_SIM_STEP_FREQ = 5
        gm.DEFAULT_RENDERING_FREQ = 5
    elif "UR5" in robot:
        gm.DEFAULT_SIM_STEP_FREQ = 30
        gm.DEFAULT_RENDERING_FREQ = 30
    else:
        gm.DEFAULT_SIM_STEP_FREQ = 15
        gm.DEFAULT_RENDERING_FREQ = 15

    gm.DEFAULT_PHYSICS_FREQ = 120
    gm.ENABLE_TRANSITION_RULES = False # this needs to be off to avoid bug with sludge state during collision: https://github.com/StanfordVL/BEHAVIOR-1K/issues/1201
    gm.ENABLE_OBJECT_STATES = True # this needs to be on because push_switch task usees the ToggledOn state
    gm.RENDER_VIEWER_CAMERA=False
    gm.ENABLE_HQ_RENDERING = False if rendering_mode == "r" else True

    seed = 1234
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def evaluate(
        task_id=0,
        perturbation_id=0,
        repeats=1,
        max_steps=500,
        horizon=8,
        model_type="pi0_FAST",
        port=8000,
        host="127.0.0.1",
        log_dir="/app/logs",
        resume=False,
        multi_view=False,
        no_record=False,
        no_render=False,
        rendering_mode=None,
        task_cfg_path=None,
        robot="DROID"
):
    start = time.perf_counter()
    og.log.info(f"DEBUG: Begin eval: {time.perf_counter() - start:.4f}s")
    if rendering_mode is None:
        rendering_mode = "rt"
    set_sim_config(rendering_mode=rendering_mode, robot=robot)

    # -------------------- Create the environment + client --------------------
    if task_cfg_path is None:
        task = SUPPORTED_TASKS[task_id]
        task_cfg_path = f"REALM_DROID10/{task}/default.yaml"
    else:
        task = task_cfg_path.split("/")[-2]
        config_name = task_cfg_path.split("/")[-1].replace(".yaml", "").replace(".cfg", "")
        if config_name != "default":
            task = f"{task}_{config_name}"

    perturbations = [SUPPORTED_PERTURBATIONS[perturbation_id]]

    os.makedirs(log_dir, exist_ok=True)

    model_type = model_type # TODO: infer type from model name, rn this will just default to a pi model inference inside the client
    client = InferenceClient(model_type, host=host, port=port)
    og.log.info(f"DEBUG: Client connected: {time.perf_counter() - start:.4f}s")

    env = RealmEnvironmentDynamic(
        config_path="/app/realm/config",
        task_cfg_path=task_cfg_path,
        perturbations=perturbations,
        multi_view=multi_view,
        no_rendering=no_render,
        rendering_mode=rendering_mode,
        robot=robot
    )
    og.log.info(f"DEBUG: Env created: {time.perf_counter() - start:.4f}s")

    results = []
    start_repeat = 0
    results_filename = None

    if resume:
        potential_csv = os.path.join(log_dir, "reports", f"{task}_{perturbations[0]}.csv")
        if os.path.exists(potential_csv):
            results_filename = potential_csv
            with open(results_filename, 'r') as f:
                reader = csv.DictReader(f)
                existing_results = list(reader)
            results = existing_results
            start_repeat = len(results)
            og.log.info(f"Resuming run from repeat {start_repeat}. Using file: {results_filename}")
        else:
            og.log.info(f"Resume requested but no report found. Starting fresh.")

    for run_id in range(repeats):
        # ------------------------ pre-configure each run --------------------------------
        # seed = 1234 + run_id
        # random.seed(seed)
        # np.random.seed(seed)
        # torch.manual_seed(seed)
        # torch.cuda.manual_seed_all(seed)

        if run_id < start_repeat:
            continue

        timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H:%M:%S")
        video_recorder = VideoRecorder(log_dir, timestamp, run_id, task, perturbations[0])

        qpos = []
        actions = []
        action_buffer = Queue()

        # -------------------- Rollout loop --------------------
        obs, _ = env.reset()
        obs, rew, terminated, truncated, info = env.warmup(obs)

        t = 0
        task_progression = 0.0
        task_progression_timestamps = []
        terminal_steps = 15

        ee_poses = []
        collisions_self = 0
        collisions_env = 0
        is_self_col_active = False
        is_env_col_active = False
        drops = 0
        was_grasping = False

        while t < max_steps and terminal_steps > 0:
            base_im, base_depth, base_im_second, base_depth_second, wrist_im, robot_state, gripper_state = extract_from_obs(obs, robot_name=env.robot.name)

            # Metrics collection
            ee_pos, ee_rot = env.get_ee_pose()
            ee_poses.append(ee_pos)

            is_self_col, is_env_col = env.check_collisions()
            if is_self_col and not is_self_col_active:
                collisions_self += 1
            is_self_col_active = is_self_col

            if is_env_col and not is_env_col_active:
                collisions_env += 1
            is_env_col_active = is_env_col

            is_grasping = env.check_grasp_condition(obs)
            if was_grasping and not is_grasping:
                is_placed = False
                if hasattr(env, "task_type") and env.task_type in ["put", "stack"] and len(env.target_objects) > 0:
                    mo = env.main_objects[0]
                    target = env.target_objects[0]
                    inside = mo.states[og.object_states.Inside].get_value(target)
                    on_top = mo.states[og.object_states.OnTop].get_value(target)
                    if inside or on_top:
                        is_placed = True

                if not is_placed:
                    drops += 1
            was_grasping = is_grasping

            if action_buffer.empty():
                # Compute robot-relative cartesian position for models that need it (e.g. DreamZero)
                _ee_pos = ee_pos.cpu().numpy() if hasattr(ee_pos, 'cpu') else np.array(ee_pos)
                _ee_rot = ee_rot.cpu().numpy() if hasattr(ee_rot, 'cpu') else np.array(ee_rot)
                _ee_euler = Rot.from_quat(_ee_rot).as_euler('xyz')
                _ee_pose_world = np.concatenate([_ee_pos, _ee_euler])
                cartesian_position = env._world2robot(_ee_pose_world).astype(np.float32)

                pred_action_chunk = client.infer(
                    env.instruction, base_im, base_im_second, wrist_im, robot_state, gripper_state,
                    use_base_im_second=(env.task_type == "open_close_drawer" if hasattr(env, "task_type") else False),
                    ee_control=env.ee_control,
                    cartesian_position=cartesian_position
                )

                if len(pred_action_chunk.shape) == 2:
                    for action in pred_action_chunk[:horizon]:
                        action = np.squeeze(action)
                        action_buffer.put(action)
                elif len(pred_action_chunk.shape) < 2:
                    action_buffer.put(pred_action_chunk)
                else:
                    assert len(pred_action_chunk.shape) <= 2, f"Unsupported number of dimensions in action chunk with shape: {pred_action_chunk.shape}. The chunk is expected to be 2D."

            if not no_record:
                video_recorder.add_frame(base_im, wrist_im, base_im_second)

            qpos.append(np.concatenate((robot_state, np.atleast_1d(np.array(gripper_state)))))

            action = action_buffer.get()
            actions.append(action)

            new_action = action.copy()
            if model_type in ["debug", "openpi", "GR00T", "GR00T_N16", "dreamzero"]: # TODO: use a model config
                new_action[-1] = 1 if action[-1] > 0.5 else -1  # Prediction: (1,0) -> Target: (1,-1)
            elif model_type == "molmoact":
                new_action[-1] = 1 if action[-1] < 0.5 else -1  # Prediction: (0,1) -> Target: (1,-1)
            else:
                raise NotImplementedError()


            # new_gripper_state = 1 if action[-1] > 0.5 else -1  # Prediction: (1,0) -> Target: (1,-1)
            # new_gripper_state = np.atleast_1d(np.array(new_gripper_state))
            # new_action = np.concatenate((new_action, new_gripper_state))

            obs, curr_task_progression, terminated, truncated, info = env.step(new_action)

            if curr_task_progression > task_progression:
                task_progression = curr_task_progression
                task_progression_timestamps.append(t)
            if task_progression >= 1.0:
                terminal_steps -= 1
            t += 1

        og.log.info(f"DEBUG: Run finished: {time.perf_counter() - start:.4f}s")
        # ------------------------------------------------------------------------------

        # Metrics calculation
        dt = 1.0 / 15.0  # Control freq is 15Hz by default

        qpos_arr = np.stack(qpos)  # (N, 8)
        qpos_joints = qpos_arr[:, :7]

        # Joint space metrics
        if len(qpos_joints) > 4:
            joint_vel = np.diff(qpos_joints, axis=0) / dt
            joint_acc = np.diff(joint_vel, axis=0) / dt
            joint_jerk = np.diff(joint_acc, axis=0) / dt

            joint_vel_var = np.mean(np.var(joint_vel, axis=0) * len(joint_vel))
            joint_acc_var = np.mean(np.var(joint_acc, axis=0) * len(joint_acc))
            joint_jerk_metric = np.mean(np.linalg.norm(joint_jerk, axis=1))
            joint_path_length = np.sum(np.linalg.norm(np.diff(qpos_joints, axis=0), axis=1))
        else:
            joint_vel_var = 0.0
            joint_acc_var = 0.0
            joint_jerk_metric = 0.0
            joint_path_length = 0.0

        # Cartesian space metrics
        ee_pos_arr = np.stack(ee_poses)
        if len(ee_pos_arr) > 4:
            cart_vel = np.diff(ee_pos_arr, axis=0) / dt
            cart_acc = np.diff(cart_vel, axis=0) / dt
            cart_jerk = np.diff(cart_acc, axis=0) / dt

            cart_jerk_metric = np.mean(np.linalg.norm(cart_jerk, axis=1))
            cart_path_length = np.sum(np.linalg.norm(np.diff(ee_pos_arr, axis=0), axis=1))
        else:
            cart_path_length = 0.0
            cart_jerk_metric = 0.0

        stage_to_log = "SUCCESS"
        if env.task_progression is not None:
            for stage, is_completed in env.task_progression.items():
                if not is_completed:
                    stage_to_log = stage
                    break
        else:
            stage_to_log = "N/A"

        if task_progression == 1.0 and hasattr(env, "task_type") and env.task_type in ["put", "stack"]:
            drops = max(0, drops - 1)

        result_entry = {
            "run_id": run_id,
            "task": task,
            "perturbation": perturbations[0],
            "instruction": env.instruction,
            "model": model_type,
            "real2sim": "Simulated",
            "env": "REALM",
            "task_progression": task_progression,
            "task_progression_timestamps": task_progression_timestamps,
            "stage": stage_to_log,
            "binary_SR": 1.0 if task_progression == 1.0 else 0.0,
            "joint_vel_var": joint_vel_var,
            "joint_acc_var": joint_acc_var,
            "joint_jerk": joint_jerk_metric,
            "joint_path_length": joint_path_length,
            "cart_path_length": cart_path_length,
            "cart_jerk": cart_jerk_metric,
            "collisions_self": collisions_self,
            "collisions_env": collisions_env,
            "object_drops": drops
        }

        result_entry["qpos"] = np.stack(qpos).tolist()
        result_entry["actions"] = np.stack(actions).tolist()
        if not no_record:
            video_bytes = video_recorder.get_video_bytes()
            result_entry["video"] = video_bytes
        
        results.append(result_entry)

        if not no_record:
            append_video(log_dir, task, perturbations[0], run_id, video_bytes)

        append_trajectory(log_dir, task, perturbations[0], run_id, np.stack(qpos), np.stack(actions))

        if not no_record:
            video_recorder.cleanup()

        client.reset()

        results_filename = save_results(results, log_dir + "/reports", task, perturbations[0], filename=results_filename)

    # ------------------------------------------------------------------------------
    save_results(results, log_dir+"/reports", task, perturbations[0])
    og.log.info("Done!")
    og.log.info(f"DEBUG: Done: {time.perf_counter() - start:.4f}s")

