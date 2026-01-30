import numpy as np
import torch
from queue import Queue
import datetime
import os
import random
import glob
import csv
import omnigibson as og
from omnigibson.macros import gm
from realm.environments.realm_environment_dynamic import RealmEnvironmentDynamic
from realm.inference import InferenceClient, extract_from_obs
from realm.logging import VideoRecorder, save_results_to_csv
import time


SUPPORTED_TASKS = [
    "put_green_block_in_bowl", #0
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
    'V-VIEW', # 2
    'V-SC', # 3
    'V-LIGHT', # 4
    'S-PROP', # 5
    'S-LANG', # 6
    'S-MO', # 7
    'S-AFF', # 8
    'S-INT', # 9
    'B-HOBJ', # 10
    'SB-NOUN', # 11
    'SB-VRB', # 12
    'VB-POSE', # 13
    'VB-MOBJ', # 14
    'VSB-NOBJ' # 15
]


def set_sim_config():
    gm.DEFAULT_SIM_STEP_FREQ = 15
    gm.DEFAULT_RENDERING_FREQ = 15
    gm.DEFAULT_PHYSICS_FREQ = 120
    gm.ENABLE_TRANSITION_RULES = False # this needs to be off to avoid bug with sludge state during collision: https://github.com/StanfordVL/BEHAVIOR-1K/issues/1201
    gm.ENABLE_OBJECT_STATES = True # this needs to be on because push_switch task usees the ToggledOn state

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
        model="pi0_FAST",
        port=8000,
        log_dir="/app/logs",
        resume_run_id=None
):
    start = time.perf_counter()
    og.log.info(f"DEBUG: Begin eval: {time.perf_counter() - start:.4f}s")
    set_sim_config()

    # -------------------- Create the environment + client --------------------
    task = SUPPORTED_TASKS[task_id]
    perturbations = [SUPPORTED_PERTURBATIONS[perturbation_id]]

    os.makedirs(log_dir, exist_ok=True)

    model_type = model # TODO: infer type from model name, rn this will just default to a pi model inference inside the client
    client = InferenceClient(model_type, port)
    og.log.info(f"DEBUG: Client connected: {time.perf_counter() - start:.4f}s")

    env = RealmEnvironmentDynamic(
        config_path="/app/realm/config",
        task=task,
        perturbations=perturbations
    )
    og.log.info(f"DEBUG: Env created: {time.perf_counter() - start:.4f}s")

    if resume_run_id:
        # find matching file
        search_pattern = os.path.join(log_dir, "reports", f"{resume_run_id}*report.csv")
        matches = glob.glob(search_pattern)
        if not matches:
            raise ValueError(f"Could not find run report to resume with ID {resume_run_id}")
        csv_filename = matches[0]
        # read existing results
        with open(csv_filename, 'r') as f:
            reader = csv.DictReader(f)
            existing_results = list(reader)
        results = existing_results
        start_repeat = len(results)
        og.log.info(f"Resuming run {resume_run_id} from repeat {start_repeat}. Using file: {csv_filename}")
    else:
        results = []
        start_repeat = 0
        csv_filename = None

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

        video_recorder = VideoRecorder(log_dir, timestamp, run_id)

        qpos = []
        actions = []
        action_buffer = Queue()

        obs, _ = env.reset()
        instruction = env.instruction

        # -------------------- Rollout loop --------------------
        obs, rew, terminated, truncated, info = env.warmup(obs)

        t = 0
        task_progression = 0.0
        task_progression_timestamps = []
        terminal_steps = 15
        while t < max_steps and terminal_steps > 0:
            base_im, base_im_second, wrist_im, robot_state, gripper_state = extract_from_obs(obs)

            if action_buffer.empty():
                pred_action_chunk = client.infer(
                    instruction, base_im, base_im_second, wrist_im, robot_state, gripper_state,
                    use_base_im_second=(env.task_type == "open_close_drawer" if hasattr(env, "task_type") else False)
                )

                if len(pred_action_chunk.shape) == 2:
                    assert pred_action_chunk.shape[-1] == 8
                    for action in pred_action_chunk[:horizon]:
                        action = np.squeeze(action)
                        action_buffer.put(action)
                else:
                    action_buffer.put(pred_action_chunk)

            video_recorder.add_frame(base_im, wrist_im)

            qpos.append(np.concatenate((robot_state, np.atleast_1d(np.array(gripper_state)))))

            action = action_buffer.get()
            actions.append(action)

            new_joint_action = action.copy()[:7]

            new_gripper_state = 1 if action[7] > 0.5 else -1  # Prediction: (1,0) -> Target: (1,-1)
            new_gripper_state = np.atleast_1d(np.array(new_gripper_state))
            new_action = np.concatenate((new_joint_action, new_gripper_state))

            obs, curr_task_progression, terminated, truncated, info = env.step(new_action)

            if curr_task_progression > task_progression:
                task_progression = curr_task_progression
                task_progression_timestamps.append(t)
            if task_progression >= 1.0:
                terminal_steps -= 1
            t += 1

        og.log.info(f"DEBUG: Run finished: {time.perf_counter() - start:.4f}s")
        # ------------------------------------------------------------------------------
        results.append({
            "run_id": run_id,
            "task": task,
            "perturbation": perturbations[0],
            "instruction": instruction,
            "model": model,
            "real2sim": "Simulated",
            "env": "REALM",
            "task_progression": task_progression,
            "task_progression_timestamps": task_progression_timestamps,
            "binary_SR": 1.0 if task_progression == 1.0 else 0.0
        })

        video_filename = os.path.join(log_dir, "videos", f"{task}_{perturbations[0]}_{run_id}")
        video_recorder.save_video(video_filename)
        video_recorder.cleanup()

        qpos_filename = os.path.join(log_dir, "qpos", f"{task}_{perturbations[0]}_{run_id}")
        os.makedirs(log_dir + "/qpos", exist_ok=True)
        np.save(qpos_filename, np.stack(qpos))

        actions_filename = os.path.join(log_dir, "actions", f"{task}_{perturbations[0]}_{run_id}")
        os.makedirs(log_dir + "/actions", exist_ok=True)
        np.save(actions_filename, np.stack(actions))

        csv_filename = save_results_to_csv(results, log_dir + "/reports", task, perturbations[0], filename=csv_filename)

    # ------------------------------------------------------------------------------
    save_results_to_csv(results, log_dir+"/reports", task, perturbations[0])
    og.log.info("Done!")
    og.log.info(f"DEBUG: Done: {time.perf_counter() - start:.4f}s")