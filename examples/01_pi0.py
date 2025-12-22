import numpy as np
import torch
from queue import Queue
import datetime
import argparse
import os
from PIL import Image
import random
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

import omnigibson as og
from omnigibson.macros import create_module_macros, gm
from omnigibson.object_states.open_state import _get_relevant_joints
from realm.environments.realm_environment_dynamic import RealmEnvironmentDynamic

from openpi_client import websocket_client_policy
from openpi_client import image_tools

import uuid

def eval(
        task_id=0,
        perturbation_id=0,
        repeats=1,
        max_steps=500,
        horizon=8,
        model_type="pi0_FAST",
        port=8000
):
    # ---------------------------------------- sim config ----------------------------------------
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

    # -------------------- Create the environment + pi0 client --------------------
    task = SUPPORTED_TASKS[task_id]
    perturbations = [SUPPORTED_PERTURBATIONS[perturbation_id]]

    print("Connecting to pi0 server...")
    client = websocket_client_policy.WebsocketClientPolicy(
        host="localhost",
        port=port
    )
    print("Connected!")

    env = RealmEnvironmentDynamic(
        config_path="/app/realm/config",
        task=task,
        perturbations=perturbations
    )

    def extract_from_obs(obs: dict):
        base_im = obs['external']['external_sensor0']['rgb'].cpu().numpy()[..., :3]
        base_im_second = obs['external']['external_sensor1']['rgb'].cpu().numpy()[..., :3]
        wrist_im = obs['franka']['franka:gripper_link_camera:Camera:0']['rgb'].cpu().numpy()[..., :3]
        proprio = obs['franka']['proprio'].cpu().numpy()
        robot_state = proprio[:7]
        gripper_state = proprio[7] / 0.05  # 0 = open, 0.05 = closed
        return base_im, base_im_second, wrist_im, robot_state, gripper_state

    global_timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H:%M:%S")
    results = []
    for run_id in range(repeats):
        # ------------------------ pre-configure each run --------------------------------
        timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H:%M:%S")
        video = []
        qpos = []
        actions = []
        action_buffer = Queue()

        obs, _ = env.reset()
        instruction = env.instruction
        print(instruction)

        # -------------------- Rollout loop --------------------
        obs, rew, terminated, truncated, info = env.warmup(obs)

        t = 0
        task_progression = 0.0
        task_progression_timestamps = []
        terminal_steps = 15
        while t < max_steps and terminal_steps > 0:
            base_im, base_im_second, wrist_im, robot_state, gripper_state = extract_from_obs(obs)

            if action_buffer.empty():
                obs_dict = {
                    "prompt": instruction,
                    "observation/joint_position": robot_state,
                    "observation/gripper_position": np.atleast_1d(np.array(gripper_state)),
                    "observation/exterior_image_1_left": image_tools.resize_with_pad(base_im, 224, 224),
                    "observation/wrist_image_left": image_tools.resize_with_pad(wrist_im, 224, 224)
                }
                pred = client.infer(obs_dict)
                pred_action_chunk = pred["actions"]


                if len(pred_action_chunk.shape) == 2:
                    assert pred_action_chunk.shape[-1] == 8
                    for action in pred_action_chunk[:horizon]:
                        action = np.squeeze(action)
                        action_buffer.put(action)
                else:
                    action_buffer.put(pred_action_chunk)

            video.append(np.concatenate((
                base_im,
                wrist_im,
            ), axis=1))
            qpos.append(np.concatenate((robot_state, np.atleast_1d(np.array(gripper_state)))))

            action = action_buffer.get()
            actions.append(action)

            new_joint_action = action.copy()[:7]

            new_gripper_state = 1 if action[7] > 0.5 else -1  # Prediction: (1,0) -> Target: (1,-1)
            new_gripper_state = np.atleast_1d(np.array(new_gripper_state))
            new_action = np.concatenate((new_joint_action, new_gripper_state))

            obs, curr_task_progression, terminated, truncated, info = env.step(new_action)
            print(f"{t}: {curr_task_progression}")

            if curr_task_progression > task_progression:
                task_progression = curr_task_progression
                task_progression_timestamps.append(t)
            if task_progression >= 1.0:
                terminal_steps -= 1
            t += 1

        # ------------------------------------------------------------------------------
        results.append({
            "task": task,
            "perturbation": perturbations,
            "model": model_type,
            "real2sim": "Simulated",
            "task_progression": task_progression,
            "task_progression_timestamps": task_progression_timestamps,
            "binary_SR": 1.0 if task_progression == 1.0 else 0.0
        })

        video = np.stack(video)
        save_filename = f"/app/logs/{timestamp}_{model_type}_rollout_{task}_{perturbations}_{run_id}"
        ImageSequenceClip(list(video), fps=15).write_videofile(save_filename + ".mp4", codec="libx264")

    # ------------------------------------------------------------------------------

    np_results = np.stack(results)
    file_uuid = str(uuid.uuid1())[:6]
    if model_type not in ("pi0", "pi0_FAST", "GR00T"):
        script_filename = model_type.split("/")[-1]
        model_type = ".".join(script_filename.split(".")[:-1])
    np_results_filename = f"/app/logs/{global_timestamp}_{model_type}_gen_eval_rollout_{task}_{perturbations[0]}_{file_uuid}_report"
    np.save(np_results_filename, np_results)

    print(f"Saved run report to {np_results_filename}")
    print("Done!")

if __name__ == "__main__":
    eval(
        task_id=1,
        perturbation_id=0,
        repeats=1,
        max_steps=500,
        model_type="pi0",
        port=8000
    )
    og.shutdown()
    os.exit(0)