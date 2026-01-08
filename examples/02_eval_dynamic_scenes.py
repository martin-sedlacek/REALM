import numpy as np
import torch
from queue import Queue
import datetime
import argparse
import os
import sys
import random

import omnigibson as og
from omnigibson.macros import gm
from realm.environments.realm_environment_dynamic import RealmEnvironmentDynamic
from realm.inference import InferenceClient, SUPPORTED_TASKS, SUPPORTED_PERTURBATIONS, extract_from_obs
from realm.logging_utils import VideoRecorder, save_results_to_csv


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
    # gm.USE_GPU_DYNAMICS = True
    # gm.ENABLE_HQ_RENDERING = False #True
    # gm.ENABLE_FLATCACHE = False #True

    seed = 1234
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # -------------------- Create the environment + client --------------------
    task = SUPPORTED_TASKS[task_id]
    perturbations = [SUPPORTED_PERTURBATIONS[perturbation_id]]

    if "push" in task and perturbations[0] in ['V-SC', 'B-HOBJ', 'SB-NOUN', 'SB-VRB', 'VB-MOBJ', 'VSB-NOBJ']:
        raise NotImplementedError()
    elif "stack" in task and perturbations[0] in ['SB-NOUN']:
        raise NotImplementedError()
    elif ("open_drawer" in task or "close_drawer" in task) and perturbations[0] in ['VB-MOBJ', 'SB-VRB']:
        raise NotImplementedError()

    client = InferenceClient(model_type, port)

    env = RealmEnvironmentDynamic(
        config_path="/app/realm/config",
        task=task,
        perturbations=perturbations
    )

    global_timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H:%M:%S")
    results = []

    log_dir = "/app/logs"
    os.makedirs(log_dir, exist_ok=True)

    for run_id in range(repeats):
        # ------------------------ pre-configure each run --------------------------------
        timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H:%M:%S")

        # Although the original 02 script didn't save temp frames and had commented out video saving for most tasks,
        # using VideoRecorder is cleaner and safer for memory.
        # But wait, original code says:
        # if task_id >= 8: ... save video
        # I'll enable it for all or follow logic? The prompt says "isolate... logic".
        # I'll just use the VideoRecorder. It handles cleanup.
        video_recorder = VideoRecorder(log_dir, timestamp, run_id)

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

            # Record frame
            video_recorder.add_frame(base_im, wrist_im)

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

        # Logic from original script: only save video if task_id >= 8.
        # But since we're using VideoRecorder, we have the frames.
        # I'll stick to original logic for saving video file, but clean up frames always.
        if task_id >= 8:
             save_filename = os.path.join(log_dir, f"{timestamp}_{model_type}_rollout_sim_{task}_{perturbations}_{run_id}")
             video_recorder.save_video(save_filename)
             print(f"Saved video for run {run_id}.")

        video_recorder.cleanup()

    # ------------------------------------------------------------------------------
    save_results_to_csv(results, log_dir, global_timestamp, model_type, task, perturbations[0])
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="dynamic sim evals")
    parser.add_argument('--perturbation_id', type=int, required=False, default=0)
    parser.add_argument('--task_id', type=int, required=False, default=0)
    parser.add_argument('--repeats', type=int, required=False, default=5)
    parser.add_argument('--max_steps', type=int, required=False, default=500)
    parser.add_argument('--model', type=str, required=True, default=None)
    parser.add_argument('--port', type=int, required=True)
    args = parser.parse_args()
    assert args.model is not None
    eval(
        task_id=args.task_id,
        perturbation_id=args.perturbation_id,
        repeats=args.repeats,
        max_steps=args.max_steps,
        model_type=args.model,
        port=args.port
    )
    og.shutdown()
    sys.exit(0)
