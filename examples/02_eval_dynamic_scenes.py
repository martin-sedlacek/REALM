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
    ] # TODO: infer from yamls in task config folder

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

    if "push" in task and perturbations[0] in ['V-SC', 'B-HOBJ', 'SB-NOUN', 'SB-VRB', 'VB-MOBJ', 'VSB-NOBJ']:
        raise NotImplementedError()
    elif "stack" in task and perturbations[0] in ['SB-NOUN']:
        raise NotImplementedError()
    elif ("open_drawer" in task or "close_drawer" in task) and perturbations[0] in ['VB-MOBJ', 'SB-VRB']:
        raise NotImplementedError()

    def enable_interactive_path_tracing(carb_settings, samples_per_pixel=16):
        carb_settings.set("/rtx/rendermode", "PathTracing")
        if samples_per_pixel is not None:
            carb_settings.set_int("/rtx/pathtracing/spp", samples_per_pixel)
            carb_settings.set_int("/rtx/pathtracing/totalSpp", samples_per_pixel)
            carb_settings.set_int(
                "/rtx/pathtracing/useDirectLightingCache", False
            )  # NOTE: This is to enable lighting cache but can add temporal noise
        carb_settings.set_bool("/rtx/pathtracing/optixDenoiser/enabled", True)

    if model_type == "debug":
        # carb_settings = lazy.carb.settings.get_settings()
        # carb_settings.set("/persistent/omnihydra/useSceneGraphInstancing", True)
        # enable_interactive_path_tracing(carb_settings, samples_per_pixel=16)
        client = None
    else:
        print("Connecting to server...")
        print(f"DEBUG: port = {port}")
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
        # gripper_state = 1 if is_gripper_closed else 0
        # gripper_state = (proprio[8] + 1) / 2  # from (1, -1) to (1, 0)
        return base_im, base_im_second, wrist_im, robot_state, gripper_state

    # ------------------------------------------------------------------------------

    global_timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H:%M:%S")
    results = []
    # x = np.load("/app/logs/block_traj.npy")
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

        # TODO: figure out what to do with drawer opening, this cannot be here at scale:
        # if env.task_type == "close_drawer":
        #     target_obj = env.omnigibson_env.scene.object_registry("name", "drawer_cabinet")
        #     relevant_joints = _get_relevant_joints(target_obj)[1]
        #     #joint = relevant_joints[1]
        #
        #     for i in range(len(relevant_joints)):
        #         if i == 2: continue
        #         relevant_joints[i].keep_still()
        #     joint = relevant_joints[2]
        #
        #     current_position = joint.get_state()[0][0]
        #     joint.set_pos(joint.lower_limit + (joint.upper_limit - joint.lower_limit) * 0.5)
        #     # joint_range = joint.upper_limit - joint.lower_limit
        #     # openness_fraction = (current_position - joint.lower_limit) / joint_range
        #     # print(openness_fraction)

        # -------------------- Rollout loop --------------------
        obs, rew, terminated, truncated, info = env.warmup(obs)

        t = 0
        task_progression = 0.0
        task_progression_timestamps = []
        terminal_steps = 15
        while t < max_steps and terminal_steps > 0:
            base_im, base_im_second, wrist_im, robot_state, gripper_state = extract_from_obs(obs)

            # TODO: tmp for making figures
            # if t == 4:
            #     img = Image.fromarray(base_im.astype('uint8'))
            #     if perturbations[0] == "V-VIEW":
            #         img = Image.fromarray(base_im_second.astype('uint8'))
            #     img.save(f"/app/{perturbations[0]}.png")
            #     #assert 1 > 2
            # if t == 0:
            #     im = Image.fromarray(base_im.astype('uint8'))
            #     im.save(f"/app/logs/debug_ims/gen_{task}_{perturbations[0]}_{run_id}_cam1.png")
                # im = Image.fromarray(base_im_second.astype('uint8'))
                # im.save(f"/app/logs/gen_{task}_cam2.png")
                # im = Image.fromarray(wrist_im.astype('uint8'))
                # im.save(f"/app/logs/debug_ims/gen_{task}_{perturbations[0]}_{run_id}_cam3.png")

            if action_buffer.empty():
                # TODO: how to adjust it to work with any model?
                #   should pi0 settings be default for any model?
                if model_type == "debug":
                    pred_action_chunk = np.atleast_1d(np.zeros(8))
                    #pred_action_chunk[:7] = np.array([0, -0.628, 0, -2.512, -0.75, 1.884, 0.0])
                elif model_type == "GR00T":
                    base_im_resized = np.asarray(Image.fromarray(base_im).resize((320, 180))).astype(np.uint8)
                    base_im_second_resized = np.asarray(Image.fromarray(base_im_second).resize((320, 180))).astype(np.uint8)
                    wrist_im_resized = np.asarray(Image.fromarray(wrist_im).resize((320, 180))).astype(np.uint8)

                    obs_dict = {
                        "prompt": [instruction],
                        "state.joint_position": np.array(robot_state).astype(np.float32).reshape(1, 7),
                        "state.gripper_position": np.atleast_1d(np.array(gripper_state)).astype(np.float32).reshape(1, 1),
                        "video.exterior_image_1": base_im_resized[None],
                        "video.exterior_image_2": base_im_second_resized[None],
                        "video.wrist_image": wrist_im_resized[None]
                    }
                    pred = client.infer(obs_dict)
                    pred_action_chunk = np.concatenate(
                        [pred["action.joint_position"],
                         pred["action.gripper_position"].reshape(-1, 1)], axis=-1)
                else:
                    obs_dict = {
                        "prompt": instruction,
                        "observation/joint_position": robot_state,
                        "observation/gripper_position": np.atleast_1d(np.array(gripper_state)),
                        "observation/exterior_image_1_left": image_tools.resize_with_pad(base_im_second if env.task_type == "open_close_drawer" else base_im, 224, 224), # TODO: the task type terminology for open/close drawer might not be globally  the same
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

            # video.append(np.concatenate((
            #     image_tools.resize_with_pad(base_im, 224, 224),
            #     image_tools.resize_with_pad(wrist_im, 224, 224),
            # ), axis=1))
            # video.append(np.concatenate((
            #     base_im,
            #     base_im_second,
            #     wrist_im,
            # ), axis=1))
            video.append(base_im)
            qpos.append(np.concatenate((robot_state, np.atleast_1d(np.array(gripper_state)))))

            action = action_buffer.get()
            actions.append(action)

            # max_joint_vel_norm = np.abs(action[:7]).max()
            # tmp = action.copy()
            # if max_joint_vel_norm > 1:
            #     tmp[:7] = tmp[:7] / max_joint_vel_norm
            # delta_qpos = tmp[:7] * 0.2
            # delta_qpos_clipped = np.clip(delta_qpos, a_min=-0.2, a_max=0.2)
            # new_joint_action = robot_state + delta_qpos_clipped
            # delta_qpos_clipped = np.clip(tmp, a_min=-0.2, a_max=0.2)

            # ---------- [for joint pos delta predictions] ----------
            # max_delta_qpos_norm = np.abs(action[:7]).max()
            # tmp = action.copy()[:7]
            # if max_delta_qpos_norm > 0.1:
            #     tmp[:7] = tmp[:7] / max_delta_qpos_norm * 0.1
            # delta_qpos_clipped = np.clip(tmp, a_min=-0.1, a_max=0.1)

            new_joint_action = action.copy()[:7] #robot_state + delta_qpos_clipped

            new_gripper_state = 1 if action[7] > 0.5 else -1  # Prediction: (1,0) -> Target: (1,-1)
            is_gripper_closed = (new_gripper_state == 1)
            new_gripper_state = np.atleast_1d(np.array(new_gripper_state))
            new_action = np.concatenate((new_joint_action, new_gripper_state))

            # new_action = x[t if t < len(x) else len(x) - 1]
            # new_action[-1] = 1 if new_action[-1] / 0.05 > 0.1 else -1
            obs, curr_task_progression, terminated, truncated, info = env.step(new_action)
            print(f"{t}: {curr_task_progression}"), #{new_action}")
            #print(new_joint_action - proprio[:7])

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

        # if task_id >= 8:
        #     video = np.stack(video)
        #     save_filename = f"/app/logs/{timestamp}_{model_type}_rollout_sim_{task}_{perturbations}_{run_id}"
        #     ImageSequenceClip(list(video), fps=15).write_videofile(save_filename + ".mp4", codec="libx264")
        #     #np.save(save_filename, video)
        #     print(f"Saved video for run {run_id}.")

    # ------------------------------------------------------------------------------
    # results = np.stack(results)
    # filename = f"/app/logs/{global_timestamp}_{model_type}_rollout_sim_{ctrl_mode}_{task}_{perturbations}_report"
    # np.save(filename, results)

    np_results = np.stack(results)
    file_uuid = str(uuid.uuid1())[:6]
    if model_type not in ("pi0", "pi0_FAST", "GR00T"):
        # assuming it is script file
        script_filename = model_type.split("/")[-1]
        model_type = ".".join(script_filename.split(".")[:-1])
    np_results_filename = f"/app/logs/{global_timestamp}_{model_type}_gen_eval_rollout_{task}_{perturbations[0]}_{file_uuid}_report"
    np.save(np_results_filename, np_results)

    # df_results = pd.DataFrame(results)
    # np_results_filename = f"/app/logs/{global_timestamp}_{model_type}_real2sim_rollout_{task}_{perturbations[perturbation_id]}_report"
    # np.save(np_results_filename, np_results)

    print(f"Saved run report to {np_results_filename}")

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
    os.exit(0)