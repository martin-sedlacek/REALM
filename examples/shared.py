import numpy as np
import os
import csv
import shutil
import uuid
from PIL import Image
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from openpi_client import websocket_client_policy, image_tools

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

def extract_from_obs(obs: dict):
    base_im = obs['external']['external_sensor0']['rgb'].cpu().numpy()[..., :3]
    base_im_second = obs['external']['external_sensor1']['rgb'].cpu().numpy()[..., :3]
    wrist_im = obs['franka']['franka:gripper_link_camera:Camera:0']['rgb'].cpu().numpy()[..., :3]
    proprio = obs['franka']['proprio'].cpu().numpy()
    robot_state = proprio[:7]
    gripper_state = proprio[7] / 0.05  # 0 = open, 0.05 = closed
    return base_im, base_im_second, wrist_im, robot_state, gripper_state

def save_results_to_csv(results, log_dir, global_timestamp, model_type, task, perturbation):
    file_uuid = str(uuid.uuid1())[:6]
    # Handle cleaning up model_type string for filename if it's a path
    if model_type not in ("pi0", "pi0_FAST", "GR00T"):
        script_filename = model_type.split("/")[-1]
        model_type_str = ".".join(script_filename.split(".")[:-1])
    else:
        model_type_str = model_type

    os.makedirs(log_dir, exist_ok=True)
    csv_results_filename = f"{log_dir}/{global_timestamp}_{model_type_str}_gen_eval_rollout_{task}_{perturbation}_{file_uuid}_report.csv"

    if len(results) > 0:
        keys = results[0].keys()
        with open(csv_results_filename, 'w', newline='') as output_file:
            dict_writer = csv.DictWriter(output_file, fieldnames=keys)
            dict_writer.writeheader()
            dict_writer.writerows(results)
    print(f"Saved run report to {csv_results_filename}")

class VideoRecorder:
    def __init__(self, log_dir, timestamp, run_id):
        self.temp_frame_dir = os.path.join(log_dir, f"{timestamp}_frames_{run_id}")
        os.makedirs(self.temp_frame_dir, exist_ok=True)
        self.frame_filenames = []
        self.count = 0

    def add_frame(self, base_im, wrist_im):
        frame_img = np.concatenate((
            base_im,
            wrist_im,
        ), axis=1)

        if frame_img.dtype.kind == 'f':
             frame_img = (frame_img * 255).astype(np.uint8)
        elif frame_img.dtype != np.uint8:
             frame_img = frame_img.astype(np.uint8)

        frame_path = os.path.join(self.temp_frame_dir, f"frame_{self.count:05d}.png")
        Image.fromarray(frame_img).save(frame_path)
        self.frame_filenames.append(frame_path)
        self.count += 1

    def save_video(self, save_filename, fps=15):
        if not self.frame_filenames:
            return
        ImageSequenceClip(self.frame_filenames, fps=fps).write_videofile(save_filename + ".mp4", codec="libx264")

    def cleanup(self):
        if os.path.exists(self.temp_frame_dir):
            shutil.rmtree(self.temp_frame_dir)

class InferenceClient:
    def __init__(self, model_type, port, host="localhost"):
        self.model_type = model_type
        self.client = None
        if model_type != "debug":
             print("Connecting to server...")
             self.client = websocket_client_policy.WebsocketClientPolicy(
                host=host,
                port=port
            )
             print("Connected!")

    def infer(self, instruction, base_im, base_im_second, wrist_im, robot_state, gripper_state, use_base_im_second=False):
        if self.model_type == "debug":
            pred_action_chunk = np.atleast_1d(np.zeros(8))
            return pred_action_chunk

        if self.model_type == "GR00T":
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
            pred = self.client.infer(obs_dict)
            pred_action_chunk = np.concatenate(
                [pred["action.joint_position"],
                 pred["action.gripper_position"].reshape(-1, 1)], axis=-1)
            return pred_action_chunk

        else:
            img_to_use = base_im_second if use_base_im_second else base_im

            obs_dict = {
                "prompt": instruction,
                "observation/joint_position": robot_state,
                "observation/gripper_position": np.atleast_1d(np.array(gripper_state)),
                "observation/exterior_image_1_left": image_tools.resize_with_pad(img_to_use, 224, 224),
                "observation/wrist_image_left": image_tools.resize_with_pad(wrist_im, 224, 224)
            }
            pred = self.client.infer(obs_dict)
            pred_action_chunk = pred["actions"]
            return pred_action_chunk
