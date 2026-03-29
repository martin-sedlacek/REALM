import time
import numpy as np
from PIL import Image
import omnigibson as og
from openpi_client import websocket_client_policy, image_tools

from realm.helpers import axisangle_to_rpy
from realm.inference.base import ExternalRobotInferenceClient
from realm.inference.hamster import HamsterClient
from realm.inference.dreamzero import DreamZeroClient


class InferenceClient:
    def __init__(self, model_type, port, host="127.0.0.1", timeout=150.0):
        self.model_type = model_type
        self.host = host
        self.port = port
        if model_type == "GR00T_N16":
            self.client = ExternalRobotInferenceClient(host=self.host, port=self.port)
        elif model_type == "hamster":
            self.client = HamsterClient(host=self.host, port=self.port)
        elif model_type == "dreamzero":
            self.client = DreamZeroClient(host=self.host, port=self.port)
        elif model_type == "openpi":
            og.log.info("Connecting to server...")
            self.client = websocket_client_policy.WebsocketClientPolicy(
                host=host,
                port=port
            )
        elif model_type == "debug":
            self.client = None
        else:
            raise NotImplementedError()

    def infer(self, instruction, base_im, base_im_second, wrist_im, robot_state, gripper_state, use_base_im_second=False, ee_control=False, cartesian_position=None):
        if self.model_type == "debug":
            if ee_control:
                pred_action_chunk = np.array([0.41402626, -0.13211727, 0.57253086, -3.09742367, 0.2580259, -0.24700592, -1])
            else:
                pred_action_chunk = np.atleast_1d(np.zeros(8))

            return pred_action_chunk

        # TODO: all DROID EE control poses need to have flip_pose_pointing_down() applied before being passed to the step
        if self.model_type == "GR00T_N16":
            base_im_resized = image_tools.resize_with_pad(base_im, 224, 224)[None, None]   # (1,1,224,224,3)
            wrist_im_resized = image_tools.resize_with_pad(wrist_im, 224, 224)[None, None]  # (1,1,224,224,3)

            obs_dict = {
                "observation": {
                    "video.wrist_image_left": wrist_im_resized,
                    "video.exterior_image_1_left": base_im_resized,
                    "state.joint_position": np.array(robot_state).astype(np.float32).reshape(1, 1, 7),
                    "state.gripper_position": np.atleast_1d(np.array(gripper_state)).astype(np.float32).reshape(1, 1, 1),
                    "annotation.language.language_instruction": [instruction]
                }
            }

            pred = self.client.get_action(obs_dict)[0]

            pred_action_chunk = np.concatenate(
                [pred["action.joint_position"].reshape(-1, 7),
                 pred["action.gripper_position"].reshape(-1, 1)], axis=-1)
            return pred_action_chunk

        elif self.model_type == "GR00T":
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

        elif self.model_type == "molmoact":
            img_to_use = base_im_second if use_base_im_second else base_im
            obs_dict = {
                "images": [img_to_use, wrist_im],
                "instruction": instruction,
            }
            _t0 = time.perf_counter()
            pred = self.client.infer(obs_dict)
            og.log.info(f"[molmoact] inference time: {time.perf_counter() - _t0:.3f}s")
            pred_action_chunk = pred["action"]

            if ee_control:
                pred_action_chunk = axisangle_to_rpy(pred_action_chunk)

            return pred_action_chunk

        elif self.model_type == "hamster":
            img_to_use = base_im_second if use_base_im_second else base_im
            # Hamster expects BGR for cv2.imencode
            import cv2
            img_bgr = cv2.cvtColor(img_to_use, cv2.COLOR_RGB2BGR)
            _t0 = time.perf_counter()
            trajectory = self.client.infer(img_bgr, instruction)
            og.log.info(f"[hamster] inference time: {time.perf_counter() - _t0:.3f}s")
            return np.array(trajectory)

        elif self.model_type == "dreamzero":
            assert base_im_second is not None, "DreamZero requires --multi-view (second external camera)"
            assert cartesian_position is not None, "DreamZero requires cartesian_position (robot-relative EE pose)"

            # DreamZero expects 180x320 RGB and strictly numpy arrays
            # H=180, W=320. Initial frames MUST be strictly 3D (H, W, 3) np.ndarray
            base_im_resized = np.array(Image.fromarray(base_im).resize((320, 180)), dtype=np.uint8)
            base_im_second_resized = np.array(Image.fromarray(base_im_second).resize((320, 180)), dtype=np.uint8)
            wrist_im_resized = np.array(Image.fromarray(wrist_im).resize((320, 180)), dtype=np.uint8)

            obs_dict = {
                "observation/exterior_image_0_left": base_im_resized,
                "observation/exterior_image_1_left": base_im_second_resized,
                "observation/wrist_image_left": wrist_im_resized,
                "observation/joint_position": np.array(robot_state, dtype=np.float32),
                "observation/cartesian_position": np.array(cartesian_position, dtype=np.float32),
                "observation/gripper_position": np.array(np.atleast_1d(gripper_state), dtype=np.float32),
                "prompt": instruction
            }

            pred_action_chunk = self.client.infer(obs_dict)
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

    def reset(self):
        if hasattr(self.client, "reset"):
            self.client.reset()
