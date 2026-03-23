import time
import base64
import re
import os
import uuid
import cv2
import numpy as np
from PIL import Image
from openai import OpenAI
import json_numpy
import json
import zmq
import msgpack
import io
from openpi_client import websocket_client_policy, image_tools
import omnigibson as og
from realm.helpers import axisangle_to_rpy


def extract_from_obs(obs: dict, robot_name='DROID', enable_depth=False):
    base_im = obs['external']['external_sensor0']['rgb'].cpu().numpy()[..., :3]
    base_depth = obs['external']['external_sensor0']['depth_linear'].cpu().numpy() if enable_depth else None
    if 'external_sensor1' in obs['external']:
        base_im_second = obs['external']['external_sensor1']['rgb'].cpu().numpy()[..., :3]
        base_depth_second = obs['external']['external_sensor1']['depth_linear'].cpu().numpy() if enable_depth else None
    else:
        base_im_second = None
        base_depth_second = None

    wrist_im = obs[robot_name]['DROID:gripper_link_camera:Camera:0']['rgb'].cpu().numpy()[..., :3]
    proprio = obs[robot_name]['proprio'].cpu().numpy()
    robot_state = proprio[:7]
    gripper_state = proprio[7] / 0.05  # 0 = open, 0.05 = closed
    return base_im, base_depth, base_im_second, base_depth_second, wrist_im, robot_state, gripper_state


class MsgSerializer:
    @staticmethod
    def to_bytes(data: dict) -> bytes:
        return msgpack.packb(data, default=MsgSerializer.encode_custom_classes)

    @staticmethod
    def from_bytes(data: bytes) -> dict:
        return msgpack.unpackb(data, object_hook=MsgSerializer.decode_custom_classes)

    @staticmethod
    def decode_custom_classes(obj):
        if "__ndarray_class__" in obj:
            obj = np.load(io.BytesIO(obj["as_npy"]), allow_pickle=False)
        return obj

    @staticmethod
    def encode_custom_classes(obj):
        if isinstance(obj, np.ndarray):
            output = io.BytesIO()
            np.save(output, obj, allow_pickle=False)
            return {"__ndarray_class__": True, "as_npy": output.getvalue()}
        return obj


class BaseInferenceClient:
    def __init__(
        self,
        host: str = "localhost",
        port: int = 5555,
        timeout_ms: int = 15000,
        api_token: str = None,
    ):
        self.context = zmq.Context()
        self.host = host
        self.port = port
        self.timeout_ms = timeout_ms
        self.api_token = api_token
        self._init_socket()

    def _init_socket(self):
        """Initialize or reinitialize the socket with current settings"""
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(f"tcp://{self.host}:{self.port}")

    def ping(self) -> bool:
        try:
            self.call_endpoint("ping", requires_input=False)
            return True
        except zmq.error.ZMQError:
            self._init_socket()  # Recreate socket for next attempt
            return False

    def kill_server(self):
        """
        Kill the server.
        """
        self.call_endpoint("kill", requires_input=False)

    def call_endpoint(
        self, endpoint: str, data: dict | None = None, requires_input: bool = True
    ) -> dict:
        """
        Call an endpoint on the server.

        Args:
            endpoint: The name of the endpoint.
            data: The input data for the endpoint.
            requires_input: Whether the endpoint requires input data.
        """
        request: dict = {"endpoint": endpoint}
        if requires_input:
            request["data"] = data
        if self.api_token:
            request["api_token"] = self.api_token

        self.socket.send(MsgSerializer.to_bytes(request))
        message = self.socket.recv()
        response = MsgSerializer.from_bytes(message)

        if "error" in response:
            raise RuntimeError(f"Server error: {response['error']}")
        return response

    def __del__(self):
        """Cleanup resources on destruction"""
        try:
            self.socket.close()
            self.context.term()
        except:
            pass


class ExternalRobotInferenceClient(BaseInferenceClient):
    """
    Client for communicating with the RealRobotServer
    """

    def get_action(self, observations: dict) -> dict:
        """
        Get the action from the server.
        """
        return self.call_endpoint("get_action", observations)


class HamsterClient:
    """
    Client for the HAMSTER (Hierarchical Action Models For Open-World Robot Manipulation) server.
    """
    GRIPPER_CLOSE = 0
    GRIPPER_OPEN = 1
    MODEL_NAME = "Hamster_dev"

    def __init__(self, host=None, port=8000, ip_file="./ip_eth0.txt"):
        # if host is None:
        #     if os.path.exists(ip_file):
        #         with open(ip_file, "r") as f:
        #             self.server_ip = f.read().strip()
        #     else:
        #         self.server_ip = "127.0.0.1"
        # else:
        #     self.server_ip = host
        self.server_ip = "127.0.0.1"
        self.base_url = f"http://{self.server_ip}:{port}/v1"
        self.client = OpenAI(base_url=self.base_url, api_key="fake-key")
        og.log.info(f"Connected to HAMSTER server at {self.base_url}")

    def _encode_image(self, image_path_or_array):
        """Encodes an image to base64 string."""
        if isinstance(image_path_or_array, str):
            image = cv2.imread(image_path_or_array)
            if image is None:
                raise ValueError(f"Could not read image from {image_path_or_array}")
        else:
            image = image_path_or_array

        _, buffer = cv2.imencode('.jpg', image)
        return base64.b64encode(buffer).decode('utf-8')

    def _parse_response(self, response_text):
        """Extracts and parses the trajectory points and actions from the model response."""
        match = re.search(r'<ans>(.*?)</ans>', response_text, re.DOTALL)
        if not match:
            og.log.info(f"Warning: No <ans> tags found in response: {response_text}")
            return []

        ans_content = match.group(1).strip()

        # Replace action tags with sentinel coordinates for easier evaluation
        ans_content = ans_content.replace('<action>Close Gripper</action>', '(1000.0, 1000.0)')
        ans_content = ans_content.replace('<action>Open Gripper</action>', '(1001.0, 1001.0)')

        try:
            # Safely evaluate the list string
            # Note: in a production environment, use a more robust parser than eval()
            keypoints = eval(ans_content)
        except Exception as e:
            og.log.info(f"Error parsing trajectory list: {e}")
            return []

        trajectory = []
        current_action = self.GRIPPER_CLOSE # Default starting state if not specified

        for point in keypoints:
            x, y = point
            if x == 1000.0 and y == 1000.0:
                current_action = self.GRIPPER_CLOSE
                if trajectory: trajectory[-1] = (trajectory[-1][0], trajectory[-1][1], current_action)
                continue
            elif x == 1001.0 and y == 1001.0:
                current_action = self.GRIPPER_OPEN
                if trajectory: trajectory[-1] = (trajectory[-1][0], trajectory[-1][1], current_action)
                continue

            trajectory.append((x, y, current_action))

        return trajectory

    def infer(self, image_path_or_array, quest, max_tokens=256, temperature=0.0):
        """
        Sends a request to the HAMSTER server.
        Returns a list of tuples: (x, y, gripper_action)
        - x, y: Normalized coordinates (0.0 to 1.0)
        - gripper_action: 0 for Close, 1 for Open
        """
        encoded_image = self._encode_image(image_path_or_array)

        prompt = (
            f"\nIn the image, please execute the command described in <quest>{quest}</quest>.\n"
            "Provide a sequence of points denoting the trajectory of a robot gripper to achieve the goal.\n"
            "Format your answer as a list of tuples enclosed by <ans> and </ans> tags. For example:\n"
            "<ans>[(0.25, 0.32), (0.32, 0.17), (0.13, 0.24), <action>Open Gripper</action>, (0.74, 0.21), <action>Close "
            "Gripper</action>]</ans>\n"
            "The tuple denotes point x and y location of the end effector in the image. The action tags indicate gripper actions.\n"
            "Coordinates should be floats between 0 and 1, representing relative positions.\n"
            "Remember to provide points between <ans> and </ans> tags and think step by step."
        )

        try:
            response = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}},
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
                max_tokens=max_tokens,
                model=self.MODEL_NAME,
                extra_body={"num_beams": 1, "use_cache": False, "temperature": temperature, "top_p": 0.95},
            )

            # OpenAI-compatible server implementation in server.py returns nested content
            response_text = response.choices[0].message.content
            if isinstance(response_text, list):
                response_text = response_text[0].get('text', '')

            return self._parse_response(response_text)

        except Exception as e:
            og.log.info(f"Request failed: {e}")
            return []

    def reset(self):
        """No-op for HamsterClient"""
        pass


class DreamZeroClient:
    """
    Client for the DreamZero server.
    """
    def __init__(self, host="localhost", port=5000):
        # DreamZero uses a specific WebsocketClientPolicy
        # We try to import it from eval_utils, fallback to openpi_client
        try:
            from eval_utils.policy_client import WebsocketClientPolicy
        except ImportError:
            from openpi_client.websocket_client_policy import WebsocketClientPolicy

        og.log.info(f"Connecting to DreamZero server at {host}:{port}...")
        self.client = WebsocketClientPolicy(host=host, port=port)
        self.session_id = str(uuid.uuid4())
        
        # Optional: Validate connection
        try:
            metadata = self.client.get_server_metadata()
            og.log.info(f"Connected to DreamZero! Server metadata: {metadata}")
        except Exception as e:
            og.log.info(f"Warning: Could not fetch DreamZero metadata: {e}")

    def infer(self, obs_dict):
        obs_dict["session_id"] = self.session_id
        # FIX: Add endpoint directly to the flat dict so openpi_client sends it properly
        obs_dict["endpoint"] = "infer"
        return self.client.infer(obs_dict)

    def reset(self):
        """Tells the server to flush buffers and saves the generated video prediction to disk"""
        reset_obs = {"session_id": self.session_id}
        
        try:
            import inspect
            if hasattr(self.client, "reset"):
                sig = inspect.signature(self.client.reset)
                # If reset takes arguments (besides self), pass the reset_obs
                if len(sig.parameters) > 0:
                    self.client.reset(reset_obs)
                else:
                    # openpi_client version takes 0 args and sends {"endpoint": "reset"}
                    self.client.reset()
            else:
                reset_obs["endpoint"] = "reset"
                self.client.infer(reset_obs)
        except Exception as e:
            og.log.info(f"Warning: DreamZero reset failed: {e}")
            
        self.session_id = str(uuid.uuid4())


class InferenceClient:
    def __init__(self, model_type, port, host="127.0.0.1"):
        self.model_type = model_type
        self.host = host
        self.port = port
        self.client = None
        if model_type == "GR00T_N16":
            self.client = ExternalRobotInferenceClient(host=self.host, port=self.port)
        elif model_type == "hamster":
            self.client = HamsterClient(host=self.host, port=self.port)
        elif model_type == "dreamzero":
            self.client = DreamZeroClient(host="192.168.0.1", port=5000)
            #self.client = DreamZeroClient(host=self.host, port=self.port)
        elif model_type != "debug":
            og.log.info("Connecting to server...")
            self.client = websocket_client_policy.WebsocketClientPolicy(
                host=host,
                port=port
            )
            og.log.info("Connected!")

    def infer(self, instruction, base_im, base_im_second, wrist_im, robot_state, gripper_state, use_base_im_second=False, ee_control=False):
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
            img_bgr = cv2.cvtColor(img_to_use, cv2.COLOR_RGB2BGR)
            _t0 = time.perf_counter()
            trajectory = self.client.infer(img_bgr, instruction)
            og.log.info(f"[hamster] inference time: {time.perf_counter() - _t0:.3f}s")
            return np.array(trajectory)
        elif self.model_type == "dreamzero":
            # DreamZero expects 180x320 RGB and strictly numpy arrays
            # H=180, W=320. Initial frames MUST be strictly 3D (H, W, 3) np.ndarray
            base_im_resized = np.array(Image.fromarray(base_im).resize((320, 180)), dtype=np.uint8)
            wrist_im_resized = np.array(Image.fromarray(wrist_im).resize((320, 180)), dtype=np.uint8)

            obs_dict = { # TODO: fix cartesian and pass second image
                "observation/exterior_image_0_left": base_im_resized,
                "observation/wrist_image_left": wrist_im_resized,
                "observation/joint_position": np.array(robot_state, dtype=np.float32),
                "observation/cartesian_position": np.zeros(6, dtype=np.float32),
                "observation/gripper_position": np.array(np.atleast_1d(gripper_state), dtype=np.float32),
                "prompt": instruction
            }
            
            # Add second camera if available, otherwise PAD WITH ZEROS to prevent crashes
            if base_im_second is not None:
                base_im_second_resized = np.array(Image.fromarray(base_im_second).resize((320, 180)), dtype=np.uint8)
                obs_dict["observation/exterior_image_1_left"] = base_im_second_resized
            else:
                # The server strictly requires this key.
                obs_dict["observation/exterior_image_1_left"] = np.zeros_like(base_im_resized)

            _t0 = time.perf_counter()
            pred_action_chunk = self.client.infer(obs_dict)
            og.log.info(f"[dreamzero] inference time: {time.perf_counter() - _t0:.3f}s")
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

