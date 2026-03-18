import time
import numpy as np
from PIL import Image
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


class InferenceClient:
    def __init__(self, model_type, port, host="127.0.0.1"):
        self.model_type = model_type
        self.host = host
        self.port = port
        self.client = None
        if model_type == "GR00T_N16":
            self.client = ExternalRobotInferenceClient(host=self.host, port=self.port)

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
