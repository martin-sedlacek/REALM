
import numpy as np
from PIL import Image
import omnigibson as og
from openpi_client import websocket_client_policy, image_tools

from realm.geometry import axisangle_to_rpy
from realm.inference.dreamzero import DreamZeroClient


class _DebugAdapter:


    client = None

    def __init__(self, host, port):
        pass

    def infer(self, instruction, base_im, base_im_second, wrist_im, robot_state, gripper_state,
              use_base_im_second=False, ee_control=False, cartesian_position=None):
        if ee_control:
            return np.array([0.41402626, -0.13211727, 0.57253086, -3.09742367, 0.2580259, -0.24700592, -1])
        return np.atleast_1d(np.zeros(8))


class _OpenPIAdapter:


    def __init__(self, host, port):
        og.log.info("Connecting to server...")
        self.client = websocket_client_policy.WebsocketClientPolicy(
            host=host,
            port=port
        )

    def infer(self, instruction, base_im, base_im_second, wrist_im, robot_state, gripper_state,
              use_base_im_second=False, ee_control=False, cartesian_position=None):
        img_to_use = base_im_second if use_base_im_second else base_im

        obs_dict = {
            "prompt": instruction,
            "observation/joint_position": robot_state,
            "observation/gripper_position": np.atleast_1d(np.array(gripper_state)),
            "observation/exterior_image_1_left": image_tools.resize_with_pad(img_to_use, 224, 224),
            "observation/wrist_image_left": image_tools.resize_with_pad(wrist_im, 224, 224)
        }
        pred = self.client.infer(obs_dict)
        return pred["actions"]


class _DreamZeroAdapter:


    def __init__(self, host, port):
        self.client = DreamZeroClient(host=host, port=port)

    def infer(self, instruction, base_im, base_im_second, wrist_im, robot_state, gripper_state,
              use_base_im_second=False, ee_control=False, cartesian_position=None):
        assert base_im_second is not None, "DreamZero requires --multi-view (second external camera)"
        assert cartesian_position is not None, "DreamZero requires cartesian_position (robot-relative EE pose)"

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

        return self.client.infer(obs_dict)


class _GR00TAdapter:


    def __init__(self, host, port):
        og.log.info("Connecting to server...")
        self.client = websocket_client_policy.WebsocketClientPolicy(host=host, port=port)

    def infer(self, instruction, base_im, base_im_second, wrist_im, robot_state, gripper_state,
              use_base_im_second=False, ee_control=False, cartesian_position=None):
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
        return np.concatenate(
            [pred["action.joint_position"],
             pred["action.gripper_position"].reshape(-1, 1)], axis=-1)


class _GR00TN16Adapter:


    def __init__(self, host, port):
        # Imported here, not at module level: realm.inference.base needs zmq + msgpack, which the
        # eval containers only carry when this server type is actually in use.
        from realm.inference.base import ExternalRobotInferenceClient
        self.client = ExternalRobotInferenceClient(host=host, port=port)

    def infer(self, instruction, base_im, base_im_second, wrist_im, robot_state, gripper_state,
              use_base_im_second=False, ee_control=False, cartesian_position=None):
        base_im_resized = image_tools.resize_with_pad(base_im, 224, 224)[None, None]
        wrist_im_resized = image_tools.resize_with_pad(wrist_im, 224, 224)[None, None]

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
        return np.concatenate(
            [pred["action.joint_position"].reshape(-1, 7),
             pred["action.gripper_position"].reshape(-1, 1)], axis=-1)


class _MolmoActAdapter:
    """MolmoAct over the openpi websocket protocol: raw images, supports EE control. DISABLED.

    TODO: all DROID EE control poses need to have flip_pose_pointing_down() applied before being
    passed to the step.
    """

    def __init__(self, host, port):
        og.log.info("Connecting to server...")
        self.client = websocket_client_policy.WebsocketClientPolicy(host=host, port=port)

    def infer(self, instruction, base_im, base_im_second, wrist_im, robot_state, gripper_state,
              use_base_im_second=False, ee_control=False, cartesian_position=None):
        img_to_use = base_im_second if use_base_im_second else base_im
        obs_dict = {
            "images": [img_to_use, wrist_im],
            "instruction": instruction,
        }
        pred = self.client.infer(obs_dict)
        pred_action_chunk = pred["action"]

        if ee_control:
            pred_action_chunk = axisangle_to_rpy(pred_action_chunk)

        return pred_action_chunk


# Keep these keys aligned with the gripper conventions in realm.rollout.
ADAPTERS = {
    "debug": _DebugAdapter,
    "openpi": _OpenPIAdapter,
    "dreamzero": _DreamZeroAdapter,
    # "GR00T": _GR00TAdapter,          # disabled -- see the block comment above
    # "GR00T_N16": _GR00TN16Adapter,   # disabled
    # "molmoact": _MolmoActAdapter,    # disabled
}


class InferenceClient:
    def __init__(self, model_type, port, host="127.0.0.1", timeout=150.0):
        adapter_cls = ADAPTERS.get(model_type)
        if adapter_cls is None:
            raise NotImplementedError(
                f"model_type {model_type!r} has no registered adapter. Registered: "
                f"{sorted(ADAPTERS)}. GR00T / GR00T_N16 / molmoact exist in "
                f"realm/inference/client.py but are deliberately disabled."
            )
        self.model_type = model_type
        self.host = host
        self.port = port
        self._adapter = adapter_cls(host, port)
        self.client = self._adapter.client

    def infer(self, instruction, base_im, base_im_second, wrist_im, robot_state, gripper_state, use_base_im_second=False, ee_control=False, cartesian_position=None):
        return self._adapter.infer(
            instruction, base_im, base_im_second, wrist_im, robot_state, gripper_state,
            use_base_im_second=use_base_im_second, ee_control=ee_control,
            cartesian_position=cartesian_position,
        )

    def reset(self):
        if hasattr(self.client, "reset"):
            self.client.reset()
