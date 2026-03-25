import uuid
import numpy as np
from PIL import Image
import omnigibson as og


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
