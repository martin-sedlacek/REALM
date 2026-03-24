import base64
import re
import cv2
import numpy as np
from openai import OpenAI
import omnigibson as og


class HamsterClient:
    """
    Client for the HAMSTER (Hierarchical Action Models For Open-World Robot Manipulation) server.
    """
    GRIPPER_CLOSE = 0
    GRIPPER_OPEN = 1
    MODEL_NAME = "Hamster_dev"

    def __init__(self, host=None, port=8000, ip_file="./ip_eth0.txt"):
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
