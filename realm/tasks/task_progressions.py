import yaml
import os
from collections import OrderedDict

# Define the path to the YAML file
# Assuming the yaml file is in the same directory as this script
_current_dir = os.path.dirname(os.path.abspath(__file__))
_yaml_path = os.path.join(_current_dir, "task_progressions.yaml")

def load_task_progressions():
    with open(_yaml_path, "r") as f:
        data = yaml.safe_load(f)

    task_progressions = {}
    for task, stages in data.items():
        # Convert the list of stages to an OrderedDict with False values
        task_progressions[task] = OrderedDict((stage, False) for stage in stages)

    return task_progressions

TASK_PROGRESSIONS = load_task_progressions()
