import yaml
import os
from collections import OrderedDict

_current_dir = os.path.dirname(os.path.abspath(__file__))
_yaml_path = os.path.join(_current_dir, "../config/tasks/task_progressions.yaml")

def load_task_progressions():
    with open(_yaml_path, "r") as f:
        data = yaml.safe_load(f)

    task_progressions = {}
    for task, stages in data.items():
        task_progressions[task] = OrderedDict((stage, False) for stage in stages)

    return task_progressions

TASK_PROGRESSIONS = load_task_progressions()
