"""Stable per-task RNG identities for evaluation."""
import json
from pathlib import Path
from uuid import UUID, NAMESPACE_URL, uuid5

import yaml


def task_seed(task_cfg_path, config_root, base_seed=1234):
    """Add a task UUID offset to the base seed within NumPy's 32-bit range."""
    path = Path(config_root) / "tasks" / task_cfg_path
    with path.open() as stream:
        config = yaml.safe_load(stream)
    identifier = config.get("task_uuid")
    if identifier is None:
        # Legacy/custom configs retain a portable identity independent of their path.
        identifier = uuid5(NAMESPACE_URL, json.dumps(config, sort_keys=True))
    identifier = UUID(str(identifier))
    seed = (base_seed + identifier.int) % (2**32)
    print(f"[REALM_RNG] task_uuid={identifier} base_seed={base_seed} seed={seed}", flush=True)
    return seed
