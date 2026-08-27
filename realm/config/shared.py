"""Shared task rules and REALM environment switches."""

import os


COMPATIBILITY_MATRIX = {
    "put": ["pick", "rotate", "stack"],
    "push": [],
    "pick": ["put", "rotate", "stack"],
    "rotate": ["put", "pick", "stack"],
    "stack": ["put", "pick", "rotate"],
    "open_drawer": ["close_drawer"],
    "close_drawer": ["open_drawer"],
}

VERB_PHRASE = {
    "pick": "pick up",
    "put": "put",
    "rotate": "rotate",
    "stack": "stack",
    "push": "push",
    "open_drawer": "open",
    "close_drawer": "close",
}

UNSUPPORTED_TASK_TYPES = {"open_drawer", "close_drawer"}


def env_flag(name: str, default: bool) -> bool:
    """Read a REALM boolean environment switch."""
    return os.environ.get(name, "1" if default else "0") == "1"


def env_value(name: str, default: str) -> str:
    """Read a REALM environment value."""
    return os.environ.get(name, default)


def env_is_set(name: str) -> bool:
    return name in os.environ
