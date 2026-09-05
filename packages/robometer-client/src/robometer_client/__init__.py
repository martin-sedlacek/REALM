"""Thin HTTP client for a running Robometer reward-model server.

See client.py's module docstring for the protocol and the shape of what comes back. The package
deliberately depends on numpy and requests only, so it installs into the REALM simulation container
next to OmniGibson's pins; the model itself runs elsewhere (packages/robometer, its own uv env).
"""
from robometer_client.client import (
    DEFAULT_PORT,
    ProgressResult,
    RobometerClient,
    RobometerServerError,
    as_frames_array,
    build_multipart_payload,
    make_progress_sample,
    parse_progress_response,
    subsample_frames,
)

__version__ = "0.2.0"

__all__ = [
    "DEFAULT_PORT",
    "ProgressResult",
    "RobometerClient",
    "RobometerServerError",
    "__version__",
    "as_frames_array",
    "build_multipart_payload",
    "make_progress_sample",
    "parse_progress_response",
    "subsample_frames",
]
