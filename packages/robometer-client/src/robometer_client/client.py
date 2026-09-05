"""HTTP client for a Robometer (RBM) reward-model eval server. No robometer dependency.

WHY A CLIENT AND NOT `pip install robometer`
--------------------------------------------
Robometer's runtime (torch 2.8, transformers>=4.57, unsloth, xformers, TensorFlow 2.19, and a hard
`requires-python == 3.10.*`) cannot share an environment with the REALM simulation container
(Python 3.11, torch 2.7.0+cu128, numpy 1.26 -- see .docker/constraints.txt). So the model runs as
its own process in its own environment -- `packages/robometer` is the pinned upstream checkout and
`scripts/run_robometer_server.sh` starts it -- and REALM talks to it over HTTP with this module,
exactly the way the openpi policy is reached through the vendored openpi-client.

PROTOCOL (robometer/evals/eval_server.py at the pinned submodule revision)
-------------------------------------------------------------------------
    POST {base_url}/evaluate_batch_npy       multipart form
        files: sample_{i}_trajectory_frames  -> .npy blob, uint8 (T, H, W, 3)
        data:  sample_{i}                    -> JSON sample; the frames field is replaced by
                                                {"__numpy_file__": "<file key>"}
               use_frame_steps               -> "true" | "false"
    GET  {base_url}/health                   -> {"status": "healthy", ...}
    GET  {base_url}/model_info               -> model path, experiment config, architecture

    Response:
        outputs_progress.progress_pred[i]    per-frame progress trace for sample i, in [0, 1]
        outputs_success.success_probs[i]     per-frame success probability (absent if the
                                             checkpoint has no success head)

WHAT THE TRACE LENGTH MEANS -- read before wiring this into per-step scoring
--------------------------------------------------------------------------
    use_frame_steps=False   ONE forward pass over the whole clip. The server linspace-subsamples
                            the frames to its training `max_frames` (16 for Robometer-4B) for the
                            forward pass, but the trace it returns has **T** entries, one per input
                            frame -- measured 2026-09-05 against the pinned revision at
                            T = 1/4/8/16/17/24/40, all returning len(progress) == T, including past
                            max_frames. This docstring previously said max_frames and NOT T; that
                            was wrong, and it mattered because it pointed at use_frame_steps=True
                            (T passes) to get a per-frame trace that one pass already provides.
                            parse_progress_response passes progress_pred through unchanged, so the
                            re-alignment is the server's. The LAST entry is the model's progress
                            estimate for the clip's final frame; ProgressResult.reward is that
                            value, clamped.
    use_frame_steps=True    T forward passes, each over frames[0:t] subsampled to 4 frames, so the
                            trace has exactly T entries aligned to the input frames. Slower by a
                            factor of T; better aligned with how the model was trained.

THE SERVER DOES NOT SUBSAMPLE ON THIS ENDPOINT -- send at most `max_frames` yourself
-----------------------------------------------------------------------------------
    /evaluate_batch_npy hands the frames straight to the collator: every frame you send becomes
    vision tokens, and the trace has T entries because T frames went through the model. The
    training-time rule (robometer/data/datasets/helpers.py::linspace_subsample_frames, max_frames
    16 for Robometer-4B) is applied only by the dataset loaders, not by the server. So a growing
    causal clip grows the forward pass with it: a 442-frame clip at 256x144 took the server past
    the memory of a 32 GB card that it shares with a policy server (measured 2026-09-05, CUDA OOM),
    and every frame past 16 is also outside the distribution the model was trained on.
    `subsample_frames` is that rule, exactly, so a caller can hold the clip at max_frames while
    keeping the first and the current frame -- ProgressResult.reward is then still the estimate
    for the current frame.

Progress is what Robometer predicts. It is a learned estimate of task completion in [0, 1], not a
rubric stage count, so it is NOT interchangeable with realm.environments.task_progression's scores
and must never be written into their columns.

The module-level helpers (as_frames_array, make_progress_sample, build_multipart_payload,
parse_progress_response) are pure and host-testable; RobometerClient is the thin transport on top.
`requests` is imported lazily so the helpers import anywhere numpy does.

Adapted from robometer's scripts/example_inference.py (MIT, Copyright (c) Anthony Liang and the
Robometer authors), trimmed to the progress path and given a batch interface.
"""
from __future__ import annotations

import io
import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

DEFAULT_PORT = 8000
DEFAULT_TIMEOUT_S = 120.0

#: Trajectory fields the server accepts as .npy blobs. Only `frames` is used here; the other two are
#: kept so a payload built for a checkpoint that wants precomputed embeddings needs no client change.
_NUMPY_FIELDS = ("frames", "lang_vector", "video_embeddings")


class RobometerServerError(RuntimeError):
    """The server answered, but not with something this client can use."""


@dataclass
class ProgressResult:
    """One sample's answer. Both arrays are float32; success_probs is EMPTY when the checkpoint has
    no success head, never None, so callers can index without a guard and check .size instead."""

    progress: np.ndarray
    success_probs: np.ndarray

    @property
    def reward(self) -> float:
        """Progress at the clip's last frame, clamped to [0, 1]; 0.0 for an empty trace. This is
        robometer's own reward convention (eval_utils.extract_rewards_from_output)."""
        if self.progress.size == 0:
            return 0.0
        return float(min(1.0, max(0.0, float(self.progress[-1]))))

    @property
    def success_prob(self) -> Optional[float]:
        """Success probability at the last frame, or None when the model has no success head."""
        if self.success_probs.size == 0:
            return None
        return float(self.success_probs[-1])


# --------------------------------------------------------------------------------------------------
# Pure helpers
# --------------------------------------------------------------------------------------------------

def as_frames_array(frames: Any) -> np.ndarray:
    """Normalise a clip to the uint8 (T, H, W, 3) array the server expects.

    Accepts an array or a sequence of per-frame arrays; (T, 3, H, W) is transposed, an RGBA channel
    is dropped (OmniGibson's rgb sensors emit 4 channels), and non-uint8 dtypes are clipped to
    [0, 255] and cast -- the same leniency as robometer's example client. Anything else raises,
    because a wrongly-shaped clip is a silent garbage score on the server side, not an error.
    """
    arr = np.asarray(frames)
    if arr.ndim == 3:
        # A single frame; promote so a one-frame clip needs no special case at the call site.
        arr = arr[None]
    if arr.ndim != 4:
        raise ValueError(f"frames must be (T, H, W, C) or (T, C, H, W); got shape {arr.shape}")
    if arr.shape[-1] not in (3, 4) and arr.shape[1] in (3, 4):
        arr = arr.transpose(0, 2, 3, 1)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.shape[-1] != 3:
        raise ValueError(f"frames must have 3 (or 4, RGBA) channels; got shape {arr.shape}")
    if arr.shape[0] == 0:
        raise ValueError("frames is empty (T == 0)")
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def subsample_frames(frames: Any, max_frames: int) -> np.ndarray:
    """Robometer's training-time clip rule (helpers.py::linspace_subsample_frames), so a clip sent
    to the server matches what the model was trained on: at most `max_frames` frames, evenly spaced
    by rounded linspace over [0, T-1], the first and the LAST frame always kept, indices
    non-decreasing. `max_frames <= 0` or T <= max_frames returns the clip unchanged. The last frame
    is what ProgressResult.reward reports on, so subsampling never moves the reward's subject."""
    arr = as_frames_array(frames)
    n = arr.shape[0]
    if max_frames is None or max_frames <= 0 or n <= max_frames:
        return arr
    if max_frames == 1:
        return arr[-1:]
    idx = np.rint(np.linspace(0, n - 1, max_frames)).astype(int).tolist()
    idx[0], idx[-1] = 0, n - 1
    for k in range(1, len(idx)):
        idx[k] = min(max(idx[k], idx[k - 1]), n - 1)
    return arr[idx]


def make_progress_sample(frames: np.ndarray, task: str, sample_id: str) -> Dict[str, Any]:
    """The sample dict for one progress query, frames still as an ndarray (build_multipart_payload
    moves them into a blob). `subsequence_length` is the full clip: the server reads it as "score
    the whole thing"."""
    frames = as_frames_array(frames)
    return {
        "sample_type": "progress",
        "trajectory": {
            "frames": frames,
            "frames_shape": tuple(int(x) for x in frames.shape),
            "task": str(task),
            "id": str(sample_id),
            "metadata": {"subsequence_length": int(frames.shape[0])},
            "video_embeddings": None,
        },
    }


def _npy_file_tuple(arr: np.ndarray, filename: str) -> Tuple[str, io.BytesIO, str]:
    buf = io.BytesIO()
    np.save(buf, arr)
    buf.seek(0)
    return (filename, buf, "application/octet-stream")


def build_multipart_payload(
    samples: Sequence[Dict[str, Any]], use_frame_steps: bool = False
) -> Tuple[Dict[str, Any], Dict[str, str]]:
    """Split samples into the (files, data) pair `requests.post(files=..., data=...)` takes.

    Every ndarray in a trajectory's numpy fields becomes an .npy blob keyed
    `sample_{i}_trajectory_{field}`, and the JSON sample carries `{"__numpy_file__": key}` in its
    place -- the wire format robometer's eval_utils.reconstruct_payload_from_npy reverses.
    """
    files: Dict[str, Any] = {}
    data: Dict[str, str] = {}

    for i, sample in enumerate(samples):
        traj = sample.get("trajectory", {})
        # Shallow-copy the shell; the arrays are pulled out below, everything else must be JSON.
        sample_copy = {k: v for k, v in sample.items() if k != "trajectory"}
        traj_copy = {k: v for k, v in traj.items() if k not in _NUMPY_FIELDS}

        for field in _NUMPY_FIELDS:
            val = traj.get(field)
            if val is None:
                traj_copy[field] = None
                continue
            if hasattr(val, "detach") and hasattr(val, "cpu"):  # torch.Tensor without importing torch
                val = val.detach().cpu().numpy()
            if isinstance(val, np.ndarray):
                key = f"sample_{i}_trajectory_{field}"
                files[key] = _npy_file_tuple(val, f"{key}.npy")
                traj_copy[field] = {"__numpy_file__": key}
            else:
                traj_copy[field] = val

        if isinstance(traj_copy.get("frames_shape"), (tuple, list)):
            traj_copy["frames_shape"] = [int(x) for x in traj_copy["frames_shape"]]

        sample_copy["trajectory"] = traj_copy
        data[f"sample_{i}"] = json.dumps(sample_copy)

    data["use_frame_steps"] = "true" if use_frame_steps else "false"
    return files, data


def parse_progress_response(outputs: Dict[str, Any], n_samples: int) -> List[ProgressResult]:
    """Turn the server JSON into one ProgressResult per sample, in request order.

    Raises RobometerServerError if the progress head is missing or the server answered for a
    different number of samples -- a silent shorter list would misattribute scores across a batch.
    """
    outputs_progress = outputs.get("outputs_progress")
    if not isinstance(outputs_progress, dict):
        raise RobometerServerError(
            f"no `outputs_progress` in server response; keys={sorted(outputs.keys())}")
    progress_pred = outputs_progress.get("progress_pred")
    if not isinstance(progress_pred, list):
        raise RobometerServerError("`outputs_progress.progress_pred` missing or not a list")
    if len(progress_pred) != n_samples:
        raise RobometerServerError(
            f"server returned {len(progress_pred)} progress traces for {n_samples} samples")

    outputs_success = outputs.get("outputs_success") or {}
    success_probs = outputs_success.get("success_probs") if isinstance(outputs_success, dict) else None
    if not isinstance(success_probs, list) or len(success_probs) != n_samples:
        success_probs = [[] for _ in range(n_samples)]

    results = []
    for trace, succ in zip(progress_pred, success_probs):
        results.append(ProgressResult(
            progress=np.asarray(trace if trace is not None else [], dtype=np.float32).reshape(-1),
            success_probs=np.asarray(succ if succ is not None else [], dtype=np.float32).reshape(-1),
        ))
    return results


# --------------------------------------------------------------------------------------------------
# Transport
# --------------------------------------------------------------------------------------------------

class RobometerClient:
    """Client for one Robometer eval server.

    `session` is anything with requests.Session's `post(url, files=, data=, timeout=)` and
    `get(url, timeout=)`; the default is a real requests.Session, created lazily so importing this
    module never needs requests. Tests inject a fake.

    Unlike the openpi websocket client, nothing here blocks forever: a connection failure raises
    immediately (requests.ConnectionError), and wait_until_healthy() gives up after its deadline.
    Do a preflight with it before starting an eval, as REALM's launchers do for the policy port.
    """

    def __init__(self, host: str = "localhost", port: int = DEFAULT_PORT, *, scheme: str = "http",
                 timeout_s: float = DEFAULT_TIMEOUT_S, use_frame_steps: bool = False, session=None):
        self.base_url = f"{scheme}://{host}:{int(port)}"
        self.timeout_s = float(timeout_s)
        self.use_frame_steps = bool(use_frame_steps)
        self._session = session

    @classmethod
    def from_url(cls, url: str, **kwargs) -> "RobometerClient":
        """Build from a full base URL such as http://node017:8000 (trailing slash tolerated)."""
        url = url.rstrip("/")
        scheme, rest = url.split("://", 1) if "://" in url else ("http", url)
        host, _, port = rest.rpartition(":")
        if not host or not port.isdigit():
            raise ValueError(f"expected scheme://host:port, got {url!r}")
        return cls(host=host, port=int(port), scheme=scheme, **kwargs)

    @property
    def session(self):
        if self._session is None:
            import requests  # deferred: see class docstring
            self._session = requests.Session()
        return self._session

    # -- endpoints ---------------------------------------------------------------------------------

    def health(self) -> Dict[str, Any]:
        resp = self.session.get(self.base_url + "/health", timeout=self.timeout_s)
        resp.raise_for_status()
        return resp.json()

    def model_info(self) -> Dict[str, Any]:
        resp = self.session.get(self.base_url + "/model_info", timeout=self.timeout_s)
        resp.raise_for_status()
        return resp.json()

    def wait_until_healthy(self, timeout_s: float = 600.0, poll_s: float = 5.0) -> Dict[str, Any]:
        """Poll /health until it answers `healthy` or the deadline passes (then TimeoutError).
        Model load on the server side takes minutes, so the default deadline is generous."""
        deadline = time.monotonic() + timeout_s
        last_error: Optional[BaseException] = None
        while True:
            try:
                status = self.health()
                if status.get("status") == "healthy":
                    return status
                last_error = RobometerServerError(f"unexpected /health payload: {status}")
            except Exception as exc:  # connection refused, 5xx, bad JSON: all mean "not yet"
                last_error = exc
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Robometer server at {self.base_url} not healthy after {timeout_s:.0f}s: {last_error}")
            time.sleep(poll_s)

    def progress(self, frames: Any, task: str) -> ProgressResult:
        """Score one clip against one instruction."""
        return self.progress_batch([frames], [task])[0]

    def progress_batch(self, frames_list: Sequence[Any], tasks: Sequence[str]) -> List[ProgressResult]:
        """Score N clips in one request; results come back in input order. Clips may differ in
        length -- each is its own sample. This is the call a vector eval wants: one round trip per
        wave rather than one per member."""
        if len(frames_list) != len(tasks):
            raise ValueError(f"{len(frames_list)} clips but {len(tasks)} tasks")
        if len(frames_list) == 0:
            return []
        samples = [make_progress_sample(f, t, str(i)) for i, (f, t) in enumerate(zip(frames_list, tasks))]
        files, data = build_multipart_payload(samples, use_frame_steps=self.use_frame_steps)
        resp = self.session.post(self.base_url + "/evaluate_batch_npy", files=files, data=data,
                                 timeout=self.timeout_s)
        resp.raise_for_status()
        return parse_progress_response(resp.json(), len(samples))
