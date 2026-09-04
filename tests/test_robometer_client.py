"""The vendored robometer-client speaks the wire format robometer's eval server reads, and reads
back what it writes -- checked against a fake transport, so no server, no GPU, no container.

WHY THIS EXISTS
---------------
packages/robometer-client is REALM's only contact with Robometer, and the two sides live in
environments that can never be installed together (see the client's module docstring). Nothing
type-checks the boundary, so the contract is pinned here: the multipart layout
(`sample_{i}_trajectory_frames` blobs, `{"__numpy_file__": key}` references, the
`use_frame_steps` flag), the response parsing (per-sample traces in request order, empty success
arrays when the head is absent), and the frame normalisation that stands between an OmniGibson
RGBA observation and the uint8 (T, H, W, 3) the server wants.

WHAT IT DOES NOT SEE: whether the LIVE server at the pinned submodule revision still accepts this.
That needs a GPU and a checkpoint; `curl /health` plus one `client.progress()` call is that check.

    python3 tests/test_robometer_client.py
"""
import io
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT / "packages" / "robometer-client" / "src"))

from robometer_client import (  # noqa: E402
    ProgressResult,
    RobometerClient,
    RobometerServerError,
    as_frames_array,
    build_multipart_payload,
    make_progress_sample,
    parse_progress_response,
)


class _FakeResponse:
    def __init__(self, payload, status=200):
        self._payload, self.status_code = payload, status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


class _FakeSession:
    """Records what the client sends and answers with a canned payload."""

    def __init__(self, payload, status=200):
        self.payload, self.status, self.calls = payload, status, []

    def post(self, url, files=None, data=None, timeout=None):
        self.calls.append(("POST", url, files, data, timeout))
        return _FakeResponse(self.payload, self.status)

    def get(self, url, timeout=None):
        self.calls.append(("GET", url, None, None, timeout))
        return _FakeResponse(self.payload, self.status)


def _clip(t, h=6, w=5, c=3, dtype=np.uint8):
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, size=(t, h, w, c), dtype=np.uint8).astype(dtype)


def main():
    failures = []

    def check(cell, cond, detail):
        print(f"[{cell}] {detail}: {'ok' if cond else 'FAIL'}")
        if not cond:
            failures.append(f"[{cell}] {detail}")

    # [1] frame normalisation --------------------------------------------------------------------
    rgba = _clip(4, c=4)
    out = as_frames_array(rgba)
    check(1, out.shape == (4, 6, 5, 3) and out.dtype == np.uint8 and np.array_equal(out, rgba[..., :3]),
          "RGBA (T,H,W,4) drops alpha to (T,H,W,3) uint8")
    chw = _clip(3).transpose(0, 3, 1, 2)
    check(1, as_frames_array(chw).shape == (3, 6, 5, 3), "(T,C,H,W) is transposed to (T,H,W,C)")
    check(1, as_frames_array(_clip(1)[0]).shape == (1, 6, 5, 3), "a single HxWxC frame becomes a 1-frame clip")
    f32 = _clip(2).astype(np.float32) + 300.0
    check(1, as_frames_array(f32).dtype == np.uint8 and as_frames_array(f32).max() == 255,
          "non-uint8 is clipped to [0,255] and cast")
    for bad in (np.zeros((0, 6, 5, 3), np.uint8), np.zeros((2, 6, 5), np.uint8), np.zeros((2, 6, 5, 2), np.uint8)):
        try:
            as_frames_array(bad)
            check(1, False, f"shape {bad.shape} is rejected")
        except ValueError:
            check(1, True, f"shape {bad.shape} is rejected")

    # [2] sample + multipart layout ----------------------------------------------------------------
    frames = _clip(7)
    sample = make_progress_sample(frames, "put the apple in the bowl", "0")
    files, data = build_multipart_payload([sample], use_frame_steps=False)
    check(2, set(files) == {"sample_0_trajectory_frames"}, "frames become the sample_0_trajectory_frames blob")
    fname, buf, ctype = files["sample_0_trajectory_frames"]
    roundtrip = np.load(io.BytesIO(buf.getvalue()))
    check(2, fname.endswith(".npy") and ctype == "application/octet-stream" and np.array_equal(roundtrip, frames),
          "the blob is a valid .npy that reloads to the exact frames")
    check(2, set(data) == {"sample_0", "use_frame_steps"} and data["use_frame_steps"] == "false",
          "data carries sample_0 JSON and the use_frame_steps flag")
    js = json.loads(data["sample_0"])
    traj = js["trajectory"]
    check(2, js["sample_type"] == "progress", "sample_type is progress")
    check(2, traj["frames"] == {"__numpy_file__": "sample_0_trajectory_frames"},
          "frames field is replaced by the __numpy_file__ reference")
    check(2, traj["frames_shape"] == [7, 6, 5, 3] and traj["metadata"]["subsequence_length"] == 7,
          "frames_shape and subsequence_length are the full clip")
    check(2, traj["task"] == "put the apple in the bowl" and traj["id"] == "0" and traj["video_embeddings"] is None,
          "task, id and a null video_embeddings are present")
    _, data_steps = build_multipart_payload([sample], use_frame_steps=True)
    check(2, data_steps["use_frame_steps"] == "true", "use_frame_steps=True is sent as 'true'")
    two_files, two_data = build_multipart_payload(
        [make_progress_sample(_clip(3), "a", "0"), make_progress_sample(_clip(5), "b", "1")])
    check(2, set(two_files) == {"sample_0_trajectory_frames", "sample_1_trajectory_frames"}
          and json.loads(two_data["sample_1"])["trajectory"]["frames_shape"][0] == 5,
          "a batch keeps per-sample blobs and shapes distinct")

    # [3] response parsing -------------------------------------------------------------------------
    resp = {"outputs_preference": None,
            "outputs_progress": {"progress_pred": [[0.1, 0.4, 1.7], [0.0, -0.2]]},
            "outputs_success": {"success_probs": [[0.2, 0.6, 0.9], [0.1, 0.05]]}}
    res = parse_progress_response(resp, 2)
    check(3, len(res) == 2 and all(isinstance(r, ProgressResult) for r in res), "one ProgressResult per sample")
    check(3, res[0].progress.dtype == np.float32 and np.allclose(res[0].progress, [0.1, 0.4, 1.7]),
          "trace is returned untouched as float32")
    check(3, res[0].reward == 1.0 and res[1].reward == 0.0, "reward is the last value clamped to [0,1]")
    check(3, abs(res[0].success_prob - 0.9) < 1e-6, "success_prob is the last success value")
    no_succ = parse_progress_response({"outputs_progress": {"progress_pred": [[0.3]]}, "outputs_success": None}, 1)
    check(3, no_succ[0].success_probs.size == 0 and no_succ[0].success_prob is None
          and abs(no_succ[0].reward - 0.3) < 1e-6,
          "a checkpoint without a success head yields an empty array and success_prob None")
    for bad, why in (({"outputs_progress": None}, "missing progress head"),
                     ({"outputs_progress": {"progress_pred": [[0.1]]}}, "trace count != sample count")):
        try:
            parse_progress_response(bad, 2)
            check(3, False, f"{why} raises RobometerServerError")
        except RobometerServerError:
            check(3, True, f"{why} raises RobometerServerError")

    # [4] transport ----------------------------------------------------------------------------------
    session = _FakeSession(resp)
    client = RobometerClient(host="node017", port=8010, session=session, use_frame_steps=True, timeout_s=33)
    out = client.progress_batch([_clip(3), _clip(2)], ["a", "b"])
    method, url, sent_files, sent_data, timeout = session.calls[-1]
    check(4, (method, url, timeout) == ("POST", "http://node017:8010/evaluate_batch_npy", 33.0),
          "progress_batch POSTs /evaluate_batch_npy at host:port with the configured timeout")
    check(4, sent_data["use_frame_steps"] == "true" and len(sent_files) == 2, "client-level use_frame_steps is honoured")
    check(4, len(out) == 2 and out[0].reward == 1.0, "batch results parse in order")
    check(4, client.progress_batch([], []) == [] and len(session.calls) == 1, "an empty batch makes no request")
    one_session = _FakeSession({"outputs_progress": {"progress_pred": [[0.2, 0.5]]}})
    one = RobometerClient(session=one_session).progress(_clip(4), "c")
    check(4, isinstance(one, ProgressResult) and abs(one.reward - 0.5) < 1e-6
          and json.loads(one_session.calls[-1][3]["sample_0"])["trajectory"]["task"] == "c",
          "progress() is the one-sample case of progress_batch")
    try:
        client.progress_batch([_clip(1)], ["a", "b"])
        check(4, False, "mismatched clips/tasks raises ValueError")
    except ValueError:
        check(4, True, "mismatched clips/tasks raises ValueError")
    client.health()
    client.model_info()
    check(4, [c[1] for c in session.calls[-2:]] == ["http://node017:8010/health", "http://node017:8010/model_info"],
          "health/model_info GET the right endpoints")
    err = RobometerClient(session=_FakeSession({}, status=503))
    try:
        err.progress(_clip(1), "a")
        check(4, False, "an HTTP error propagates from raise_for_status")
    except RuntimeError:
        check(4, True, "an HTTP error propagates from raise_for_status")
    check(4, RobometerClient.from_url("https://gpu-1:9000/").base_url == "https://gpu-1:9000"
          and RobometerClient().base_url == "http://localhost:8000", "from_url and defaults resolve the base URL")
    try:
        RobometerClient(session=_FakeSession({"status": "loading"})).wait_until_healthy(timeout_s=0.0, poll_s=0.0)
        check(4, False, "wait_until_healthy gives up at the deadline with TimeoutError")
    except TimeoutError:
        check(4, True, "wait_until_healthy gives up at the deadline with TimeoutError")
    check(4, RobometerClient(session=_FakeSession({"status": "healthy"})).wait_until_healthy(timeout_s=1)["status"]
          == "healthy", "wait_until_healthy returns the healthy payload")

    print("\n" + "=" * 78)
    if failures:
        print(f"FAILED -- {len(failures)} problem(s):")
        for f in failures:
            print(f"  - {f}")
    else:
        print("PASSED -- robometer-client payload, response parsing and transport match the server contract")
    print("=" * 78)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
