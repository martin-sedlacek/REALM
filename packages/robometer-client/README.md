# robometer-client

Thin HTTP client for a running [Robometer](https://github.com/robometer/robometer) reward-model
eval server. Depends on `numpy` and `requests` only, so it installs into the REALM simulation
container next to OmniGibson's pins. The model itself runs elsewhere: `packages/robometer` is the
pinned upstream checkout, and `scripts/run_robometer_server.sh` starts it in its own uv environment.

This mirrors how `packages/openpi-client` reaches an openpi policy server. Read the module docstring
of `src/robometer_client/client.py` for the wire protocol and, more importantly, for what the
returned trace length means under `use_frame_steps` on and off.

```python
from robometer_client import RobometerClient

client = RobometerClient(host="node017", port=8010)
client.wait_until_healthy(timeout_s=600)          # model load takes minutes; do not skip
result = client.progress(frames, "put the apple in the bowl")   # frames: uint8 (T, H, W, 3)
result.reward          # progress at the last frame, clamped to [0, 1]
result.progress        # the whole trace
result.success_prob    # None if the checkpoint has no success head

batch = client.progress_batch([frames_a, frames_b], [task_a, task_b])   # one round trip
```

Protocol pinned against the submodule revision recorded in the repo's `.gitmodules` /
`git submodule status`. Adapted from robometer's `scripts/example_inference.py` (MIT).
