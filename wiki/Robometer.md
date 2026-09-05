# Robometer (optional reward-model add-on)

[Robometer](https://github.com/robometer/robometer) ([paper](https://arxiv.org/abs/2603.02115),
[Robometer-4B on Hugging Face](https://huggingface.co/robometer/Robometer-4B)) is a learned
reward model: given a video clip and a language instruction it predicts per-frame **task progress**
in [0, 1] and, for checkpoints with a success head, a per-frame **success probability**.

**Status in REALM: an opt-in scorer.** Pass `--robometer` to either evaluator and
`task_progression` becomes Robometer's progress estimate instead of the rubric's fraction of
completed stages; without the flag nothing changes, and the benchmark's scores are exactly what
[Tasks and perturbations](Tasks-and-Perturbations) describes. The two are different quantities and
their rows must never be averaged together, so a Robometer run gets its own `--experiment_name` and
its report carries a `scorer` column. The flag-level details (cadence, success threshold, extra
columns) are in [Running evaluations](Running-Evaluations#scoring-with-robometer---robometer); this
page is about the pieces and the server.

## Why it runs as a separate server

The same arrangement as the openpi policy server, and for the same reason: the two stacks cannot be
installed together.

| | REALM simulation container | Robometer |
|---|---|---|
| Python | 3.11 (conda env `behavior`) | `== 3.10.*` (hard pin) |
| torch | 2.7.0+cu128 (Isaac Sim wheels are built against it) | 2.8.0 |
| numpy | 1.26 (`< 2`, OmniGibson) | any |
| transformers / xformers / unsloth / TensorFlow | absent | required |

So the model lives in its own environment on its own GPU (Robometer-4B needs roughly 10–12 GB) and
REALM reaches it over HTTP through a client that depends on `numpy` and `requests` only.

## What is in the repository

| Path | What | Where it runs |
|---|---|---|
| `packages/robometer` | git **submodule**: the upstream Robometer checkout, pinned to one revision | its own `uv` env, never inside the image |
| `packages/robometer-client` | vendored client, `robometer_client.RobometerClient` | installed into the image by both recipes |
| `scripts/run_robometer_server.sh` | starts the server from the submodule | the GPU node hosting the model |
| `realm/progress_scorer.py` | `RobometerScorer`, the `--robometer` seam in both evaluators; `RubricScorer` is the default passthrough | inside the image |
| `tests/test_robometer_client.py` | pins the wire format against a fake transport; tier-1, container-free | host |
| `tests/test_progress_scorer.py` | pins the scorer's cadence, threshold and columns against a fake client; container, no GPU | container |

The submodule is a **pin**, not just a convenience: the client was written against the server at
that revision, and `git submodule status` tells you which one. `.dockerignore` excludes the
submodule from the build context, so cloning it does not change the image.

## Bringing a server up

```sh
git submodule update --init packages/robometer     # once per checkout
./scripts/run_robometer_server.sh                  # Robometer-4B on 0.0.0.0:8010, one GPU
```

The first start downloads the checkpoint from Hugging Face (several GB; the host needs outbound
HTTPS to huggingface.co and, for gated models, `hf auth login`).

> **Two repositories are needed, not one.** The `robometer/Robometer-4B` snapshot is **weights
> only** -- no `tokenizer.json`, no `preprocessor_config.json`, no chat template. The loader takes
> those from the base model named inside the checkpoint's own `config.yaml`
> (`base_model_id: Qwen/Qwen3-VL-4B-Instruct`). On a cluster node without outbound HTTPS, pre-fetch
> **both** on a host that has it and run the node with `HF_HUB_OFFLINE=1`:
>
> ```sh
> python -c "from huggingface_hub import snapshot_download as d; d('robometer/Robometer-4B'); d('Qwen/Qwen3-VL-4B-Instruct')"
> ```
>
> Pre-fetching only the Robometer repo gets you through tag resolution and then dies ~90 s into
> model load with `OSError: We couldn't connect to 'https://huggingface.co' ... and couldn't find
> them in the cached files`. Unrelated and harmless in the same log: `find_best_model_tag` calls the
> HF **API** and logs `Error finding best tag ... offline mode is enabled`; resolution falls through
> to the local snapshot root, which is where the shards are. Model load then takes minutes.
`ROBOMETER_MODEL`, `ROBOMETER_HOST`, `ROBOMETER_PORT` and `ROBOMETER_NUM_GPUS` override the
defaults; anything else on the command line is passed through to the server's hydra config. The
default port is `8010`, matching the evaluators' `--robometer_port`, so it does not collide with a
policy server on `8000`. Then run an evaluation with `--robometer --robometer_host <host>`.

Verify from anywhere that can reach the node:

```sh
curl -s http://<host>:<port>/health        # {"status":"healthy","available_gpus":1,"total_gpus":1}
curl -s http://<host>:<port>/model_info    # model path, experiment config, parameter counts
```

## Using the client directly

`--robometer` does all of this for you; the client is documented for scripts and analyses.

```python
from robometer_client import RobometerClient

client = RobometerClient(host="node017", port=8010)
client.wait_until_healthy(timeout_s=600)              # raises TimeoutError; never blocks forever
result = client.progress(frames, "put the apple in the bowl")   # frames: uint8 (T, H, W, 3)
result.reward        # progress at the last frame, clamped to [0, 1]
result.progress      # the whole trace
result.success_prob  # None when the checkpoint has no success head

batch = client.progress_batch([frames_a, frames_b], [task_a, task_b])   # one round trip per wave
```

Frames are normalised for you: OmniGibson's RGBA observations lose their alpha channel, `(T, C, H,
W)` is transposed, non-`uint8` is clipped and cast. A wrongly shaped clip raises instead of being
scored as garbage.

**Read this before wiring per-step scoring.** The length of `result.progress` depends on the
`use_frame_steps` flag on the client:

| `use_frame_steps` | Server does | Trace length | Cost |
|---|---|---|---|
| `False` (default) | one forward pass; frames linspace-subsampled to the training `max_frames` (16), short clips padded by repeating the last frame | **`T`** -- see the note below | 1 pass |
| `True` | one pass per prefix `frames[0:t]`, each subsampled to 4 frames; results re-aligned to the input | exactly `T` | `T` passes |

> **Measured 2026-09-05 against the pinned revision (`352d160`), and it contradicts what this table
> used to say.** One query per clip length, `use_frame_steps=False`, live server:
>
> | frames sent `T` | 1 | 4 | 8 | 16 | 17 | 24 | 40 |
> |---|---|---|---|---|---|---|---|
> | `len(result.progress)` | 1 | 4 | 8 | 16 | 17 | 24 | 40 |
>
> `len(result.progress) == T` at every length, including past `max_frames`. The client is not doing
> it -- `parse_progress_response` passes `outputs_progress.progress_pred` through with a bare
> `np.asarray(...).reshape(-1)` -- so the server re-aligns to the input. This row previously claimed
> `max_frames`, **not** `T`, which would send you to `use_frame_steps=True` (T forward passes) to
> obtain a per-frame trace you already get in one. Nothing in REALM's scores moves: the scorer reads
> `result.reward`, the last entry, which is the final-frame estimate under either rule. But
> `robometer_progress_trace` in the reports is as long as the clip sent, not a fixed 16.

Either way the **last** entry is the model's estimate for the clip's final frame, which is what
`result.reward` returns. This is Robometer's own reward convention, and it is what `--robometer`
records: the scorer uses the default (`use_frame_steps=False`), one forward pass per query.

Unlike the openpi client, a connection failure raises immediately (`requests.ConnectionError`).
Do a preflight with `wait_until_healthy()` before starting an eval, as REALM's launchers already do
for the policy port.

## Verifying

- **Host, no container:** `python3 tests/test_robometer_client.py` (also part of
  `tests/run_suite.py --only local`). It checks the multipart layout the server reads and the
  response parsing, against a fake transport.
- **Inside the image:** `python -c "import robometer_client; print(robometer_client.__version__)"`.
  `apptainer test realm.sif` imports it alongside the other runtime dependencies.
  `./scripts/run_apptainer.sh python -u tests/test_progress_scorer.py` checks the `--robometer`
  seam with a fake client (no GPU).
- **Live:** a `curl /health` followed by one `client.progress()` call against a running server. The
  host test cannot tell you whether the server at the pinned revision still accepts the format; only
  this can. Then a short eval:
  `python -u examples/02_evaluate.py --task_id 0 --repeats 1 --max_steps 40 --model_type debug
  --model_name debug --port 8000 --experiment_name robometer_smoke --robometer --robometer_host <host>`
  should leave a report whose `scorer` column reads `robometer` and whose `robometer_queries` is
  about `max_steps / horizon`.

## Updating the pinned revision

```sh
cd packages/robometer && git fetch && git checkout <new revision> && cd -
git add packages/robometer
```

Then re-read `robometer/evals/eval_server.py` and `robometer/evals/eval_utils.py` for changes to
`/evaluate_batch_npy`, run the host test, and do one live call. Record all three in the pull request:
the client's contract is pinned by hand, nothing type-checks it across the two environments.

## See also

- [Installation](Installation)
- [Running evaluations](Running-Evaluations)
- [Tasks and perturbations](Tasks-and-Perturbations) — the scoring Robometer will sit beside
