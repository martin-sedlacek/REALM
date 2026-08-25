# Performance and scaling

What is actually expensive, and which levers have been measured.

> **Read the dates on numbers.** These figures come from profiling runs during the 3.9.1 port and
> the vectorization work. Several were superseded by later measurements *within the same project*,
> and where that happened the later one is what appears here. Nothing on this page is a promise
> about your hardware.

## Startup dominates short runs

On a small evaluation, roughly **64% of wall-clock time is startup** — importing Isaac, building the
scene, initialising physics — not rollout stepping.

Two consequences, and they are the most useful things on this page:

- **Do not compare configurations by wall-clock time.** You will mostly be comparing startup.
  Compare stepping time.
- **Amortise.** More repeats per process beats more processes. Sweeping the matrix with one process
  per cell pays the startup cost 160 times.

Reset is also not free, and it got worse — see below.

## The port changed the shape of the cost

Moving from OmniGibson 1.1.1 to 3.9.1:

| | Change |
|---|---|
| rollout stepping | **~1.9× faster**, and ~3.4× with the OG-lite fork |
| reset | **2.2–3.2× slower** |

So the port is a clear win for long rollouts and can be a loss for workloads dominated by resets.
If you run many short rollouts, measure before assuming the newer stack is faster for you.

## Rendering is the lever you already have

`--render_on_demand` is **on by default**. It renders only on steps whose observation feeds
inference and runs physics on the rest, which roughly **halves median step time**.

The cost is video: roughly **one recorded frame per action chunk**, so a 300-step rollout yields
tens of frames rather than hundreds. Turn it off with `--no-render_on_demand` when the video matters.

`--rendering_mode r` is cheaper than the default `rt`, and `pt` is more expensive. **The speed
multipliers quoted in older documentation for these were never measured — do not repeat them.** More
importantly, `r` changes what the policy sees, so it is a change to the experiment. Switching to it
for throughput needs a success-rate A/B, not just a timing comparison.

## Vectorization

`--num_envs N` shares one simulator across N environments.

**Around 4 environments is the safe operating point on a single L40S.** Above that the project's own
measurements disagree: one batch found 8 members meaningfully worse in aggregate, a later batch under
a more aggressive configuration found 8 better, and 16 was never shown to be economic. Treat 4 as the
default, measure if you want more, and **say which measurement you are relying on** if you publish a
number.

Two things that cost you:

- **Four perturbations force a stopped-simulator cycle** — `V-SC`, `VB-MOBJ`, `VSB-NOBJ`, `SB-VRB`,
  the ones that add or remove objects. They still vectorize: the cycle is batched once across the
  whole wave rather than per member. They are the expensive resets, not excluded ones.
- **Scene import cost grows worse than linearly.** Each import triggers a global play/stop, so
  building many scenes in one process is quadratic-ish. Importing a large number of scenes has taken
  over an hour.

There is also a **renderer descriptor-pool ceiling** that causes a segfault once enough scenes are
resident. The release image includes the raised descriptor-set limits from OG-lite. If you test a
different image and hit an unexplained segfault during high-`--num_envs` scene construction, verify
that it carries the same limit change.

## The contact cache: a lever that was spent

Early profiling found the non-physics step was almost entirely contact-cache work — about half of
all stepping time — which made it the single biggest lever. An incremental contact cache was
implemented and measured at roughly **−23% of total simulator step time** under a real policy.

**By the later vectorized measurements that lever is spent**: the non-physics step had dropped to a
fraction of a millisecond, well under 1% of a step. If you find older notes ranking the contact cache
as the top optimisation target, they predate this.

Turning every available flag on at once measured about **−9%** under the `debug` model, not the
larger figure a naive sum of individual levers would predict.

## GPU physics is not available

Running rigid-body dynamics on the GPU is blocked by device-consistency gaps upstream — the
simulation crashes rather than running slowly. It has been attempted and abandoned twice. Separately,
GPU dynamics was measured to be **near-irrelevant** to the physical behaviour being chased at the
time: the reference stack forced onto CPU retained 94% of its measured compliance.

So this is not a performance lever waiting to be pulled.

## Running vectorized REALM 3.9.1 at full speed

The supported high-throughput configuration is OG-lite on OmniGibson 3.9.1, CPU physics,
real-time rendering, render-on-demand, the incremental contact cache, and the proximity gate.
Both optimization flags currently default on in `realm/sim_config.py`, but set them explicitly in
recorded runs so the configuration remains reproducible:

Run this inside the release container on an allocated GPU node:

```sh
REALM_INCREMENTAL_CONTACT_CACHE=1 \
REALM_PROXIMITY_GATE=1 \
REALM_GPU_DYNAMICS=0 \
python -u examples/04_vector_evaluate.py \
    --num_envs 4 \
    --task_id 0 --perturbation_id 0 \
    --repeats 25 --max_steps 500 --horizon 8 \
    --model_type openpi --model_name <checkpoint-name> \
    --host 127.0.0.1 --port 8000 \
    --robot DROID_mounted \
    --experiment_name <experiment> --run_id <run-id> \
    --log_dir /app/logs --rendering_mode rt \
    --render_on_demand
```

The policy server must already be reachable at the supplied host and port. Start with
`--num_envs 4`; eight has measured better throughput on suitable hardware, while sixteen has hit
the renderer descriptor-pool ceiling during scene loading. Render-on-demand intentionally records
roughly one frame per action chunk. Use `--no-render_on_demand` when complete videos matter, at a
substantial throughput cost. Do not enable `REALM_GPU_DYNAMICS`: it is unsupported and has crashed
at reset. Do not enable contact-report filtering patterns without a separately validated pattern,
because omitted links silently disappear from task contact queries.

## Instrumentation notes

- **Wrist cameras render at 1280×720.** An earlier profiling note claiming 128×128 was retracted;
  128×128 is the shape of a zero-filled placeholder in one code path, not the render resolution.
- **Exit codes tell you nothing** about whether a profiling run succeeded.
- Run-to-run variance of around **17%** has been observed on the same configuration. Do not read a
  single pair of runs as a result.

## See also

- [Cluster and parallel runs](Cluster-and-Parallel-Runs)
- [Running evaluations](Running-Evaluations)
