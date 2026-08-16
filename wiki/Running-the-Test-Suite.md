# Running the test suite

REALM's tests are not a pytest suite, and running them the way the filenames suggest gets you
nothing. This page is how to actually run them.

## `pytest tests/` collects zero tests. Do not use it.

Every file in `tests/` is named `test_*.py`, and **none of them defines a collectable test** — no
`def test_*`, no `class Test*`, no `import pytest`. They are standalone scripts with an
`if __name__ == "__main__":` block that `sys.exit(1)` on failure.

So `pytest tests/` collects nothing — and it collects nothing *expensively*, because collection
works by **importing** each module, and four of these import `omnigibson` at module scope. That is a
full Isaac boot, about a minute, to discover zero tests.

(pytest is not missing. It is installed in the container. It is simply the wrong tool here.)

The driver is **`tests/run_suite.py`**, wrapped by `make`.

## The commands

The pipeline has **two tiers**, split on "does this need Isaac, the container and a GPU".

### Tier 1 — static. No container, no GPU, no dataset. ~1 s.

```sh
make check         # lint + the container-free tests — tier 1 in full
make lint          # ruff, with .ruff.toml's deliberately narrow ruleset
make test-static   # the container-free tests
```

Runs in GitHub Actions on every push and pull request
(`.github/workflows/static-checks.yml`).

### Tier 2 — GPU. Needs the image, the dataset and a card.

```sh
ALLOC=<jobid> make test-smoke    # ~11 min   the cheap gate
ALLOC=<jobid> make test-suite    # ~1.7 h    the gate before trusting a change
ALLOC=<jobid> make test-matrix   # hours     the task × perturbation sweep
ALLOC=<jobid> make test-server   #           needs a policy server on :8000
```

| Level | What it runs | Cost |
|---|---|---|
| `smoke` | the static test, joint-reset scheduling, one task end to end, and the scene check at `num_envs=2` | ~11 min |
| `suite` | `smoke` plus both drawer paths, all 10 tasks, all 16 perturbations | ~1.7 h |
| `matrix` | the full task × perturbation sweep through the vector path | hours; no completed run on record |

Costs are **measured**, from `logs/suite_results_v2.json` (2026-08-16, one L40S) — except
`test_single_task`, which had no completed measurement when the levels were written.

`make test-suite` additionally **re-runs the OG-lite-sensitive cells at `MODE=oglite`** as a second
invocation, because at `MODE=stock` a drawer scene can be physically wrong while every artifact
check passes. See below.

### Either tier

```sh
make test          # tier 1 only, then a list of exactly what it SKIPPED
make test-list     # what is in the suite and what each member needs
make test-report   # re-print the last run's table, running nothing
```

> **`make test` does not run the suite, and says so.** It runs tier 1 — **1 of the suite's 12
> entries** — and then prints which eleven it did not. That is deliberate: a `make test` that
> quietly covered a twelfth of the suite while looking like a full pass would be this project's
> worst failure mode, which is things passing while being wrong.

`ALLOC` is a **RUNNING** Slurm allocation — see [Quick start](Quick-Start) step 0. Every tier-2
target refuses without it, because `rr` starts the container **wherever it is invoked**: with no
allocation the container tests run on the login node, get no GPU, and fail confusingly rather than
obviously.

**Before trusting any of this, read [Test coverage](Test-Coverage)** — what a pass in either tier
does and does not establish. It is short, and it is the point.

Knobs: `SUITE_OUT=` (results JSON, default `tmp/suite/results.json`), `SUITE_MODE=`
(`stock`/`stockfix`/`oglite`, default `stock`), `SUITE_ARGS=` (passed through to `run_suite.py`).

To run one test rather than a tier, name it:

```sh
ALLOC=<jobid> python3 tests/run_suite.py --only test_single_task_drawer --mode oglite --strict
```

## The tiers

`make test-list` prints this live. `--only` accepts a tier name or a comma-separated list of test
names.

| Tier | Needs | Roughly |
|---|---|---|
| `local` | nothing | 0.06 s |
| `fast` | the container | ~35 s |
| `medium` | container + GPU | minutes each |
| `slow` | container + GPU | tens of minutes to hours each |
| `server` | container + GPU + a live policy server on `:8000` | ~1 h |

> **`local` is the only container-free tier.** `fast` is *fast*, not portable:
> `test_joint_reset_batching` stubs `og.sim` and needs no GPU, but it still does
> `import omnigibson` at module scope, so on a login python it dies with `ModuleNotFoundError`.
> Do not read `needs_gpu=0` in `make test-list` as "runs anywhere" — that column is about the
> **device**, not about the container.

## Verdicts come from printed lines, never from exit codes

This is the same hazard as everywhere else in REALM (see
[Known issues](Known-Issues-and-Gotchas)), and it is why the driver exists in this shape.

Isaac's shutdown hard-exits with status **0** on an unhandled exception, and can segfault at
teardown *after* a test has already printed a pass. So a child's exit code carries no information
about the test. `run_suite.py` records it in the JSON as an observation and **never gates on it**;
every verdict comes from matching the test's own printed verdict line (`PASSED -- `,
`ALL TASKS PASSED INTEGRITY CHECK!`, `N passed, M known-broken, 0 failed`, …).

Two things *are* gateable, and both are opt-in:

- **`--strict`** (every `make` target passes it) makes the **driver's** exit status mean "every
  test I ran ended `PASS` or `SKIP`". That is the thing to gate a script on.
- **`--junit-xml PATH`** writes a JUnit report **once, after the last test**. Its *absence* is the
  signal that **the driver itself** died — an OOM, a walltime kill, a node failure — which no exit
  code could have told you. Both workflows gate on exactly that, following upstream BEHAVIOR-1K's
  pattern (`.github/workflows/tests.yml` there): run the tests with `continue-on-error`, then

  ```sh
  if [ ! -f results.xml ]; then echo "driver died"; exit 1
  elif grep -Eq 'failures="[1-9][0-9]*"|errors="[1-9][0-9]*"' results.xml; then exit 1
  ```

The JSON (`--out`) is the **record** and is rewritten after every test, so a driver that is killed
still leaves a complete account of the tests that finished. The XML is the **gate** and is written
once. That difference is deliberate — writing the XML incrementally would destroy the signal it
exists to carry.

## Two tier-1 checks are expected to FAIL right now

Do not treat either as a broken install. Both are the repository's real state.

**`make lint` reports 25 findings** — `F401`/`F811`, in `realm/` (3), `scripts/` (19) and `tests/`
(3), all `--fix`-able. `.ruff.toml` records the baseline and explains why the ruleset is two rules
wide. (Separately, `ruff check --select E9,F63,F7,F82` finds 3 × `F821 Undefined name 'np'` in
`scripts/clara/interactive/t10_bhobj_props.py` — a real bug, not a style finding.)

**`make test-static` reports `FAILED -- 2 problem(s)`.**

`tests/test_task_progression_rubrics.py` was committed red on purpose. It reports two real defects:

1. the `pour` rubric in `realm/config/tasks/task_progressions.yaml` names a `POUR` stage that
   `success_conditions` has no key for. `get_task_progression()` does
   `checker_function = self.success_conditions.get(stage)` and then calls the result **anyway**, so
   an unknown stage is not a skipped stage — it is `TypeError: 'NoneType' object is not callable`,
   thrown mid rollout;
2. `success_conditions["POURED"] -> check_pour` does not accept `obs`, though it is invoked as
   `checker(obs)`.

Both are **latent**: no shipped task config declares `pour`, so neither fires today. They are
reported rather than the test being weakened to green. Expect:

```
FAILED -- 2 problem(s):
```

Everything else in the suite is expected to pass. If `make test-static` reports anything other than
those two problems, that is new.

## Which `MODE` to run under, and why it changes the answer

`SUITE_MODE` selects the OmniGibson bind, exactly as `MODE` does for `rr` — see
[Running evaluations](Running-Evaluations). The default is `stock`, the image's own 3.9.1.

**`stock` is not sufficient where scene correctness matters.** The current image carries most of the
OmniGibson patches but **not the up-axis fix**, which lives only in the OG-lite fork. Without it, a
referenced layer whose `upAxis` disagrees with the stage's gets an `xformOp:rotateX:unitsResolve`
appended that no OmniGibson pose setter can see — and it is materialised only for the **first**
reference to the asset. Measured on `impact_drawer`'s `cabinet.usd` at `num_envs=2`: scene 0's
cabinet lay on its back with its drawers jammed at 0.229 m of a 0.300 m range, while scene 1's stood
upright.

The trap is that **the drawer tasks still load, still run, and still pass** under `stock` — 10/10 on
`test_integrity`, 2/2 on the vector drawer cells. Nothing in `tests/` notices, because almost every
check is "the artifacts exist with the right row count" and the `debug` policy never touches the
drawer. `test_scene_object_placement` is the exception and the reason it exists: it is the only test
that looks at the **scene** rather than at the artifacts.

So:

```sh
ALLOC=<jobid> SUITE_MODE=oglite make test-suite  # when an OUTCOME matters
ALLOC=<jobid> make test-suite                    # when "does it build and write artifacts" is the question
```

**Always say which mode a result came from.** `run_suite.py` prints `mode` as a per-result column
rather than a header field, precisely because a results file accumulates across invocations that
differ in exactly that.

## Continuous integration: two workflows, one of which cannot run yet

**`static-checks.yml` — tier 1, GitHub-hosted, every push and PR.** Static-only by necessity: a
GitHub-hosted runner has no NVIDIA device, the ~13 GB image is published nowhere and currently
cannot even be rebuilt, and the dataset is ~36 GB behind an EULA. **No simulation is exercised** —
no scene, no rollout, no artifact. Eleven of the suite's twelve entries cannot run there and do not.

It checks: every Python file byte-compiles; every shell script under `scripts/` passes `bash -n`,
including `rr`, `go` and `lib/paths.sh`; every YAML under `realm/config/` parses (the
trailing-comma bug class — `use_cc_compensation: False,` → the truthy **string** `"False,"` — lives
there); every `SUITE` entry and every `LEVELS` member still points at a file that exists; every
tier-2 `make` target refuses without an allocation; and the `local` tier runs, non-blocking,
because it is known red.

It is called `static-checks` rather than `tests` or `CI` for that reason.

**`gpu-suite.yml` — tier 2, self-hosted, `workflow_dispatch` only. It cannot run yet.** No
self-hosted runner is registered for this repository. The file is committed because the *same
entry point* runs both ways — its steps are `ALLOC=<jobid> make test-suite` with an
allocation-shaped hole — and because it makes the runner question concrete rather than theoretical.
Its `schedule:` block is commented out on purpose: a scheduled job with no matching runner queues
until it times out and reports a failure that means nothing.

Upstream BEHAVIOR-1K, which REALM forks, **does** run its GPU tests in CI on
`[self-hosted, linux, gpu, dataset-enabled]`. So this is not impossible — it is a decision about
whether Clara can host a runner. That decision is Martin's; `gpu-suite.yml`'s header sets out what
it would take (outbound HTTPS, the labels, how the runner reaches a GPU on a Slurm cluster, and the
fork-PR security constraint).

### There is deliberately no badge in the README

For either workflow. A "passing" badge on a repository whose simulation is untested by that badge
would be worse than no badge, and this project has a documented history of precisely that failure
mode — a perturbation that passed every numeric check while being a complete no-op, and the drawer
tasks that pass today with scene 0's cabinet lying on its back.

What a green run is actually worth, per tier, is tabulated in [Test coverage](Test-Coverage).

## See also

- [Installation](Installation) — the single-command install check
- [Quick start](Quick-Start) — getting an allocation, and the smoke tests
- [Running evaluations](Running-Evaluations) — `MODE`, and the full flag surface
- [Test coverage](Test-Coverage) — what a pass in either tier does and does not establish
- [Known issues and gotchas](Known-Issues-and-Gotchas) — why exit code 0 proves nothing
