# Running the test suite

REALM's tests are not a pytest suite, and running them the way the filenames suggest gets you
nothing. This page is how to actually run them.

## `pytest tests/` boots Isaac just to collect. Do not use it.

Most files in `tests/` are standalone scripts — an `if __name__ == "__main__":` block with printed
verdict lines that `sys.exit(1)` on failure — and pytest cannot run those. The exceptions are
**four real pytest modules**, all host-safe by design (they read code and configs as text with
`ast`/`yaml` instead of importing omnigibson):

```sh
pytest tests/test_perturbation_task_types.py tests/test_cell_classification.py \
       tests/test_robot_base_column.py tests/test_robot_definition_parity.py
```

Run them **by filename, exactly as above** — never as `pytest tests/`. Collection works by
**importing** each module, and three of the script-style files pull in `omnigibson` at module
scope: `test_joint_reset_batching` and `test_scene_object_placement` import it directly, and
`test_rollout_camera_selection` reaches it through `realm.rollout`. That is a full Isaac boot,
about a minute, before a single test runs. (The eval drivers — `test_integrity`,
`test_single_task`, `test_perturbations_integrity`, `test_vector_integrity` — used to boot Isaac
at import too; they now `ast`-parse the task/perturbation lists instead and boot it only in
their child processes.)

The host-only uv environment supplies pytest, Ruff and PyYAML without attempting to reproduce the
simulation runtime. `uv run make check` runs the four pytest modules plus the script-style static
checks. Pytest remains the wrong tool for the other files.

The driver is **`tests/run_suite.py`**, wrapped by `make`.

## The commands

The pipeline has **two tiers**, split on "does this need Isaac, the container and a GPU".

### Tier 1 — static. No container, no GPU, no dataset. ~1 s.

```sh
uv sync --locked   # create/update the host-only tool environment
uv run make check  # lint + the container-free tests — tier 1 in full
uv run make lint   # ruff, with .ruff.toml's deliberately narrow ruleset
```

`.github/workflows/static-checks.yml` runs this same locked command on every push and pull request.
It does not build or run the simulation container.

### Tier 2 — GPU. Needs the image, the dataset and a card.

The Make targets accept `ALLOC` as a scheduler allocation identifier through an operator-provided
site adapter:

```sh
ALLOC=<jobid> make test-smoke    # ~12 min   the cheap gate
ALLOC=<jobid> make test-suite    # ~1.7 h    the gate before trusting a change
ALLOC=<jobid> make test-matrix   # hours     the task × perturbation sweep
ALLOC=<jobid> make test-server   #           needs a policy server on :8000
```

| Level | What it runs | Cost |
|---|---|---|
| `smoke` | the static test, joint-reset scheduling, one task end to end, and the scene check at `num_envs=2` | ~12 min |
| `suite` | `smoke` plus both drawer paths, all 10 tasks, all 16 perturbations | ~1.7 h |
| `matrix` | the full task × perturbation sweep through the vector path | hours; no completed run on record |

The rough costs were measured on one L40S-class GPU and will vary with hardware, storage, caches,
rendering settings, and policy-server placement.

The release image contains the required OmniGibson fixes and is validated in its default `stock`
mode. `SUITE_MODE=oglite` is for testing a host checkout of the separate OG-lite fork.

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

`ALLOC` must identify a running allocation understood by your site adapter. The repository does not
ship cluster-specific allocation or launch wrappers. On another scheduler, run the corresponding
test scripts inside the release container on an allocated GPU node or provide a local adapter for
`tests/run_suite.py`.

**Before trusting any of this, read [Test coverage](Test-Coverage)** — what a pass in either tier
does and does not establish. It is short, and it is the point.

Knobs: `SUITE_OUT=` (results JSON, default `tmp/suite/results.json`), `SUITE_MODE=`
(`stock`/`oglite`, default `stock`), `SUITE_ARGS=` (passed through to `run_suite.py`).

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

## Tier 1 is a green gate

`uv run make check` is expected to pass. It runs Ruff's deliberately narrow `F401`/`F811` rules,
two standalone AST/YAML contract checks, and the four explicitly named host pytest modules. Missing
tools are resolved by `uv sync --locked`; a failure after that is a regression, not a known baseline.

## Which mode to run

The default `stock` mode uses the OmniGibson package baked into `realm.sif` and is the release gate.
Use `oglite` only when deliberately binding and testing a host OG-lite checkout:

```sh
ALLOC=<jobid> make test-suite
ALLOC=<jobid> SUITE_MODE=oglite make test-suite  # host-fork development only
```

**Always say which mode a result came from.** `run_suite.py` prints `mode` as a per-result column
rather than a header field, precisely because a results file accumulates across invocations that
differ in exactly that.

## Continuous integration

**`static-checks.yml` — tier 1, GitHub-hosted, every push and PR.** It installs the locked virtual
uv project and runs `uv run --locked make check`, then uploads the JUnit report. It exercises no
simulation, scene, rollout, Docker/SIF image, dataset, or GPU path.

There is no GPU workflow or self-hosted runner. Tier 2 remains an explicit manual gate through the
same `tests/run_suite.py` entry point used by the Make targets.

### There is deliberately no badge in the README

For either workflow. A "passing" badge on a repository whose simulation is untested by that badge
would be worse than no badge, and this project has a documented history of precisely that failure
mode — a perturbation that passed every numeric check while being a complete no-op, and the drawer
tasks that pass today with scene 0's cabinet lying on its back.

What a green run is actually worth, per tier, is tabulated in [Test coverage](Test-Coverage).

## See also

- [Installation](Installation) — the single-command install check
- [Quick start](Quick-Start) — GPU setup and the smoke test
- [Running evaluations](Running-Evaluations) — container execution and the full flag surface
- [Test coverage](Test-Coverage) — what a pass in either tier does and does not establish
- [Known issues and gotchas](Known-Issues-and-Gotchas) — why exit code 0 proves nothing
