# Contributing to REALM

Thank you for improving REALM. Bug reports, documentation fixes, new robot or policy integrations,
and carefully validated benchmark changes are welcome.

## Environments

REALM has two deliberately separate environments:

- **Docker or Apptainer/SIF is the runtime.** It contains OmniGibson, Isaac Sim, REALM's runtime
  dependencies, and the simulator patches. The current recipes are
  `.docker/realm.Dockerfile` and `.docker/realm.def`; follow `wiki/Installation.md`.
- **uv is only for host checks.** Run `uv sync --locked`, then the tier-1 commands under
  [Testing](#testing). This environment cannot run simulations and must not accumulate runtime
  dependencies.

See `wiki/Installation.md` for setup and GPU commands.

## Making Changes

REALM is a benchmark, so preserve observable behavior and random-number draw order during refactors.
Do not silently fix a result-changing defect. Such changes need an explicit compatibility decision
and a `VERSION` bump.

Keep single- and vector-evaluation semantics shared in `realm/rollout.py`. Add tasks and
perturbations through the existing YAML layers and registries; avoid hard-coded parallel lists.

## Testing

Run tier 1 before submitting — lint and the container-free tests, as separate commands so a lint
failure does not hide the test result:

```bash
uv run ruff check realm examples tests scripts
uv run python tests/run_suite.py --only local --strict \
    --out tmp/suite/results.json --junit-xml tmp/suite/results.xml
uv run python -m pytest -q \
    tests/test_perturbation_task_types.py tests/test_cell_classification.py \
    tests/test_robot_base_column.py tests/test_robot_definition_parity.py
```

Do not run `pytest tests/`: most files are standalone Isaac scripts and collection can boot the
simulator. GPU-capable changes should also run the smallest relevant suite through
`tests/run_suite.py`; typically:

```bash
python tests/run_suite.py --jobid <slurm jobid> --mode stock --level smoke --strict \
    --out tmp/suite/results.json --junit-xml tmp/suite/results.xml
```

Record verification that must be completed on a GPU cluster in the pull request.

## Pull Requests

Use a focused, imperative commit subject, such as `rollout: validate camera selection`. Explain the
behavioral impact, exact commands and modes tested, affected assets/configuration, and any result
compatibility implications. Link related issues and include logs or images when output changes.
