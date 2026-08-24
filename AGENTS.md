# Repository Guidelines

## Project Structure & Module Organization

`realm/` contains evaluation entry points, rollout logic, environments, inference adapters, robot definitions, and layered YAML configuration. Put usage examples in `examples/`, tests in `tests/`, cluster and debugging utilities in `scripts/`, and operator documentation in `wiki/`. Longer investigations belong in `docs/`. Simulation assets live under `realm/robots/`, `custom_assets/`, and `images/`; avoid committing generated output from `logs/` or `tmp/`.

REALM is a reproducibility-sensitive benchmark. Preserve behavior and RNG draw order during refactors. Changes that intentionally alter benchmark numbers require explicit review and a `VERSION` bump.

## Build, Test, and Development Commands

- `./setup.sh --docker --dataset` prepares the recommended container and dataset.
- `uv sync --locked` creates the host-only lint/static-test environment; it is not a runtime.
- `uv run make check` runs all host-safe checks: Ruff plus static tests.
- `make lint` runs the narrow ruleset in `.ruff.toml` over `realm`, `examples`, `tests`, and `scripts`.
- `make test-list` displays suite entries and their runtime requirements.
- `ALLOC=<jobid> make test-smoke` runs the approximately 12-minute GPU gate.
- `ALLOC=<jobid> make test-suite` runs the full approximately 1.7-hour suite.

Use `SUITE_MODE=oglite` when validating scene correctness. GPU work requires the container, dataset, and an active Slurm allocation.

## Coding Style & Naming Conventions

Use four-space indentation and conventional Python naming: `snake_case` for modules, functions, and variables; `PascalCase` for classes; and `UPPER_CASE` for constants. Keep configuration names aligned with existing YAML identifiers. Prefer shared behavior in `realm/rollout.py` over duplicating logic between single and vector evaluation. Do not reorder or remove random draws without treating the change as a benchmark-semantic modification.

## Testing Guidelines

Most files in `tests/` are standalone scripts whose printed verdicts are interpreted by `tests/run_suite.py`. Do **not** run `pytest tests/`; collection can boot Isaac. Run `uv run make check`, or invoke only the four host-safe pytest modules listed by `HOST_PYTESTS` in the `Makefile`. Name new tests `test_<behavior>.py`, register script-style tests with the suite driver, and use `--strict` for reliable gating. Record required off-cluster verification in the pull request.

## Commit & Pull Request Guidelines

Recent commits use concise, imperative, sentence-style subjects, often naming the affected component (for example, `b_hobj: cast the scaled mass to float`). Keep each commit focused. Pull requests should explain behavioral impact, list exact checks run and their mode, link relevant issues, and include logs or screenshots for simulation or visual changes. Call out any result-compatibility or asset changes explicitly.
