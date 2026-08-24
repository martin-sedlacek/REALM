# Contributing to REALM

Thank you for improving REALM. Bug reports, documentation fixes, new robot or policy integrations,
and carefully validated benchmark changes are welcome.

## Environments

REALM has two deliberately separate environments:

- **Docker or Apptainer/SIF is the runtime.** It contains OmniGibson, Isaac Sim, REALM's runtime
  dependencies, and the simulator patches. The current recipes are
  `.docker/realm.Dockerfile` and `.docker/realm.def`; follow `wiki/Installation.md`.
- **uv is only for host checks.** Run `uv sync --locked`, then `uv run make check`. This environment
  cannot run simulations and must not accumulate runtime dependencies.

See `wiki/Installation.md` and `wiki/Running-the-Test-Suite.md` for full setup and GPU commands.

## Making Changes

REALM is a benchmark, so preserve observable behavior and random-number draw order during refactors.
Do not silently fix a result-changing defect. Such changes need an explicit compatibility decision,
a `VERSION` bump, and an entry in `CHANGE_LEDGER.md`.

Keep single- and vector-evaluation semantics shared in `realm/rollout.py`. Add tasks and
perturbations through the existing YAML layers and registries; avoid hard-coded parallel lists.

## Testing

Run `uv run make check` before submitting. Do not run `pytest tests/`: most files are standalone
Isaac scripts and collection can boot the simulator. GPU-capable changes should also run the
smallest relevant suite through `tests/run_suite.py`; typically:

```bash
ALLOC=<jobid> make test-smoke
```

Record verification that must be completed on a GPU cluster in the pull request.

## Pull Requests

Use a focused, imperative commit subject, such as `rollout: validate camera selection`. Explain the
behavioral impact, exact commands and modes tested, affected assets/configuration, and any result
compatibility implications. Link related issues and include logs or images when output changes.
