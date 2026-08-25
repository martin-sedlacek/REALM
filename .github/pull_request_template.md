## Summary

Describe the problem and the change.

## Validation

- [ ] Tier 1: `uv run ruff check realm examples tests scripts` **and**
      `uv run python tests/run_suite.py --only local --strict --out tmp/suite/results.json --junit-xml tmp/suite/results.xml`
      **and** the four host pytest modules
- [ ] Relevant Docker/SIF or GPU checks (list command, image, `MODE`, and allocation below)
- [ ] Documentation and configuration updated where needed

Commands and results:

## Benchmark impact

- [ ] This preserves benchmark behavior and RNG draw order.
- [ ] This intentionally changes results; the compatibility impact and `VERSION` change are
      explained below.
- [ ] Not applicable (documentation or tooling only).

## Assets and output

List changed assets/configuration and attach representative logs, screenshots, or videos when the
observable simulation output changes.
