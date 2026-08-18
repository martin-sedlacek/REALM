"""Pin SB-VRB's COMPATIBILITY_MATRIX against the task configs, on the host, with no GPU.

WHY THIS EXISTS. The matrix used to carry "open": ["close"] / "close": ["open"] while the task YAMLs
declare task_type "open_drawer" / "close_drawer". Neither half of that lined up:

  - the KEYS matched nothing, so `COMPATIBILITY_MATRIX.get(task_type, [])` fell through to `[]` and
    SB-VRB silently perturbed NOTHING on tasks 8 and 9. The cells stopped crashing, which is exactly
    what a fix looks like from the outside -- the 45-cell matrix would have reported them PASS.
  - the VALUES were not valid task_types either, so even a matching key would have raised KeyError
    on TASK_PROGRESSIONS[env.task_type] a few lines later.

The 2026-08-16 matrix verdict recommended a static test for precisely this and it never landed, so
the regression had nothing watching it. Every assertion below reads the SAME sources the
perturbation reads at runtime -- the task YAMLs and task_progressions.yaml -- so the test cannot
drift from them by holding its own copy of the answer.

A GPU cell run cannot replace this: a silent no-op passes a crash test.
"""

from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TASK_CONFIG_GLOB = "realm/config/tasks/**/*.yaml"
PROGRESSIONS = PROJECT_ROOT / "realm/config/tasks/task_progressions.yaml"


def declared_task_types():
    """Every `task_type` the task configs declare, mapped to the files that declare it."""
    declared = {}
    for path in sorted(PROJECT_ROOT.glob(TASK_CONFIG_GLOB)):
        try:
            cfg = yaml.safe_load(path.read_text())
        except Exception:
            continue
        if not isinstance(cfg, dict):
            continue
        task_type = cfg.get("task_type")
        if isinstance(task_type, str) and task_type:
            declared.setdefault(task_type, []).append(path.relative_to(PROJECT_ROOT).as_posix())
    return declared


def progression_task_types():
    return set(yaml.safe_load(PROGRESSIONS.read_text()).keys())


@pytest.fixture(scope="module")
def matrix():
    # Imported inside a fixture, not at module scope: sb_vrb pulls in omnigibson, so a collection-time
    # import would make this whole file unrunnable outside the container -- which would defeat the
    # point of a host-side test.
    from realm.environments.perturbations.sb_vrb import COMPATIBILITY_MATRIX

    return COMPATIBILITY_MATRIX


@pytest.fixture(scope="module")
def verb_phrase():
    from realm.environments.perturbations.sb_vrb import VERB_PHRASE

    return VERB_PHRASE


def test_every_declared_task_type_is_a_matrix_key(matrix):
    """The bug that hid for a day: a declared task_type absent from the matrix is a silent no-op."""
    declared = declared_task_types()
    missing = {t: files for t, files in declared.items() if t not in matrix}
    assert not missing, (
        "task_type(s) declared by task configs but absent from COMPATIBILITY_MATRIX -- SB-VRB would "
        "no-op on them, or raise, rather than perturbing: "
        + "; ".join(f"{t!r} (declared in {', '.join(files)})" for t, files in sorted(missing.items()))
    )


def test_every_matrix_value_is_a_real_task_type(matrix):
    """A value outside the task_type namespace KeyErrors on the TASK_PROGRESSIONS lookup."""
    valid = progression_task_types()
    bad = {key: [v for v in vals if v not in valid] for key, vals in matrix.items()}
    bad = {k: v for k, v in bad.items() if v}
    assert not bad, (
        "COMPATIBILITY_MATRIX value(s) are not keys of task_progressions.yaml, so drawing one would "
        f"KeyError in sb_vrb's TASK_PROGRESSIONS[env.task_type] lookup. Valid: {sorted(valid)}. "
        f"Offending: {bad}"
    )


def test_every_matrix_key_is_a_real_task_type(matrix):
    """Catches the other direction: a key nothing declares is dead weight that looks like coverage."""
    valid = progression_task_types()
    unknown = sorted(k for k in matrix if k not in valid)
    assert not unknown, (
        "COMPATIBILITY_MATRIX key(s) are not keys of task_progressions.yaml, so no task can ever "
        f"match them -- this is what 'open'/'close' looked like. Offending: {unknown}"
    )


def test_no_key_lists_itself(matrix):
    """sb_vrb's docstring promises the drawn task_type always differs, which tests rely on."""
    self_listed = sorted(k for k, vals in matrix.items() if k in vals)
    assert not self_listed, (
        "COMPATIBILITY_MATRIX key(s) list themselves, so SB-VRB can draw the task_type it already "
        f"had and 'task_type changed' stops being a sound assertion: {self_listed}"
    )


def test_opt_outs_are_deliberate_and_few(matrix):
    """An empty list is a documented opt-out. Keep it explicit, so a typo cannot become one."""
    empty = sorted(k for k, vals in matrix.items() if not vals)
    assert empty == ["push"], (
        "The only deliberate SB-VRB opt-out is 'push' (commented out in the matrix rather than "
        f"deleted). Found: {empty}. If another task_type genuinely does not apply, say why in a "
        "comment there and update this test -- an empty list must never be arrived at by accident."
    )


def test_every_reachable_task_type_has_a_verb_phrase(matrix, verb_phrase):
    """Reachable from the matrix but missing a phrasing = the NotImplementedError at the end."""
    reachable = {v for vals in matrix.values() for v in vals}
    missing = sorted(reachable - set(verb_phrase))
    assert not missing, (
        "task_type(s) reachable from COMPATIBILITY_MATRIX have no VERB_PHRASE entry, so sb_vrb would "
        f"raise NotImplementedError when one is drawn: {missing}"
    )


def test_verb_phrases_are_not_task_types_verbatim(verb_phrase):
    """The drawer entries are the whole point: "open_drawer the top drawer" must not be reachable."""
    leaked = sorted(k for k, phrase in verb_phrase.items() if "_" in phrase)
    assert not leaked, (
        "VERB_PHRASE value(s) still contain an underscore, so the instruction would read like "
        f"'open_drawer the top drawer' -- and sb_vrb's trailing .replace('_', ' ') would launder it "
        f"into the plausible 'open drawer the top drawer' instead of failing: {leaked}"
    )
