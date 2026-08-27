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

RUNS ON THE HOST, and that is now true rather than aspirational. The two dicts are read out of
sb_vrb.py with `ast`, because importing it drags in omnigibson -- see module_level_dict below.
"""

import ast
from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TASK_CONFIG_GLOB = "realm/config/tasks/**/*.yaml"
PROGRESSIONS = PROJECT_ROOT / "realm/config/tasks/task_progressions.yaml"
SHARED = PROJECT_ROOT / "realm/config/shared.py"


def module_level_dict(path, name):
    """Read a module-level literal dict out of `path` WITHOUT importing the module.

    sb_vrb.py does `import omnigibson as og` at module scope, so any import of it -- including one
    deferred into a fixture -- needs the container. Deferring only moves the ModuleNotFoundError
    from collection time to fixture-setup time; it does not make the test host-runnable, and this
    file's whole reason to exist is being runnable without a GPU or a container. Measured
    2026-08-19: every test here ERROR'd on the host with `No module named 'omnigibson'`.

    Parsing is not a second copy of the answer -- it reads the SAME file the perturbation is loaded
    from, so it cannot drift from what runs. ast.literal_eval, not eval: it accepts only literals,
    so a non-literal definition raises here rather than executing anything.
    """
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == name:
                return ast.literal_eval(node.value)
    raise AssertionError(f"no module-level `{name} = ...` literal in {path}")


def declared_task_types():

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
    return module_level_dict(SHARED, "COMPATIBILITY_MATRIX")


@pytest.fixture(scope="module")
def verb_phrase():
    return module_level_dict(SHARED, "VERB_PHRASE")


def test_every_declared_task_type_is_a_matrix_key(matrix):

    declared = declared_task_types()
    missing = {t: files for t, files in declared.items() if t not in matrix}
    assert not missing, (
        "task_type(s) declared by task configs but absent from COMPATIBILITY_MATRIX -- SB-VRB would "
        "no-op on them, or raise, rather than perturbing: "
        + "; ".join(f"{t!r} (declared in {', '.join(files)})" for t, files in sorted(missing.items()))
    )


def test_every_matrix_value_is_a_real_task_type(matrix):

    valid = progression_task_types()
    bad = {key: [v for v in vals if v not in valid] for key, vals in matrix.items()}
    bad = {k: v for k, v in bad.items() if v}
    assert not bad, (
        "COMPATIBILITY_MATRIX value(s) are not keys of task_progressions.yaml, so drawing one would "
        f"KeyError in sb_vrb's TASK_PROGRESSIONS[env.task_type] lookup. Valid: {sorted(valid)}. "
        f"Offending: {bad}"
    )


def test_every_matrix_key_is_a_real_task_type(matrix):

    valid = progression_task_types()
    unknown = sorted(k for k in matrix if k not in valid)
    assert not unknown, (
        "COMPATIBILITY_MATRIX key(s) are not keys of task_progressions.yaml, so no task can ever "
        f"match them -- this is what 'open'/'close' looked like. Offending: {unknown}"
    )


def test_no_key_lists_itself(matrix):

    self_listed = sorted(k for k, vals in matrix.items() if k in vals)
    assert not self_listed, (
        "COMPATIBILITY_MATRIX key(s) list themselves, so SB-VRB can draw the task_type it already "
        f"had and 'task_type changed' stops being a sound assertion: {self_listed}"
    )


def test_opt_outs_are_deliberate_and_few(matrix):

    empty = sorted(k for k, vals in matrix.items() if not vals)
    assert empty == ["push"], (
        "The only deliberate SB-VRB opt-out is 'push' (commented out in the matrix rather than "
        f"deleted). Found: {empty}. If another task_type genuinely does not apply, say why in a "
        "comment there and update this test -- an empty list must never be arrived at by accident."
    )


def test_every_reachable_task_type_has_a_verb_phrase(matrix, verb_phrase):

    reachable = {v for vals in matrix.values() for v in vals}
    missing = sorted(reachable - set(verb_phrase))
    assert not missing, (
        "task_type(s) reachable from COMPATIBILITY_MATRIX have no VERB_PHRASE entry, so sb_vrb would "
        f"raise NotImplementedError when one is drawn: {missing}"
    )


def test_verb_phrases_are_not_task_types_verbatim(verb_phrase):

    leaked = sorted(k for k, phrase in verb_phrase.items() if "_" in phrase)
    assert not leaked, (
        "VERB_PHRASE value(s) still contain an underscore, so the instruction would read like "
        f"'open_drawer the top drawer' -- and sb_vrb's trailing .replace('_', ' ') would launder it "
        f"into the plausible 'open drawer the top drawer' instead of failing: {leaked}"
    )


# --------------------------------------------------------------------------------------------------
# The two 2026-08-19 fixes: SB-VRB's deliberate refusal of the drawer tasks, and the
# instruction_obj_to_replace field that has to be a substring of its own instruction.
# --------------------------------------------------------------------------------------------------


def module_level_set(path, name):

    tree = ast.parse(path.read_text(), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return set(ast.literal_eval(node.value))
    raise AssertionError(f"no module-level `{name} = ...` literal in {path}")


@pytest.fixture(scope="module")
def unsupported():
    return module_level_set(SHARED, "UNSUPPORTED_TASK_TYPES")


def test_unsupported_task_types_are_exactly_the_drawer_tasks(unsupported):

    assert unsupported == {"open_drawer", "close_drawer"}, (
        "SB-VRB's UNSUPPORTED_TASK_TYPES changed. It refuses the two drawer tasks because their "
        "configs declare target_objects: [], which sends it down the receiver-adding branch and "
        "drops an unplaceable object from the air. If a task_type is added here, say why at the set "
        f"and update this test deliberately. Found: {sorted(unsupported)}"
    )


def test_unsupported_task_types_stay_matrix_keys(matrix, unsupported):

    missing = sorted(t for t in unsupported if t not in matrix)
    assert not missing, (
        "task_type(s) in UNSUPPORTED_TASK_TYPES are absent from COMPATIBILITY_MATRIX. sb_vrb would "
        "then raise KeyError ('table and configs disagree') instead of the NotImplementedError that "
        f"says the refusal is deliberate: {missing}"
    )


def test_instruction_obj_to_replace_occurs_in_its_instruction():
    """sb_noun.py:21 and vsb_nobj.py:36 do instruction.replace(field, ...).

    A field that is not a substring of the instruction replaces nothing, so the perturbation reports
    PASS while not perturbing. close_drawer/default.yaml carried the VERB ("close drawer") instead of
    the object ("top drawer") and was the only one of the ten REALM_DROID10 tasks to do so; 9:VSB-NOBJ
    is a recorded matrix PASS earned that way.
    """
    offenders = []
    for path in sorted(PROJECT_ROOT.glob("realm/config/tasks/**/*.yaml")):
        try:
            cfg = yaml.safe_load(path.read_text())
        except Exception:
            continue
        if not isinstance(cfg, dict):
            continue
        instruction, field = cfg.get("instruction"), cfg.get("instruction_obj_to_replace")
        if not (isinstance(instruction, str) and isinstance(field, str) and field):
            continue
        if field not in instruction:
            offenders.append(f"{path.relative_to(PROJECT_ROOT).as_posix()}: {field!r} not in {instruction!r}")
    assert not offenders, (
        "instruction_obj_to_replace is not a substring of its own instruction, so SB-NOUN and "
        "VSB-NOBJ would silently fail to substitute and still report PASS:\n  "
        + "\n  ".join(offenders)
    )
