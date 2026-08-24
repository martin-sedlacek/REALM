"""The two robolab_v2 RobotDefinitions must differ ONLY in usd_path. Host-side, no GPU.

WHY THIS EXISTS. `droid_robolab_v2` (mounted asset) and `droid_robolab_v2_bare` (bare arm) are two
copies of the same ~110-line definition, because OmniGibson selects a robot by `model` and the model's
definition YAML is the only place usd_path can be named -- nothing in REALM reads usd_path, so a config
cannot override it and one definition cannot include another. Duplication was therefore forced.

The hazard duplication creates is silent divergence: 36 disabled_collision_pairs, a 13-entry
default_joint_pos and the whole manipulation block are all safety-relevant, and a fix applied to one
file and not the other would show up as a physics difference between two robots that are supposed to be
the same arm. Nothing else in the repo would notice.

So this pins the invariant that makes the duplication acceptable: byte-identical apart from the one line
that is meant to differ.
"""

from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFINITIONS = PROJECT_ROOT / "realm/robots/definitions"

# (model name, expected USD basename). The pair is named explicitly rather than discovered, so that
# ADDING a third variant is a deliberate edit here and not a silently untested file.
PAIR = [
    ("droid_robolab_v2", "droid_robolab_v2_mounted.usd"),
    ("droid_robolab_v2_bare", "droid_robolab_v2.usd"),
]


def definition_path(model):
    return DEFINITIONS / model / f"{model}.yaml"


@pytest.mark.parametrize("model,usd", PAIR)
def test_definition_exists_and_names_its_usd(model, usd):
    p = definition_path(model)
    assert p.is_file(), (
        f"{p} is missing. install_robot_definitions.py requires <dir>/<dir>.yaml with matching stems, "
        f"so a mismatch here means OmniGibson cannot discover the model at all")
    got = (yaml.safe_load(p.read_text()) or {}).get("usd_path")
    assert got and Path(got).name == usd, (
        f"{model} should load {usd}, but its usd_path is {got!r}")
    asset = PROJECT_ROOT / "realm/robots/panda_robotiq" / usd
    assert asset.is_file(), f"{model} points at {usd}, which does not exist at {asset}"


def test_parsed_definitions_agree_key_by_key():
    """Line equality would also be satisfied by two identically-broken files; this checks the parsed
    content, so a YAML-level difference (duplicate key, changed type) is caught too."""
    a, b = (yaml.safe_load(definition_path(m).read_text()) for m, _ in PAIR)
    a.pop("usd_path", None)
    b.pop("usd_path", None)
    assert a == b, (
        "parsed definitions differ outside usd_path; keys not equal: "
        f"{sorted(set(a) ^ set(b)) or [k for k in a if a.get(k) != b.get(k)]}")
