"""Pin `has_base_column` against the USD each robot config actually loads. Host-side, no GPU.

WHY THIS EXISTS. `has_base_column` describes THE ASSET, and env_config.py:111 turns it into geometry:

    if env.use_droid_with_base and not cfg_robot["robots"][0].pop("has_base_column", True):
        spawn_pos[2] += DROID_BASE_HEIGHT          # 0.863891

So the flag and the asset must agree, and the two failure modes are both silent-ish and both severe:

  * flag FALSE + asset that HAS a base  -> the spawn is raised on top of a base the asset already
    carries, and the arm floats 0.86 m in the air.
  * flag TRUE + BARE arm                -> the arm spawns at the bottom of the column, i.e. buried in
    the table; the contact forces NaN the sim within a few steps.

Neither is a crash at load, and a NaN a few steps in reads like a physics problem rather than a
config one. On 2026-08-19 `droid_robolab_v2` was switched from the bare arm to
`droid_robolab_v2_mounted.usd`, which is exactly the edit that trips the first case if the flag is
forgotten -- and the flag lives in TWO files (DROID_robolab_v2.yaml and
DROID_robolab_v2_ee_control.yaml) that both name the same `model`, so it can also be half-changed.

The release invariant is now stronger: every DROID config uses the single mounted RoboLab v2 model.
"""

from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROBOT_CONFIGS = sorted((PROJECT_ROOT / "realm/config/robots").glob("*.yaml"))
DEFINITIONS = PROJECT_ROOT / "realm/robots/definitions"


def robot_entries():
    """(config path, robot dict) for every robot entry that declares a model."""
    out = []
    for path in ROBOT_CONFIGS:
        try:
            cfg = yaml.safe_load(path.read_text())
        except Exception:
            continue
        if not isinstance(cfg, dict):
            continue
        for robot in cfg.get("robots") or []:
            if isinstance(robot, dict) and robot.get("model"):
                out.append((path.relative_to(PROJECT_ROOT).as_posix(), robot))
    return out


def usd_for_model(model):
    """The usd_path the RobotDefinition for `model` loads, or None if there is no definition."""
    d = DEFINITIONS / model / f"{model}.yaml"
    if not d.is_file():
        return None
    return (yaml.safe_load(d.read_text()) or {}).get("usd_path")


def test_there_are_robot_entries_to_check():
    """Guard against the whole suite passing because a glob silently matched nothing."""
    entries = robot_entries()
    assert entries, f"no robot entries found under {ROBOT_CONFIGS!r} -- the test is inert"


@pytest.mark.parametrize("cfg_path,robot", robot_entries(),
                         ids=lambda v: v if isinstance(v, str) else v.get("name", "?"))
def test_has_base_column_matches_its_usd(cfg_path, robot):
    usd = usd_for_model(robot["model"])
    if usd is None:
        pytest.skip(f"{robot['model']} has no RobotDefinition yaml (legacy `type`-based config)")

    stem = Path(usd).stem
    mounted_asset = stem.endswith("_mounted")
    # Default TRUE, matching env_config.py's `.pop("has_base_column", True)` -- an absent flag means
    # "no offset added", which is only right for an asset that has its own base.
    flag = robot.get("has_base_column", True)

    assert flag == mounted_asset, (
        f"{cfg_path}: robot {robot.get('name')!r} sets has_base_column={flag} but model "
        f"{robot['model']!r} loads {stem}.usd, which by the _mounted naming convention "
        f"{'DOES' if mounted_asset else 'does NOT'} carry its own base.\n"
        f"  flag False + mounted asset -> env_config.py:111 adds DROID_BASE_HEIGHT on top of a base "
        f"the asset already has: the arm floats ~0.86 m.\n"
        f"  flag True + bare arm       -> the arm spawns at the bottom of the column, buried in the "
        f"table, and NaNs a few steps in.\n"
        f"If the asset genuinely disagrees with its filename, rename the asset -- do not weaken this "
        f"test, because the filename is the only base-height signal available without booting Isaac."
    )


def test_configs_sharing_a_model_agree_on_has_base_column():
    """The half-changed case. Two configs naming one model share its USD, so a flag that differs
    between them means one entry point spawns the arm at the wrong height while the other is fine --
    and whichever one you happen to run decides whether you notice."""
    by_model = {}
    for cfg_path, robot in robot_entries():
        by_model.setdefault(robot["model"], {})[cfg_path] = robot.get("has_base_column", True)

    disagreeing = {m: v for m, v in by_model.items() if len(set(v.values())) > 1}
    assert not disagreeing, (
        "config(s) naming the same `model` disagree on has_base_column, so they load the same USD "
        f"with different spawn offsets: {disagreeing}"
    )


def test_all_droid_profiles_use_robolab_v2():
    """Prevent a generic profile from silently reintroducing a retired stock or v1 asset."""
    droid_configs = [path for path in ROBOT_CONFIGS if path.name.startswith("DROID")]
    assert droid_configs
    for path in droid_configs:
        robot = yaml.safe_load(path.read_text())["robots"][0]
        expected_model = (
            "droid_robolab_v2_bare"
            if path.name == "DROID_robolab_v2_bare.yaml"
            else "droid_robolab_v2"
        )
        assert robot.get("model") == expected_model, path
        assert robot.get("name") == "DROID_robolab_v2", path
        assert robot.get("dof") == 13, path
        assert robot.get("has_base_column") is (expected_model == "droid_robolab_v2"), path

    retired = (
        "droid.usd",
        "droid_mounted.usd",
        "droid_robolab.usd",
    )
    asset_root = PROJECT_ROOT / "realm/robots/panda_robotiq"
    assert not [name for name in retired if (asset_root / name).exists()]
