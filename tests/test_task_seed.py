"""Host-safe checks for persistent task RNG identities."""
from pathlib import Path
from uuid import UUID

import yaml

from realm.task_seed import task_seed


ROOT = Path(__file__).resolve().parents[1] / 'realm/config'


def test_bundled_task_identities_and_seeds_are_unique():
    identifiers = []
    seeds = []
    for path in sorted((ROOT / 'tasks').rglob('*.yaml')):
        if path.name == 'task_progressions.yaml':
            continue
        config = yaml.safe_load(path.read_text())
        identifiers.append(UUID(config['task_uuid']))
        relative = str(path.relative_to(ROOT / 'tasks'))
        seed = task_seed(relative, ROOT)
        assert seed == task_seed(relative, ROOT)
        assert 0 <= seed < 2**32
        seeds.append(seed)
    assert len(identifiers) == len(set(identifiers))
    assert len(seeds) == len(set(seeds))


def test_explicit_uuid_survives_config_edits_and_base_seed_offsets(tmp_path):
    tasks = tmp_path / 'tasks'
    tasks.mkdir()
    path = tasks / 'custom.yaml'
    identity = 'task_uuid: 4b97a399-9339-4b44-884b-58dddf866612\n'
    path.write_text(identity + 'instruction: first\n')
    seed = task_seed('custom.yaml', tmp_path)
    path.write_text(identity + 'instruction: changed\n')
    assert task_seed('custom.yaml', tmp_path) == seed
    assert task_seed('custom.yaml', tmp_path, 1235) == (seed + 1) % 2**32


def test_legacy_identity_is_portable(tmp_path):
    tasks = tmp_path / 'tasks'
    tasks.mkdir()
    (tasks / 'one.yaml').write_text('task_type: rotate\ninstruction: rotate mug\n')
    (tasks / 'two.yaml').write_text('instruction: rotate mug\ntask_type: rotate\n')
    assert task_seed('one.yaml', tmp_path) == task_seed('two.yaml', tmp_path)
