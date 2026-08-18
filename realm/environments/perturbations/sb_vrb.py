"""SB-VRB: switch the task to a compatible task_type, spawning or swapping the receiver it needs.

What one call mutates on @env: ``task_type`` and ``task_progression`` (rebased to the new type's
rubric), ``instruction``, and -- for task types that need a receiver -- ``target_objects``,
``cfg["objects"]``, ``init_poses`` and ``cfg["instruction_target_to_replace"]``.

NOT idempotent across resets by design: the receiver is spawned only on a reset that finds NO
target object (``len(env.target_objects) == 0``), i.e. the first perturbed reset of a task without
one. On later resets ``target_objects`` still holds the receiver, so only the put/stack target swap
runs -- reset 1 and reset N do different amounts of work.
"""
from __future__ import annotations

import copy
import random
from typing import TYPE_CHECKING

import torch

import omnigibson as og
from omnigibson.objects import DatasetObject

from realm.environments.perturbations._helpers import (
    backfill_object_cfgs,
    rebase_after_play,
    set_scene_positions,
    settle,
    sim_play,
    sim_step,
    sim_stop,
)
from realm.environments.perturbations.object_sampling import (
    replace_obj,
    rescale_to_max_dim,
    sample_objects,
)
from realm.environments.utils import load_task_progressions
from realm.placement import place_within

if TYPE_CHECKING:
    from realm.environments.env_dynamic import RealmEnvironmentDynamic

TASK_PROGRESSIONS = load_task_progressions()

# Which task_types may replace which. Module level rather than a local, so a test can assert the
# drawn task_type against the SAME table the perturbation drew from instead of keeping its own copy
# in sync. Note that no key lists itself: the new task_type always differs from the current one,
# which is what makes "task_type changed" a sound assertion rather than a probabilistic one.
#
# KEYS AND VALUES ARE BOTH IN THE `task_type` NAMESPACE -- the one the task YAMLs declare and
# task_progressions.yaml is keyed by: put / pick / rotate / stack / push / open_drawer /
# close_drawer. NOT the natural-language verb, which is a separate field in the task YAMLs
# (`instruction_verb_to_replace`) and is mapped in VERB_PHRASE below.
#
# That distinction was got wrong once and BOTH halves of the error mattered. The drawer entries read
# "open": ["close"] / "close": ["open"], where:
#   - the KEYS never matched anything. The configs declare task_type "open_drawer"/"close_drawer",
#     so COMPATIBILITY_MATRIX.get(task_type, []) fell through to [] and SB-VRB silently perturbed
#     NOTHING on tasks 8 and 9. It stopped crashing, which read as fixed.
#   - the VALUES were not valid task_types either: "close" is not a key of task_progressions.yaml,
#     so even a matching key would have KeyError'd on the TASK_PROGRESSIONS lookup below.
# tests/test_perturbation_task_types.py now pins both halves against the configs, with no GPU.
COMPATIBILITY_MATRIX = {
    "put": ["pick", "rotate", "stack"],
    "push": [], #["put", "pick", "rotate", "stack"]
    "pick": ["put", "rotate", "stack"],
    "rotate": ["put", "pick", "stack"],
    "stack": ["put", "pick", "rotate"],
    "open_drawer": ["close_drawer"],
    "close_drawer": ["open_drawer"],
}

# task_type -> the natural-language verb phrase the instruction is rebuilt with. Separate from the
# table above because the namespaces genuinely differ: task_type "open_drawer" reads "open the top
# drawer". The task YAMLs already carry this split as `instruction_verb_to_replace`.
VERB_PHRASE = {
    "pick": "pick up",
    "put": "put",
    "rotate": "rotate",
    "stack": "stack",
    "push": "push",
    "open_drawer": "open",
    "close_drawer": "close",
}

#: Largest bbox dimension (metres) a spawned or swapped receiver may have.
RECEIVER_MAX_DIM = 0.185


def sb_vrb(env: "RealmEnvironmentDynamic") -> None:
    # An ABSENT key and a key with an EMPTY list mean different things, and collapsing them with
    # .get(task_type, []) is exactly what hid the drawer-namespace bug:
    #   - empty list -> DELIBERATE opt-out. "push" is commented out above rather than deleted. No-op,
    #                   rather than letting random.choice([]) raise the IndexError that killed
    #                   task 7 + SB-VRB.
    #   - absent key -> the table and the task configs DISAGREE. That is a bug in one of them and it
    #                   must be LOUD: silently no-op'ing is how a perturbation gets reported as
    #                   passing while measuring nothing at all.
    if env.task_type not in COMPATIBILITY_MATRIX:
        raise KeyError(
            f"SB-VRB: task_type {env.task_type!r} is not in COMPATIBILITY_MATRIX "
            f"(known: {sorted(COMPATIBILITY_MATRIX)}). Add it -- with an empty list if the "
            f"perturbation genuinely does not apply -- rather than leaving it absent."
        )
    available_task_types = COMPATIBILITY_MATRIX[env.task_type]
    if not available_task_types:
        og.log.info(
            f"SB-VRB: no-op, task_type {env.task_type!r} is a deliberate opt-out (empty candidate list)"
        )
        return

    new_task_type = _draw_new_task_type(env, available_task_types)

    included_categories = None
    if env.task_type == "put":
        included_categories = ["bowl", "wineglass"]

    if len(env.target_objects) == 0:
        _spawn_receiver(env, included_categories)

    # sim_step: og.sim.step() asserts a playing sim. Single-env that holds here; in a vector env the
    # shared stop is already in force for the whole perturbation, so this no-ops and the shared
    # settle after the shared play does the equivalent work.
    sim_step(env)

    if env.task_type in ["put", "stack"]:
        _swap_target(env, included_categories)

    # SB-VRB adds a "receiver" and/or replaces the target, so like V-SC it MUST rebase the scene's
    # initial file in a vector env; see rebase_after_play for the failure mode. vec-only because
    # single-env SB-VRB has never rebased and works.
    # STILL UNVERIFIED END TO END vectorized: only V-SC has actually been run that way
    # (t9_vbpose_nostopplay.py --perturbation V-SC, 2 members x 3 resets). SB-VRB also ADDS a
    # "receiver" that the other members' scenes do not have, which is a case V-SC never exercises,
    # so run it before trusting it.
    rebase_after_play(env, vec_only_rebase=True)

    _rebuild_instruction(env, new_task_type)


def _draw_new_task_type(env: "RealmEnvironmentDynamic", available_task_types) -> str:
    """Draw the replacement task_type; rebase env.task_type and the progression rubric onto it."""
    new_task_type = random.choice(available_task_types)
    env.task_type = new_task_type
    # deepcopy is load-bearing, for the same reason as in task_progression.py: TASK_PROGRESSIONS is
    # built ONCE at module import and recompute_task_progression MUTATES its dict in place.
    # Assigning it directly gave every member that drew the same new task_type ONE SHARED
    # progression dict, so one member grasping marked GRASP done for all of them -- precisely the
    # bug that invalidated the SR 0.960 run.
    env.task_progression = copy.deepcopy(TASK_PROGRESSIONS[env.task_type])
    return new_task_type


def _spawn_receiver(env: "RealmEnvironmentDynamic", included_categories) -> None:
    """Sample and place a receiver object for a task_type that needs a target the task lacks.

    Adds the object to the scene and to env.cfg["objects"], re-places everything collision-free
    (pre-existing objects keep their poses via main_object_names), and records the receiver's
    world pose in env.init_poses.
    """
    sampled = sample_objects(num_objects=1, included_categories=included_categories)
    if not sampled:
        raise RuntimeError(
            f"SB-VRB could not sample a receiver: no installed object model matches "
            f"included_categories={included_categories}")
    nobj_cfg = sampled[0]
    env.cfg['instruction_target_to_replace'] = nobj_cfg["category"]
    nobj_cfg["name"] = "receiver"

    new_obj = DatasetObject(
        name="receiver",
        relative_prim_path="/receiver",
        category=nobj_cfg["category"],
        model=nobj_cfg["model"],
    )
    env.omnigibson_env.scene.add_object(new_obj)
    env.target_objects = [new_obj]

    # Records the bbox EXTENT and shrinks oversized receivers -- see rescale_to_max_dim for why the
    # extent (a size) and never the center. A receiver may still legitimately fail to place on a
    # crowded task (e.g. task 4's 0.4 x 0.5 m spawn box already holds a plate and a tray) and be
    # dropped from the air; that is a task-config problem, not a placement bug.
    rescale_to_max_dim(new_obj, nobj_cfg, RECEIVER_MAX_DIM)

    env.cfg["objects"].append(nobj_cfg)

    # --------------- Translation ---------------
    obj_cfgs = copy.deepcopy(env.cfg["objects"])
    n_non_receiver = len(obj_cfgs) - 1  # every pre-existing object; only the receiver moves

    backfill_object_cfgs(env.main_objects + env.distractors + env.target_objects, obj_cfgs)

    env.cfg["objects"] = place_within(
        env.spawn_bbox,
        obj_cfgs,
        objects_to_skip=[obj.name for obj in env.main_objects + env.distractors],
        main_object_names=[o["name"] for o in obj_cfgs[:n_non_receiver]],
    )

    _record_receiver_pose(env, new_obj)

    # --------------- Set Position ---------------
    set_scene_positions(env, env.cfg["objects"])


def _record_receiver_pose(env: "RealmEnvironmentDynamic", new_obj) -> None:
    """Move the freshly placed receiver to its drawn pose and record it in env.init_poses.

    The drawn position is SCENE-relative (it comes from the scene-relative spawn_bbox), but
    set_bbox_center_position_orientation is world-frame by construction -- it takes no frame
    argument at all -- so convert explicitly. In scene 0 the conversion is the identity, which is
    why the world-frame version was correct single-env and wrong for every other member of a
    vector env. env.init_poses holds WORLD poses everywhere else, so the world pose is what gets
    recorded -- storing the scene-relative one here would restore to the wrong place.
    """
    receiver_cfg = env.cfg["objects"][-1]
    pos = torch.tensor(receiver_cfg["position"])
    rot = torch.tensor(receiver_cfg["orientation"] if "orientation" in receiver_cfg else [0, 0, 0, 1])
    pos_world, rot_world = env.omnigibson_env.scene.convert_scene_relative_pose_to_world(pos, rot)
    new_obj.set_bbox_center_position_orientation(pos_world, rot_world)
    env.init_poses[new_obj._relative_prim_path] = {"pos": pos_world, "rot": rot_world}


def _swap_target(env: "RealmEnvironmentDynamic", included_categories) -> None:
    """Replace the existing target with a freshly sampled one (put/stack task types only).

    Genuinely needs the stopped sim -- replace_obj removes and adds an object. The sim wrappers
    and the settle all no-op in a vector env, where RealmVectorEnvironment.reset() has already
    stopped once for all members and settles once after.
    """
    sim_stop(env)
    nobj, nobj_cfg = replace_obj(env, env.target_objects[0],
                                 included_categories=included_categories,
                                 maximum_dim=RECEIVER_MAX_DIM)
    env.target_objects = [nobj]
    env.cfg['instruction_target_to_replace'] = nobj_cfg["category"]
    sim_play(env)
    settle(env)


def _rebuild_instruction(env: "RealmEnvironmentDynamic", new_task_type: str) -> None:
    """Rewrite env.instruction for the drawn task_type.

    VERB_PHRASE, not the task_type itself: "open_drawer" as a verb would read "open_drawer the top
    drawer", and the trailing .replace("_", " ") would launder that into the plausible-looking
    "open drawer the top drawer" rather than failing. Single-object phrasings share a branch; the
    two that name a second object do not.
    """
    if new_task_type in ("rotate", "push", "pick", "open_drawer", "close_drawer"):
        env.instruction = f"{VERB_PHRASE[new_task_type]} the {env.cfg['instruction_obj_to_replace']}"
    elif new_task_type == "stack":
        env.instruction = f"stack the {env.cfg['instruction_obj_to_replace']} on top of the {env.cfg['instruction_target_to_replace']}"
    elif new_task_type == "put":
        env.instruction = f"put the {env.cfg['instruction_obj_to_replace']} into the {env.cfg['instruction_target_to_replace']}"
    else:
        raise NotImplementedError(
            f"SB-VRB: no instruction phrasing for task_type {new_task_type!r}. It is reachable "
            f"from COMPATIBILITY_MATRIX, so add a branch here rather than widening the matrix alone."
        )
    env.instruction = env.instruction.replace("_", " ")
    og.log.info(f"New instruction: {env.instruction}")
