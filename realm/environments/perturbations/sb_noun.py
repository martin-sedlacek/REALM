from __future__ import annotations

import random
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from realm.environments.env_dynamic import RealmEnvironmentDynamic


def sb_noun(env: "RealmEnvironmentDynamic") -> None:
    """Point the task at a different object that is already in the scene.

    Drawer tasks re-target a different drawer of the same cabinet. Every other task promotes a
    random distractor to main object -- the old main object becomes a distractor -- and swaps the
    noun in the instruction. Precondition: at least one distractor (``env.distractors`` non-empty),
    or the randint below raises; env_dynamic refuses the push+SB-NOUN combination up front.
    """
    if env.task_type in ["open_drawer", "close_drawer"]:
        adjective = random.choice(["middle", "top"])
        env.instruction = env.cfg["instruction"].replace("top", adjective)
        env.reset_joints(target_drawer_loc=adjective)
        return

    i = np.random.randint(len(env.distractors))
    new_mo = env.distractors.pop(i)
    new_obj_for_task = new_mo.category
    env.instruction = env.cfg["instruction"].replace(env.cfg["instruction_obj_to_replace"], new_obj_for_task)
    env.instruction = env.instruction.replace("_", " ")

    env.distractors.append(env.main_objects[0])
    env.main_objects[0] = new_mo
