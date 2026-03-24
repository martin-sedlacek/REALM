from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING

from realm.environments.perturbations._helpers import apply_cached_semantic_perturbations

if TYPE_CHECKING:
    from realm.environments.env_dynamic import RealmEnvironmentDynamic


def s_prop(env: "RealmEnvironmentDynamic") -> None:
    apply_cached_semantic_perturbations(env, "S-PROP")


def s_mo(env: "RealmEnvironmentDynamic") -> None:
    apply_cached_semantic_perturbations(env, "S-MO")


def s_aff(env: "RealmEnvironmentDynamic") -> None:
    apply_cached_semantic_perturbations(env, "S-AFF")


def s_int(env: "RealmEnvironmentDynamic") -> None:
    apply_cached_semantic_perturbations(env, "S-INT")


def s_lang(env: "RealmEnvironmentDynamic") -> None:
    synonyms: dict[str, list[str]] = env.cfg.get("synonyms", None)
    if synonyms is None:
        apply_cached_semantic_perturbations(env, "S-LANG")
        return
    n_synonyms_comb = np.prod([(len(v) + 1) for v in synonyms.values()]) - 1
    s_langs = env.cfg["cached_semantic_perturbations"].get("S-LANG", None)
    if s_langs is not None:
        n_s_langs = len(s_langs)
        if np.random.random() < n_s_langs / (n_synonyms_comb + n_s_langs):
            apply_cached_semantic_perturbations(env, "S-LANG")
            return

    orig_instruction: str = env.cfg["instruction"]
    instruction = orig_instruction.lower()
    instruction_words = instruction.split()

    synonyms: dict[str, list[str]] = env.cfg["synonyms"]
    number_words_which_can_be_replaced = len(synonyms)
    # Picking with 50% which words to replace with synonyms
    word_idx_to_replace = np.random.randint(2, size=number_words_which_can_be_replaced)
    # Making sure that at least one word will be replaced
    guaranteed_replaced_word_idx = np.random.randint(number_words_which_can_be_replaced)
    word_idx_to_replace[guaranteed_replaced_word_idx] = 1

    for word_idx, (word, syns) in enumerate(synonyms.items()):
        if not word_idx_to_replace[word_idx]:
            continue
        for i, w in enumerate(instruction_words):
            if w == word:
                s = np.random.choice(syns)
                instruction_words[i] = s

    env.instruction = " ".join(instruction_words).capitalize()
