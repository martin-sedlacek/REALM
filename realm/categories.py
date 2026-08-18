"""Object-category catalogue loaded from realm/config/objects/categories.yaml."""
import copy
import os

import yaml

_CATEGORIES_DATA = None


def _load_categories_from_yaml():
    yaml_path = os.path.join(os.path.dirname(__file__), "config/objects/categories.yaml")
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)


def _get_categories_data():
    global _CATEGORIES_DATA
    if _CATEGORIES_DATA is None:
        _CATEGORIES_DATA = _load_categories_from_yaml()
    return _CATEGORIES_DATA


def get_non_droid_categories():
    return list(_get_categories_data()["non_droid_categories"])


def get_droid_categories_by_theme():
    return copy.deepcopy(_get_categories_data()["droid_categories_by_theme"])


def droid_categories_excluding_theme(obj_category):
    """Every DROID category name, minus ALL categories sharing @obj_category's theme.

    Used to draw a replacement object that is not a lookalike of the one it replaces: excluding
    the whole theme (e.g. all drinkware, not just "mug") is what makes the swap visually
    meaningful. An @obj_category that appears in no theme excludes nothing.
    """
    categories_by_theme = get_droid_categories_by_theme()
    theme = next((theme for theme, subcategories in categories_by_theme.items()
                  if any(obj_category in obj_list for obj_list in subcategories.values())),
                 None)
    if theme is not None:
        categories_by_theme.pop(theme)
    return [obj for subcategories in categories_by_theme.values()
            for obj_list in subcategories.values() for obj in obj_list]
