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


def find_and_remove_category(categories_dict, obj_category):
    for theme, sub_categories in categories_dict.items():
        for category, obj_list in sub_categories.items():
            if obj_category in obj_list:
                return theme
    return None


def process_droid_categories(original_dict, obj_category):
    processed_dict = original_dict.copy()

    theme_to_pop = find_and_remove_category(processed_dict, obj_category)

    if theme_to_pop:
        processed_dict.pop(theme_to_pop)

    flattened_list = []
    for sub_categories in processed_dict.values():
        for obj_list in sub_categories.values():
            flattened_list.extend(obj_list)

    return flattened_list
