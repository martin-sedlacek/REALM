

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Optional

import streamlit as st
import streamlit.components.v1 as components
import yaml

from authoring import (
    default_dataset_roots,
    demo_assets,
    discover_assets,
    discover_existing_task_names,
    discover_task_types,
    load_drawer_cabinet_models,
    load_droid_categories,
    load_camera_extrinsics,
    load_panda_preview_meshes,
    load_scene_regions,
    sample_opposite_camera_pair,
)
from save_server import start_save_server


REPO_ROOT = Path(__file__).resolve().parents[2]


@st.cache_data(show_spinner="Loading full DROID camera-pose pool…")
def get_camera_poses(path_text: str):
    return load_camera_extrinsics(Path(path_text))


@st.cache_resource
def get_save_server(repo_root_text: str, camera_path_text: str, _camera_poses):
    del camera_path_text
    return start_save_server(Path(repo_root_text), _camera_poses)

st.set_page_config(
    page_title="REALM Task Authoring",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="collapsed",
)
st.title("REALM Task Authoring")
st.caption("Arrange asset bounding boxes on a scene footprint and export a REALM task YAML draft.")

roots = default_dataset_roots(REPO_ROOT)
camera_path = (
    REPO_ROOT
    / "realm"
    / "config"
    / "env"
    / "external_sensors"
    / "camera_extrinsics_droid_realm.yaml"
)
all_camera_extrinsics = get_camera_poses(str(camera_path))
save_server, save_url, camera_sample_url = get_save_server(
    str(REPO_ROOT), str(camera_path), all_camera_extrinsics
)
default_root = next((root for root in roots if root.is_dir()), roots[0] if roots else REPO_ROOT / "datasets")

source_column, limit_column = st.columns([4, 1])
with source_column:
    dataset_text = st.text_input("OmniGibson dataset root", value=str(default_root))
with limit_column:
    scan_limit_text = st.text_input("Maximum USD files", value="N/A")

normalized_limit = scan_limit_text.strip().upper()
if normalized_limit in {"", "N/A", "NA", "NONE", "ALL"}:
    scan_limit = None
else:
    try:
        scan_limit = int(normalized_limit)
        if scan_limit <= 0:
            raise ValueError
    except ValueError:
        st.error("Maximum USD files must be a positive integer or N/A.")
        st.stop()


@st.cache_data(show_spinner="Indexing USD assets…")
def load_assets(root_text: str, limit: Optional[int]):
    return discover_assets(Path(root_text).expanduser().resolve(), limit)


assets = load_assets(dataset_text, scan_limit)
using_demo = not assets
if using_demo:
    assets = demo_assets()
    st.warning("No USD assets found at that path. Showing demo objects while the dataset downloads.")
else:
    st.success(f"Indexed {len(assets):,} USD assets from `{dataset_text}`.")

task_config_root = REPO_ROOT / "realm" / "config" / "tasks"
task_paths = sorted(task_config_root.rglob("*.yaml"))
task_options = {str(path.relative_to(task_config_root)): path for path in task_paths}
load_column, upload_column = st.columns(2)
with load_column:
    selected_task = st.selectbox(
        "Load existing REALM task YAML",
        ["New task"] + list(task_options),
    )
with upload_column:
    uploaded_task = st.file_uploader("Or upload task YAML", type=("yaml", "yml"))

initial_task = None
initial_task_source = None
try:
    if uploaded_task is not None:
        initial_task = yaml.safe_load(uploaded_task.getvalue())
        initial_task_source = uploaded_task.name
    elif selected_task != "New task":
        initial_task = yaml.safe_load(task_options[selected_task].read_text(encoding="utf-8"))
        initial_task_source = selected_task
    if initial_task is not None and not isinstance(initial_task, dict):
        raise ValueError("the YAML root must be a mapping")
except (OSError, ValueError, yaml.YAMLError) as error:
    st.error(f"Could not load task YAML: {error}")
    initial_task = None
if initial_task_source and initial_task is not None:
    st.success(f"Loaded editable draft from `{initial_task_source}`.")

asset_json = json.dumps(assets).replace("</", "<\\/")
scene_regions = load_scene_regions(REPO_ROOT / "realm" / "config" / "scenes" / "scenes.yaml")
scene_json = json.dumps(scene_regions).replace("</", "<\\/")
task_types = discover_task_types(REPO_ROOT / "realm" / "config" / "tasks")
task_type_options = "".join(f'<option value="{value}">{value}</option>' for value in task_types)
existing_task_names = discover_existing_task_names(REPO_ROOT / "realm" / "config" / "tasks")
drawer_models = load_drawer_cabinet_models(
    REPO_ROOT / "realm" / "environments" / "perturbations" / "object_sampling.py"
)
droid_categories = load_droid_categories(REPO_ROOT / "realm" / "config" / "objects" / "categories.yaml")
camera_extrinsics = sample_opposite_camera_pair(all_camera_extrinsics, random.SystemRandom())
if initial_task:
    for value in (initial_task.get("camera_extrinsics") or {}).values():
        if isinstance(value, str) and value in all_camera_extrinsics:
            camera_extrinsics[value] = all_camera_extrinsics[value]
robot_mesh_root = (
    REPO_ROOT
    / "data"
    / "datasets_og391"
    / "omnigibson-robot-assets"
    / "source"
    / "franka"
    / "meshes"
    / "collision"
)
robot_mesh_json = json.dumps(load_panda_preview_meshes(robot_mesh_root), separators=(",", ":"))
template_path = Path(__file__).with_name("workspace.html")
workspace = template_path.read_text(encoding="utf-8")
workspace = workspace.replace("__ASSET_CATALOG__", asset_json).replace("__SCENE_REGIONS__", scene_json)
workspace = workspace.replace("__TASK_TYPE_OPTIONS__", task_type_options)
workspace = workspace.replace("__EXISTING_TASK_NAMES__", json.dumps(existing_task_names))
workspace = workspace.replace("__DRAWER_MODELS__", json.dumps(drawer_models))
workspace = workspace.replace("__DROID_CATEGORIES__", json.dumps(droid_categories))
workspace = workspace.replace("__CAMERA_EXTRINSICS__", json.dumps(camera_extrinsics))
workspace = workspace.replace("__INITIAL_TASK__", json.dumps(initial_task).replace("</", "<\\/"))
workspace = workspace.replace("__SAVE_URL__", json.dumps(save_url))
workspace = workspace.replace("__CAMERA_SAMPLE_URL__", json.dumps(camera_sample_url))
workspace = workspace.replace("__ROBOT_MESHES__", robot_mesh_json)
workspace = workspace.replace("__THREE_JS__", Path(__file__).with_name("three.min.js").read_text(encoding="utf-8"))
workspace = workspace.replace(
    "__ORBIT_CONTROLS_JS__",
    Path(__file__).with_name("OrbitControls.js").read_text(encoding="utf-8"),
)
components.html(workspace, height=940, scrolling=True)

with st.expander("How coordinates map to REALM"):
    st.markdown(
        "The square is the top-down XY spawn rectangle from `realm/config/scenes/scenes.yaml`. "
        "Those regions are typically about 0.4 × 0.6 m (40 × 60 cm). Dragging changes the first two values of "
        "`relative_bbox_position`; Z, bounding-box dimensions, orientation, role, and fixed-base "
        "status are editable in the inspector. Generated YAML follows the object lists used by "
        "`realm/config/tasks/REALM_DROID10/*/default.yaml`."
    )
