import streamlit as st
import os
import pandas as pd
import glob
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import io

st.set_page_config(layout="wide", page_title="Experiment Dashboard")

# Define the logs directory
LOGS_DIR = "logs"

SUPPORTED_TASKS = [
    "put_green_block_in_bowl", #0
    "put_banana_into_box", #1
    "rotate_marker", #2
    "rotate_mug", #3
    "pick_spoon", #4
    "pick_water_bottle", #5
    "stack_cubes", #6
    "push_switch", #7
    "open_drawer", #8
    "close_drawer", #9
]

SUPPORTED_PERTURBATIONS = [
    'Default', #0
    'V-AUG', # 1
    'V-VIEW', # 2
    'V-SC', # 3
    'V-LIGHT', # 4
    'S-PROP', # 5
    'S-LANG', # 6
    'S-MO', # 7
    'S-AFF', # 8
    'S-INT', # 9
    'B-HOBJ', # 10
    'SB-NOUN', # 11
    'SB-VRB', # 12
    'VB-POSE', # 13
    'VB-MOBJ', # 14
    'VSB-NOBJ' # 15
]

# Initialize session state for selection
if "selected_experiment" not in st.session_state:
    st.session_state.selected_experiment = None

def get_subdirectories(path):
    if not os.path.exists(path):
        return []
    try:
        return sorted([d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))])
    except OSError:
        return []

def is_experiment_folder(path):
    """Check if the folder contains 'reports' or 'videos' subdirectories."""
    return os.path.isdir(os.path.join(path, "reports")) or os.path.isdir(os.path.join(path, "videos"))

def load_reports(experiment_path):
    reports_path = os.path.join(experiment_path, "reports")
    if not os.path.exists(reports_path):
        return None

    csv_files = glob.glob(os.path.join(reports_path, "*.csv"))
    if not csv_files:
        return None

    dfs = []
    for f in csv_files:
        try:
            df = pd.read_csv(f)
            dfs.append(df)
        except Exception as e:
            st.error(f"Error reading {f}: {e}")

    if not dfs:
        return None

    try:
        aggregated_df = pd.concat(dfs, axis=0, ignore_index=True)
        return aggregated_df
    except Exception as e:
        st.error(f"Error aggregating CSVs: {e}")
        return None

def get_videos(experiment_path):
    videos_path = os.path.join(experiment_path, "videos")
    if not os.path.exists(videos_path):
        return []
    return sorted(glob.glob(os.path.join(videos_path, "*.mp4")))

def load_experiment_metadata(experiment_path):
    """Loads metadata.json from the experiment directory."""
    metadata_path = os.path.join(experiment_path, "metadata.json")
    if not os.path.exists(metadata_path):
        return None, f"Metadata file not found at {metadata_path}"

    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        return metadata, None
    except Exception as e:
        return None, f"Error reading metadata: {e}"

def check_experiment_status(df, tasks_indices, perts_indices, required_repeats):
    if df is None:
        return False, "No data loaded."

    target_tasks = []
    for i in tasks_indices:
        if 0 <= i < len(SUPPORTED_TASKS):
            target_tasks.append(SUPPORTED_TASKS[i])

    target_perts = []
    for i in perts_indices:
        if 0 <= i < len(SUPPORTED_PERTURBATIONS):
            # Format to match CSV string representation of list
            target_perts.append(f"['{SUPPORTED_PERTURBATIONS[i]}']")

    if not target_tasks or not target_perts:
        return False, "Invalid task or perturbation indices in experiment name."

    # Check existence
    all_good = True
    missing = []

    for t in target_tasks:
        for p in target_perts:
            # Filter df
            count = len(df[(df['task'] == t) & (df['perturbation'] == p)])
            if count < required_repeats:
                all_good = False
                missing.append(f"{t} | {p}: Found {count}/{required_repeats}")

    if all_good:
        return True, "All required tasks and perturbations evaluated with sufficient samples."
    else:
        return False, "Missing evaluations:\n" + "\n".join(missing)


def render_tree(path, depth=0):
    """Recursive function to render the directory tree using expanders."""
    if depth > 5:
        return

    subdirs = get_subdirectories(path)
    if not subdirs:
        return

    for d in subdirs:
        full_path = os.path.join(path, d)
        is_exp = is_experiment_folder(full_path)

        # Unique key for widgets
        key_base = full_path

        sub_subdirs = get_subdirectories(full_path)
        has_children = len(sub_subdirs) > 0

        if has_children:
            label = f"📁 {d}" if not is_exp else f"🔬 {d}"

            with st.sidebar.expander(label, expanded=False):
                # If this node itself is selectable
                if is_exp:
                    if st.button(f"👉 Select {d}", key=f"btn_{key_base}_inner"):
                        st.session_state.selected_experiment = full_path

                # Recurse
                render_tree(full_path, depth + 1)
        else:
            # Leaf node
            if is_exp:
                # Leaf experiment button
                if st.sidebar.button(f"🔬 {d}", key=f"btn_{key_base}_leaf"):
                    st.session_state.selected_experiment = full_path
            else:
                 # It's a folder but has no subdirectories and isn't an experiment?
                 st.sidebar.markdown(f"📁 {d}")

# Sidebar
st.sidebar.title("Experiment Browser")
render_tree(LOGS_DIR)

# Main Content
if st.session_state.selected_experiment and os.path.exists(st.session_state.selected_experiment):
    selected_path = st.session_state.selected_experiment

    # Header Parsing
    rel_path = os.path.relpath(selected_path, LOGS_DIR)
    path_parts = rel_path.split(os.sep)

    experiment_name = path_parts[0] if len(path_parts) > 0 else "N/A"
    model_name = path_parts[1] if len(path_parts) > 1 else "N/A"
    run_id = path_parts[2] if len(path_parts) > 2 else "N/A"

    st.title("Experiment Dashboard")

    c1, c2, c3 = st.columns(3)
    c1.metric("Experiment", experiment_name)
    c2.metric("Model", model_name)
    c3.metric("Run ID", run_id)

    st.divider()

    # Load Reports
    df = load_reports(selected_path)

    # Plots Section (Now at Top)
    st.header("Plots")
    try:
        c1, c2 = st.columns(2)

        with c1:
            st.subheader("Performance under Perturbations")
            # Placeholder: Radar plot
            categories = ['Perturbation A', 'Perturbation B', 'Perturbation C', 'Perturbation D', 'Perturbation E']
            values = [4, 3, 2, 5, 4]

            N = len(categories)
            angles = [n / float(N) * 2 * np.pi for n in range(N)]
            values += values[:1]
            angles += angles[:1]

            fig, ax = plt.subplots(figsize=(4, 4), subplot_kw=dict(polar=True))
            ax.plot(angles, values, linewidth=1, linestyle='solid')
            ax.fill(angles, values, 'b', alpha=0.1)
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories)
            # st.pyplot(fig)
            buf = io.BytesIO()
            fig.savefig(buf, format="png")
            st.image(buf)

        with c2:
            st.subheader("Binary Success Rate")
            # Placeholder: Binary SR
            tasks = ['Task 1', 'Task 2', 'Task 3']
            sr = [0.85, 0.60, 0.95]

            fig, ax = plt.subplots(figsize=(4, 4))
            ax.bar(tasks, sr, color=['green', 'orange', 'blue'])
            ax.set_ylim(0, 1)
            ax.set_ylabel("Success Rate")
            # st.pyplot(fig)
            buf = io.BytesIO()
            fig.savefig(buf, format="png")
            st.image(buf)
    except Exception as e:
        st.error(f"Error in Plots: {e}")

    # Experiment Status Section (New)
    st.header("Experiment Status")
    try:
        # Resolve experiment directory from selected_path
        # selected_path is logs/Experiment/Model/Run or similar.
        # We need logs/Experiment.
        # Use relative path parts to find Experiment root.
        rel_path = os.path.relpath(selected_path, LOGS_DIR)
        parts = rel_path.split(os.sep)

        if len(parts) > 0:
            experiment_name = parts[0]
            experiment_path = os.path.join(LOGS_DIR, experiment_name)

            metadata, err = load_experiment_metadata(experiment_path)

            if metadata:
                tasks_indices = metadata.get("task_ids", [])
                perts_indices = metadata.get("perturbation_ids", [])
                required_repeats = metadata.get("repeats", 0)

                st.write(f"**Target Configuration (from {experiment_name}/metadata.json):** Tasks: {tasks_indices}, Perturbations: {perts_indices}, Repeats: {required_repeats}")

                status, msg = check_experiment_status(df, tasks_indices, perts_indices, required_repeats)

                if status:
                    st.success("✅ " + msg)
                else:
                    st.error("❌ " + msg)
            else:
                st.warning(err)
        else:
            st.warning("Could not determine experiment directory.")

    except Exception as e:
        st.error(f"Error in Experiment Status: {e}")

    # Aggregated Reports Section
    st.header("Aggregated Reports")
    try:
        if df is not None:
            st.dataframe(df, height=300)
        else:
            st.info("No reports found.")
    except Exception as e:
        st.error(f"Error in Aggregated Reports: {e}")

    # Videos Section
    st.header("Videos")
    videos = get_videos(selected_path)
    if videos:
        # Tiled viewer with 3 columns
        cols = st.columns(3)
        for i, video_path in enumerate(videos):
            with cols[i % 3]:
                st.video(video_path)
                st.caption(os.path.basename(video_path))
    else:
        st.info("No videos found.")
else:
    if not os.path.exists(LOGS_DIR):
         st.error(f"Logs directory '{LOGS_DIR}' not found.")
    else:
        st.info("Please select an experiment from the sidebar.")
