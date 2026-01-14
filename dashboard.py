import streamlit as st
import os
import pandas as pd
import glob

st.set_page_config(layout="wide", page_title="Experiment Dashboard")

# Define the logs directory
LOGS_DIR = "logs"

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

    # User requested: "label the thing at the top it should be one level deeper that oyu currently display"
    # Indices: 1, 2, 3.

    experiment_name = path_parts[1] if len(path_parts) > 1 else "N/A"
    model_name = path_parts[2] if len(path_parts) > 2 else "N/A"
    run_id = path_parts[3] if len(path_parts) > 3 else "N/A"

    st.title("Experiment Dashboard")

    c1, c2, c3 = st.columns(3)
    c1.metric("Experiment", experiment_name)
    c2.metric("Model", model_name)
    c3.metric("Run ID", run_id)

    st.divider()

    # Reports Section
    st.header("Aggregated Reports")
    df = load_reports(selected_path)
    if df is not None:
        st.table(df)
    else:
        st.info("No reports found.")

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
