import streamlit as st
import os
import pandas as pd
import glob

st.set_page_config(layout="wide", page_title="Experiment Dashboard")

# Define the logs directory
LOGS_DIR = "logs"

def get_subdirectories(path):
    if not os.path.exists(path):
        return []
    # Only return directories
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

    # Aggregate: concatenate them
    try:
        # Try to concat. If columns are different, it will introduce NaNs, which is fine for aggregation.
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

# Sidebar Navigation
st.sidebar.title("Navigation")

current_path = LOGS_DIR
selected_experiment = None

# Max depth logic:
# The user said "stop at depth 4 - e.g. ./logs/something/something/something"
# logs is depth 0.
# loop for depth 1, 2, 3, 4.

# We will traverse down. If a selection is made, we go deeper.
# If at any point the current path is an experiment folder, we flag it.
# However, user might have nested experiments? Let's assume leaf-ish nodes are experiments.

cols_depth = 4
found_experiment = False

# Store the path components for display
path_components = []

for depth in range(cols_depth):
    subdirs = get_subdirectories(current_path)

    if not subdirs:
        break

    # Add a selectbox for this level
    # Use key to make it unique per level
    selection = st.sidebar.selectbox(f"Level {depth + 1}", [""] + subdirs, key=f"level_{depth}")

    if selection and selection != "":
        current_path = os.path.join(current_path, selection)
        path_components.append(selection)

        # Check if this is an experiment
        if is_experiment_folder(current_path):
            selected_experiment = current_path
            found_experiment = True
    else:
        # Stop traversing if user hasn't selected anything at this level
        break

# Main Content
if selected_experiment:
    # Display the relative path from LOGS_DIR
    rel_path = os.path.relpath(selected_experiment, LOGS_DIR)
    st.title(f"Experiment: {rel_path}")

    # Reports Section
    st.header("Aggregated Reports")
    df = load_reports(selected_experiment)
    if df is not None:
        st.table(df)
    else:
        st.info("No reports found.")

    # Videos Section
    st.header("Videos")
    videos = get_videos(selected_experiment)
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
        st.write("Please select an experiment directory using the sidebar.")
        st.write("Navigate through the folders until you reach an experiment (folder containing 'reports' or 'videos').")
