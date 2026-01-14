import streamlit as st
import os
import pandas as pd
import glob

st.set_page_config(layout="wide", page_title="Experiment Dashboard")

# Define the logs directory
LOGS_DIR = "logs"

def get_experiments():
    """Recursively finds all subdirectories in the LOGS_DIR."""
    if not os.path.exists(LOGS_DIR):
        return []
    
    experiment_dirs = []
    for dirpath, _, _ in os.walk(LOGS_DIR):
        rel_path = os.path.relpath(dirpath, LOGS_DIR)
        if rel_path != ".":
            experiment_dirs.append(rel_path)
    return sorted(experiment_dirs)

def load_reports(experiment_path):
    """Loads all CSV reports from the selected experiment directory."""
    if not os.path.exists(experiment_path):
        return None

    csv_files = glob.glob(os.path.join(experiment_path, "*.csv"))
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
    """Gets all videos from the selected experiment directory."""
    if not os.path.exists(experiment_path):
        return []
    return sorted(glob.glob(os.path.join(experiment_path, "*.mp4")))

# Sidebar
st.sidebar.title("Experiments")
experiments = get_experiments()

if not experiments:
    st.sidebar.warning("No experiments found in 'logs' folder.")
    selected_experiment = None
else:
    selected_experiment = st.sidebar.radio("Select Experiment", experiments)

if selected_experiment:
    st.title(f"Experiment: {selected_experiment}")
    exp_path = os.path.join(LOGS_DIR, selected_experiment)

    # Reports Section
    st.header("Aggregated Reports")
    df = load_reports(exp_path)
    if df is not None:
        st.table(df)
    else:
        st.info("No reports found.")

    # Videos Section
    st.header("Videos")
    videos = get_videos(exp_path)
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
    st.write("Please select an experiment from the sidebar.")
