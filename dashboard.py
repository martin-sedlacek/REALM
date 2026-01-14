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
    """Recursive function to render the directory tree."""
    # Safety break for recursion depth
    if depth > 5:
        return

    subdirs = get_subdirectories(path)
    if not subdirs:
        return

    for d in subdirs:
        full_path = os.path.join(path, d)
        is_exp = is_experiment_folder(full_path)
        has_subdirs = len(get_subdirectories(full_path)) > 0

        # Unique key for widgets
        key_base = full_path

        # Layout: Indentation + Widget
        # We use columns to simulate the tree layout: [Indent, Expand/Label, SelectButton]

        # Calculate indentation
        # Streamlit doesn't support fine-grained indentation easily.
        # We'll use a container or simple markdown for visual hierarchy?
        # Actually, standard approach is just to render widgets.
        # But we need visual hierarchy.

        # Using checkboxes for expansion (acting as folders)
        # If it has subdirs, it can be expanded.
        # If it is an experiment, it can be selected.

        # Label generation
        icon = "📂" if has_subdirs else "📄"
        if is_exp:
            icon = "🔬"

        label = f"{icon} {d}"

        # To create a "tree" feel, sub-items are only shown if parent is expanded.
        # We use st.checkbox to handle expansion state.

        # Note: Nested checkboxes work fine.

        # Indent using markdown? No, checkbox has to be the toggle.
        # We can't easily indent the checkbox itself.
        # Workaround: Use unicode spaces in label? '    ' * depth
        indent_spaces = '\u2003' * depth # Em spaces
        display_label = f"{indent_spaces}{label}"

        # If it has subdirectories, we use a checkbox to expand/collapse.
        # If it is ONLY an experiment (leaf), we might just show it?
        # But an experiment might also have subdirectories (if structure is weird).

        # Logic:
        # 1. Render a row.
        # 2. If allow selection, show a button.

        # To put checkbox and button on same line is hard in sidebar (columns are cramped).
        # We'll try:
        # Checkbox for expansion (if has subdirs or just always?).
        # If checked -> show children.

        expanded = False

        # If we are strictly implementing "click + to expand", checkbox is the closest native "toggle".
        # But if we also want "click to select", we need a separate action.

        # Let's try this:
        # Use columns.

        # Problem: 'st.columns' inside sidebar inside loop works, but width is small.

        if has_subdirs:
             # Checkbox for expansion
             expanded = st.sidebar.checkbox(display_label, key=f"chk_{key_base}")
        else:
             st.sidebar.markdown(display_label)

        # If it is an experiment, we need a way to select it.
        # If we used a checkbox for expansion, how do we select?
        # Add a small button underneath? Or next to it?

        if is_exp:
            # Add a select button.
            # To make it look associated, maybe indent it or put it right below.
            # Using a button with a unique key.
            # "Select [d]"
            btn_label = f"Select {d}"
            # Indent the button slightly more
            btn_col1, btn_col2 = st.sidebar.columns([0.1 + (0.05 * depth), 0.9 - (0.05 * depth)])
            with btn_col2:
                if st.button(f"👉 Load {d}", key=f"btn_{key_base}"):
                    st.session_state.selected_experiment = full_path
                    # Force rerun? Streamlit reruns on button click anyway.

        # Recursion
        if expanded:
            render_tree(full_path, depth + 1)

# Sidebar
st.sidebar.title("Experiment Browser")
st.sidebar.write("Expand folders and click 'Load' to view.")
render_tree(LOGS_DIR)

# Main Content
if st.session_state.selected_experiment and os.path.exists(st.session_state.selected_experiment):
    selected_path = st.session_state.selected_experiment

    # Header Parsing
    rel_path = os.path.relpath(selected_path, LOGS_DIR)
    path_parts = rel_path.split(os.sep)

    # Safe unpacking
    experiment_name = path_parts[0] if len(path_parts) > 0 else "N/A"
    model_name = path_parts[1] if len(path_parts) > 1 else "N/A"
    run_id = path_parts[2] if len(path_parts) > 2 else "N/A"

    # Display Metadata
    st.title("Experiment Dashboard")

    # Using columns for the "Three different rows at the top" - wait, user said "three different rows".
    # "Experiment, model/ and run id in three different rows at the top"
    # Rows usually means vertical stack.
    # "Metric" style is good for this.

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
