import streamlit as st
import os
import dashboard_utils
import ast
import pandas as pd

from plots import (
    plot_bayesian_violin,
    plot_grouped_bars_with_symbols,
    render_metric_bar_chart,
    plot_stage_frequency,
    plot_stage_frequency_per_task,
    plot_task_progression_timesteps_per_task,
    render_model_legend,
    plot_dirichlet_progression_violin,
)
from report import generate_comprehensive_report
import analytics

st.set_page_config(layout="wide", page_title="Experiment Dashboard")

MARKERS = ['o', 's', '^', 'v', '<', '>', 'D', 'P', 'X', '*']
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

# --- Custom CSS for tree-style sidebar buttons ---
st.markdown("""
<style>
section[data-testid="stSidebar"] .stButton > button {
    text-align: left !important;
    background-color: transparent !important;
    border: none !important;
    border-radius: 4px !important;
    padding: 0.1rem 0.4rem !important;
    font-size: 0.875rem !important;
    width: 100% !important;
    color: inherit !important;
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
    box-shadow: none !important;
    min-height: 0 !important;
    line-height: 1.6 !important;
}
section[data-testid="stSidebar"] .stButton > button:hover {
    background-color: rgba(151, 166, 195, 0.15) !important;
}
section[data-testid="stSidebar"] .stButton > button:focus {
    box-shadow: none !important;
    outline: none !important;
}
section[data-testid="stSidebar"] .stButton {
    margin-bottom: -0.6rem !important;
}
</style>
""", unsafe_allow_html=True)

# Define the logs directory
LOGS_DIR = os.environ.get("REALM_LOGS", "logs")
VIDEOS_PER_PAGE = 6

# Initialize session state
if "selected_runs" not in st.session_state:
    st.session_state.selected_runs = []
if "expanded_folders" not in st.session_state:
    st.session_state.expanded_folders = set()
if "video_page" not in st.session_state:
    st.session_state.video_page = 0
if "last_selected_runs" not in st.session_state:
    st.session_state.last_selected_runs = []
if "search_query" not in st.session_state:
    st.session_state.search_query = ""

# --- Caching Wrapper Functions ---
@st.cache_data(ttl=60)
def get_cached_subdirectories(path):
    return dashboard_utils.get_subdirectories(path)

@st.cache_data(ttl=60)
def get_cached_is_experiment_folder(path):
    return dashboard_utils.is_experiment_folder(path)

@st.cache_data(ttl=10)
def get_cached_reports(path):
    return dashboard_utils.load_reports(path)

@st.cache_data(ttl=10)
def get_cached_videos(path):
    return dashboard_utils.get_videos(path)

@st.cache_data(ttl=60)
def get_cached_experiment_metadata(path):
    return dashboard_utils.load_experiment_metadata(path)

# --- Tree callbacks ---
def _toggle_folder(path):
    if path in st.session_state.expanded_folders:
        st.session_state.expanded_folders.discard(path)
    else:
        st.session_state.expanded_folders.add(path)

def _update_selection_recursive(path, val):
    """Recursively update the selection state of all experiments under a folder."""
    if get_cached_is_experiment_folder(path):
        st.session_state[f"chk_{path}"] = val

    st.session_state[f"fold_{path}"] = val

    subdirs = get_cached_subdirectories(path)
    for d in subdirs:
        _update_selection_recursive(os.path.join(path, d), val)

def _on_folder_checkbox_change(path):
    """Callback when a folder checkbox is toggled."""
    val = st.session_state.get(f"fold_{path}", False)
    _update_selection_recursive(path, val)

def _on_experiment_checkbox_change(path):
    """Callback when an individual experiment checkbox is toggled."""
    val = st.session_state.get(f"chk_{path}", False)
    _update_selection_recursive(path, val)

def has_experiments_recursively(base_path, search_query="", match_found_in_path=False):
    """Check if the given path or any of its subdirectories contain an experiment matching the search query."""
    if search_query and not match_found_in_path:
        match_found_in_path = search_query.lower() in os.path.basename(base_path).lower()

    if get_cached_is_experiment_folder(base_path):
        if search_query:
            return match_found_in_path
        return True

    subdirs = get_cached_subdirectories(base_path)
    for d in subdirs:
        full_path = os.path.join(base_path, d)
        if has_experiments_recursively(full_path, search_query, match_found_in_path):
            return True
    return False

# --- Tree rendering ---
def render_tree(base_path, depth=0, search_query="", match_found_in_path=False):
    """Render a collapsible directory tree in the sidebar with proper indentation."""
    if depth > 5:
        return

    subdirs = get_cached_subdirectories(base_path)
    if not subdirs:
        return

    for d in subdirs:
        full_path = os.path.join(base_path, d)

        current_match = match_found_in_path or (search_query.lower() in d.lower() if search_query else False)

        if not has_experiments_recursively(full_path, search_query, current_match):
            continue

        is_exp = get_cached_is_experiment_folder(full_path)
        sub_subdirs = get_cached_subdirectories(full_path)

        has_children = len(sub_subdirs) > 0 and not is_exp

        is_expanded = (full_path in st.session_state.expanded_folders) or (search_query != "")

        indent = "\u2003" * (depth * 2)

        if has_children:
            chevron = "▾" if is_expanded else "▸"
            label = f"{indent}{chevron} 📁 {d}"

            cols = st.sidebar.columns([0.15, 0.85])
            with cols[0]:
                st.checkbox("Select All", key=f"fold_{full_path}", on_change=_on_folder_checkbox_change, args=(full_path,), label_visibility="collapsed")
            with cols[1]:
                st.button(
                    label,
                    key=f"tree_{full_path}",
                    on_click=_toggle_folder,
                    args=(full_path,),
                )
        elif is_exp:
            label = f"{indent}🔬 {d}"
            st.sidebar.checkbox(label, key=f"chk_{full_path}", on_change=_on_experiment_checkbox_change, args=(full_path,))
        else:
            st.sidebar.markdown(
                f"<div style='padding-left:{depth * 1.5}rem; font-size:0.85rem; color:gray; "
                f"padding-top:0.1rem; padding-bottom:0.1rem;'>\u2003 📁 {d}</div>",
                unsafe_allow_html=True,
            )

        if has_children and is_expanded:
            render_tree(full_path, depth + 1, search_query, current_match)

# --- Video page callbacks ---
def _video_prev_page():
    st.session_state.video_page = max(0, st.session_state.video_page - 1)

def _video_next_page():
    st.session_state.video_page += 1

# --- Video section (fragment for partial reruns) ---
@st.fragment
def render_video_section(video_paths, selected_tasks, selected_perts, run_paths=None):
    """Render paginated video gallery. Runs as a fragment so page navigation
    does not trigger a full dashboard rerun."""
    filtered_videos = dashboard_utils.filter_videos(video_paths, selected_tasks, selected_perts)

    if not filtered_videos:
        if video_paths:
            st.info("No videos match the selected filters.")
        else:
            # Check if there are parquets in the selected runs
            has_parquets = False
            if run_paths:
                for rp in run_paths:
                    v_dir = os.path.join(rp, "videos")
                    if os.path.exists(v_dir) and any(f.endswith(".parquet") for f in os.listdir(v_dir)):
                        has_parquets = True
                        break
            
            if has_parquets:
                st.info("No MP4 videos found, but parquet videos are available. Use the **Unpack Parquet Videos** button above to view them.")
            else:
                st.info("No videos found.")
        return

    total_videos = len(filtered_videos)
    total_pages = max(1, (total_videos + VIDEOS_PER_PAGE - 1) // VIDEOS_PER_PAGE)

    if st.session_state.video_page >= total_pages:
        st.session_state.video_page = total_pages - 1

    start_idx = st.session_state.video_page * VIDEOS_PER_PAGE
    end_idx = min(start_idx + VIDEOS_PER_PAGE, total_videos)

    nav1, nav2, nav3 = st.columns([1, 3, 1])
    with nav1:
        st.button("← Prev", on_click=_video_prev_page,
                  disabled=(st.session_state.video_page == 0))
    with nav2:
        st.markdown(
            f"<div style='text-align:center; padding-top:0.4rem;'>"
            f"Showing {start_idx + 1}–{end_idx} of {total_videos} videos "
            f"(Page {st.session_state.video_page + 1}/{total_pages})</div>",
            unsafe_allow_html=True,
        )
    with nav3:
        st.button("Next →", on_click=_video_next_page,
                  disabled=(st.session_state.video_page >= total_pages - 1))

    page_videos = filtered_videos[start_idx:end_idx]
    cols = st.columns(3)
    for i, video_path in enumerate(page_videos):
        with cols[i % 3]:
            st.video(video_path, format="video/mp4")
            st.caption(os.path.relpath(video_path, LOGS_DIR))

# ===== SIDEBAR =====
st.sidebar.header("Filter Data & Videos")

with st.sidebar.expander("Filter by Task", expanded=False):
    selected_tasks = []
    all_tasks = st.checkbox("All Tasks", value=False)
    if all_tasks:
        selected_tasks = dashboard_utils.SUPPORTED_TASKS
    else:
        for task in dashboard_utils.SUPPORTED_TASKS:
            if st.checkbox(task, key=f"chk_task_{task}"):
                selected_tasks.append(task)

with st.sidebar.expander("Filter by Perturbation", expanded=False):
    selected_perts = []
    all_perts = st.checkbox("All Perturbations", value=False)
    if all_perts:
        selected_perts = dashboard_utils.SUPPORTED_PERTURBATIONS
    else:
        for pert in dashboard_utils.SUPPORTED_PERTURBATIONS:
            if st.checkbox(pert, key=f"chk_pert_{pert}"):
                selected_perts.append(pert)

st.sidebar.markdown("---")

st.sidebar.title("Experiment Browser")
search_query = st.sidebar.text_input("Search Experiments", key="search_query", placeholder="e.g. baseline")

if os.path.exists(LOGS_DIR):
    st.sidebar.checkbox("Select All Experiments", key=f"fold_{LOGS_DIR}", on_change=_on_folder_checkbox_change, args=(LOGS_DIR,))

render_tree(LOGS_DIR, search_query=search_query)

# Collect selected runs from checkboxes
selected_runs = [key[4:] for key, value in st.session_state.items() if key.startswith("chk_") and value and os.path.isdir(key[4:]) and get_cached_is_experiment_folder(key[4:])]

# ===== MAIN CONTENT =====
if selected_runs:
    # Reset video page when experiment selection changes
    if sorted(selected_runs) != st.session_state.get('last_selected_runs', []):
        st.session_state.video_page = 0
        st.session_state.last_selected_runs = sorted(selected_runs)

    # --- Data Loading and Aggregation ---
    all_reports = []
    all_videos = []
    for run_path in selected_runs:
        report = get_cached_reports(run_path)
        if report is not None:
            rel_path = os.path.relpath(run_path, LOGS_DIR)
            path_parts = rel_path.split(os.sep)
            model_name = path_parts[1] if len(path_parts) > 1 else "Unknown"
            report['model'] = model_name
            all_reports.append(report)
        all_videos.extend(get_cached_videos(run_path))

    if all_reports:
        df_full = pd.concat(all_reports, ignore_index=True)
        df = dashboard_utils.filter_dataframe(df_full, selected_tasks, selected_perts)
    else:
        df_full = pd.DataFrame()
        df = pd.DataFrame()

    # --- Header ---
    if len(selected_runs) == 1:
        rel_path = os.path.relpath(selected_runs[0], LOGS_DIR)
        path_parts = rel_path.split(os.sep)
        experiment_name = path_parts[0] if len(path_parts) > 0 else "N/A"
        model_name = path_parts[1] if len(path_parts) > 1 else "N/A"
    else:
        common_path = os.path.commonpath(selected_runs)
        rel_common_path = os.path.relpath(common_path, LOGS_DIR)
        path_parts = rel_common_path.split(os.sep)
        experiment_name = path_parts[0] if len(path_parts) > 0 and path_parts[0] != '.' else "Multiple Experiments"
        model_name = path_parts[1] if len(path_parts) > 1 else "Multiple Models"

    run_ids = [os.path.basename(run) for run in selected_runs]

    unique_models = sorted(df_full['model'].unique())
    model_to_marker = {model: MARKERS[i % len(MARKERS)] for i, model in enumerate(unique_models)}
    model_to_color = {model: COLORS[i % len(COLORS)] for i, model in enumerate(unique_models)}

    st.title("Experiment Dashboard")
    col_title, col_download, col_csv = st.columns([2, 1, 1])
    with col_title:
        st.markdown(f"<h3><b>Experiment:</b> {experiment_name}</h3><h3><b>Model:</b> {model_name}</h3>", unsafe_allow_html=True)
    with col_download:
        if st.button("📄 Generate PDF Report", width="stretch"):
            with st.spinner("Generating comprehensive PDF report..."):
                pdf_buffer = generate_comprehensive_report(df_full, model_to_marker, model_to_color)
                st.download_button(
                    label="⬇️ Download PDF",
                    data=pdf_buffer,
                    file_name=f"REALM_Report_{experiment_name.replace(' ', '_')}.pdf",
                    mime="application/pdf",
                    width="stretch"
                )
    with col_csv:
        csv_data = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📊 Download CSV",
            data=csv_data,
            file_name=f"REALM_Data_{experiment_name.replace(' ', '_')}.csv",
            mime="text/csv",
            width="stretch"
        )

    if len(run_ids) == 1:
        st.markdown(f"<h3><b>Run ID:</b> {run_ids[0]}</h3>", unsafe_allow_html=True)
    else:
        with st.expander(f"Selected {len(run_ids)} Runs"):
            st.write(", ".join(run_ids))

    st.divider()

    # --- Experiment Status ---
    st.header("Experiment Status")
    try:
        first_run_path = selected_runs[0]
        exp_path_for_meta = os.path.dirname(os.path.dirname(first_run_path))

        metadata, err = get_cached_experiment_metadata(exp_path_for_meta)

        if metadata:
            tasks_indices = metadata.get("task_ids", [])
            perts_indices = metadata.get("perturbation_ids", [])
            required_repeats = metadata.get("repeats", 0) * len(selected_runs)

            st.markdown(f"**Target Configuration (from {os.path.basename(exp_path_for_meta)}/metadata.json):**\n"
                        f"* Tasks: {tasks_indices}\n"
                        f"* Perturbations: {perts_indices}\n"
                        f"* Repeats per run: {metadata.get('repeats', 0)}")

            status, msg = dashboard_utils.check_experiment_status(df, tasks_indices, perts_indices, required_repeats)

            if status:
                st.success("✅ All required tasks and perturbations evaluated with sufficient samples across all selected runs.")
            else:
                st.error("❌ Missing evaluations.")
                with st.expander("Missing Combinations", expanded=False):
                    missing_lines = msg.split("Missing evaluations:\n")
                    if len(missing_lines) > 1:
                        missing_items = missing_lines[1].split("\n")
                        formatted_missing = "\n".join([f"- {item}" for item in missing_items if item.strip()])
                        st.write(formatted_missing)
                    else:
                        st.write(msg)
        else:
            st.warning(f"Could not load metadata to check experiment status. {err}")

    except Exception as e:
        st.error(f"Error in Experiment Status: {e}")

    st.divider()

    # --- Model Legend (Global) ---
    if df is not None and not df.empty:
        render_model_legend(model_to_marker, model_to_color)

    # --- Plots ---
    st.header("Success Metrics")
    try:
        if df is not None and not df.empty:
            c1, c2 = st.columns(2)

            with c1:
                st.subheader("Success Rate per Task")
                if 'binary_SR' in df.columns:
                    task_stats = df.groupby(['task', 'model'])['binary_SR'].agg(['sum', 'count']).reset_index()
                    task_stats = task_stats.sort_values(['task', 'model'])

                    labels = [f"{row['task']} - {row['model']}" for _, row in task_stats.iterrows()]
                    successes = task_stats['sum'].tolist()
                    failures = (task_stats['count'] - task_stats['sum']).tolist()
                    colors = [model_to_color[row['model']] for _, row in task_stats.iterrows()]
                    symbols = [model_to_marker[row['model']] for _, row in task_stats.iterrows()]
                    model_names = task_stats['model'].tolist()

                    all_tasks_stats = df.groupby('model')['binary_SR'].agg(['sum', 'count']).reset_index()
                    all_tasks_stats = all_tasks_stats.sort_values('model')
                    for _, row in all_tasks_stats.iterrows():
                        labels.append(f"All Tasks (Mean) - {row['model']}")
                        successes.append(row['sum'])
                        failures.append(row['count'] - row['sum'])
                        colors.append(model_to_color[row['model']])
                        symbols.append(model_to_marker[row['model']])
                        model_names.append(row['model'])

                    fig = plot_bayesian_violin(labels, successes, failures, colors, symbols=symbols, model_names=model_names, model_colors=model_to_color, fontsize=12)
                    st.plotly_chart(fig, use_container_width=True)

            with c2:
                st.subheader("Success Rate per Perturbation")
                df['clean_pert'] = df['perturbation'].apply(lambda x: x.replace("['", "").replace("']", "") if isinstance(x, str) else str(x))

                if 'binary_SR' in df.columns:
                    pert_stats = df.groupby(['clean_pert', 'model'])['binary_SR'].agg(['sum', 'count']).reset_index()
                    pert_stats = pert_stats.sort_values(['clean_pert', 'model'])

                    labels = [f"{row['clean_pert']} - {row['model']}" for _, row in pert_stats.iterrows()]
                    successes = pert_stats['sum'].tolist()
                    failures = (pert_stats['count'] - pert_stats['sum']).tolist()
                    colors = [model_to_color[row['model']] for _, row in pert_stats.iterrows()]
                    symbols = [model_to_marker[row['model']] for _, row in pert_stats.iterrows()]
                    model_names = pert_stats['model'].tolist()

                    fig = plot_bayesian_violin(labels, successes, failures, colors, symbols=symbols, model_names=model_names, model_colors=model_to_color, fontsize=12)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No binary_SR column found for plots.")

            st.divider()
            c3, c4 = st.columns(2)

            with c3:
                st.subheader("Task Progression per Task")
                if 'task_progression' in df.columns:
                    fig = plot_dirichlet_progression_violin(
                        df,
                        group_col='task',
                        model_to_color=model_to_color,
                        model_to_marker=model_to_marker,
                        show_average=True,
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No task_progression column found for plots.")

            with c4:
                st.subheader("Task Progression per Perturbation")
                if 'task_progression' in df.columns:
                    fig = plot_dirichlet_progression_violin(
                        df,
                        group_col='clean_pert',
                        model_to_color=model_to_color,
                        model_to_marker=model_to_marker,
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No task_progression column found for plots.")

            st.divider()

            c5, c6 = st.columns(2)
            with c5:
                st.subheader("Failure Stage Frequency")
                fig = plot_stage_frequency(df, model_to_marker=model_to_marker, model_to_color=model_to_color)
                st.plotly_chart(fig, use_container_width=True)

            with c6:
                st.subheader("Failure Stage Frequency per Task")
                fig = plot_stage_frequency_per_task(df, model_to_marker=model_to_marker, model_to_color=model_to_color)
                st.plotly_chart(fig, use_container_width=True)

            st.caption("*Note: Data for perturbation 'SB-VRB' is excluded from the 'Failure Stage Frequency per Task' plot.*")

            st.divider()

            # --- Auxiliary Metrics ---
            st.header("Auxiliary metrics")

            with st.expander("Metric Interpretations (How to read these numbers?)"):
                st.markdown(r"""
                These metrics are computed from joint positions ($q$) and time steps ($dt$) using numerical differentiation:

                *   **Joint Velocity Variance** ($rad^2/s^2 \cdot samples$): Measures total velocity variation over the rollout ($ \sum (v - \bar{v})^2 $). Higher values indicate oscillating or non-smooth speed profiles.
                *   **Joint Acceleration Variance** ($rad^2/s^4 \cdot samples$): Measures the consistency of acceleration. High values indicate frequent changes in force, often seen as vibration or jitter.
                *   **Jerk (Joint/Cartesian)** ($rad/s^3$ or $m/s^3$): The time derivative of acceleration ($da/dt$). Represents the "shakiness" of the motion. High jerk is mechanically damaging and indicates poor control stability.
                *   **Path Length** ($rad$ or $m$): The total distance traveled in joint space or Cartesian space ($ \sum ||\Delta q|| $). Lower is more efficient; high values suggest the robot is "wandering" or taking an inefficient route.
                *   **Collisions/Drops**: Raw count of safety or reliability failures during the rollout. Ideally should be zero.
                """)

            c7, c8 = st.columns(2)
            with c7:
                st.subheader("Time to Completion per Task (Successful Only)")
                if 'task_progression' in df.columns:
                    col_name = None
                    for col in ['task_progression_timesteps', 'task_progression_timestamps', 'task_progression_timstamps']:
                        if col in df.columns:
                            col_name = col
                            break
                    if col_name:
                        success_df = df[df['task_progression'] == 1].copy()
                        if not success_df.empty:
                            try:
                                success_df['parsed_ts'] = success_df[col_name].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
                                success_df['completion_time'] = success_df['parsed_ts'].apply(lambda x: x[-1] if isinstance(x, list) and len(x) > 0 else 0)

                                task_time = success_df.groupby(['task', 'model'])['completion_time'].mean().reset_index()
                                task_time = task_time.sort_values(['task', 'model'])

                                labels = [f"{row['task']} - {row['model']}" for _, row in task_time.iterrows()]
                                values = task_time['completion_time'].tolist()
                                colors = [model_to_color[row['model']] for _, row in task_time.iterrows()]
                                symbols = [model_to_marker[row['model']] for _, row in task_time.iterrows()]
                                model_names = task_time['model'].tolist()

                                fig = plot_grouped_bars_with_symbols(labels, values, colors, symbols, model_names, model_colors=model_to_color, ylabel="Average Completion Time")
                                st.plotly_chart(fig, use_container_width=True)
                            except Exception as e:
                                st.error(f"Error parsing timestamps: {e}")
                        else:
                            st.info("No successful runs (task_progression == 1) to calculate time to completion.")
                    else:
                        st.info("Required column for timestamps not found.")
                else:
                    st.info("Required column 'task_progression' not found.")

            with c8:
                st.subheader("Stage Timesteps per Task")
                fig = plot_task_progression_timesteps_per_task(df, model_to_marker=model_to_marker, model_to_color=model_to_color)
                st.plotly_chart(fig, use_container_width=True)

            st.caption("*Note: Data for perturbation 'SB-VRB' is excluded from the 'Stage Timesteps per Task' plot.*")

            st.divider()

            # --- Physical and Kinematic Metrics ---
            c9, c10 = st.columns(2)
            with c9:
                render_metric_bar_chart(df, 'joint_vel_var', "Joint Velocity Variance", "Mean Variance (rad²/s²·samples)", model_to_marker, model_to_color, color='lightblue')
            with c10:
                render_metric_bar_chart(df, 'joint_acc_var', "Joint Acceleration Variance", "Mean Variance (rad²/s⁴·samples)", model_to_marker, model_to_color, color='lightblue')

            st.divider()

            c11, c12 = st.columns(2)
            with c11:
                render_metric_bar_chart(df, 'joint_jerk', "Joint Jerk", "Mean Jerk (rad/s³)", model_to_marker, model_to_color, color='lightblue')
            with c12:
                render_metric_bar_chart(df, 'cart_jerk', "Cartesian Jerk", "Mean Jerk (m/s³)", model_to_marker, model_to_color, color='lightblue')

            st.divider()

            c13, c14 = st.columns(2)
            with c13:
                render_metric_bar_chart(df, 'joint_path_length', "Joint Path Length", "Mean Path Length (rad)", model_to_marker, model_to_color, color='lightgreen')
            with c14:
                render_metric_bar_chart(df, 'cart_path_length', "Cartesian Path Length", "Mean Path Length (m)", model_to_marker, model_to_color, color='lightgreen')

            st.divider()

            # --- Interaction and Collision Metrics ---
            c15, c16 = st.columns(2)
            with c15:
                render_metric_bar_chart(df, 'collisions_env', "Environment Collisions", "Count", model_to_marker, model_to_color, color='salmon')
            with c16:
                render_metric_bar_chart(df, 'object_drops', "Object Drops", "Count", model_to_marker, model_to_color, color='salmon')

        else:
            if df is not None:
                st.info("No data matches the selected filters.")
            else:
                st.info("No data available.")
    except Exception as e:
        st.error(f"Error in Plots: {e}")

    # --- Causal Insights ---
    st.header("Causal Insights")

    with st.expander("How this analysis works — methodology & thresholds", expanded=False):
        st.markdown("""
### Overview

The Causal Insights engine automatically scans every **(task × perturbation) cell** in the
full unfiltered dataset (i.e. it ignores any sidebar filters, so that baselines are always
computed over the complete experiment). It runs four independent analytics passes and presents
all findings sorted by a **unified merit score**.

A cell must contain **at least 10 trials** to appear in any analysis.

---

### Unified Merit Score

Every finding — regardless of which pass produced it — is scored by:

```
merit = ES × SF × SW × CW     (all components ∈ [0, 1])
```

| Component | Meaning | Formula |
|---|---|---|
| **ES** — Effect Size | How large is the detected deviation? | Pass-specific (see below) |
| **SF** — Significance Factor | How statistically confident are we? | `min(1, −log₁₀(p) / 4)` |
| **SW** — Sample Weight | How many trials back this finding? | `0.6 + 0.4 × min(1, (n−10) / 15)` |
| **CW** — Category Weight | Epistemic value of the finding type | stage=1.0, SR=0.9, metric=0.7, ranking≤0.10 |

**SF calibration** — `min(1, −log₁₀(p) / 4)` maps:

| p-value | SF |
|---|---|
| ≤ 0.0001 | 1.00 (saturates) |
| 0.001 | 0.75 |
| 0.01 | 0.50 |
| 0.05 | 0.33 |
| 0.15 | 0.21 |

**SW calibration** — penalises cells near the 10-trial minimum:

| n (trials) | SW |
|---|---|
| 10 (minimum) | 0.60 |
| 17 | 0.79 |
| ≥ 25 | 1.00 |

Because all four components are in [0, 1], the final merit score is always in [0, 1], making
findings from different passes **directly comparable**. This replaces the previous system where
stage-deviation scores ranged 0–6, success-rate scores ranged 0–0.7, and metric-anomaly scores
were raw sigma values — three incompatible scales that produced mis-orderings.

---

### Pass 1 — Failure Stage Deviation

**Underlying distribution: Categorical / Multinomial.** The failure stage is a discrete
label drawn from a task-specific set of categories (e.g. GRASP, REACH, LIFT_LARGE for
pick tasks; ROTATED for rotation tasks). The number of possible categories differs by task
and is determined by the task's stage graph — it is *not* a continuous signal and not
normally distributed. The appropriate test for comparing two categorical distributions is
the **chi-squared test**, which makes no distributional assumption beyond the multinomial
sampling model.

**Question:** Does a specific (task, perturbation) combination shift the failure-stage
distribution in a statistically unusual way compared to how that task normally fails?

**Method:**
1. For every task, compute the **task-wide baseline** stage distribution — proportions of
   trials ending in each stage pooled across all perturbations of that task.
2. For every (task, perturbation) cell, compute the **cell stage distribution**.
3. Run a **Pearson chi-squared test** (`scipy.stats.chi2_contingency`) with a 2×k contingency
   table: row 0 = cell stage counts, row 1 = (task total − cell) stage counts.
   This tests whether the cell's categorical distribution is consistent with the task baseline.
4. Find the single stage with the **largest absolute deviation** (cell % − task baseline %).
   Pre-filter: only proceed if this deviation is **≥ 10 percentage points**.
5. Collect all raw p-values and compute **BH-adjusted p-values**:
   `p_adj[i] = min(1, p_raw[i] × m / rank_ascending[i])`
   where m = total candidates and rank is 1-based ascending by p-value.
   Only findings with `p_adj < 0.15` (FDR ≤ 15%) are shown.

**Merit components:**
- **ES** = `|Δpp|` — deviation in proportion for the most-shifted stage (a mean deviation,
  valid for categorical data regardless of the full distribution shape)
- **SF** = `min(1, −log₁₀(p_bh_adj) / 4)` — uses the BH-adjusted p-value
- **SW** = sample weight from `n_cell`
- **CW** = 1.0 (most causal finding type — shows *which mechanism* changes)

---

### Pass 2 — Success Rate Anomalies

**Underlying distribution: Binomial.** `binary_SR` is 0 or 1 per trial, so the number of
successes in n trials follows a Binomial(n, θ) distribution. We use **Bayesian inference
with a uniform Beta(1,1) prior** — the same conjugate update used by the violin plots —
giving posterior Beta(S+1, F+1) for S successes and F=n−S failures.

**Question:** Is a cell's success rate a genuine outlier relative to *both* the task average
and the perturbation average simultaneously?

**Admission criteria (all three must hold):**
- |cell posterior mean SR − task posterior mean SR| ≥ **15 percentage points**
- |cell posterior mean SR − perturbation posterior mean SR| ≥ **15 percentage points**
- Both deltas in the **same direction** (both above average, or both below)

All baseline means are themselves Bayesian posterior means `(S+1)/(N+2)` rather than raw
proportions — this shrinks extreme estimates from small groups toward 0.5.

**Method:**
1. Compute Bayesian posterior means for the task and perturbation baselines.
2. Apply the three admission criteria above.
3. Measure significance via **Bayesian posterior overlap (Monte Carlo)**:
   - Draw 8000 samples from `Beta(S_cell+1, F_cell+1)` and `Beta(S_rest+1, F_rest+1)`
     (where "rest" = all other perturbations of the same task).
   - Compute P(θ_cell > θ_rest).
   - Convert to a two-tailed p-equivalent: `p_equiv = 2 × (1 − max(P, 1−P))`.
   - This makes no normality assumption and is exact for any sample size.
4. Compute a **Wilson score 95% CI** for the cell SR (shown in headline).

**Merit components:**
- **ES** = `min(1, |Cohen's h| / π)` — arc-sine effect size on Bayesian posterior means:
  `h = 2·arcsin(√SR_cell_bayes) − 2·arcsin(√SR_task_bayes)`
  The arc-sine transform is the variance-stabilising transform for proportions.
- **SF** = `min(1, −log₁₀(p_bayes_equiv) / 4)`
- **SW** = sample weight from `n`
- **CW** = 0.9

---

### Pass 3 — Metric Anomalies in Failures

**Question:** Among trials that *failed*, does a cell show unusually high values on any
safety or motion-quality metric?

**Metrics examined:**
| Metric | Description |
|---|---|
| `collisions_env` | Environment collisions per trial |
| `object_drops` | Object drops per trial |
| `joint_vel_var` | Joint velocity variance (rad²/s²·samples) |
| `cart_jerk` | Cartesian jerk (m/s³) |

These are continuous auxiliary measurements for which a **normal approximation is
reasonable**. Unlike binary SR (Binomial) or failure stage (Categorical/Multinomial),
these signals are real-valued and approximately symmetric around their means in practice.

**Method:**
1. Restrict to **failed trials only** (`binary_SR == 0`).
2. Compute the **global failure mean and std** for each metric, pooled over all cells.
3. For each (task, perturbation) cell (with ≥10 failed trials), compute the cell mean.
4. Pre-filter: only proceed if the cell mean is **> 1.5σ above the global failure mean**.
5. Convert sigma to a one-tailed p-value via `scipy.stats.norm.sf(sigma)`.

**Merit components:**
- **ES** = `min(1, sigma / 4)` — normalised deviation in mean, capped at 4σ=1.0
- **SF** = `min(1, −log₁₀(norm.sf(sigma)) / 4)` — σ=1.5→SF≈0.29; σ=3.0→SF≈0.72
- **SW** = sample weight from `n_failed` in cell
- **CW** = 0.7 (informative but less directly causal than stage or SR findings)

---

### Pass 4 — Ranking Summaries

Context findings always shown last via fixed low merit scores (0.04–0.10):

| Finding | Merit | How computed |
|---|---|---|
| Best / Worst (task × pert) combo | 0.10 / 0.09 | Cell with highest / lowest mean SR (min 5 trials) |
| Most consistent perturbation | 0.08 | Lowest std of per-task SR across tasks |
| Most variable perturbation | 0.07 | Highest std of per-task SR across tasks |
| Task most sensitive to perturbation | 0.06 | Largest (max SR − min SR) range across its perturbations |
| Hardest / Easiest task | 0.05 / 0.04 | Lowest / highest mean SR across all perturbations |

These are always below any real statistical finding (which have merit ≥ 0.12 in practice).

---

### Sorting & Deduplication

All findings are merged and **sorted descending by merit score**. A cell can appear in multiple
passes — a co-occurrence is itself informative (e.g. the same condition causing both a stage
spike and elevated collisions).

---

### What is *not* shown

- Cells with **< 10 trials** (unreliable proportions)
- Stage deviations **< 10pp** (not practically meaningful even if significant)
- SR anomalies that are outliers vs. only *one* baseline (task or perturbation), not both
- Metric anomalies **< 1.5σ**
- Stage findings with BH-adjusted p ≥ 0.15 (FDR > 15%)
""")

    try:
        if df_full is not None and not df_full.empty:
            findings = analytics.run_all_analytics(df_full, min_cell_size=10)
            if findings:
                for f in findings:
                    st.markdown(f"- {f['headline']}")
            else:
                st.info("Not enough data to compute causal insights (need ≥10 trials per cell).")
        else:
            st.info("No data available for analysis.")
    except Exception as e:
        st.error(f"Error computing causal insights: {e}")

    # --- Aggregated Reports ---
    st.header("Aggregated Reports")
    try:
        if df is not None:
            st.write(f"Showing {len(df)} rows.")
            st.dataframe(df, height=300)
        else:
            st.info("No reports found.")
    except Exception as e:
        st.error(f"Error in Aggregated Reports: {e}")

    # --- Videos (paginated fragment) ---
    vid_col1, vid_col2 = st.columns([1, 1])
    with vid_col1:
        st.header("Videos")
    
    # Check for parquets to show the button
    has_parquets = False
    for rp in selected_runs:
        v_dir = os.path.join(rp, "videos")
        if os.path.exists(v_dir) and any(f.endswith(".parquet") for f in os.listdir(v_dir)):
            has_parquets = True
            break
            
    if has_parquets:
        with vid_col2:
            st.write("") # Spacer
            if st.button("📦 Unpack All Parquet Videos", width="stretch"):
                unpacked_any = False
                for rp in selected_runs:
                    v_dir = os.path.join(rp, "videos")
                    if os.path.exists(v_dir):
                        p_files = [f for f in os.listdir(v_dir) if f.endswith(".parquet")]
                        for pf in p_files:
                            p_path = os.path.join(v_dir, pf)
                            success, msg = dashboard_utils.unpack_parquet_videos(p_path, v_dir)
                            if success:
                                unpacked_any = True
                
                if unpacked_any:
                    st.success("Successfully unpacked parquet videos!")
                    st.cache_data.clear()
                    st.rerun()
                else:
                    st.error("Failed to unpack any parquet videos.")

    render_video_section(all_videos, selected_tasks, selected_perts, run_paths=selected_runs)
else:
    if not os.path.exists(LOGS_DIR):
        st.error(f"Logs directory '{LOGS_DIR}' not found.")
    else:
        st.info("Please select an experiment from the sidebar.")
