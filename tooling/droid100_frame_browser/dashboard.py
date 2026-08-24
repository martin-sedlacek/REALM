"""Streamlit dashboard for visually reviewing DROID100 first-frame panels."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from tooling.droid100_frame_browser.core import (
    REVIEW_FILE,
    Frame,
    discover_frames,
    display_name,
    load_reviews,
    save_reviews,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOG_ROOT = REPO_ROOT / "logs" / "droid100_first_frames"


st.set_page_config(page_title="DROID100 first-frame review", page_icon="🖼️", layout="wide")
st.title("DROID100 first-frame review")

root = Path(st.text_input("First-frame log root", str(DEFAULT_LOG_ROOT))).expanduser()
frames = discover_frames(root)
if not frames:
    st.error(f"No panel.jpg or panel.jpeg files found below {root}.")
    st.stop()

runs = sorted({frame.run for frame in frames}, reverse=True)
run = st.selectbox("Render run", runs)
run_frames = [frame for frame in frames if frame.run == run]
run_dir = root / run
reviews = load_reviews(run_dir)

controls = st.columns([3, 2, 2])
query = controls[0].text_input("Search tasks", placeholder="lid, bowl, 006 …").strip().lower()
status_filter = controls[1].selectbox(
    "Review status",
    ("All", "Unreviewed", "Needs review", "Valid", "Invalid"),
)
view = controls[2].radio("View", ("Single", "Gallery"), horizontal=True)


def visible(frame: Frame) -> bool:
    if query and query not in frame.task.lower():
        return False
    status = reviews.get(frame.task, {}).get("status", "Unreviewed")
    return status_filter == "All" or status == status_filter


filtered = [frame for frame in run_frames if visible(frame)]
counts = {label: 0 for label in ("Unreviewed", "Needs review", "Valid", "Invalid")}
for frame in run_frames:
    counts[reviews.get(frame.task, {}).get("status", "Unreviewed")] += 1
st.caption(
    f"{len(filtered)}/{len(run_frames)} panels · "
    + " · ".join(f"{label}: {count}" for label, count in counts.items())
)
if not filtered:
    st.warning("No panels match the current filters.")
    st.stop()

if view == "Gallery":
    for offset in range(0, len(filtered), 3):
        columns = st.columns(3)
        for column, frame in zip(columns, filtered[offset : offset + 3]):
            annotation = reviews.get(frame.task, {})
            column.image(str(frame.path), use_container_width=True)
            column.markdown(f"**{display_name(frame.task)}**")
            column.caption(annotation.get("status", "Unreviewed"))
    st.stop()

task_names = [frame.task for frame in filtered]
current_name = st.session_state.get("current_task")
if current_name not in task_names:
    current_name = task_names[0]
index = task_names.index(current_name)

navigation = st.columns([1, 5, 1])
if navigation[0].button("← Previous", use_container_width=True, disabled=index == 0):
    st.session_state.current_task = task_names[index - 1]
    st.rerun()
selected_name = navigation[1].selectbox(
    "Task",
    task_names,
    index=index,
    format_func=display_name,
    label_visibility="collapsed",
)
if selected_name != current_name:
    st.session_state.current_task = selected_name
    st.rerun()
if navigation[2].button("Next →", use_container_width=True, disabled=index == len(task_names) - 1):
    st.session_state.current_task = task_names[index + 1]
    st.rerun()

frame = next(item for item in filtered if item.task == selected_name)
image_column, review_column = st.columns([4, 1])
image_column.image(str(frame.path), use_container_width=True)
image_column.caption(str(frame.path.relative_to(REPO_ROOT)) if frame.path.is_relative_to(REPO_ROOT) else str(frame.path))

existing = reviews.get(frame.task, {})
statuses = ("Unreviewed", "Needs review", "Valid", "Invalid")
status = review_column.radio(
    "Verdict",
    statuses,
    index=statuses.index(existing.get("status", "Unreviewed")),
    key=f"status_{frame.run}_{frame.task}",
)
notes = review_column.text_area(
    "Notes",
    value=existing.get("notes", ""),
    height=180,
    key=f"notes_{frame.run}_{frame.task}",
)
if review_column.button("Save review", type="primary", use_container_width=True):
    reviews[frame.task] = {"status": status, "notes": notes.strip()}
    save_reviews(run_dir, reviews)
    review_column.success(f"Saved to {run_dir / REVIEW_FILE}")
