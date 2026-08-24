"""Documentation page for REALM's prompt-driven task authoring workflow."""

import streamlit as st


st.set_page_config(page_title="Agentic Task Authoring", page_icon="✨", layout="wide")

st.title("Agentic Task Authoring")
st.caption("Turn one manipulation instruction into an editable REALM task draft.")

st.info(
    "This is currently a deterministic local planner, not a hosted language-model call. "
    "It uses the indexed OmniGibson catalogue and REALM's validation rules, so it reports "
    "unknown objects instead of inventing assets."
)

st.header("Using it")
st.markdown(
    """
1. Return to **dashboard** using the page navigation.
2. Enter an instruction in **Describe the task**.
3. Press Enter or click **Draft task from instruction**.
4. Review the inferred task type, selected assets, roles, placement, and camera views.
5. Correct the draft with the regular 2D/3D editing tools.
6. Copy or download incomplete YAML at any time. Saving to REALM becomes available once all red
   validation errors are resolved.
"""
)

st.header("Example instructions")
examples = {
    "Put": "Put the apple in the bowl",
    "Pick": "Pick up the spoon",
    "Stack": "Stack the plate on the bowl",
    "Rotate": "Rotate the bottle of water",
    "Push": "Push the apple",
    "Drawer": "Open the drawer",
}
st.table([{"Task type": task_type, "Instruction": instruction} for task_type, instruction in examples.items()])

st.header("What the planner does")
st.markdown(
    """
The workflow follows the same broad pattern as RoboLab's agent-assisted generation tools:

1. **Intent** — match the instruction to a task type supported by existing REALM configs.
2. **Grounding** — match category phrases to the currently indexed OmniGibson asset catalogue.
   Exact matches are preferred; small misspellings use edit-distance matching, and generic receiving
   containers such as “box” or “bin” use role-aware container preferences. Any substitution is
   displayed in the draft status for review.
3. **Roles** — assign one main object and, for `put` or `stack`, one receiving target.
4. **Asset constraints** — drawer tasks select only articulated `bottom_cabinet` models from
   REALM's drawer allowlist.
5. **Scene draft** — place task objects on the selected scene's valid spawn rectangle and avoid
   obvious bounding-box overlap when choosing starter positions.
   Objects start 50 mm above the support after accounting for half their authored bbox height.
   Oversized assets are scaled uniformly, never squeezed per axis; receiver alignment uses yaw
   before any uniform main-object reduction.
6. **Clutter** — add up to three distinct DROID-category distractors that fit the support region.
7. **Validation** — run the same red errors and yellow recommendations used by manually authored
   drafts.
8. **Human review** — leave every object, role, transform, camera, and generated YAML field in the
   existing visual editor.
"""
)

st.header("Interpretation rules")
st.markdown(
    """
- `put`, `place`, `move`, or `drop` together with `in`, `into`, or `inside` infer `put`.
- `stack`, `on top of`, `onto`, or a placement instruction using `on` infer `stack`.
- `pick`, `grab`, `lift`, or `take` infer `pick`.
- `rotate`, `reorient`, or `turn` infer `rotate`.
- `push` infers `push`.
- Explicit open/close drawer wording infers `open_drawer` or `close_drawer`.
- For two-object tasks, category mention order determines main object first and target second.
"""
)

st.header("Current limitations")
st.markdown(
    """
- Resolved categories must exist in the indexed dataset. Fuzzy matches are suggestions and should
  be reviewed, especially for broad words such as “box” or “container.”
- The planner handles the task types already represented in REALM; it does not synthesize new task
  implementations or success criteria.
- Starter placement uses bounding-box collision checks, not OmniGibson physics or reachability.
- It does not yet conduct a clarification conversation when an instruction is ambiguous.
- Asset choice among matching models and distractor choice are sampled; the draft must be reviewed.
- Camera extrinsics and object transforms remain author-controlled after drafting.
- Receiver checks are outer-bbox proxies. Mesh interiors and first-step settling still require an
  OmniGibson scene-correctness run.
"""
)

st.header("Planned agentic extension")
st.markdown(
    """
The deterministic planner is the safe grounding layer for a later model-backed agent. A model can
propose structured intent, category choices, spatial relationships, and clarification questions;
the local layer should still resolve real asset IDs, enforce task-specific constraints, calculate
placements, run validation, and require review before saving. This keeps natural-language reasoning
separate from repository and dataset truth.
"""
)
