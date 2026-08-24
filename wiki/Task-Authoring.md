# Task authoring

REALM includes a browser-based editor for building task YAML from OmniGibson assets. It supports
manual 2D/3D layout, prompt-to-draft generation, existing-config import, task-specific validation,
camera placement, and direct creation of a config under `realm/config/tasks/REALM_DROID10`.

The editor is an authoring aid, not a simulator. Its geometry checks catch common configuration
mistakes before a GPU run, but every saved task still needs rendered and physics-based review.

## Start the editor

The editor runs in the host-side uv environment; OmniGibson and Isaac Sim do not need to be imported.
From the repository root:

```sh
uv sync --locked
OMNIGIBSON_DATASET_PATH=/path/to/datasets \
  uv run streamlit run tooling/task_authoring/dashboard.py --server.port 8503
```

Open the URL printed by Streamlit. Use a different port if the results dashboard is already running.

The dataset root may also be changed at the top of the page. It should contain the extracted
OmniGibson object catalogue. Set **Maximum USD files** to `N/A` to index the full dataset, or use a
positive number for a faster development scan. If no assets are found, the editor loads a small demo
catalogue; do not mistake a demo-only draft for a dataset-backed task.

## Authoring workflow

1. **Choose a scene support.** Select a scene and support region loaded from
   `realm/config/scenes/scenes.yaml`. The orange grid is the valid spawn footprint. In the 2D view,
   hatched red padding is outside the valid region.
2. **Add assets.** Filter by category or model name, optionally enable **DROID objects only**, then
   drag assets onto the workspace. New objects start as distractors.
3. **Assign semantic roles.** Select an object and choose `main_objects`, `target_objects`,
   `distractors`, or `immutables`. The editor permits exactly one main and one target; assigning a
   new unique role demotes the previous object to a distractor.
4. **Place and orient.** Move objects in 2D or 3D, then edit exact position, bounding-box dimensions,
   and XYZ rotation in the inspector. Rotations are entered as degrees and serialized as XYZW
   quaternions.
5. **Review both cameras.** Select a real DROID extrinsic for each external camera, edit its pose, or
   use **Shuffle camera extrinsics** to draw another valid pair. The two 16:9 previews show the
   resulting views.
6. **Resolve red validation errors.** Yellow messages are recommendations; red messages block direct
   saving.
7. **Export or save.** Copy or download YAML at any time. **Save to REALM config** becomes available
   only when the draft has no red errors.
8. **Run simulation review.** Load the saved config in OmniGibson, allow objects to settle, and check
   support contact, reachability, camera framing, initial predicates, and task completion.

## Workspace controls

| Action | Control |
|---|---|
| Select an object or camera | Left click |
| Move an object on the ground plane | Left drag |
| Move along one 3D axis | Drag the red X, green Y, or blue Z gizmo handle |
| Orbit the editor camera | Right drag |
| Pan without rotation | Middle drag |
| Zoom | Mouse wheel |
| Delete the selected object | Delete or Backspace |
| Exact transform editing | Position and rotation fields in the inspector |

The 2D and 3D views edit the same draft. Bounding boxes use their authored XY dimensions in the
top-down view; large objects therefore occupy proportionally more of the support. The robot base and
support pedestal provide a consistent reference for near/far placement. Dataset objects are rendered
as scale-accurate outlined boxes because encrypted BEHAVIOR USD geometry is not decoded in the host
viewer. The Panda preview uses lightweight local visual meshes.

## Prompt-to-draft authoring

Enter an instruction such as `Put the apple in the bowl` under **Describe the task**, then choose
**Draft task from instruction**. This is a deterministic local grounding pipeline, not a hosted
language-model call. It:

1. infers a task type already supported by REALM;
2. resolves noun phrases against the indexed asset catalogue;
3. uses edit distance and role-aware container preferences for near matches such as `box` or `bin`;
4. assigns main, target, and required immutable-source roles;
5. selects only allowed articulated assets for drawer tasks;
6. creates a non-overlapping starter layout and up to three DROID distractors;
7. applies the same validation as a manual draft.

Approximate category substitutions are reported in the draft status and require human review. The
planner never invents a category that is absent from the current index, and it does not create new
task implementations or success criteria. Use the **Agentic Task Authoring** page in the Streamlit
navigation for examples and the current interpretation rules.

## Geometry and semantic contract

The editor and batch generator share the rules in
`tooling/task_authoring/AUTHORING_RULES.md`. The most important are:

- **Support clearance:** generated objects start with their bounding-box bottom 50 mm above the
  configured support plane. A short settling drop is safer than initial interpenetration.
- **Proportion-preserving resize:** automatic fitting uses one uniform scale for X, Y, and Z. Do not
  squeeze individual axes to force an object into a receiver.
- **Yaw first:** prefer no rotation; use Z rotation for footprint alignment. Roll or pitch needs a
  task-specific reason and a simulation review.
- **Receiver capacity:** `put` requires the receiver footprint to cover the oriented main footprint
  with a 1.15 proxy margin. `stack` uses a 0.65 support proxy. These are outer-bbox checks, not proof
  of interior volume or stability.
- **Instruction closure:** every object needed to make the instruction meaningful must exist. For
  example, removing a lid requires a pot or pan in the immutable role and the lid must begin above
  that source.
- **Spawn-region margin:** keep complete object footprints inside the configured support; the object
  centre alone is insufficient.
- **Clutter diversity:** three distractors are recommended, excluding categories already used by
  the instructed objects.

## Validation and saving

Red errors include missing or duplicate main/target roles, unsupported role combinations, absent
source objects, support-plane intersection, inadequate receiver capacity, invalid initial source
relationships, and non-articulated drawer assets. Yellow warnings include fewer than three
distractors, roll/pitch use, and excessive settling gaps.

Copy and download remain enabled for incomplete drafts so they can be reviewed externally. Direct
save is stricter: it writes a new
`realm/config/tasks/REALM_DROID10/<task_name>/default.yaml`, validates the YAML shape and task name,
and refuses to overwrite an existing task. Generated names are derived from the task and object
categories; an existing name receives a numeric suffix in the editor.

## Loading an existing task

Use **Load existing task config** to select a tracked YAML, or upload a `.yaml`/`.yml` file. The
editor reconstructs the scene, object roles, transforms, bounding boxes, and camera extrinsics.
Always review the reconstruction before saving: custom object types and legacy fields may not have a
one-to-one visual representation.

## What must still be checked in simulation

Bounding boxes cannot establish mesh-level clearance, containment volume, frictional stability,
robot reachability, collision-free settling, or whether a visual attribute in the instruction is
actually present. A task is release-ready only after inspecting rendered external and wrist views
and exercising its success predicate in OmniGibson. When a generated family is repaired, encode the
correction in its generator or manifest as well as the emitted YAML so regeneration cannot restore a
known failure.

## Troubleshooting

- **No assets indexed:** verify the dataset root and decryption key installation; the demo catalogue
  is only a fallback.
- **An object appears as a box:** expected for encrypted dataset USDs in the host editor.
- **Save is disabled:** resolve every red message below the YAML controls; warnings do not block it.
- **Save reports that the task exists:** choose a new generated name or load and edit the existing
  config. The save endpoint intentionally never overwrites.
- **A valid-looking draft explodes on reset:** increase support clearance, inspect the model origin,
  and verify mesh-level contact in simulation.

## See also

- [Tasks and perturbations](Tasks-and-Perturbations)
- [Robots and configs](Robots-and-Configs)
- [Running evaluations](Running-Evaluations)
