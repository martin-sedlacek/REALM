# Task Authoring

The task authoring dashboard builds REALM task YAML from OmniGibson assets. It supports manual 2D
and 3D editing, prompt-based drafts, existing YAML files, camera placement and direct saving to
`realm/config/tasks/REALM_DROID10`.

## Running the dashboard

From the repository root:

```sh
uv sync --locked
OMNIGIBSON_DATASET_PATH=/path/to/datasets \
  uv run streamlit run tooling/task_authoring/dashboard.py --server.port 8503
```

The dataset path and USD scan limit can also be changed at the top of the page. `N/A` indexes all
USD files. If the path is empty, the dashboard uses a small demo catalogue.

## Creating a task

1. Select a scene and support surface.
2. Filter the asset list and drag objects into the workspace.
3. Assign each object a role: main, target, distractor or immutable.
4. Set the task type and adjust object transforms.
5. Check both external camera previews.
6. Resolve the red validation errors.
7. copy, download or save the generated YAML.

New objects start as distractors. Only one main object and one target object are allowed. Assigning
either role to another object moves the previous object back to distractors.

Copy and download work for incomplete drafts. **Save to REALM config** is enabled only when all red
errors are fixed. Saving creates:

```text
realm/config/tasks/REALM_DROID10/<task_name>/default.yaml
```

Existing tasks are never overwritten.

## Workspace controls

| Action | Control |
|---|---|
| Select | Left click |
| Move on XY | Left drag |
| Move on one axis | Drag the red X, green Y or blue Z arrow |
| Rotate the view | Right drag |
| Pan the view | Middle drag |
| Zoom | Mouse wheel |
| Delete | Delete or Backspace |

The 2D and 3D views edit the same task. Exact position, bounding box and XYZ rotation values are in
the object inspector. Rotations are exported as XYZW quaternions.

Dataset objects are shown as bounding boxes because encrypted BEHAVIOR USD meshes cannot be loaded
by the host viewer. The bounding boxes use the real object dimensions. The Panda robot preview uses
local visual meshes.

## Prompt-based drafts

Enter an instruction such as `Put the apple in the bowl` and click **Draft task from instruction**.
The local planner:

- selects a supported task type;
- matches object names to indexed categories;
- assigns object roles;
- creates a starter layout;
- adds up to three DROID distractors;
- runs the normal task validation.

This is a deterministic local planner, not an online language model. Small misspellings and generic
container names can use a nearby valid category. The selected substitution is shown for review.
Unknown categories are not invented.

The planner only uses task types already supported by REALM. It does not create new success criteria.

## Placement rules

The complete rules are in `tooling/task_authoring/AUTHORING_RULES.md`. The main ones are:

- Objects start with their bounding-box bottom 5 cm above the support plane.
- Resizing keeps the original XYZ proportions.
- Prefer Z rotation. Use roll or pitch only when the instruction needs it.
- A receiving object must be large enough for the main object.
- Every object named by the instruction must exist in the scene.
- A lid-removal task must start with the lid on the pot or pan.
- Keep the full object bounding box inside the support area.
- Use at least three distractors when possible.

Red messages make the task invalid. Yellow messages are recommendations and do not block saving.

## Cameras

The dashboard loads robot-relative camera poses from
`realm/config/env/external_sensors/camera_extrinsics_droid_realm.yaml`.

Both cameras can be moved or rotated manually. **Shuffle camera extrinsics** samples another real
DROID pair. The two 16:9 views below the editor show what the cameras see.

## Loading an existing task

Select a tracked task from the dropdown or upload a YAML file. The dashboard reconstructs its scene,
objects, roles, transforms and cameras. Review the result before saving, especially for custom or
older object types.

## Simulation check

The dashboard checks bounding boxes, not physics. Before adding a task to a benchmark, render it in
OmniGibson and check:

- objects settle without collisions or large forces;
- the main object is reachable;
- receiving objects are actually usable, not only large enough by bounding box;
- the instruction matches the initial state;
- both external cameras and the wrist camera show the task.

## See also

- [Tasks and perturbations](Tasks-and-Perturbations)
- [Robots and configs](Robots-and-Configs)
