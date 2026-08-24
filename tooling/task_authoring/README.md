# REALM task-authoring dashboard

Start the dashboard from the repository root on a port separate from the results viewer:

```bash
OMNIGIBSON_DATASET_PATH=/path/to/og_dataset \
  uv run streamlit run tooling/task_authoring/dashboard.py --server.port 8503
```

The workspace scans USD files, supports drag-and-drop 3D placement, exposes object roles and
bounding boxes, and downloads a draft matching the object-list structure in `realm/config/tasks`.
Left-click selects and drags objects on XY, the axis gizmo supports constrained XYZ movement,
right-drag orbits the camera, middle-drag pans, and the mouse wheel zooms. Exact transforms remain
editable in the selected-object inspector. The 3D view renders a lightweight
version of REALM's unencrypted Panda OBJ visual meshes in the scene's standard reset pose. Encrypted
BEHAVIOR object assets remain scale-accurate outlined bounding boxes. New objects default to
`distractors`; assigning the unique `main_objects` or `target_objects` role automatically demotes
the previous object in that role, while any number of distractors and immutables are allowed. Object
rotation is edited as XYZ Euler angles in degrees and exported to task YAML as an XYZW quaternion.
Live validation blocks saving to REALM, but still allows copying or downloading incomplete drafts.
Valid drafts can be created directly under `realm/config/tasks/REALM_DROID10`; existing configs are
never overwritten.
Drawer
tasks additionally require the custom articulated cabinet or a whitelisted `bottom_cabinet` model
from `object_sampling.py`; a yellow recommendation asks for at least three distractors.

The **Describe the task** field provides a local prompt-to-draft workflow inspired by RoboLab's
intent → objects → placement → validation pipeline. For supported REALM task verbs it infers the
task type, resolves category names against the indexed OmniGibson catalogue, assigns main/target
roles, adds up to three DROID distractors, and creates an editable starter layout. It does not call
a hosted language model; ambiguous or unknown categories are reported instead of invented.
Generated and manually reviewed layouts follow [AUTHORING_RULES.md](AUTHORING_RULES.md), including
support-plane clearance, uniform scaling, yaw-first alignment, and receiver-capacity checks.

Three.js 0.128.0 and its OrbitControls helper are vendored so the embedded workspace runs without
a CDN connection. Three.js is distributed under the MIT license; see `THREE-LICENSE.txt`.
