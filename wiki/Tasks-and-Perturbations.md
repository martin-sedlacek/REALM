# Tasks and perturbations

REALM evaluates a policy on **10 manipulation tasks** crossed with **16 perturbation settings** — a
dense 160-cell matrix. This page is the reference for both axes: the exact identifiers, what each
one does, and how you select them.

Both lists are defined once, in `realm/eval.py`, as `SUPPORTED_TASKS` and `SUPPORTED_PERTURBATIONS`.
**Position in those lists is the ID you pass on the command line.** Everything else in the repo
imports from there — the vectorized path, the integrity tests, and the cluster sweep drivers (which
parse `eval.py` at runtime rather than keeping a second copy), so the numbering cannot drift.

## Selecting them

```sh
--task_id <0-9>            # index into SUPPORTED_TASKS,        default 0
--perturbation_id <0-15>   # index into SUPPORTED_PERTURBATIONS, default 0
```

Both flags exist on `examples/02_evaluate.py` (single environment) and `examples/04_vector_evaluate.py`
(vectorized). There is **no name-based CLI flag** — the IDs are the interface.

`--task_cfg_path <suite>/<task>/<variant>.yaml` bypasses the index and names a config directly, e.g.
`REALM_DROID10/pick_spoon/default.yaml`. When set, `--task_id` is ignored.

The Python API does take names: `RealmEnvironmentDynamic(perturbations=[...])` accepts the exact
strings below. Both eval paths pass **exactly one** perturbation. Composing two is untested, and at
least one pair is known to break — see [Composition](#composition).

## The 10 tasks

Each identifier is also the directory name under `realm/config/tasks/REALM_DROID10/<task>/default.yaml`.

| ID | Identifier | Type | Instruction | Scene / surface |
|---:|---|---|---|---|
| 0 | `put_green_block_into_bowl` | `put` | "put the green block in the bowl" | `Pomaria_1_int` / `Table` |
| 1 | `put_banana_into_box` | `put` | "put the banana in the box" | `office_cubicles_left` / `Circular_Table` |
| 2 | `rotate_marker` | `rotate` | "rotate the marker" | `Benevolence_1_int` / `Kitchen_Counter` |
| 3 | `rotate_mug` | `rotate` | "rotate the mug" | `office_cubicles_left` / `Office_Desk` |
| 4 | `pick_spoon` | `pick` | "pick up the spoon" | `Merom_1_int` / `Table` |
| 5 | `pick_water_bottle` | `pick` | "pick up the water bottle" | `Wainscott_0_int` / `Dining_Table` |
| 6 | `stack_cubes` | `stack` | "stack the green block on the yellow block" | `Benevolence_1_int` / `Dining_Table` |
| 7 | `push_switch` | `push` | "push the light switch" | `Pomaria_1_int` / `Light_Switch` |
| 8 | `open_drawer` | `open_drawer` | "open the top drawer" | `Pomaria_1_int` / `Drawers_Near_Table` |
| 9 | `close_drawer` | `close_drawer` | "close the top drawer" | `Pomaria_1_int` / `Drawers_Near_Table` |

Ten tasks, **seven distinct skills** — the `Type` column is the skill, and `put`, `pick` and `rotate`
each appear twice.

> **Watch the spelling of task 0.** It is `put_green_block_into_bowl`, with *into*. A
> `put_green_block_in_bowl` directory also exists, but under the legacy `IMPACT/` suite — passing
> that spelling to `--task_cfg_path` under `REALM_DROID10/` will not resolve. The repo README
> currently prints the wrong one.

> ### ⚠ Two tasks currently have unusable camera views
>
> **Task 6 (`stack_cubes`) renders essentially nothing but sky.** Its spawn region sits roughly a
> metre beyond the nearest wall, so the camera ends up outside the room.
>
> **Task 2 (`rotate_marker`) also gives an unusable external view, but for a different reason** — its
> spawn region is *inside* the wall envelope and the scene does render an interior; the region is
> simply short of any floor surface. Do not assume the two share a cause.
>
> What makes both dangerous is that **every artifact and metric check still passes** — the run
> completes, videos are written, reports are produced, and nothing warns you. A vision-conditioned
> policy evaluated on task 6 in this state is being scored on pictures of the sky.
>
> **Status: observed and parked, not diagnosed.** The project's own note says these may simply
> differ from the pre-port configuration rather than be a port bug, and flags them for eyeballing in
> the GUI before anything is changed. Do not report numbers from these two tasks without looking at
> the frames first.

### Scoring: partial credit, not pass/fail

A rollout is scored against a **progression ladder** for its task type, defined in
`realm/config/tasks/task_progressions.yaml`, with the per-stage predicates in the
`success_conditions` dict built by `TaskProgressionMixin`
(`realm/environments/task_progression.py`), which `RealmEnvironmentBase` inherits — so
`env.success_conditions` is still where you read them at runtime. The last stage is full
success; reaching an earlier stage is partial credit.

| Type | Ladder |
|---|---|
| `pick` | REACH → GRASP → LIFT_LARGE |
| `rotate` | REACH → GRASP → ROTATED |
| `push` | REACH → TOUCH → TOGGLED_ON |
| `put` | REACH → GRASP → LIFT_SLIGHT → MOVE_CLOSE → PLACE_INTO |
| `stack` | REACH → GRASP → LIFT_SLIGHT → MOVE_CLOSE → PLACE_ONTO |
| `open_drawer` | REACH → TOUCH_AND_MOVE_JOINT → OPEN_JOINT_SMALL → OPEN_JOINT_LARGE → OPEN_JOINT_FULL |
| `close_drawer` | mirrors `open_drawer`, ending at CLOSE_JOINT_FULL |

`task_progressions.yaml` also defines ladders for `pour` and `turn_faucet`. **Those are scaffolding,
not tasks** — no task config uses them and the pour predicate returns `False` unconditionally.

### Other task suites

`realm/config/tasks/` also holds `IMPACT/` and `other/` suites, reachable only via `--task_cfg_path`.
They are ablation and real2sim configs, not part of the benchmark 10. Note that
`RealmEnvironmentDynamic` selects the base-mounted DROID variant by checking whether the task config
path starts with `REALM_DROID10`, so the other suites deliberately get a different robot mount.

## The 16 perturbations

ID 0 is the unperturbed control. So: **16 selectable settings, 15 of which actually perturb
something.** Both numbers are correct; say which you mean.

Implementations live in `realm/environments/perturbations/`. The name→implementation registry is
`RealmEnvironmentDynamic.supported_pertrubations` (the misspelling is in the source), and the
constructor asserts every requested name is a key of it.

### Control

| ID | Name | What it does |
|---:|---|---|
| 0 | `Default` | Nothing. The unperturbed baseline every other cell is compared against. |

### Visual

| ID | Name | What it perturbs |
|---:|---|---|
| 1 | `V-AUG` | Image quality: Gaussian blur and contrast scaling applied to every rendered view, including the wrist view. |
| 2 | `V-VIEW` | External camera pose: re-draws a calibrated viewpoint, then jitters position and pitch/yaw. |
| 3 | `V-SC` | Scene clutter: re-places the scene's objects collision-free within the spawn region, then swaps each **distractor** for a different object drawn from a different category theme. It does not add objects — it works with the distractors the task config already declares. See the caveats below. |
| 4 | `V-LIGHT` | Illumination: randomises every light's intensity across a wide range and shifts its colour. |

`V-AUG` is the odd one out: it changes no scene state, so its registry entry is the no-op. The
augmentation is applied in the **observation path** instead — the environment draws the blur and
contrast parameters once per reset and applies them to each observation.

> ### ⚠ `V-SC` does nothing on three tasks
>
> The number of distractors is **declared per task**, and `V-SC` re-places and re-models whichever
> ones exist rather than spawning new ones. Read from the task configs:
>
> | distractors | tasks |
> |---:|---|
> | 5 | `put_green_block_into_bowl` |
> | 4 | `put_banana_into_box`, `rotate_marker` |
> | 3 | `rotate_mug`, `pick_spoon`, `pick_water_bottle` |
> | 2 | `stack_cubes` |
> | **0** | **`push_switch`, `open_drawer`, `close_drawer`** |
>
> On the three tasks with no distractors there is nothing for `V-SC` to clutter with or swap out, so
> it is effectively inert — while still costing a full stopped-simulator reset. Averaging `V-SC`
> across all ten tasks therefore averages in three near-no-ops.
>
> *Read from the task configs and `v_sc.py`, not measured at runtime.*
>
> Separately, on the tasks that do have distractors, the spawn region is over-subscribed: roughly two
> objects **per environment** per reset fail collision-free placement and are dropped in from above —
> at `--num_envs 4` that is about eight.

### Semantic — instruction only, scene untouched

| ID | Name | How the instruction is rewritten |
|---:|---|---|
| 5 | `S-PROP` | By physical properties — colour, size, texture — instead of the object's name. |
| 6 | `S-LANG` | By wording and synonymy: different phrasing for the same request. |
| 7 | `S-MO` | By spatial relation to other objects ("the mug between the two chocolate items"). |
| 8 | `S-AFF` | By affordance or human purpose ("...for safekeeping", "as if clearing a table"). |
| 9 | `S-INT` | By world knowledge — material, shape class, typical use. |

These five change the language and nothing else. **Four of them are data-defined, not code-defined:**
`S-PROP`, `S-MO`, `S-AFF` and `S-INT` are one-line wrappers around a shared helper that picks a
string from the `cached_semantic_perturbations` block of the task's own YAML. What distinguishes them
is entirely the content of those lists — every one of the 10 task configs carries all five keys with
ten paraphrases each. If you want to know what `S-INT` means for a given task, read that task's YAML;
the code will not tell you.

`S-LANG` is the only one with real logic. `open_drawer` and `close_drawer` define a `synonyms` block,
so on those two it can generate a fresh substitution; on the other eight it falls back to the cache.

### Behavioural

| ID | Name | What it perturbs |
|---:|---|---|
| 10 | `B-HOBJ` | The target object's physics: per-link mass is rescaled and clipped, and joint effort/stiffness/damping are scaled log-uniformly. Factors are computed from a pristine baseline snapshot, so they cannot compound across resets. |

### Composite

| ID | Name | Axes | What it perturbs |
|---:|---|---|---|
| 11 | `SB-NOUN` | S+B | Re-targets the instruction at a **different object already in the scene**. On drawer tasks it swaps which drawer is named and re-homes the arm instead. |
| 12 | `SB-VRB` | S+B | Changes the required **skill**: draws a different verb, swaps in that verb's progression ladder, and spawns a receiver object if the new verb needs one. |
| 13 | `VB-POSE` | V+B | Object placement: re-samples collision-free positions for all objects and adds yaw noise to the target. |
| 14 | `VB-MOBJ` | V+B | Target object size and shape: anisotropic rescale, capped and clipped to a task-dependent range. |
| 15 | `VSB-NOBJ` | V+S+B | Object identity: replaces the target with a different, unseen category and model, and rewrites the noun in the instruction to match. |

### Two groupings worth knowing

- **`NEEDS_STOPPED_SIM`** (`realm/environments/perturbations/_helpers.py`) — `V-SC`, `VB-MOBJ`,
  `VSB-NOBJ`, `SB-VRB`. These add or remove objects and so require a stopped simulator. The rest only
  write poses. This is why those four are the expensive ones to reset.
- **Five names that are not implemented** — `V-OBJ`, `VB-ISC`, `VS-PROP`, `SB-ADV`, `SB-SMO`. They
  appear in the perturbation taxonomy but REALM has no code for them, as the module docstring of
  `realm/environments/perturbations/registry.py` states. They are not in `PERTURBATION_FNS`, so they
  are not in `SUPPORTED_PERTURBATIONS`, there is no ID to pass, and
  `RealmEnvironmentDynamic.__init__` asserts every requested name is a key of its
  `supported_pertrubations`. Do not treat them as available. (A `MISSING_PERTURBATIONS` constant
  used to list them; it was removed when `realm/environments/` was split. Nothing behavioural
  changed — the constant was never what did the rejecting.)

### Known incompatibilities

- `SB-NOUN` on task 7 (`push_switch`, type `push`) raises `NotImplementedError` by design.
- `SB-VRB` on `push` has an empty compatible-verb list, so it has nothing to draw from. *Read from
  the compatibility matrix and the unguarded selection call — no explicit guard or test was found, so
  treat the exact failure mode as unverified.*

### Composition

Both eval entry points pass exactly one perturbation per process. The environment's reference-capture
logic depends on this: composing `SB-NOUN` with `VB-MOBJ`, for instance, would break the size anchor
that `VB-MOBJ` measures against. **Composition is untested and the code says so.** If you need it,
verify it yourself first.

## Running the matrix

Per-cell output is keyed `<task>_<perturbation>`, e.g. a report named `pick_spoon_VSB-NOBJ.csv`. That
naming is what the integrity tests and the log viewer rely on.

A single cell:

```sh
python examples/02_evaluate.py --task_id 4 --perturbation_id 15 \
    --model_type <type> --model_name <name> --port <port> \
    --experiment_name <exp>
```

For a scheduler sweep, pass explicit task and perturbation IDs to one evaluation process per cell.
A useful scheduler-facing convention accepts comma-separated values and `a-b` ranges:

```sh
--task_ids 0,4,8 --perturbation_ids 3-7
```

When implementing this convention in a site-local launcher, validate IDs against `realm/eval.py`,
make omitted ranges explicit, assign unique run IDs and ports, and skip a cell only after verifying
that its complete expected artifact set already exists. See
[Cluster and parallel runs](Cluster-and-Parallel-Runs).

## Rollout budget

Defaults in `examples/02_evaluate.py` are `--repeats 5 --max_steps 500`, which is a smoke-test
budget. The published benchmark configuration is `--repeats 25 --max_steps 800`. The vectorized entry
point defaults to `--repeats 25` and runs them in waves of `--num_envs`.

## Task authoring

The task authoring dashboard builds REALM task YAML from OmniGibson assets. It supports manual 2D
and 3D editing, prompt-based drafts, existing YAML files, camera placement and direct saving to
`realm/config/tasks/REALM_DROID10`.

### Running the dashboard

From the repository root:

```sh
uv sync --locked
OMNIGIBSON_DATASET_PATH=/path/to/datasets \
  uv run streamlit run tooling/task_authoring/dashboard.py --server.port 8503
```

The dataset path and USD scan limit can also be changed at the top of the page. `N/A` indexes all
USD files. If the path is empty, the dashboard uses a small demo catalogue.

### Creating a task

1. Select a scene and support surface.
2. Filter the asset list and drag objects into the workspace.
3. Assign each object a role: main, target, distractor or immutable.
4. Set the task type and adjust object transforms.
5. Check both external camera previews.
6. Resolve the red validation errors.
7. Copy, download or save the generated YAML.

New objects start as distractors. Only one main object and one target object are allowed. Assigning
either role to another object moves the previous object back to distractors.

Copy and download work for incomplete drafts. **Save to REALM config** is enabled only when all red
errors are fixed. Saving creates:

```text
realm/config/tasks/REALM_DROID10/<task_name>/default.yaml
```

Existing tasks are never overwritten.

### Workspace controls

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

### Prompt-based drafts

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

### Placement rules

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

### Cameras

The dashboard loads robot-relative camera poses from
`realm/config/env/external_sensors/camera_extrinsics_droid_realm.yaml`.

Both cameras can be moved or rotated manually. **Shuffle camera extrinsics** samples another real
DROID pair. The two 16:9 views below the editor show what the cameras see.

### Loading an existing task

Select a tracked task from the dropdown or upload a YAML file. The dashboard reconstructs its scene,
objects, roles, transforms and cameras. Review the result before saving, especially for custom or
older object types.

### Simulation check

The dashboard checks bounding boxes, not physics. Before adding a task to a benchmark, render it in
OmniGibson and check:

- objects settle without collisions or large forces;
- the main object is reachable;
- receiving objects are actually usable, not only large enough by bounding box;
- the instruction matches the initial state;
- both external cameras and the wrist camera show the task.

## See also

- [Running evaluations](Running-Evaluations) — the full flag surface
- [Cluster and parallel runs](Cluster-and-Parallel-Runs) — sweeping the matrix
