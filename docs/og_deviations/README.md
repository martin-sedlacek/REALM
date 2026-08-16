# OmniGibson deviations — an internals reference

REALM runs on OmniGibson, which runs on Isaac Sim. OmniGibson is not a thin wrapper: between the
USD asset on disk and what PhysX actually simulates, it substitutes defaults, overwrites values the
asset authored, imposes constraints assets must satisfy, and in a few places composes transforms to
the wrong frame. Most of this is silent — no warning, no log line.

These four chapters are the record of what this project found while chasing one such difference.
They exist so that the next person debugging "why does REALM behave differently from a raw Isaac Lab
setup" starts from evidence instead of from scratch.

**This is a read-only audit. Nothing here is fixed** — a handful of findings were patched in the
OG-lite fork and those are marked in place; everything else is flagged only.

## The four chapters

| chapter | domain |
|---|---|
| [Control and actuation](control_and_actuation.md) | drive gains, control modes, effort/velocity/position limits, controller lifecycle, the joint↔controller partition OmniGibson enforces, and the drive rewrites it applies to every dataset object |
| [Rigid bodies, mass and collision](rigid_bodies_and_collision.md) | mass, inertia, centre of mass, collision shapes and approximations, contact/rest offsets, physics materials, solver iteration counts, contact reporting |
| [Simulator, physics scene and lifecycle](simulator_and_scene.md) | how `/physicsScene` is actually configured (read back from a live run), CCD, fabric, GPU dynamics, timestep and substepping, and the fact that `stop`/`play`/`step`/`render` act on every scene at once |
| [Transforms, articulation state and asset import](transforms_and_assets.md) | pose composition, xformOps and the up-axis problem, articulation-root rewriting, state dump/restore, and what the URDF→USD import path does to an asset |

Each chapter is a table of numbered rows — site, what OmniGibson does, what Isaac Lab does or what
the asset authored, whether it is silent, and the measured impact — followed by three sections that
are worth as much as the table:

- **Benign / by design** — things that look like deviations and are not. Checked so they are not
  re-checked.
- **Traps** — things that are not deviations but will mislead you if you read the code quickly.
- **Not covered** — named gaps, so absence is not mistaken for cleanliness.

Two findings recur across chapters and are cross-referenced rather than duplicated: the
`frame="parent"` frame-composition defect (three live sites) and OmniGibson's habit of computing a
value and then overwriting whatever the asset authored.

## How paths are cited

**An unqualified filename in these tables is never a REALM file.** `simulator.py`, `env_base.py`,
`joint_prim.py`, `macros.py` and friends are OmniGibson's, relative to the `omnigibson/` package —
so `env_base.py:730` means `omnigibson/envs/env_base.py:730`, **not** REALM's
`realm/environments/env_base.py`, which shares the basename and is a quarter the length. Likewise
`base.py` in the "what raw Isaac / Isaac Lab does" column is RoboLab's
`robolab/core/environments/base.py`, spelled out in full the first time it appears in each table.
**REALM's own files are always written with the `realm/` prefix.**

Line numbers on OmniGibson and Isaac Lab files are against the versions in the container
(`omnigibson 3.9.1`, Isaac Sim 5.1.0, Isaac Lab 2.2.0) and the OG-lite fork; those trees do not move
under this repo's refactors. Line numbers on `realm/` files do, so REALM sites are cited by
**function or class** wherever the exact line does not carry the point.

## How much to trust this

**The OmniGibson side is what this project measured or read directly. It is not exhaustive**, and it
was not written by auditing OmniGibson systematically — it was written by four agents auditing four
lanes, each starting from a real symptom. A domain absent from these chapters was probably never
looked at.

The Isaac Lab side varies by chapter, and each chapter states its own standard:

- **Control and actuation** — the Isaac Lab side is a **complete, verified extraction**. This is the
  one chapter where "Isaac Lab does not do this" is a claim about the whole Isaac Lab source tree
  rather than about the files that happened to be opened.
- **Rigid bodies** — a verified in-SIF extraction, complete for the files in scope.
- **Simulator and scene** — verified per row, against both Isaac Sim 5.1.0 and Isaac Lab 2.2.0. No
  whole-tree completeness claim.
- **Transforms and assets** — an in-SIF extraction that is explicitly partial. Cells that were not
  resolved say `NOT ESTABLISHED` rather than carrying a guess.

Within the tables, claims are labelled where they were **measured**, and labelled **inferred** where
they were reasoned from source but never run. Take those labels literally — several rows are
mechanism-established-but-unmeasured, and at least one row (the articulation state-restore path) is
explicitly marked "do not act on this until someone measures it."

Two corrections are recorded in place rather than quietly fixed, because a document that only logs
successes will mislead: an earlier "1e5× stiffer" gain ratio was a degrees/radians error (the
reconciled figure is ~1745×), and an earlier "hull-derived separations survive the dropped
transform" belief turned out to be false between the left and right gripper pads.

## Versions this was measured against

Everything here is pinned to a moment. It is not a general statement about OmniGibson.

- **OmniGibson 3.9.1**, via the OG-lite fork. Every row answers "OG-lite only?" — the great majority
  are stock upstream behaviour, decided by diffing against the port commit.
- **Isaac Sim 5.1.0**, inside REALM's own Apptainer image — the Isaac that OmniGibson actually runs
  on here.
- **Isaac Lab 2.2.0**, inside a separate Apptainer image, used as the reference stack.
- **PhysX USD schema defaults** read from the shipped `generatedSchema.usda`, not from memory.

## What this reference does not cover

Named gaps, so that absence is not read as a clean bill of health:

- **Sensors, cameras and rendering.** Touched only where they affect physics readback. Camera pose
  composition in particular is a known unaudited pose path.
- **Object states and transition rules.** The lifecycle coupling is covered; the rule semantics —
  what fires, when, and what each recipe changes — are not, and the engine runs them by default.
- **Grasping.** Assisted grasping can override actuation by welding a joint. REALM runs physical
  grasping so it never fires here, and it was left to a grasping audit that does not exist.
- **Cloth and particle systems**, beyond the trigger that causes a cloth mesh to be remeshed at load.
- **The BEHAVIOR dataset itself.** A few objects were sampled to check specific claims; the dataset
  was not audited, and it is not on disk on the host most of this was measured from.
- **Task, reward and termination logic** — none of this is REALM-side benchmark behaviour. For that,
  see the task and perturbation reference.
- **Anything about performance.** Cost is mentioned only where a deviation happens to be a cost
  rather than a correctness difference.

Per-chapter "Not covered" sections are narrower and more specific than this list. Read the one at
the end of whichever chapter you are relying on.

## A note on the RoboLab citations

Several rows compare against **RoboLab**, a private codebase, cited by `file:line` and by a
description of what the code does. Its source is deliberately not reproduced here. Those particular
citations are therefore not checkable by a reader outside the project — treat them as the one class
of claim in these chapters you cannot verify yourself. Every claim about OmniGibson, Isaac Sim,
Isaac Lab and PhysX is checkable from open source.

## Related

- [Gripper compliance findings](../GRIPPER_COMPLIANCE_FINDINGS.md) — the investigation these audits
  came out of: what was tried, what failed, and what was reverted. Reads as the applied companion to
  this reference.
