Welcome to the REALM wiki!

**REALM** is a simulation benchmark for **generalization** in robotic manipulation. It runs 10
manipulation tasks covering 7 distinct skills, and stress-tests a policy against **15 perturbations**
— visual, semantic, behavioural, and combinations of those — plus an unperturbed control, so **16
selectable settings** in total. (The paper says 15 and this wiki mostly says 16; both are right, and
[Tasks and perturbations](Tasks-and-Perturbations) spells out which is which.) The benchmark's
premise, argued in the paper, is that results measured this way correlate with real-world
performance.

Concretely: you point REALM at a policy server, pick a task and a perturbation, and it runs rollouts
in a photorealistic scene and scores them.

> **Reproducing the paper's table is not the same as running the benchmark.** Those results are π₀,
> π₀-FAST and GR00T N1.5, and **none of those three can be constructed on this branch** — see
> [Running evaluations](Running-Evaluations). This wiki documents the code as it currently stands.

- **Project page** — <https://martin-sedlacek.com/realm>
- **Paper** — <https://arxiv.org/abs/2512.19562>
- **Issues** — <https://github.com/martin-sedlacek/REALM/issues>
- **Discussions** — <https://github.com/martin-sedlacek/REALM/discussions>

## What a run produces

Every rollout is scored on a **progression ladder** rather than pass/fail, so a policy that grasps
the object but fails to place it gets credit for the grasp. A run writes four things:

| | |
|---|---|
| `reports/` | one CSV row per rollout: progression, binary success, smoothness and path-length metrics, collision and drop counts, and the instruction actually given |
| `qpos/` | joint trajectories |
| `actions/` | the actions the policy emitted |
| `videos/` | recorded rollouts |

See [Logs, outputs and the viewer](Logs-Outputs-and-Viewer) for the full schema.

## Getting started

| Page | |
|---|---|
| [Installation](Installation) | the container, the dataset, and the robot-definition step that is easy to miss |
| [Quick start](Quick-Start) | allocation → smoke test → real evaluation, in four steps |

## Reference

| Page | |
|---|---|
| [Running evaluations](Running-Evaluations) | every flag, `MODE`, rendering modes, model types |
| [Tasks and perturbations](Tasks-and-Perturbations) | the 10 × 16 matrix, with what each cell means |
| [Robots and configs](Robots-and-Configs) | what `--robot` selects, and how they differ |
| [Logs, outputs and the viewer](Logs-Outputs-and-Viewer) | what a run writes, and how to read it |
| [Cluster and parallel runs](Cluster-and-Parallel-Runs) | SLURM, sweeps, and vectorized evaluation |

## Operating notes

| Page | |
|---|---|
| [Known issues and gotchas](Known-Issues-and-Gotchas) | **read this before debugging anything** |
| [Performance and scaling](Performance-and-Scaling) | what is expensive and which levers were measured |

## Internals

How the simulator behaves underneath REALM. Not needed to run the benchmark; needed to trust or
debug it.

| Page | |
|---|---|
| [OmniGibson deviations](OmniGibson-Deviations) | index — start here |
| [· Control and actuation](OmniGibson-Deviations-Control-and-Actuation) | drive gains, control modes, actuation limits |
| [· Rigid bodies and collision](OmniGibson-Deviations-Rigid-Bodies-and-Collision) | mass, inertia, collision shapes, materials |
| [· Simulator and scene](OmniGibson-Deviations-Simulator-and-Scene) | physics-scene configuration and lifecycle |
| [· Transforms and assets](OmniGibson-Deviations-Transforms-and-Assets) | pose composition, xformOps, asset import |
| [Gripper compliance findings](Gripper-Compliance-Findings) | a worked investigation, including what failed |

## A note on scope

Some of these pages document things that are **wrong or stale in the repository** — an install script
that references deleted files, older cluster pipelines that cannot run against the current container,
two tasks whose camera views are unusable. Those are recorded rather than hidden, because finding out
the hard way is worse. [Known issues](Known-Issues-and-Gotchas) is the index of them.

Where a number here was measured, it says so. Where something was reasoned from source but never run,
it says that too.
