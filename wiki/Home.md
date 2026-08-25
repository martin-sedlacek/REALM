Welcome to the REALM wiki!

**REALM** is a simulation benchmark for **generalization** in robotic manipulation. It runs 10
manipulation tasks covering 7 distinct skills, and stress-tests a policy against **15 perturbations**
— visual, semantic, behavioural, and combinations of those — plus an unperturbed control, so **16
selectable settings** in total. (The paper says 15 and this wiki mostly says 16; both are right, and
[Tasks and perturbations](Tasks-and-Perturbations) spells out which is which.) The benchmark's
premise, argued in the paper, is that results measured this way correlate with real-world
performance.

- **Project page** — <https://martin-sedlacek.com/realm>
- **Paper** — <https://arxiv.org/abs/2512.19562>
- **Issues** — <https://github.com/martin-sedlacek/REALM/issues>
- **Discussions** — <https://github.com/martin-sedlacek/REALM/discussions>

> Please keep in mind - **reproducing the paper's tables/figures is not the same as running the current evolution of the benchmark.**
> There is much inherent noise in every evaluation of robotic policies and since first releasing the paper we migrated
> to a newer of Omnigibson with a different renderer and scene lighting. For more details please see TBA.

## Getting started

| Page | |
|---|---|
| [Installation](Installation) | the container, the dataset, and the robot-definition step that is easy to miss |
| [Quick start](Quick-Start) | GPU setup → smoke test → real evaluation |

## Reference

| Page | |
|---|---|
| [Running evaluations](Running-Evaluations) | container execution, flags, rendering modes, model types |
| [Tasks and perturbations](Tasks-and-Perturbations) | the 10 × 16 matrix, with what each cell means |
| [Logging and results dashboard](Logging) | inspect logs, compare runs, watch videos, and export reports |
| [Task authoring](Task-Authoring) | build, validate, import, and save task YAML in 2D/3D |
| [Cluster and parallel runs](Cluster-and-Parallel-Runs) | scheduler-neutral sweeps and vectorized evaluation |

## Operating notes

| Page | |
|---|---|
| [Performance and scaling](Performance-and-Scaling) | what is expensive and which levers were measured |
