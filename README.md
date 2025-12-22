# REALM: A Real-to-Sim Validated Benchmark for Generalization in Robotic Manipulation

<p align="center">
  <a href="https://martin-sedlacek.com/realm"><img src="https://img.shields.io/badge/project-page-brightgreen" alt="Project Page"></a>
  <a href="https://arxiv.org/"><img src="https://img.shields.io/badge/paper-preprint-red" alt="arXiv"></a>
  <a href="https://github.com/martin-sedlacek/REALM/wiki"><img src="https://img.shields.io/badge/doc-page-orange" alt="Documentation"></a>
  <a href="https://github.com/martin-sedlacek/REALM/issues"><img src="https://img.shields.io/github/issues/martin-sedlacek/REALM?color=yellow" alt="Issues"></a>
  <a href="https://github.com/martin-sedlacek/REALM/discussions"><img src="https://img.shields.io/github/discussions/martin-sedlacek/REALM?color=blueviolet" alt="Discussions"></a>
</p>

![](./images/realm_overview_fig.png)

REALM is a large-scale realistic simulation environment and benchmark for generalization 
in robotic manipulation. It supports 7 distinct manipulation skills and stress-tests them 
against 15 perturbations. Through empirical validation, we show that evaluation results 
in simulation are strongly correlated to real-world performance. 

# 🚧 Roadmap
- [x] Streamlined installation
- [x] Example scripts for getting started
- [ ] Improved benchmarking UX:
  - [ ] End-to-end scripts for producing result plots and tables
- [ ] Extended documentation
- [ ] Performance:
  - [ ] Support vectorized environments
  - [ ] Improve parallelism and overall execution speed

# Installation 🛠️
1. Clone the project repository:
```
git clone https://github.com/martin-sedlacek/REALM.git
cd REALM
```

2. Run the set-up script and download sim assets: 
```
# [RECOMMENDED OPTION] Docker installation (with downloading the dataset):
./setup.sh --docker --dataset

# Using a custom dataset path: 
./setup.sh --docker --dataset --data-path /path/to/dataset

# [UNSTABLE] Apptainer installation (with downloading the dataset):
./setup.sh --apptainer --dataset

# Using a custom apptainer .sif path: 
./setup.sh --apptainer --dataset --sif-path /path/to/realm.sif
```

> ❗ **Please note that running with apptainer is currently not stable.**
> We noticed that the apptainer can crash inexplicably on some systems. 
> It is recommended to use the stable Docker container if possible.


# Easy run 🏃

The main entrypoint for running evaluations is:

```bash
./scripts/eval.sh -c /path/to/checkpoint
```

This script:

- Selects a free TCP port and starts a model server (pi0, pi0_FAST, GR00T, or a custom server script).
- Runs the REALM / OmniGibson evaluation inside Singularity/Apptainer, Docker, or the current environment.

## Requirements

Before running `eval.sh`, make sure:
- You are in the project root (so the script is at `./scripts/eval.sh`).
- `REALM_DATA_PATH` is set and points to the directory containing OmniGibson datasets (mounted as /data in the container). `setup.sh --dataset` should maintain this variable automatically.
- Either `nc` or `ss` is available on the host (used to probe free ports).

Additional requirements depend on the chosen model and environment:

**For** `--model pi0` **or** `--model pi0_FAST`
- Singularity or Apptainer is installed.
- The following environment variables are set:
    ```
    export OPENPI_ROOT="/path/to/openpi/checkout"
    export OPENPI_SIF="/path/to/openpi.sif"
    ```
The checkpoint passed via `--ckpt-path` is mounted into the container at `/checkpoint`.

**For** `--model GR00T`
- The following environment variable is set:
    ```
    export GR00T_ROOT="/path/to/Isaac-GR00T"
    ```
The checkpoint passed via `--ckpt-path` is used as `--model_path`.

**For** `--environment docker`
- docker must be installed and available on $PATH.
- NVIDIA Omniverse EULA must be accepted:
    - Either set OMNIVERSE_EULA_ACCEPTED=1 in your environment, _or_
    - Run interactively and accept when the script prompts you.

## Usage

Basic usage:
```
./scripts/eval.sh -c /path/to/checkpoint [OPTIONS]
```
**Required:**
- `-c, --ckpt-path PATH`
    
    Host path to the model checkpoint.

**Task options:**
- `-t, --task-id ID` (default: `0`)
    
    Task ID in `[0–9]`:
    | ID | Task                         |
    |----|------------------------------|
    | 0  | put\_green\_block\_in\_bowl   |
    | 1  | put\_banana\_into\_box        |
    | 2  | rotate\_marker               |
    | 3  | rotate\_mug                  |
    | 4  | pick\_spoon                  |
    | 5  | pick\_water\_bottle          |
    | 6  | stack\_cubes                 |
    | 7  | push\_switch                 |
    | 8  | open\_drawer                 |
    | 9  | close\_drawer                |

**Perturbations:**
- `-p, --perturbation-id ID` (default: `0`)
    
    Perturbation ID in `[0–15]`:
    | ID | Perturbation |
    |----|--------------|
    | 0  | Default      |
    | 1  | V-AUG        |
    | 2  | V-VIEW       |
    | 3  | V-SC         |
    | 4  | V-LIGHT      |
    | 5  | S-PROP       |
    | 6  | S-LANG       |
    | 7  | S-MO         |
    | 8  | S-AFF        |
    | 9  | S-INT        |
    | 10 |  B-HOBJ      |
    | 11 |  SB-NOUN     |
    | 12 |  SB-VRB      |
    | 13 |  VB-POSE     |
    | 14 |  VB-MOBJ     |
    | 15 |  VSB-NOBJ    |

**Run configuration:**
- `-r, --repeats N` (default: `25`)
    
    Number of episodes.
- `-s, --max-steps N` (default: `500`)

    Maximum number of environment steps per episode before termination.

**Model selection:**
- `-m, --model MODEL` (default: `pi0`)
    
    Either:
    - One of the built-in models: `pi0`, `pi0_FAST`, `GR00T`, or
    - A path to an executable script that starts a model server.

    For a custom script, it will be called as:
    ```
    bash /path/to/custom_model.sh CKPT_PATH PORT
    ```
    where:
    - `CKPT_PATH` is the value from `--ckpt-path`
    - `PORT` is an auto-selected free TCP port on the host.

**Environment:**
- `-e, --environment ENV` (default: `singularity`)
    Where to run the evaluation:
    - `singularity` — uses Singularity/Apptainer image given by `REALM_SIF`
    - `docker` — uses the `stanfordvl/omnigibson:1.1.1` Docker image
    - `current` — runs in the current Python/OS environment

## Typical usage examples

From the project root:
```
# 1) Default pi0 evaluation on task 0 (put_green_block_in_bowl)
./scripts/eval.sh -c /path/to/pi0/checkpoint

# 2) GR00T evaluation on 'rotate_marker' (task 2), V-SC perturbation, 1 episode, 50 steps, via Docker
./scripts/eval.sh \
    -p 3 \
    -t 2 \
    -r 1 \
    -s 50 \
    -m GR00T \
    -c /path/to/gr00t/checkpoint \
    -e docker

# 3) pi0_FAST evaluation on 'stack_cubes' (task 6) with S-MO perturbation
./scripts/eval.sh \
    -p 7 \
    -t 6 \
    -m pi0_FAST \
    -c /path/to/pi0_FAST/checkpoint
```


# Benchmarking models in REALM
TBA

# Citation

If you use REALM or found our results useful for your research, please consider citing this work:
```
@article{sedlacek2025realm,
         title={TBA},
         author={TBA},
         journal = {arXiv preprint arXiv:TBA},
         year={2025}
}
```