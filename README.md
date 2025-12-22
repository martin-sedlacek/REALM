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


# Example Workflow: Pi0 evaluation
> ⚠️ This example is provided for a single evaluation run on local hardware using an NVIDIA GPU with at least 16GB of VRAM. 
> This is required to run both the VLA model and underlying isaacsim on the same card.


1. Setup pi0 from openpi (https://github.com/Physical-Intelligence/openpi):
```
git clone https://github.com/Physical-Intelligence/openpi.git
cd openpi
uv sync
```
The add s3 to the uv environment and add (or create) your AWS credentials:
```
uv add s3fs
```
Run the model:
```
XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi0_fast_droid_jointpos --policy.dir=s3://openpi-assets-simeval/pi0_fast_droid_jointpos
```
> ❗ Set XLA_PYTHON_CLIENT_MEM_FRACTION such that you have at least 8GB+ free on the GPU for isaacsim.

> ⚠️ In general, make sure you are using models that output **absolute joint configurations** as REALM currently expects action to be in this format.

2. From REALM project root, open the containerized environment:
```
# [RECOMMENDED OPTION] Docker:
source ./scripts/run_docker.sh

# [UNSTABLE] Apptainer:
source ./scripts/run_apptainer.sh
```

3. Inside the container run:
```
python /app/examples/01_pi0_eval.py
```

This should produce a rollout video and a report numpy file with the evaluation results in logs.

# Benchmarking models in REALM

Instructions on using REALM for benchmarking custom models and how to systematically test on all tasks and 
preturbations will be provided soon.

# 🚧 Roadmap
- [x] Streamlined installation
- [x] Example scripts for getting started
- [ ] Improved benchmarking UX:
  - [ ] End-to-end scripts for producing result plots and tables
- [ ] Extended documentation
- [ ] Performance:
  - [ ] Support vectorized environments
  - [ ] Improve parallelism and overall execution speed


# Acknowledgments and Licensing
We build on top of essential simulation tooling and the dataset from BEHAVIOR-1K and adhere to their licensing and terms of usage. 
For more information, please see https://behavior.stanford.edu/.

This work was supported by the European Union's Horizon Europe projects AGIMUS (No. 101070165), euROBIN (No. 101070596), 
ERC FRONTIER (No. 101097822), and ELLIOT (No. 101214398). Pavlo Yefanov (PY) and Georgy Ponimatkin (GP) were also partly 
supported by Grant Agency of the Czech Technical University in Prague under allocations SGS25/158/OHK3/3T/13 (PY) and 
SGS25/156/OHK3/3T/13 (GP). Martin Sedlacek was partly supported by the ELLIS Unit Amsterdam as part of the MSc Honours Programme. 
Compute resources and infrastructure were supported by the Ministry of Education, Youth and Sports of the Czech Republic 
through the e- INFRA CZ (ID:90254) and by the European Union's Horizon Europe project CLARA (No. 101136607).

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