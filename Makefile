# ==================================================================================================
# THE TESTING PIPELINE
#
# Two tiers. The boundary is "does this need Isaac, the ~13 GB container and a GPU".
#
#   TIER 1 -- static. No container, no GPU, no dataset. Runs locally in ~1 s. (No CI workflow
#             invokes it yet -- .github/workflows/ holds only release.yml; wiring tier 1 into a
#             push-time workflow is an open task.)
#
#               make lint          ruff, with .ruff.toml's deliberately narrow ruleset
#               make test-static   the container-free tests
#               make check         both of the above
#
#   TIER 2 -- GPU. Needs the image, the dataset and a card. Runs by hand on Clara against a held
#             Slurm allocation, and is written so a self-hosted GitHub runner could invoke the
#             SAME entry point unchanged (no such runner is registered yet, and no workflow file
#             exists for it -- that is a decision for Martin).
#
#               ALLOC=<jobid> make test-smoke    ~12 min   the cheap gate
#               ALLOC=<jobid> make test-suite    ~1.7 h    the gate before trusting a change
#               ALLOC=<jobid> make test-matrix   hours     the task x perturbation sweep
#               ALLOC=<jobid> make test-server             needs a policy server on :8000
#
#   Either tier:  make test-list     what is in the suite, what each member needs
#                 make test-report   re-print the last run's table, running nothing
#
# `make test` runs TIER 1 ONLY and prints what it skipped. It is not the suite. See below.
#
# WHAT COUNTS AS A PASS, AND WHAT DOES NOT
# ----------------------------------------
# Never an exit code. Isaac's shutdown hard-exits 0 on an unhandled exception and can segfault at
# teardown after a pass, so a child's status carries no information; run_suite.py records it and
# does not gate on it. Two things are gateable:
#
#   * `--strict` (every target below passes it) makes the DRIVER's status mean "every test I ran
#     ended PASS or SKIP".
#   * `--junit-xml` writes a JUnit report ONCE, after the last test. Its ABSENCE means the driver
#     itself died -- an OOM, a walltime kill, a node failure -- which no exit code can tell you.
#     This is upstream BEHAVIOR-1K's gate (their .github/workflows/tests.yml); adopt the same
#     grep-the-XML pattern in any future CI workflow here.
#
# ONE PYTEST FILE, OTHERWISE SCRIPTS. tests/test_perturbation_task_types.py is a real pytest
# module (and host-safe: it delays its omnigibson import into fixtures) -- run it directly with
# `pytest tests/test_perturbation_task_types.py`. Every OTHER test_*.py is a standalone script
# with a printed verdict, driven by run_suite.py. Do NOT run `pytest tests/`: collection imports
# every module, and several boot a full Isaac instance at import time just to be collected.
#
# `make test-static` and `make lint` are both expected GREEN (re-measured 2026-08-18: lint is at
# zero findings, and the rubrics test passes since the POUR key/signature repair). A finding in
# either is a real regression, not "the repository's known state".
#
# AND EXPECT `make test-smoke` / `make test-suite` TO REPORT A FAILURE AT THE DEFAULT MODE=stock.
# test_scene_object_placement is MODE-sensitive BY DESIGN: it is the only test that looks at the
# SCENE, and the v2 image lacks the up-axis fix, so a drawer scene really is wrong under stock.
# Measured on job 191496 (2026-08-16): FAIL at stock in 428.5 s; PASS at oglite. Run
# `SUITE_MODE=oglite` when the scene has to be right. Do NOT loosen that test's tolerance.
# Coverage, and the gaps that matter more than either: the wiki's Test coverage page.
#
# Knobs: SUITE_OUT= (results JSON), SUITE_XML= (JUnit report), SUITE_MODE= (stock/stockfix/oglite),
#        SUITE_ARGS= (passed through), RUFF= (ruff binary), PYTHON=.
# ==================================================================================================

SUITE      ?= tests/run_suite.py
SUITE_OUT  ?= tmp/suite/results.json
SUITE_XML  ?= tmp/suite/results.xml
SUITE_MODE ?= stock
SUITE_ARGS ?=
PYTHON     ?= python3
RUFF       ?= ruff

# Every GPU target starts with this. `rr` starts the container WHEREVER it is invoked, so without
# an allocation these would run on the login node, get no GPU, and fail confusingly.
define require_alloc
	@[ -n "$(ALLOC)" ] || { \
	  echo "ERROR: $@ needs a RUNNING Slurm allocation. Set ALLOC=<jobid>:"; \
	  echo "         ALLOC=191496 make $@"; \
	  echo "       squeue -u \$$USER          to find one"; \
	  echo "       salloc --no-shell --partition=l40s --gres=gpu:L40S:1 ...   to get one"; \
	  echo "       'make check' needs no allocation, but covers tier 1 only."; \
	  exit 1; }
endef

.PHONY: check lint test test-static test-smoke test-suite test-matrix test-server \
        test-list test-report

# --- tier 1 ---------------------------------------------------------------------------------------

check: ## Tier 1 in full: lint + the container-free tests. Runs BOTH, then fails if either did.
	@rc=0; \
	 $(MAKE) --no-print-directory lint || rc=1; \
	 echo ""; \
	 $(MAKE) --no-print-directory test-static || rc=1; \
	 echo ""; \
	 if [ $$rc -ne 0 ]; then \
	   echo "make check: one or more tier-1 checks failed (see above)."; \
	   echo "  On this branch BOTH are expected to: 25 ruff findings (baseline in .ruff.toml),"; \
	   echo "  and 2 real defects reported by test_task_progression_rubrics. Neither is your"; \
	   echo "  install. See the wiki's Test coverage page."; \
	 else \
	   echo "make check: tier 1 clean."; \
	 fi; \
	 exit $$rc
	@# NOT `check: lint test-static`. As prerequisites, a lint failure aborts the target and the
	@# test result is never printed -- so the second half of tier 1 would silently stop being run
	@# the moment the first half went red, which on this branch is always.

lint: ## ruff, using .ruff.toml (F401/F811 only -- see that file before widening)
	@command -v $(RUFF) >/dev/null 2>&1 || { \
	  echo "ruff is not installed. It is a developer tool, not a REALM dependency:"; \
	  echo "  pip install ruff     (or: pipx install ruff)"; \
	  echo "CI installs it per-run; nothing in REALM imports it."; exit 1; }
	$(RUFF) check realm examples tests scripts

test: ## Tier 1 only (the suite's 2 container-free entries), then print what it skipped
	@echo "======================================================================================"
	@echo " make test runs TIER 1 ONLY: 2 of the suite's 14 entries."
	@echo ""
	@echo " NOT RUN -- needs a GPU, the ~13 GB image and the ~36 GB dataset:"
	@echo "   test_joint_reset_batching        joint-reset scheduling, in-container"
	@echo "   test_single_task / _drawer       one task end to end"
	@echo "   test_scene_object_placement      the only test that looks at the SCENE"
	@echo "   test_integrity                   all 10 tasks"
	@echo "   test_perturbations_integrity     all 16 perturbations"
	@echo "   test_vector_integrity_*          the vectorized matrix (4 configurations)"
	@echo "   test_pi0_integration             needs a live policy server on :8000"
	@echo ""
	@echo " No scene is loaded and no rollout is run below. For tier 2:"
	@echo "   ALLOC=<slurm jobid> make test-smoke     (~12 min)"
	@echo "   ALLOC=<slurm jobid> make test-suite     (~1.7 h)"
	@echo "======================================================================================"
	@echo ""
	@$(MAKE) --no-print-directory test-static

test-static: ## The container-free tests (no GPU, no allocation, no container)
	$(PYTHON) $(SUITE) --only local --strict \
	    --out $(SUITE_OUT) --junit-xml $(SUITE_XML) $(SUITE_ARGS)

# --- tier 2 ---------------------------------------------------------------------------------------

test-smoke: ## ~12 min. Static + scheduling + one task end to end + the scene check at num_envs=2
	$(require_alloc)
	$(PYTHON) $(SUITE) --jobid $(ALLOC) --mode $(SUITE_MODE) --level smoke --strict \
	    --out $(SUITE_OUT) --junit-xml $(SUITE_XML) $(SUITE_ARGS)

test-suite: ## ~1.7 h. Every task, every perturbation, both drawer paths. Re-runs drawers at oglite.
	$(require_alloc)
	$(PYTHON) $(SUITE) --jobid $(ALLOC) --mode $(SUITE_MODE) --level suite --strict \
	    --out $(SUITE_OUT) --junit-xml $(SUITE_XML) $(SUITE_ARGS)
	@echo ""
	@echo "--- re-running the OG-lite-sensitive cells at MODE=oglite ------------------------------"
	@echo "The v2 image lacks the up-axis fix, so at MODE=stock a drawer scene can be physically"
	@echo "wrong while every artifact check passes. The JSON and the table record mode per result."
	$(PYTHON) $(SUITE) --jobid $(ALLOC) --mode oglite --strict \
	    --only test_single_task_drawer,test_vector_integrity_drawers,test_scene_object_placement \
	    --out $(SUITE_OUT) $(SUITE_ARGS)

test-matrix: ## Hours. The full task x perturbation sweep through the vector path.
	$(require_alloc)
	$(PYTHON) $(SUITE) --jobid $(ALLOC) --mode $(SUITE_MODE) --level matrix --strict \
	    --out $(SUITE_OUT) --junit-xml $(SUITE_XML) $(SUITE_ARGS)

test-server: ## The tier that needs a live policy server on :8000
	$(require_alloc)
	$(PYTHON) $(SUITE) --jobid $(ALLOC) --mode $(SUITE_MODE) --only server --strict \
	    --out $(SUITE_OUT) --junit-xml $(SUITE_XML) $(SUITE_ARGS)

# --- either tier ----------------------------------------------------------------------------------

test-list: ## List the suite: tier, whether each member needs a GPU or a policy server
	$(PYTHON) $(SUITE) --list

test-report: ## Re-print the table from the last run's JSON. Runs nothing.
	$(PYTHON) $(SUITE) --report --out $(SUITE_OUT)

# ==================================================================================================

put_karolina: ## Put source to remote source
	rsync -av \
	    --exclude /.venv \
	    --exclude-from .gitignore \
	    --exclude /.git \
	    --exclude /singularity \
	    --exclude /logs \
	    --exclude /OmniGibson \
        --exclude /datasets \
        --exclude /data \
	    --exclude /.idea \
	    --exclude /.claude \
	    --exclude /isaac-sim \
		. sedlam@karolina.it4i.cz:/scratch/project/open-34-32/sedlam/projects/REALM

get_logs_karolina:
	rsync -av \
		--exclude slurm-* \
		--exclude *.npy \
		--exclude *.png \
		--exclude 'appdata/' \
	 	sedlam@karolina.it4i.cz:/scratch/project/open-34-32/sedlam/projects/REALM/logs/ ./logs/

run_interactive_karolina:
	salloc -A OPEN-34-32 -p qgpu_exp --gpus 1 -t 60

# ========================================================================================================

put_clara: ## Put source to remote source
	rsync -av \
	    --exclude-from .gitignore \
	    --exclude /.git \
	    --exclude /.venv \
	    --exclude '*.sif' \
	    --exclude '*.zip' \
	    --exclude '*/__pycache__' \
	    --exclude 'slurm-*.out' \
	    --exclude hf_cache/ \
	    --exclude pip_cache/ \
	    --exclude mamba_cache/ \
	    --exclude /singularity \
	    --exclude /logs \
	    --exclude /OmniGibson \
	    --exclude datasets/ \
	    --exclude data/ \
	    --exclude tmp/ \
	    --exclude real2sim_perf_data/ \
	    --exclude /.idea \
	    --exclude /.claude \
		. sedlam56@login01.clara.ciirc.cvut.cz:/home/sedlam56/projects/REALM

put_sif_clara: ## Put the og391 Apptainer image to clara (put_clara excludes *.sif)
	rsync -av --partial --info=progress2 \
		realm_og391.sif sedlam56@login01.clara.ciirc.cvut.cz:/home/sedlam56/projects/REALM/

# The BEHAVIOR-1K 3.9.1 dataset (behavior-1k-assets/ + omnigibson-robot-assets/) replaces the 1.1.1
# assets/ + og_dataset/ layout. It goes to datasets_og391/ so the existing datasets/ -- still needed
# by the 1.1.1 image -- is left alone. No --delete, deliberately.
OG391_DATASET ?= ../BEHAVIOR-1K/docker/behavior_docker_data/datasets
put_dataset_clara: ## Put the BEHAVIOR-1K 3.9.1 dataset to clara (alongside the 1.1.1 one)
	rsync -a --partial --info=progress2 \
		$(OG391_DATASET)/ sedlam56@login01.clara.ciirc.cvut.cz:/home/sedlam56/projects/REALM/data/datasets_og391/

get_clara: ## Get source from remote source
	rsync -av \
	    --exclude-from .gitignore \
	    --exclude .git \
	    --exclude .venv \
	    --exclude '*/__pycache__' \
	    --exclude 'slurm-*.out' \
	    --exclude hf_cache/ \
	    --exclude pip_cache/ \
	    --exclude mamba_cache/ \
	    --exclude singularity \
	    --exclude logs \
	    --exclude OmniGibson \
	    --exclude datasets/ \
	    --exclude data/ \
	    --exclude real2sim_perf_data/ \
	    --exclude .idea \
	    --exclude .claude \
	    --exclude tmp \
		sedlam56@login01.clara.ciirc.cvut.cz:/home/sedlam56/projects/REALM/ .

get_logs_clara:
	rsync -av \
		--exclude slurm-* \
		--exclude *.log \
		--exclude 'appdata/' \
	 	sedlam56@login01.clara.ciirc.cvut.cz:/home/sedlam56/projects/REALM/logs/ ./logs/

#		--exclude *.npy \