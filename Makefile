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
		--exclude *.png \
	 	sedlam56@login01.clara.ciirc.cvut.cz:/home/sedlam56/projects/REALM/logs/ ./logs/

#		--exclude *.npy \