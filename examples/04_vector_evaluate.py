"""Run a REALM evaluation with N rollouts in flight at once.

Same artifacts as examples/02_evaluate.py -- reports/*.csv plus qpos/, actions/, videos/ parquets --
so anything that reads a single-env run reads this too.

    python examples/04_vector_evaluate.py --num_envs 4 --repeats 25 --max_steps 500 \
        --task_id 0 --perturbation_id 0 --model_type openpi --model_name pi05 --port 8000

Inference is sequential: one policy call per member per action-chunk boundary, never batched.
Requires the OG-lite bind (MODE=oglite) -- the scene z-offset fix that makes scenes 1..N-1 usable
lives in the fork, not the image.
"""
from realm.evaluation_cli import run_evaluation_cli

if __name__ == "__main__":
    run_evaluation_cli(vectorized=True)
