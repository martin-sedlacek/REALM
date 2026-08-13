"""Run a REALM evaluation with N rollouts in flight at once.

Same artifacts as examples/02_evaluate.py -- reports/*.csv plus qpos/, actions/, videos/ parquets --
so anything that reads a single-env run reads this too.

    python examples/04_vector_evaluate.py --num_envs 4 --repeats 25 --max_steps 500 \
        --task_id 0 --perturbation_id 0 --model_type openpi --model_name pi05 --port 8000

Inference is sequential: one policy call per member per action-chunk boundary, never batched.
Requires the OG-lite bind (MODE=oglite) -- the scene z-offset fix that makes scenes 1..N-1 usable
lives in the fork, not the image.
"""
import argparse

import omnigibson as og

from realm.vector_eval import evaluate_vectorized

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Vectorized REALM evaluation")
    p.add_argument("--num_envs", type=int, default=4)
    p.add_argument("--repeats", type=int, default=25, help="total rollouts, run in waves of num_envs")
    p.add_argument("--max_steps", type=int, default=500)
    p.add_argument("--horizon", type=int, default=8)
    p.add_argument("--task_id", type=int, default=0)
    p.add_argument("--perturbation_id", type=int, default=0)
    p.add_argument("--task_cfg_path", type=str, default=None)
    p.add_argument("--model_type", type=str, required=True)
    p.add_argument("--model_name", type=str, required=True)
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--experiment_name", type=str, required=True)
    p.add_argument("--run_id", type=str, default=None)
    p.add_argument("--log_dir", type=str, default="/logs")
    p.add_argument("--robot", type=str, default="DROID")
    p.add_argument("--rendering_mode", type=str, default="rt")
    p.add_argument("--multi-view", dest="multi_view", action="store_true")
    p.add_argument("--no_record", action="store_true")
    a = p.parse_args()

    run_id = a.run_id or "vec"
    log_dir = f"{a.log_dir}/{a.experiment_name}/{a.model_name}/{run_id}"

    evaluate_vectorized(
        num_envs=a.num_envs, task_id=a.task_id, perturbation_id=a.perturbation_id,
        repeats=a.repeats, max_steps=a.max_steps, horizon=a.horizon,
        model_type=a.model_type, model_name=a.model_name, port=a.port, host=a.host,
        log_dir=log_dir, rendering_mode=a.rendering_mode, robot=a.robot,
        multi_view=a.multi_view, no_record=a.no_record, task_cfg_path=a.task_cfg_path,
    )
    og.shutdown()
