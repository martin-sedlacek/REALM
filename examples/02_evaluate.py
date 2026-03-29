import argparse
from realm.eval import evaluate
import sys

import omnigibson as og

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="dynamic sim evals")
    parser.add_argument('--perturbation_id', type=int, required=False, default=0)
    parser.add_argument('--task_id', type=int, required=False, default=0)
    parser.add_argument('--repeats', type=int, required=False, default=5)
    parser.add_argument('--max_steps', type=int, required=False, default=500)
    parser.add_argument('--horizon', type=int, required=False, default=8)
    parser.add_argument('--task_cfg_path', type=str, required=False, default=None)
    parser.add_argument('--model_name', type=str, required=True, default=None)
    parser.add_argument('--model_type', type=str, required=True, default=None)
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--host', type=str, required=False, default="127.0.0.1", help='Inference server host')
    parser.add_argument('--experiment_name', type=str, required=True)
    parser.add_argument('--run_id', type=str, required=False, default=None)
    parser.add_argument('--log_dir', type=str, required=False, default=None)
    parser.add_argument('--rendering_mode', type=str, required=False, default=None, help='Omnigibson rendering mode (pt, rt, r)')
    parser.add_argument('--multi-view', action='store_true', help='Enable second external camera')
    parser.add_argument('--resume', action='store_true', help='Resume from existing run report if found')
    parser.add_argument('--no_record', action='store_true', help='Do not record videos from runs.')
    parser.add_argument('--no_render', action='store_true', help='Disable rendering completely')
    parser.add_argument('--robot', type=str, required=False, default="DROID", help='Robot type')
    args = parser.parse_args()

    assert args.model_name is not None
    assert args.model_type is not None
    assert args.experiment_name is not None
    #assert not (args.task_cfg_path and args.task_id), f"Either task --task_cfg_path or --task_id should be specified, but not both."

    log_dir = args.log_dir if args.log_dir is not None else "/app/logs"
    log_dir += f"/{args.experiment_name}"
    log_dir += f"/{args.model_name}"
    log_dir += f"/{args.run_id}" if args.run_id is not None else ""

    evaluate(
        task_id=args.task_id,
        perturbation_id=args.perturbation_id,
        repeats=args.repeats,
        max_steps=args.max_steps,
        horizon=args.horizon,
        model_type=args.model_type,
        port=args.port,
        host=args.host,
        log_dir=log_dir,
        multi_view=args.multi_view,
        resume=args.resume,
        no_record=args.no_record,
        no_render=args.no_render,
        rendering_mode=args.rendering_mode,
        task_cfg_path=args.task_cfg_path,
        robot=args.robot
    )
    og.shutdown()
    sys.exit(0)
