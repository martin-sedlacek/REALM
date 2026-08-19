import argparse
from realm.eval import evaluate
from realm.paths import run_log_dir
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
    parser.add_argument('--render_on_demand', action=argparse.BooleanOptionalAction, default=True,
                        help='Render only on steps whose observation feeds inference; run physics '
                             'only on the rest. Native in OG 3.9.1 (og.sim.render_on_step); this is '
                             'what OG-lite used to provide. DEFAULT ON: it roughly halves the median '
                             'step time (140 -> 79 ms measured on put_banana_into_box / DROID_robolab). '
                             'Pass --no-render_on_demand to render every step, which is what you want '
                             'when the recorded video matters: on-demand rendering drops the mp4 to '
                             '~1 frame per action chunk (~39 frames per 300 steps instead of 300).')
    args = parser.parse_args()

    assert args.model_name is not None
    assert args.model_type is not None
    assert args.experiment_name is not None
    #assert not (args.task_cfg_path and args.task_id), f"Either task --task_cfg_path or --task_id should be specified, but not both."

    # The /app/logs default only works when the container binds a real directory there. Under the
    # clara `rr` binds it is a DANGLING SYMLINK and the first makedirs dies -- cluster scripts pass
    # --log_dir explicitly; see tests/_paths.scratch_log_root for the measured failure.
    log_root = args.log_dir if args.log_dir is not None else "/app/logs"
    log_dir = run_log_dir(log_root, args.experiment_name, args.model_name, args.run_id)

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
        robot=args.robot,
        render_on_demand=args.render_on_demand,
    )
    og.shutdown()
    sys.exit(0)
