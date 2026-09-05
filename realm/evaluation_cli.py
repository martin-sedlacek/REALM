

import argparse


def build_evaluation_parser(*, vectorized=False):

    parser = argparse.ArgumentParser(
        description="Vectorized REALM evaluation" if vectorized else "REALM evaluation"
    )

    if vectorized:
        parser.add_argument("--num_envs", type=int, default=4)

    parser.add_argument("--repeats", type=int, default=25 if vectorized else 5,
                        help="total rollouts, run in waves of num_envs" if vectorized else None)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--task_id", type=int, default=0)
    parser.add_argument("--perturbation_id", type=int, default=0)
    parser.add_argument("--task_cfg_path", type=str, default=None)
    parser.add_argument("--model_type", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Inference server host")
    parser.add_argument("--experiment_name", type=str, required=True)
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--log_dir", type=str, default="/logs" if vectorized else None)
    parser.add_argument("--robot", type=str, default="DROID_mounted", help="Robot type")
    parser.add_argument("--rendering_mode", type=str, default="rt" if vectorized else None,
                        help="OmniGibson rendering mode (pt, rt, r)")
    parser.add_argument("--multi-view", dest="multi_view", action="store_true",
                        help="Enable second external camera")
    parser.add_argument("--no_record", action="store_true", help="Do not record rollout videos")
    parser.add_argument(
        "--render_on_demand",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Render only when an observation feeds inference; pass "
             "--no-render_on_demand to render every step.",
    )

    if vectorized:
        parser.add_argument("--n_pre_obs_renders", type=int, default=2)
        parser.add_argument("--max_render_interval", type=int, default=8)
    else:
        parser.add_argument("--resume", action="store_true",
                            help="Resume from an existing run report")
        parser.add_argument("--no_render", action="store_true",
                            help="Disable rendering completely")

    return parser


def run_evaluation_cli(*, vectorized=False):

    args = build_evaluation_parser(vectorized=vectorized).parse_args()

    # Heavy simulator imports stay behind argument parsing so --help remains lightweight.
    import omnigibson as og

    from realm.paths import run_log_dir

    if vectorized:
        from realm.vector_eval import evaluate_vectorized

        log_dir = run_log_dir(args.log_dir, args.experiment_name, args.model_name,
                              args.run_id or "vec")
        evaluate_vectorized(
            num_envs=args.num_envs,
            task_id=args.task_id,
            perturbation_id=args.perturbation_id,
            repeats=args.repeats,
            max_steps=args.max_steps,
            horizon=args.horizon,
            model_type=args.model_type,
            model_name=args.model_name,
            port=args.port,
            host=args.host,
            log_dir=log_dir,
            rendering_mode=args.rendering_mode,
            robot=args.robot,
            multi_view=args.multi_view,
            no_record=args.no_record,
            task_cfg_path=args.task_cfg_path,
            render_on_demand=args.render_on_demand,
            n_pre_obs_renders=args.n_pre_obs_renders,
            max_render_interval=args.max_render_interval,
        )
    else:
        from realm.eval import evaluate

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
