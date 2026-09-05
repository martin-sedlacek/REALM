

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
    parser.add_argument("--robot", type=str, default="DROID", help="Robot type")
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

    # Task-progression scorer. Defaults are literals (not imports from realm.progress_scorer) so
    # --help stays free of the simulator imports that module pulls in.
    scoring = parser.add_argument_group(
        "scoring",
        "By default task_progression is the rubric's fraction of completed stages. --robometer "
        "replaces it with a Robometer reward model's progress estimate from a separate server "
        "(scripts/run_robometer_server.sh); see wiki/Robometer.md. Not comparable with rubric "
        "numbers -- use a distinct --experiment_name.")
    scoring.add_argument("--robometer", action="store_true",
                         help="Score task progression with a Robometer server instead of the rubric")
    scoring.add_argument("--robometer_host", type=str, default="127.0.0.1",
                         help="Robometer server host")
    scoring.add_argument("--robometer_port", type=int, default=8010,
                         help="Robometer server port (keep it distinct from --port)")
    scoring.add_argument("--robometer_success_threshold", type=float, default=1.0,
                         help="CALIBRATED Robometer progress at or above which a rollout counts as a "
                              "success (binary_SR, and the terminal countdown). 1.0 = the raw score "
                              "reached the task's calibrated ceiling")
    scoring.add_argument("--robometer_cameras", type=str, default="base,wrist",
                         help="Comma-separated cameras scored per query: base (the exterior view the "
                              "policy sees) and/or wrist")
    scoring.add_argument("--robometer_fusion", type=str, default="max", choices=["max", "min", "mean"],
                         help="How the cameras' raw scores are combined before calibration")
    scoring.add_argument("--robometer_calibration", type=str,
                         default="realm/config/robometer_calibration.yaml",
                         help="Per-task raw->0-1 calibration table (floor/ceiling per task). "
                              "Relative paths resolve against the repo root")
    scoring.add_argument("--robometer_frame_size", type=int, default=256,
                         help="Longest side, in pixels, of the frames sent to the server")
    scoring.add_argument("--robometer_max_frames", type=int, default=16,
                         help="Frames per query: the clip so far is linspace-subsampled to this many "
                              "(first and current frame kept). 16 is the model's training clip "
                              "length; 0 sends every frame (grows without bound)")

    return parser


def build_scorer(args):
    """The progress scorer the parsed flags ask for; None means the rubric (the default).

    Imports realm.progress_scorer lazily -- it pulls in omnigibson through realm.rollout -- and
    constructs the Robometer scorer BEFORE the simulator boots, so a server that is not up fails
    the run in seconds rather than after an Isaac start.
    """
    if not getattr(args, "robometer", False):
        return None
    import os

    from realm.progress_scorer import RobometerScorer

    calibration = args.robometer_calibration
    if not os.path.isabs(calibration):
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        calibration = os.path.join(repo_root, calibration)
    return RobometerScorer(
        host=args.robometer_host,
        port=args.robometer_port,
        success_threshold=args.robometer_success_threshold,
        frame_size=args.robometer_frame_size,
        max_frames=args.robometer_max_frames,
        cameras=tuple(c.strip() for c in args.robometer_cameras.split(",") if c.strip()),
        fusion=args.robometer_fusion,
        calibration=calibration,
    )


def run_evaluation_cli(*, vectorized=False):

    args = build_evaluation_parser(vectorized=vectorized).parse_args()

    # Heavy simulator imports stay behind argument parsing so --help remains lightweight.
    import omnigibson as og

    from realm.paths import run_log_dir

    scorer = build_scorer(args)

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
            scorer=scorer,
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
            scorer=scorer,
        )

    og.shutdown()
