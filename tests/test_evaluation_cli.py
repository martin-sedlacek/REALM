from realm.evaluation_cli import build_evaluation_parser


REQUIRED_ARGS = [
    "--model_type", "openpi",
    "--model_name", "pi05",
    "--port", "8000",
    "--experiment_name", "test",
]


def test_single_evaluation_defaults():
    args = build_evaluation_parser().parse_args(REQUIRED_ARGS)

    assert args.repeats == 5
    assert args.log_dir is None
    assert args.rendering_mode is None
    assert args.resume is False
    assert args.no_render is False
    assert not hasattr(args, "num_envs")


def test_vector_evaluation_defaults():
    args = build_evaluation_parser(vectorized=True).parse_args(REQUIRED_ARGS)

    assert args.num_envs == 4
    assert args.repeats == 25
    assert args.log_dir == "/logs"
    assert args.rendering_mode == "rt"
    assert args.n_pre_obs_renders == 2
    assert args.max_render_interval == 8
    assert not hasattr(args, "resume")
    assert not hasattr(args, "no_render")
