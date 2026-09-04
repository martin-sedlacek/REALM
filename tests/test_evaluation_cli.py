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


def test_robometer_flags_default_off_on_both_parsers():
    # Off by default, so build_scorer returns None and the rubric path is untouched.
    from realm.evaluation_cli import build_scorer

    for vectorized in (False, True):
        args = build_evaluation_parser(vectorized=vectorized).parse_args(REQUIRED_ARGS)
        assert args.robometer is False
        assert args.robometer_host == "127.0.0.1"
        assert args.robometer_port == 8010
        assert args.robometer_success_threshold == 0.9
        assert args.robometer_frame_size == 256
        assert build_scorer(args) is None


def test_robometer_flags_parse():
    args = build_evaluation_parser().parse_args(REQUIRED_ARGS + [
        "--robometer", "--robometer_host", "node017", "--robometer_port", "9000",
        "--robometer_success_threshold", "0.8", "--robometer_frame_size", "224"])
    assert args.robometer is True
    assert args.robometer_host == "node017"
    assert args.robometer_port == 9000
    assert args.robometer_success_threshold == 0.8
    assert args.robometer_frame_size == 224
