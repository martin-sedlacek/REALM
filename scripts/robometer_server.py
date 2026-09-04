#!/usr/bin/env python
"""Robometer's eval server, with the checkpoint loaded through plain transformers instead of unsloth.

Run through scripts/run_robometer_server.sh (it needs robometer's own environment, not REALM's).
Hydra overrides pass straight through: `model_path=... server_port=... num_gpus=...`.

WHY THIS SHIM EXISTS
--------------------
Robometer-4B's training config says `use_unsloth: true`, and robometer's loader honours that at
inference time too: `FastVisionModel.from_pretrained(..., full_finetuning=True, dtype=bfloat16)`.
Current unsloth releases (2026.9 at the time of writing) keep every LayerNorm of the Qwen3-VL vision
tower in float32 under full finetuning, while the surrounding weights are bfloat16. Unsloth's own
patched forwards cope with that; the eval server calls the stock transformers forward on the
extracted inner model, which does not, and the first request dies in the vision tower with

    RuntimeError: expected scalar type BFloat16 but found Float

Unsloth is a training accelerator and the server only does inference, so the smallest fix is to
turn it off for loading: `robometer.utils.setup_utils.setup_model_and_processor` then takes its
standard `AutoModelForImageTextToText.from_pretrained(..., torch_dtype=bfloat16)` path, every
weight is bfloat16, and the same checkpoint files are loaded on top. Nothing in the submodule is
modified; the flag is flipped on the in-memory config just before the loader reads it.

Upstream has no switch for this (the eval config only carries model_path / GPUs / port), which is
why it is done here rather than on the command line. Revisit when the pinned robometer revision
grows one, or when its unsloth pin changes.
"""
import os

from hydra import main as hydra_main

import robometer.utils.setup_utils as setup_utils

_setup_model_and_processor = setup_utils.setup_model_and_processor


def setup_without_unsloth(cfg, *args, **kwargs):
    if getattr(cfg, "use_unsloth", False):
        print("[realm] robometer_server: loading with use_unsloth=False (inference only; see the "
              "module docstring of scripts/robometer_server.py)", flush=True)
        cfg.use_unsloth = False
    return _setup_model_and_processor(cfg, *args, **kwargs)


# save.load_model_from_hf imports the function lazily by name from this module, so patching the
# module attribute is enough for the server's load path; eval_server's own top-level import is
# patched too in case a later revision calls it directly.
setup_utils.setup_model_and_processor = setup_without_unsloth

import robometer.evals.eval_server as eval_server  # noqa: E402  (must follow the patch)

eval_server.setup_model_and_processor = setup_without_unsloth

# eval_server.main is decorated with @hydra_main(config_path="../configs"). Run as its own script
# that resolves on the filesystem; imported from here, hydra reads it as the package
# `robometer.configs`, which has no __init__.py, and refuses to start. Re-wrap the undecorated
# function with the configs directory as an absolute path instead.
CONFIGS_DIR = os.path.abspath(os.path.join(os.path.dirname(eval_server.__file__), "..", "configs"))
main = hydra_main(version_base=None, config_path=CONFIGS_DIR,
                  config_name="eval_config_server")(eval_server.main.__wrapped__)

if __name__ == "__main__":
    main()
