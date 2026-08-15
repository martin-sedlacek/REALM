"""Single-environment REALM evaluation: `repeats` rollouts, one after another, in one simulator.

`realm/vector_eval.py` is the concurrent counterpart -- `num_envs` rollouts stepped together -- and
writes the same artifacts, so downstream tooling does not care which path produced a run. Everything
the two must agree on lives in `realm/rollout.py`.

`SUPPORTED_TASKS` and `SUPPORTED_PERTURBATIONS` must stay top-level list literals in this module:
tests/test_vector_integrity.py reads them with `ast.parse` rather than importing this file, which
would boot an Isaac instance in the test driver just to read two lists of strings.
"""
import csv
import datetime
import os
import time

import omnigibson as og

from realm.environments.env_dynamic import RealmEnvironmentDynamic
from realm.inference import InferenceClient
from realm.realm_logging import VideoRecorder, save_results
from realm.rollout import (
    RenderSchedule,
    Rollout,
    build_result_entry,
    gripper_is_inverted,
    resolve_task,
    write_rollout_artifacts,
)
from realm.sim_config import set_sim_config

CONFIG_ROOT = "/app/realm/config"

# Index into each list is the --task_id / --perturbation_id the entry points take.
SUPPORTED_TASKS = [
    "put_green_block_into_bowl",  # 0
    "put_banana_into_box",        # 1
    "rotate_marker",              # 2
    "rotate_mug",                 # 3
    "pick_spoon",                 # 4
    "pick_water_bottle",          # 5
    "stack_cubes",                # 6
    "push_switch",                # 7
    "open_drawer",                # 8
    "close_drawer",               # 9
]

SUPPORTED_PERTURBATIONS = [
    "Default",   # 0
    "V-AUG",     # 1
    "V-VIEW",    # 2
    "V-SC",      # 3
    "V-LIGHT",   # 4
    "S-PROP",    # 5
    "S-LANG",    # 6
    "S-MO",      # 7
    "S-AFF",     # 8
    "S-INT",     # 9
    "B-HOBJ",    # 10
    "SB-NOUN",   # 11
    "SB-VRB",    # 12
    "VB-POSE",   # 13
    "VB-MOBJ",   # 14
    "VSB-NOBJ",  # 15
]


def evaluate(
        task_id=0,
        perturbation_id=0,
        repeats=1,
        max_steps=500,
        horizon=8,
        model_type="pi0_FAST",
        port=8000,
        host="127.0.0.1",
        log_dir="/app/logs",
        resume=False,
        multi_view=False,
        no_record=False,
        no_render=False,
        rendering_mode=None,
        task_cfg_path=None,
        robot="DROID",
        render_on_demand=True,
        n_pre_obs_renders=2,
        max_render_interval=8,
):
    """Evaluate one policy on one (task, perturbation) for `repeats` rollouts.

    Writes `reports/{task}_{perturbation}.csv` plus `{qpos,actions,videos}/{task}.parquet` under
    `log_dir`, rewriting the report in full after every rollout so a run that dies part way still
    leaves a readable prefix -- which is what `resume` picks up from.

    Repeats deliberately share the single seed set by `set_sim_config()`; they are not reseeded per
    run, so each repeat continues the same RNG stream and diverges naturally.

    `model_type` selects the inference client and the policy's gripper convention. It is taken
    verbatim and never inferred from the model's name.
    """
    start = time.perf_counter()
    og.log.info(f"DEBUG: Begin eval: {time.perf_counter() - start:.4f}s")
    if rendering_mode is None:
        rendering_mode = "rt"
    set_sim_config(robot=robot)

    task, task_cfg_path = resolve_task(task_id, task_cfg_path, SUPPORTED_TASKS,
                                       name_includes_config=True)
    perturbation = SUPPORTED_PERTURBATIONS[perturbation_id]
    os.makedirs(log_dir, exist_ok=True)

    client = InferenceClient(model_type, host=host, port=port)
    og.log.info(f"DEBUG: Client connected: {time.perf_counter() - start:.4f}s")

    env = RealmEnvironmentDynamic(
        config_path=CONFIG_ROOT,
        task_cfg_path=task_cfg_path,
        perturbations=[perturbation],
        multi_view=multi_view,
        no_rendering=no_render,
        rendering_mode=rendering_mode,
        robot=robot,
    )
    og.log.info(f"DEBUG: Env created: {time.perf_counter() - start:.4f}s")

    if resume:
        results, first_run_id, results_filename = _load_previous_results(log_dir, task, perturbation)
    else:
        results, first_run_id, results_filename = [], 0, None

    for run_id in range(first_run_id, repeats):
        rollout = Rollout(
            env,
            run_id,
            recorder=None if no_record else _new_recorder(log_dir, run_id, task, perturbation),
            gripper_inverted=gripper_is_inverted(model_type),
        )
        _run_rollout(rollout, client, max_steps, horizon,
                     render_on_demand, n_pre_obs_renders, max_render_interval)
        og.log.info(f"DEBUG: Run finished: {time.perf_counter() - start:.4f}s")

        entry = build_result_entry(rollout, task, perturbation, model_type)
        write_rollout_artifacts(log_dir, task, perturbation, rollout, entry)
        results.append(entry)

        client.reset()
        results_filename = save_results(results, log_dir + "/reports", task, perturbation,
                                        filename=results_filename)

    save_results(results, log_dir + "/reports", task, perturbation)
    og.log.info("Done!")
    og.log.info(f"DEBUG: Done: {time.perf_counter() - start:.4f}s")


def _run_rollout(rollout, client, max_steps, horizon,
                 render_on_demand, n_pre_obs_renders, max_render_interval):
    """Reset, warm up and step one rollout to its end.

    Ends once the task has been complete for `realm.rollout.TERMINAL_STEPS` control steps, or at
    `max_steps`, whichever comes first.
    """
    env = rollout.env
    obs, _ = env.reset()
    obs, _, _, _, _ = env.warmup(obs)

    renders = RenderSchedule(max_render_interval, n_pre_obs_renders)
    step = 0
    while step < max_steps and rollout.active:
        command = rollout.next_command(obs, client, horizon, renders.obs_is_fresh)

        if render_on_demand:
            render, n_render_iterations = renders.schedule(rollout.needs_fresh_obs())
            with og.sim.render_on_step(render):
                obs, task_progression, _, _, _ = env.step(
                    command, n_render_iterations=n_render_iterations)
        else:
            obs, task_progression, _, _, _ = env.step(command)

        rollout.record_progression(task_progression, step)
        step += 1


def _new_recorder(log_dir, run_id, task, perturbation):
    """A video recorder for one rollout, named after the wall-clock time the rollout started."""
    timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H:%M:%S")
    return VideoRecorder(log_dir, timestamp, run_id, task, perturbation)


def _load_previous_results(log_dir, task, perturbation):
    """Rows already written for this (task, perturbation), so `--resume` can skip those rollouts.

    Returns (results so far, first run_id still to do, report path to keep writing to). The report
    is the only record of how far a previous run got.
    """
    report = os.path.join(log_dir, "reports", f"{task}_{perturbation}.csv")
    if not os.path.exists(report):
        og.log.info("Resume requested but no report found. Starting fresh.")
        return [], 0, None

    with open(report, "r") as report_file:
        results = list(csv.DictReader(report_file))
    og.log.info(f"Resuming run from repeat {len(results)}. Using file: {report}")
    return results, len(results), report
