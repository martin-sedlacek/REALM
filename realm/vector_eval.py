"""Vectorized REALM evaluation: N rollouts in one simulator, stepped together.

`realm/eval.py:evaluate()` is single-env -- one rollout at a time, one Isaac boot amortised over
`repeats` sequential rollouts. This runs `num_envs` rollouts concurrently inside one
`RealmVectorEnvironment`, in waves of `num_envs`, and writes the same artifacts
(`reports/*.csv`, `qpos/`, `actions/`, `videos/` parquets) so downstream tooling does not care which
path produced them. Metric definitions, termination, the gripper convention and the drops correction
come from `realm/rollout.py`, which both paths share.

Deliberately NOT batched inference: the policy is called once per member per chunk boundary, in a
loop. Batching is a separate change and would hide desync bugs behind a fixed-shape batch.

**Members desync.** Rollouts end at different times -- one succeeds at step 180, another runs to
`max_steps`. `og.sim.step()` advances every scene regardless, so a finished member cannot simply
stop. It is marked inactive, its result is finalised immediately, and it keeps receiving its last
action as a hold command while the others run on. Nothing about an inactive member is recorded.
"""
import datetime
import time

import numpy as np
import omnigibson as og

from realm.environments.env_vector import RealmVectorEnvironment
from realm.eval import SUPPORTED_PERTURBATIONS, SUPPORTED_TASKS
from realm.inference import InferenceClient
from realm.realm_logging import VideoRecorder, save_results
from realm.rollout import (
    RenderSchedule,
    Rollout,
    build_result_entry,
    resolve_task,
    write_rollout_artifacts,
)
from realm.sim_config import set_sim_config


def evaluate_vectorized(
        num_envs=4, task_id=0, perturbation_id=0, repeats=25, max_steps=500, horizon=8,
        model_type="openpi", model_name="model", port=8000, host="127.0.0.1",
        log_dir="/logs", rendering_mode="rt", robot="DROID", multi_view=False,
        no_record=False, task_cfg_path=None,
        render_on_demand=True, n_pre_obs_renders=2, max_render_interval=8,
):
    """Run `repeats` rollouts of one (task, perturbation) in waves of `num_envs` concurrent members.

    Returns the result rows, and writes the same four artifacts as the single-env path. The report
    is rewritten after every wave, so a run that dies part way still leaves a readable prefix.

    `model_name` is part of the CLI surface (it names the log directory) and is not used here.
    """
    start = time.perf_counter()
    set_sim_config(robot=robot)

    task, task_cfg_path = resolve_task(task_id, task_cfg_path, SUPPORTED_TASKS,
                                       name_includes_config=False)
    perturbation = SUPPORTED_PERTURBATIONS[perturbation_id]

    print(f"[vec_eval] building {num_envs} environments...", flush=True)
    vec_env = RealmVectorEnvironment(
        num_envs, task_cfg_path=task_cfg_path, perturbations=[perturbation],
        robot=robot, rendering_mode=rendering_mode, multi_view=multi_view,
    )
    print(f"[vec_eval] {num_envs} envs ready at {time.perf_counter() - start:.1f}s", flush=True)

    # One client per member. Inference is still strictly sequential -- this only avoids sharing one
    # websocket's state across concurrent rollouts.
    clients = [InferenceClient(model_type, port, host) for _ in range(num_envs)]

    results, results_filename = [], None
    run_id = 0
    wave = 0
    while run_id < repeats:
        n_record = min(num_envs, repeats - run_id)
        wave += 1
        print(f"[vec_eval] wave {wave}: rollouts {run_id}..{run_id + n_record - 1} "
              f"({n_record} of {num_envs} members recorded)", flush=True)

        members, step_results = _start_wave(vec_env, clients, run_id, n_record, log_dir, task,
                                            perturbation, no_record)
        wave_start = time.perf_counter()
        steps = _step_wave(vec_env, clients, members, step_results, max_steps, horizon,
                           render_on_demand, n_pre_obs_renders, max_render_interval)
        print(f"[vec_eval] wave {wave} stepped {steps} times in "
              f"{time.perf_counter() - wave_start:.1f}s", flush=True)

        for member in members:
            if member is None:
                continue
            entry = build_result_entry(member, task, perturbation, model_type)
            write_rollout_artifacts(log_dir, task, perturbation, member, entry)
            results.append(entry)
            print(f"[vec_eval]   run {member.run_id}: SR={entry['binary_SR']} "
                  f"TP={entry['task_progression']} stage={entry['stage']} "
                  f"steps={member.metrics.steps} col_env={entry['collisions_env']}", flush=True)

        results_filename = save_results(results, log_dir + "/reports", task, perturbation,
                                        filename=results_filename)
        run_id += n_record

    sr = float(np.mean([r["binary_SR"] for r in results]))
    tp = float(np.mean([r["task_progression"] for r in results]))
    print(f"[vec_eval] DONE {len(results)} rollouts in "
          f"{time.perf_counter() - start:.1f}s -- SR={sr:.3f} TP={tp:.3f}", flush=True)
    save_results(results, log_dir + "/reports", task, perturbation)
    return results


def _start_wave(vec_env, clients, first_run_id, n_record, log_dir, task, perturbation, no_record):
    """Settle the whole batch and open one `Rollout` per recorded member.

    Returns (members, the warmup's per-member step results). A wave shorter than `num_envs` -- the
    last one, when `repeats` is not a multiple -- leaves the surplus members as None: their scenes
    still step, but nothing about them is recorded.
    """
    step_results = vec_env.warmup()

    members = []
    for i in range(vec_env.num_envs):
        if i >= n_record:
            members.append(None)
            continue
        recorder = None
        if not no_record:
            timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H:%M:%S")
            recorder = VideoRecorder(log_dir, timestamp, first_run_id + i, task, perturbation)
        members.append(Rollout(vec_env.envs[i], first_run_id + i, recorder=recorder))

    for client in clients:
        client.reset()
    return members, step_results


def _step_wave(vec_env, clients, members, step_results, max_steps, horizon,
               render_on_demand, n_pre_obs_renders, max_render_interval):
    """Step every member of one wave until all have finished or `max_steps` is reached.

    Returns how many steps the wave took.
    """
    renders = RenderSchedule(max_render_interval, n_pre_obs_renders)

    step = 0
    while step < max_steps and any(m is not None and m.active for m in members):
        commands = []
        for i, member in enumerate(members):
            if member is None or not member.active:
                commands.append(_hold_command(vec_env.envs[i], member))
            else:
                commands.append(member.next_command(
                    step_results[i][0], clients[i], horizon, renders.obs_is_fresh))

        if render_on_demand:
            # og.sim.render_on_step() is GLOBAL -- one flag for every scene -- so the decision has
            # to be the OR across active members: if ANY of them needs fresh images next iteration,
            # the whole batch renders. In practice members stay in phase, because each active member
            # pops exactly one action per step and they all refill on the same boundary, so this
            # costs no more renders than the single-env path.
            needs_fresh_obs = any(m is not None and m.active and m.needs_fresh_obs()
                                  for m in members)
            render, n_render_iterations = renders.schedule(needs_fresh_obs)
            with og.sim.render_on_step(render):
                step_results = vec_env.step(commands, n_render_iterations=n_render_iterations)
        else:
            step_results = vec_env.step(commands)

        for i, member in enumerate(members):
            if member is None or not member.active:
                continue
            if not member.record_progression(step_results[i][1], step):
                print(f"[vec_eval]   member {i} (run {member.run_id}) finished at step {step}, "
                      f"TP={member.metrics.task_progression}", flush=True)
        step += 1

    return step


def _hold_command(env, member):
    """What to send a member that is not running its own rollout.

    `og.sim.step()` advances its scene either way, so it has to be given something. Hold the last
    commanded action rather than zeros, which would drive the arm toward its zero pose; a member
    that never started gets the warmup's hold-still command instead.
    """
    if member is not None and member.last_command is not None:
        return member.last_command
    return env.warmup_action(0, env.warmup_ee_cmd())
