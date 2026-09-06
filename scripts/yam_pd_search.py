"""Brute-force search for ONE (kp, kd) shared by every YAM arm joint, scored by replaying ABC's real
episodes. Container + GPU; no rendering.

    python scripts/yam_pd_search.py --robot YAM_ABC --data /abc --out /app/tmp/yam_pd_search/<tag>

WHAT IT MEASURES. ABC's `states_actions.bin` holds the commanded joint targets a[t] next to the measured
joints s[t] at 30 Hz, so the real controller's lag is recorded. This boots one environment, sets the
candidate drive gains on all 12 arm joints at runtime (the stock position drive with
`use_impedances: False` only forwards targets, so the PhysX drive IS the controller), puts the robot at
each episode's first recorded pose, replays the recorded actions open-loop and compares the simulated joint
trajectory with the recorded one. Cost = RMSE over all arm joints, steps and episodes, both arms pooled.
`--mode teacher` instead resets the sim to s[t] (with the recorded velocity) before every step and scores
the one-step prediction, which removes any drift from the comparison.

Stepping (`--stepper`) defaults to apply_action + `og.sim.step_physics()` per physics substep, which skips
Kit's per-step app update; `blind` (OG-lite `Environment.step_blind`) and `env` (`env.step`) are the same PhysX
path with more overhead and exist for the equivalence check. Per cell it appends a row to <out>/grid.csv (cost, per-joint RMSE, per-joint first-order tau of the sim
trajectory) and saves the sim trajectories to <out>/traj_<tag>.npz, so plots are made on the host.
Reference rows `high_pd` and `base` (the per-group sets in realm/robots/yam.py) run first.
"""
import argparse
import csv
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.append(str(PROJECT_ROOT))

TASK = "put_the_plastic_bottles_in_the_bin"
N_ARM_JOINTS = 6
FPS = 30.0


def load_episodes(root, include_sim=False, max_steps=None):
    """[(name, states (T,14), actions (T,14))] for the real put-bottles episodes, sorted by name."""
    episodes = []
    for meta_path in sorted(glob.glob(f"{root}/*/episode_*/episode_metadata.json")):
        meta = json.load(open(meta_path))
        task = meta.get("task_name", "")
        is_sim = task.startswith("sim_")
        if task.removeprefix("sim_") != TASK or (is_sim and not include_sim):
            continue
        ep_dir = Path(meta_path).parent
        data = np.fromfile(ep_dir / "states_actions.bin", dtype=np.float64).reshape(-1, 28)
        if max_steps:
            data = data[:max_steps]
        episodes.append((("sim_" if is_sim else "real_") + ep_dir.name[8:16], data[:, :14], data[:, 14:]))
    return episodes


def fit_tau(states, actions, targets_are_next=True):
    """First-order lag per joint: s[t+1] - s[t] = alpha (a[t] - s[t]); tau = -dt / ln(1 - alpha) in ms."""
    taus = []
    for j in range(states.shape[1]):
        ds = states[1:, j] - states[:-1, j]
        e = actions[:-1, j] - states[:-1, j]
        denom = float(e @ e)
        alpha = float(ds @ e) / denom if denom > 1e-12 else np.nan
        alpha = min(max(alpha, 1e-6), 1 - 1e-6) if np.isfinite(alpha) else np.nan
        taus.append(-1000.0 / FPS / np.log(1 - alpha) if np.isfinite(alpha) else np.nan)
    return np.array(taus)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", default="YAM_ABC")
    ap.add_argument("--task_cfg_path", default="REALM_DROID10/put_green_block_into_bowl/default.yaml")
    ap.add_argument("--data", default="/abc", help="abc_preview root (train/, val/)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--kp", type=float, nargs="*", default=[5, 10, 20, 40, 80, 160, 320, 640])
    ap.add_argument("--kd", type=float, nargs="*", default=[0.5, 1, 2, 5, 10, 20, 50])
    ap.add_argument("--no-refs", action="store_true", help="skip the high_pd / base reference rows")
    ap.add_argument("--mode", choices=["openloop", "teacher"], default="openloop")
    ap.add_argument("--hold-steps", type=int, default=15, help="settle steps at s[0] before each replay")
    ap.add_argument("--max-steps", type=int, default=None, help="truncate every episode (throughput test)")
    ap.add_argument("--include-sim", action="store_true", help="also replay ABC's own sim episodes")
    ap.add_argument("--cells", nargs="*", default=None, metavar="KP:KD",
                    help="explicit unified cells (e.g. 20:2 160:25) instead of the kp x kd product")
    ap.add_argument("--shard", type=int, nargs=2, metavar=("I", "N"), default=(0, 1),
                    help="run only cells I::N of the (refs + kp x kd) list -- for spreading a grid over processes")
    ap.add_argument("--stepper", choices=["env", "blind", "physics"], default="physics",
                    help="env: env.step (obs + task progression); blind: Environment.step_blind (Kit update, no "
                         "render/obs); physics: apply the action and call og.sim.step_physics() for each physics "
                         "substep, skipping Kit's app update. Same PhysX path in all three -- the numbers must agree")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    episodes = load_episodes(args.data, include_sim=args.include_sim, max_steps=args.max_steps)
    assert episodes, f"no {TASK} episodes under {args.data}"
    total_steps = sum(len(s) for _, s, _ in episodes)
    print(f"episodes: {[(n, len(s)) for n, s, _ in episodes]} total {total_steps} steps", flush=True)

    import torch as th
    import omnigibson as og
    from omnigibson.macros import gm
    from realm.eval import CONFIG_ROOT
    from realm.environments.env_dynamic import RealmEnvironmentDynamic
    from realm.inference.utils import get_robot_obs_profile
    from realm.robots.yam import YamRobot
    from realm.sim_config import set_sim_config

    set_sim_config(robot=args.robot)
    assert gm.DEFAULT_SIM_STEP_FREQ == 30, gm.DEFAULT_SIM_STEP_FREQ
    env = RealmEnvironmentDynamic(config_path=CONFIG_ROOT, task_cfg_path=args.task_cfg_path,
                                  perturbations=["Default"], robot=args.robot, no_rendering=True)
    robot = env.robot
    print(f"control_freq {robot._control_freq} Hz, physics dt {og.sim.get_physics_dt():.5f} s, "
          f"sim step dt {og.sim.get_sim_step_dt():.5f} s", flush=True)
    assert abs(robot._control_freq - 30) < 1e-6, robot._control_freq

    profile = get_robot_obs_profile(robot.name)
    arms = list(profile["arms"])
    assert len(arms) == 2 and profile["arm_dof"] == N_ARM_JOINTS, profile
    dof_names = list(robot.dof_names_ordered)
    arm_joint_names = [j for arm in arms for j in profile["arm_joint_names"][arm]]     # [L j1..6, R j1..6]
    arm_dof_idx = th.tensor([dof_names.index(j) for j in arm_joint_names])
    finger_dof_idx = th.tensor([dof_names.index(j) for arm in arms for j in profile["finger_joint_names"][arm]])
    # Fingers open at whichever limit is farther from 0 (crank: +0.0475 left / -0.0475 right; YAMLab: -0.0475).
    open_qpos = th.tensor([max((robot.joints[dof_names[i]].lower_limit, robot.joints[dof_names[i]].upper_limit), key=abs)
                           for i in finger_dof_idx.tolist()], dtype=th.float32)
    action_dim = len(arms) * (N_ARM_JOINTS + 1)
    grip_cols = list(profile["gripper_action_idx"])
    arm_cols = [c for c in range(action_dim) if c not in grip_cols]
    # ABC's columns are [L j1..6, L grip, R j1..6, R grip] -- the same layout as the REALM action.
    data_arm_cols = [c for c in range(14) if c not in (6, 13)]
    data_grip_cols = [6, 13]
    joints = [robot.joints[j] for j in arm_joint_names]
    limits = [(float(jt.lower_limit), float(jt.upper_limit)) for jt in joints]
    print("arm DOF idx", arm_dof_idx.tolist(), "finger DOF idx", finger_dof_idx.tolist(), "open", open_qpos.tolist())
    print("limits: " + ", ".join(f"{n} [{lo:+.2f}, {hi:+.2f}]" for n, (lo, hi) in zip(arm_joint_names, limits)))
    for name, s, _ in episodes:
        lo = s[:, data_arm_cols].min(0)
        hi = s[:, data_arm_cols].max(0)
        viol = [(arm_joint_names[k], float(lo[k]), float(hi[k])) for k in range(12)
                if lo[k] < limits[k][0] - 1e-3 or hi[k] > limits[k][1] + 1e-3]
        if viol:
            print(f"WARNING {name}: recorded states outside the USD limits: {viol}")

    obs, _ = env.reset()
    env.warmup(obs)

    def set_gains(kp_vec, kd_vec):
        for jt, kp, kd in zip(joints, kp_vec, kd_vec):
            jt.stiffness = float(kp)
            jt.damping = float(kd)
        got_kp = [float(jt.stiffness) for jt in joints]
        got_kd = [float(jt.damping) for jt in joints]
        assert np.allclose(got_kp, kp_vec, rtol=1e-3) and np.allclose(got_kd, kd_vec, rtol=1e-3), (got_kp, got_kd)

    def build_action(a_row):
        act = np.zeros(action_dim)
        act[arm_cols] = a_row[data_arm_cols]
        act[grip_cols] = np.where(a_row[data_grip_cols] > 0.5, 1.0, -1.0)
        return act

    def teleport(q_arm, qd_arm=None):
        robot.set_joint_positions(th.as_tensor(q_arm, dtype=th.float32), indices=arm_dof_idx)
        robot.set_joint_positions(open_qpos.clone(), indices=finger_dof_idx)
        vel = th.zeros(len(arm_dof_idx)) if qd_arm is None else th.as_tensor(qd_arm, dtype=th.float32)
        robot.set_joint_velocities(vel, indices=arm_dof_idx)
        robot.set_joint_velocities(th.zeros(len(finger_dof_idx)), indices=finger_dof_idx)

    def q_arm():
        q = robot.get_joint_positions()
        return q[arm_dof_idx].cpu().numpy().astype(np.float64)

    og_env = env.omnigibson_env
    n_sub = og.sim.n_physics_timesteps_per_render
    assert n_sub == 4, n_sub
    print(f"stepper {args.stepper}, {n_sub} physics substeps per control step", flush=True)

    def sim_step(action):
        if args.stepper == "env":
            env.step(action)
        elif args.stepper == "blind":
            og_env.step_blind(action)   # apply_action + physics substeps only; no render, no obs
        else:
            og_env._pre_step(action)    # robot.apply_action; ControllerView.step_all runs in the PhysX pre-step callback
            for _ in range(n_sub):
                og.sim.step_physics()

    def replay(states, actions):
        """Sim arm trajectory (T,12): row t is the sim state after applying a[t] (compare with s[t+1])."""
        T = len(states)
        sim = np.zeros((T, 12))
        teleport(states[0, data_arm_cols])
        hold = build_action(states[0])  # targets = the recorded pose, grippers open
        hold[grip_cols] = 1.0
        for _ in range(args.hold_steps):
            sim_step(hold)
        for t in range(T):
            if args.mode == "teacher" and t > 0:
                teleport(states[t, data_arm_cols], (states[t, data_arm_cols] - states[t - 1, data_arm_cols]) * FPS)
            sim_step(build_action(actions[t]))
            sim[t] = q_arm()
        return sim

    cells = []
    if not args.no_refs:
        for name in ("high_pd", "base"):
            kp, kd = YamRobot.arm_gains(name)
            cells.append((name, kp * 2, kd * 2))
    pairs = [(float(c.split(":")[0]), float(c.split(":")[1])) for c in args.cells] if args.cells is not None \
        else [(kp, kd) for kp in args.kp for kd in args.kd]
    for kp, kd in pairs:
        cells.append((f"kp{kp:g}_kd{kd:g}", [kp] * 12, [kd] * 12))

    i, n = args.shard
    cells = cells[i::n]
    print(f"shard {i}/{n}: {len(cells)} cells: {[c[0] for c in cells]}", flush=True)

    grid_path = out / "grid.csv"
    header = ["cell", "kp", "kd", "mode", "stepper", "rmse_all", "steps", "sec"] + \
             [f"rmse_j{j+1}" for j in range(6)] + [f"tau_sim_j{j+1}_ms" for j in range(6)] + \
             [f"tau_real_j{j+1}_ms" for j in range(6)]
    if not grid_path.exists():
        with open(grid_path, "w", newline="") as f:
            csv.writer(f).writerow(header)
    real_tau = None

    for tag, kp_vec, kd_vec in cells:
        set_gains(kp_vec, kd_vec)
        t0 = time.time()
        errs, sims, refs, acts = [], {}, [], []
        for name, s, a in episodes:
            sim = replay(s, a)
            sims[name] = sim.astype(np.float32)
            errs.append(sim[:-1] - s[1:, data_arm_cols])
            refs.append(s[:, data_arm_cols])
            acts.append(a[:, data_arm_cols])
        sec = time.time() - t0
        E = np.concatenate(errs)                          # (N, 12)
        rmse_all = float(np.sqrt((E ** 2).mean()))
        rmse_j = [float(np.sqrt((E[:, [j, 6 + j]] ** 2).mean())) for j in range(6)]
        # tau of the sim trajectory, per episode then averaged, L/R pooled by joint number
        tau_sim = np.nanmean([fit_tau(sims[n], acts[i]) for i, (n, _, _) in enumerate(episodes)], axis=0)
        tau_sim6 = [float(np.nanmean(tau_sim[[j, 6 + j]])) for j in range(6)]
        if real_tau is None:
            tr = np.nanmean([fit_tau(r, a) for r, a in zip(refs, acts)], axis=0)
            real_tau = [float(np.nanmean(tr[[j, 6 + j]])) for j in range(6)]
        row = [tag, kp_vec[0] if len(set(kp_vec)) == 1 else "group", kd_vec[0] if len(set(kd_vec)) == 1 else "group",
               args.mode, args.stepper, f"{rmse_all:.5f}", len(E), f"{sec:.1f}"] + [f"{v:.5f}" for v in rmse_j] + \
              [f"{v:.1f}" for v in tau_sim6] + [f"{v:.1f}" for v in real_tau]
        with open(grid_path, "a", newline="") as f:
            csv.writer(f).writerow(row)
        np.savez_compressed(out / f"traj_{tag}.npz", **sims)
        print(f"[{tag:>14s}] rmse {rmse_all:.4f} | per joint " + " ".join(f"{v:.3f}" for v in rmse_j) +
              f" | tau_sim " + " ".join(f"{v:.0f}" for v in tau_sim6) + f" | {len(E)} steps in {sec:.0f}s "
              f"({len(E) / sec:.0f} steps/s)", flush=True)

    print(f"real tau (ms): " + " ".join(f"{v:.0f}" for v in real_tau))
    print(f"DONE -- {len(cells)} cells -> {grid_path}")
    og.shutdown()


if __name__ == "__main__":
    main()
