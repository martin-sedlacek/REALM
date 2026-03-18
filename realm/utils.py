import numpy as np
import matplotlib.pyplot as plt
import os
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

import omnigibson as og

from realm.environments.env_dynamic import RealmEnvironmentDynamic


def replay_traj(env: RealmEnvironmentDynamic, trajectory_actions, trajectory_gt_qpos, trajectory_gt_ee=None, max_steps=1000, dof=7):
    max_steps = min(len(trajectory_actions), max_steps)

    qpos = []
    video = [] if not env.no_rendering else None
    ee_pos_list = []
    ee_rot_list = []

    obs, _ = env.reset()
    obs, rew, terminated, truncated, info = env.warmup(obs)

    # Simple warmup: go to initial GT position
    for _ in range(150):
        action = np.concatenate((trajectory_gt_qpos[0, :dof], np.atleast_1d(np.array([-1]))))
        obs, curr_task_progression, terminated, truncated, info = env.step(action)

    for t in range(max_steps):
        base_im = None if env.no_rendering else obs['external']['external_sensor0']['rgb'].cpu().numpy()[..., :3]
        if base_im is not None and video is not None:
            video.append(base_im)

        robot_state = obs[env.robot.name]['proprio'].cpu().numpy()
        qpos.append(robot_state[:dof])

        ee_pos, ee_rot = env.get_ee_pose()
        ee_pos_list.append(ee_pos)
        ee_rot_list.append(ee_rot)

        action = np.concatenate((trajectory_actions[t, :dof], np.atleast_1d(np.array([-1]))))
        obs, curr_task_progression, terminated, truncated, info = env.step(action)

    # Save debug replay video:
    if video is not None:
        video = np.stack(video)
        save_filename = f"/app/logs/debug_ur5_replay"
        ImageSequenceClip(list(video), fps=15).write_videofile(save_filename + ".mp4", codec="libx264")

    # Stack trajectories
    qpos_joints = np.stack(qpos)
    ee_pos_arr = np.stack(ee_pos_list)
    ee_rot_arr = np.stack(ee_rot_list)

    # Calculate errors
    # Note: ensure GT matches the length of replayed steps
    qpos_err = qpos_joints - trajectory_gt_qpos[:max_steps, :dof]

    ee_pos_err = None
    if trajectory_gt_ee is not None:
        ee_pos_err = ee_pos_arr - trajectory_gt_ee[:max_steps, :3]

    return {
        "qpos_err": qpos_err,
        "ee_pos_err": ee_pos_err,
        "qpos_joints": qpos_joints,
        "ee_pos": ee_pos_arr,
        "ee_rot": ee_rot_arr,
        "trajectory_gt_qpos": trajectory_actions,
        "trajectory_gt_ee": trajectory_gt_ee,
    }


def plot_err(res_dict, ep_name, log_dir, plot_title=None):
    plot_title = ep_name if plot_title is None else plot_title
    qpos_err = res_dict["qpos_err"]
    ee_pos_err = res_dict["ee_pos_err"]
    dof = qpos_err.shape[1]

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    # Plot joint errors
    axes[0].plot(qpos_err)
    axes[0].set_title(f"Joint Position Error: {plot_title}")
    axes[0].set_ylabel("Error (rad)")
    axes[0].set_xlabel("Time steps")
    axes[0].legend([f"Joint {i}" for i in range(dof)], loc='upper right')
    axes[0].grid(True)

    # Auto-scale if error is small
    if np.max(np.abs(qpos_err)) < 0.2:
        axes[0].set_ylim(-0.1, 0.1)

    # Plot EE xyz errors
    if ee_pos_err is not None:
        axes[1].plot(ee_pos_err)
        axes[1].set_title(f"EE XYZ Errors: {plot_title}")
        axes[1].set_ylabel("Error (m)")
        axes[1].set_xlabel("Time steps")
        axes[1].legend(['X', 'Y', 'Z'], loc='upper right')
        axes[1].grid(True)
        if np.max(np.abs(ee_pos_err)) < 0.1:
            axes[1].set_ylim(-0.05, 0.05)
    else:
        axes[1].text(0.5, 0.5, "EE Ground Truth Not Available", ha='center', va='center')

    plt.tight_layout()

    plots_dir = os.path.join(log_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    plot_path = os.path.join(plots_dir, f"{ep_name}.png")
    plt.savefig(plot_path)
    plt.close(fig)
