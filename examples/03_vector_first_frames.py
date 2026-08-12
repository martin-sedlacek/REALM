"""Load N REALM environments in parallel, step once, and save the first frame of each.

A visual smoke test for vectorized environments: it proves the scenes tile without overlapping,
that each member's external and wrist cameras are placed inside its own scene, and that one shared
og.sim.step() advances all of them.

    python examples/03_vector_first_frames.py --num_envs 4 --task_id 0

Writes <out_dir>/env<i>_external.png, env<i>_wrist.png and a montage.png of all members.
"""
import argparse
import os

import numpy as np
from PIL import Image

import omnigibson as og

from realm.environments.env_vector import RealmVectorEnvironment
from realm.eval import SUPPORTED_PERTURBATIONS, SUPPORTED_TASKS
from realm.inference import extract_from_obs
from realm.sim_config import set_sim_config


def _to_uint8(im):
    im = np.asarray(im)
    if im.dtype.kind == "f":
        im = (im * 255).clip(0, 255)
    return im.astype(np.uint8)[..., :3]


def montage(images, cols=2, pad=8, bg=32):
    """Tile equally sized images into a grid so all members can be eyeballed in one picture."""
    h, w = images[0].shape[:2]
    rows = int(np.ceil(len(images) / cols))
    canvas = np.full((rows * h + (rows + 1) * pad, cols * w + (cols + 1) * pad, 3), bg, dtype=np.uint8)
    for i, im in enumerate(images):
        r, c = divmod(i, cols)
        y, x = pad + r * (h + pad), pad + c * (w + pad)
        canvas[y:y + h, x:x + w] = im
    return canvas


def main(num_envs, task_id, perturbation_id, robot, out_dir, task_cfg_path, rendering_mode):
    set_sim_config(robot=robot)

    if task_cfg_path is None:
        task_cfg_path = f"REALM_DROID10/{SUPPORTED_TASKS[task_id]}/default.yaml"
    perturbation = SUPPORTED_PERTURBATIONS[perturbation_id]

    vec_env = RealmVectorEnvironment(
        num_envs,
        task_cfg_path=task_cfg_path,
        perturbations=[perturbation],
        robot=robot,
        rendering_mode=rendering_mode,
    )

    # Settle every member, then take the single shared step whose observations we save.
    vec_env.warmup()
    ee_cmds = [env.warmup_ee_cmd() for env in vec_env.envs]
    actions = [env.warmup_action(0, ee_cmd) for env, ee_cmd in zip(vec_env.envs, ee_cmds)]
    results = vec_env.step(actions)

    os.makedirs(out_dir, exist_ok=True)
    externals, wrists = [], []
    for i, (obs, task_progression, terminated, truncated, info) in enumerate(results):
        base_im, _, _, _, wrist_im, robot_state, gripper_state = extract_from_obs(obs, robot_name=robot)
        base_im, wrist_im = _to_uint8(base_im), _to_uint8(wrist_im)
        externals.append(base_im)
        wrists.append(wrist_im)
        Image.fromarray(base_im).save(os.path.join(out_dir, f"env{i}_external.png"))
        Image.fromarray(wrist_im).save(os.path.join(out_dir, f"env{i}_wrist.png"))
        scene_prim = vec_env.envs[i].omnigibson_env.scene.prim_path
        print(f"env{i}: scene={scene_prim} external={base_im.shape} wrist={wrist_im.shape} "
              f"progression={task_progression} gripper={gripper_state:.3f} "
              f"q0={np.round(robot_state[:3], 4)}")

    Image.fromarray(montage(externals)).save(os.path.join(out_dir, "montage_external.png"))
    Image.fromarray(montage(wrists)).save(os.path.join(out_dir, "montage_wrist.png"))
    print(f"\nWrote {2 * len(results) + 2} images to {out_dir}")

    # Distinct pixels across members confirm the cameras really are in different scenes rather than
    # all four rendering the same tile.
    for i in range(1, len(externals)):
        same = np.array_equal(externals[0], externals[i])
        print(f"env0 vs env{i} external frames identical: {same}")

    og.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parallel-env first-frame smoke test")
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--task_id", type=int, default=0)
    parser.add_argument("--perturbation_id", type=int, default=0)
    parser.add_argument("--robot", type=str, default="DROID")
    parser.add_argument("--task_cfg_path", type=str, default=None)
    parser.add_argument("--rendering_mode", type=str, default="rt")
    parser.add_argument("--out_dir", type=str, default="/app/logs/vector_first_frames")
    args = parser.parse_args()
    main(args.num_envs, args.task_id, args.perturbation_id, args.robot,
         args.out_dir, args.task_cfg_path, args.rendering_mode)
