"""Single-step eval that projects the main/target object 3D bbox onto each
external camera and saves annotated frames for visual verification.

Run inside the REALM container (matches the env requirements of debug_eval.py).
"""
import argparse
import datetime
import os
import sys

import cv2
import numpy as np
import omnigibson as og

from realm.environments.env_dynamic import RealmEnvironmentDynamic
from realm.eval import SUPPORTED_TASKS, SUPPORTED_PERTURBATIONS, set_sim_config


def _draw_bbox(img_bgr, bbox, color, label):
    if bbox is None:
        return
    x0, y0, x1, y1 = bbox
    cv2.rectangle(img_bgr, (x0, y0), (x1, y1), color, 2)
    cv2.putText(
        img_bgr, label, (x0, max(15, y0 - 6)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_id", type=int, default=0)
    parser.add_argument("--perturbation_id", type=int, default=0)
    parser.add_argument("--multi_view", action="store_true")
    parser.add_argument("--rendering_mode", type=str, default="rt")
    parser.add_argument("--output_dir", type=str, default="/app/logs/bbox_debug")
    args = parser.parse_args()

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.output_dir, timestamp)
    os.makedirs(out_dir, exist_ok=True)

    set_sim_config(rendering_mode=args.rendering_mode)

    task = SUPPORTED_TASKS[args.task_id]
    perturbation = SUPPORTED_PERTURBATIONS[args.perturbation_id]
    task_cfg_path = f"REALM_DROID10/{task}/default.yaml"

    env = RealmEnvironmentDynamic(
        config_path="/app/realm/config",
        task_cfg_path=task_cfg_path,
        perturbations=[perturbation],
        multi_view=args.multi_view,
        no_rendering=False,
        rendering_mode=args.rendering_mode,
    )

    obs, _ = env.reset()
    # Warmup settles the robot + renderer so the obs reflects the actual eval-start state.
    obs, _, _, _, _ = env.warmup(obs)

    main_bbox = env.get_main_object_2d_bbox()
    target_bbox = env.get_target_object_2d_bbox()
    print(f"[bbox] task={task} perturbation={perturbation}")
    print(f"[bbox] main_object  bbox per sensor: {main_bbox}")
    print(f"[bbox] target_object bbox per sensor: {target_bbox}")

    for sensor_name, sensor_obs in obs["external"].items():
        rgb = sensor_obs["rgb"].cpu().numpy()[..., :3].astype(np.uint8)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()

        if isinstance(main_bbox, dict):
            _draw_bbox(bgr, main_bbox.get(sensor_name), (0, 255, 0), "main")
        if isinstance(target_bbox, dict):
            _draw_bbox(bgr, target_bbox.get(sensor_name), (0, 0, 255), "target")

        out_path = os.path.join(
            out_dir, f"{task}_{perturbation}_{sensor_name}.png"
        )
        cv2.imwrite(out_path, bgr)
        # Also save the raw frame for side-by-side comparison.
        raw_path = os.path.join(
            out_dir, f"{task}_{perturbation}_{sensor_name}_raw.png"
        )
        cv2.imwrite(raw_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        print(f"[bbox] saved {out_path}")

    og.log.info("Done!")
    og.shutdown()
    sys.exit(0)


if __name__ == "__main__":
    main()
