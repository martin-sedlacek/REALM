"""Render the YAM_bimanual policy views for candidate top-camera poses (and both wrist-camera poses) on a
REALM task, next to MolmoAct2 reference frames, to close the visual gap to the training station by eye.

    python scripts/yam_camera_sweep.py --out /logs/<exp> --dataset /molmo   (container-side; GPU)

Writes <out>/top_<name>.png (full 1280x720), <out>/tile_top_<name>.png (224 letterboxed as the policy sees it),
<out>/tile_<arm>_<wristpose>.png, and two grids: <out>/grid_top.png, <out>/grid_wrist.png (last row = dataset).
Candidates are expressed like the robot config's exterior_camera (position in the mount frame, USD camera
convention); the pitch is applied about the camera's own x axis relative to YAMLab's 60-deg-down pose.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# (name, x forward of the mount midpoint [m], z above it [m], pitch below horizontal [deg], focal length [mm])
CANDIDATES = [
    ("yamlab", -0.1664949, 0.9443205, 60.0, 12.8413),
    # v2 pick D3 (x0.25 z0.45 p85 f15) framed the table like MolmoAct2's top camera; refine around it
    ("D3_x0.25_z0.45_p85_f15", 0.25, 0.45, 85.0, 15.0),
    ("E1_x0.25_z0.40_p85_f15", 0.25, 0.40, 85.0, 15.0),
    ("E2_x0.30_z0.40_p90_f15", 0.30, 0.40, 90.0, 15.0),
    ("E3_x0.20_z0.45_p80_f15", 0.20, 0.45, 80.0, 15.0),
    ("E4_x0.25_z0.45_p85_f13", 0.25, 0.45, 85.0, 13.0),
    ("E5_x0.25_z0.50_p85_f16.5", 0.25, 0.50, 85.0, 16.5),
    ("E6_x0.30_z0.45_p85_f15", 0.30, 0.45, 85.0, 15.0),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", default="YAM_bimanual")
    ap.add_argument("--task_cfg_path", default="REALM_DROID10/put_green_block_into_bowl/default.yaml")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dataset", default="/molmo", help="dir with frame000_{top,left,right}.png reference frames")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    import omnigibson as og
    from openpi_client import image_tools
    from realm.eval import CONFIG_ROOT
    from realm.environments.env_dynamic import RealmEnvironmentDynamic
    from realm.inference.openpi_yam import crop_to_aspect
    from realm.inference.utils import extract_from_obs, extract_wrist_images
    from realm.robots.yam import YamBimanualRobot, YamRobot
    from realm.sim_config import set_sim_config

    set_sim_config(robot=args.robot)
    env = RealmEnvironmentDynamic(config_path=CONFIG_ROOT, task_cfg_path=args.task_cfg_path,
                                  perturbations=["Default"], robot=args.robot, rendering_mode="rt")
    obs, _ = env.reset()
    env.warmup(obs)
    robot = env.robot
    og_env = env.omnigibson_env

    def grab():
        # a moved camera needs a few frames before the RT render product shows the new view; some renders
        # still come back as a uniform smear (sweeps 204837/204838) -- detect (low pixel std) and retry
        for attempt in range(4):
            for _ in range(2):
                og.sim.step()
            for _ in range(12 * (attempt + 1)):
                og.sim.render()
            o = og_env.get_obs()[0]
            base_im = np.asarray(extract_from_obs(o, robot.name)[0])[..., :3]
            wrists = [np.asarray(w)[..., :3] for w in extract_wrist_images(o, robot.name)]
            if base_im.std() > 15 and all(w.std() > 15 for w in wrists):
                return base_im, wrists
            print(f"[sweep] uniform render (std top {base_im.std():.1f}), retrying", flush=True)
        return base_im, wrists

    def tile(im):
        return image_tools.resize_with_pad(crop_to_aspect(np.ascontiguousarray(im)), 224, 224)

    # --- exterior camera candidates -------------------------------------------------------------
    sensor_cfg = env.cfg["env"]["external_sensors"][0]
    sensor = og_env.external_sensors[sensor_cfg["name"]]
    # construct_ext_cam_pose_by_name wants the base pose as xyz + rpy radians (env_config keeps the rpy)
    robot_pos, robot_rot = env.cfg["robots"][0]["position"], np.asarray(env.robot_rot_rad, dtype=float).tolist()
    r_yam = R.from_quat(list(YamBimanualRobot.EXTERIOR_CAMERA_QUAT_XYZW))  # looks +x, 60 deg down

    def quat_for_pitch(pitch_deg):
        target_fwd_z = -np.sin(np.radians(pitch_deg))
        best = None
        for sign in (1.0, -1.0):
            r = r_yam * R.from_rotvec([sign * np.radians(pitch_deg - 60.0), 0.0, 0.0])
            fwd = r.apply([0.0, 0.0, -1.0])
            err = abs(fwd[2] - target_fwd_z) + abs(fwd[1])
            if best is None or err < best[0]:
                best = (err, r, fwd)
        return best[1].as_quat().tolist(), best[2]

    top_tiles = []
    for name, x, z, pitch, focal in CANDIDATES:
        quat, fwd = quat_for_pitch(pitch)
        pos, rot = env.construct_ext_cam_pose_by_name({"pos": [x, YamBimanualRobot.EXTERIOR_CAMERA_POSITION[1], z], "rot": quat},
                                                      robot_pos, robot_rot)
        sensor.set_position_orientation(pos, rot, sensor_cfg["pose_frame"])
        got_pos, got_rot = sensor.get_position_orientation()
        print(f"[sweep] {name}: set world pos {np.round(pos, 3).tolist()} -> read back {np.round(np.asarray(got_pos), 3).tolist()} "
              f"rot {np.round(np.asarray(got_rot), 3).tolist()}", flush=True)
        try:
            sensor.focal_length = focal
        except Exception as e:  # noqa: BLE001
            print(f"[sweep] could not set focal length on {name}: {e}")
        base_im, wrists = grab()
        Image.fromarray(base_im).save(out / f"top_{name}.png")
        t = tile(base_im)
        Image.fromarray(t).save(out / f"tile_top_{name}.png")
        top_tiles.append((name, t))
        print(f"[sweep] {name}: pos {np.round(pos, 3).tolist()} fwd {np.round(fwd, 3).tolist()} focal {focal}", flush=True)
        if name == "yamlab":
            base_wrists = wrists

    # --- wrist camera poses: current USD (ABC bracket) vs YAMLab's calibration -------------------
    wrist_sensors = {k: s for k, s in robot.sensors.items() if ":Camera:" in k and "link_6" in k}
    print("[sweep] all robot sensors:", sorted(robot.sensors))
    print("[sweep] wrist sensors:", sorted(wrist_sensors))
    wrist_rows = [("abc_bracket_50deg", [tile(w) for w in base_wrists])]
    saved = {k: s.get_local_pose() for k, s in wrist_sensors.items()}
    # interpolate between YAMLab's calibration (~25 deg below the flange axis) and ABC's bracket (50 deg)
    from scipy.spatial.transform import Slerp
    yw = YamRobot.YAMLAB_WRIST_CAMERA_QUAT_WXYZ
    aw = YamRobot.WRIST_CAMERA_QUAT_WXYZ
    key = R.from_quat([[yw[1], yw[2], yw[3], yw[0]], [aw[1], aw[2], aw[3], aw[0]]])
    slerp = Slerp([0.0, 1.0], key)
    p0, p1 = np.array(YamRobot.YAMLAB_WRIST_CAMERA_POSITION), np.array(YamRobot.WRIST_CAMERA_POSITION)
    for frac, label in ((0.0, "yamlab_25deg"), (0.25, "mix25_31deg"), (0.5, "mix50_37deg"), (0.75, "mix75_44deg")):
        q = slerp([frac])[0].as_quat()  # xyzw
        pos = p0 + frac * (p1 - p0)
        for k, s_ in wrist_sensors.items():
            s_.set_local_pose(pos, q)
        _, wrists = grab()
        wrist_rows.append((label, [tile(w) for w in wrists]))
        print(f"[sweep] wrist {label}: pos {np.round(pos, 4).tolist()} quat_xyzw {np.round(q, 4).tolist()}", flush=True)
    for k, s_ in wrist_sensors.items():
        s_.set_local_pose(*saved[k])
    for label, tiles in wrist_rows:
        for arm, t in zip(("left", "right"), tiles):
            Image.fromarray(t).save(out / f"tile_{arm}_{label}.png")

    # --- grids with the dataset reference in the last row --------------------------------------
    ds = {}
    for cam in ("top", "left", "right"):
        p = Path(args.dataset) / f"frame000_{cam}.png"
        if p.exists():
            ds[cam] = image_tools.resize_with_pad(np.asarray(Image.open(p).convert("RGB")), 224, 224)

    def grid(rows, path):
        w = 224 * max(len(r[1]) for r in rows)
        im = Image.new("RGB", (w, (224 + 16) * len(rows)), "black")
        d = ImageDraw.Draw(im)
        for i, (label, tiles) in enumerate(rows):
            y = i * 240
            d.text((4, y + 2), label, fill="white")
            for j, t in enumerate(tiles):
                im.paste(Image.fromarray(t), (j * 224, y + 16))
        im.save(path)
        print("[sweep] wrote", path, flush=True)

    rows = [(n, [t]) for n, t in top_tiles]
    if "top" in ds:
        rows.append(("DATASET frame 0", [ds["top"]]))
    grid(rows, out / "grid_top.png")
    rows = [(label, tiles) for label, tiles in wrist_rows]
    if "left" in ds:
        rows.append(("DATASET frame 0", [ds["left"], ds["right"]]))
    grid(rows, out / "grid_wrist.png")
    print("[sweep] done", flush=True)
    og.shutdown()


if __name__ == "__main__":
    main()
