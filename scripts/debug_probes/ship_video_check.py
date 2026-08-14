"""PHASE 1 cross-check: sample frames from the curl close-ups and see which way the tip rolls.

Pixels are not the ground truth here -- the numeric observables are -- but they are INDEPENDENT of
`collision_boundary_points_world`, which is the thing under suspicion, so they are worth having on a
sign that two numeric measures disagree about.

For each clip: the first (unloaded) frame, the peak-load frame, a red/cyan overlay of the two so any
motion shows as colour fringing, and an amplified absolute difference. Written as one contact sheet
per clip plus the raw stills.
"""
import argparse
import os

import numpy as np
from PIL import Image, ImageDraw

ap = argparse.ArgumentParser()
ap.add_argument("--clips", default="curl_B_nf100_L_open_closeup_ZOOM.mp4,"
                                   "curl_A_nf1000a_open_closeup_ZOOM.mp4")
ap.add_argument("--dir", default="/logs/gripper_squeeze")
ap.add_argument("--prefix", default="ship_vid")
ap.add_argument("--peak-back", type=int, default=None,
                help="frames back from the end to call 'peak'. Default: the probe retracts at the "
                     "end, so peak is taken before the retract.")
ap.add_argument("--retract", type=int, default=6)
args = ap.parse_args()

import imageio.v3 as iio


def load(path):
    return list(iio.imiter(path, plugin="pyav", format="rgb24"))


for clip in args.clips.split(","):
    p = os.path.join(args.dir, clip)
    if not os.path.exists(p):
        print(f"[skip] {p}")
        continue
    fr = load(p)
    n = len(fr)
    ip = n - 1 - (args.peak_back if args.peak_back is not None else args.retract)
    ip = max(0, min(n - 1, ip))
    a, b = np.asarray(fr[0], float), np.asarray(fr[ip], float)
    print(f"{clip}: {n} frames {fr[0].shape}, rest=0 peak={ip}")

    # where did anything move? the moving-pixel centroid tells you which side of the frame the
    # action is on without needing to know the camera's handedness.
    d = np.abs(b - a).mean(2)
    thr = max(8.0, np.percentile(d, 99.5))
    ys, xs = np.nonzero(d > thr)
    if len(xs):
        print(f"   moving pixels: {len(xs)} above {thr:.1f}; centroid x={xs.mean():.1f}/"
              f"{d.shape[1]} y={ys.mean():.1f}/{d.shape[0]}; "
              f"x range {xs.min()}..{xs.max()}, y range {ys.min()}..{ys.max()}")
        # split top/bottom half of the moving region: a CURL moves the distal end more than the
        # proximal one, so the motion should be bottom-heavy in a tips-down view.
        mid = (ys.min() + ys.max()) / 2.0
        top, bot = (ys < mid).sum(), (ys >= mid).sum()
        print(f"   motion split about the mid-line of the moving region: {top} upper / {bot} lower")

    overlay = np.zeros_like(a)
    overlay[..., 0] = a.mean(2)                    # rest  -> red
    overlay[..., 1] = b.mean(2)                    # peak  -> cyan
    overlay[..., 2] = b.mean(2)
    amp = np.clip(np.abs(b - a) * 6.0, 0, 255)

    tiles = [("rest", a), ("peak", b), ("rest=RED peak=CYAN", overlay), ("|diff| x6", amp)]
    h, w = a.shape[:2]
    sheet = Image.new("RGB", (w * len(tiles), h + 18), (0, 0, 0))
    dr = ImageDraw.Draw(sheet)
    for i, (name, arr) in enumerate(tiles):
        sheet.paste(Image.fromarray(arr.astype(np.uint8)), (i * w, 18))
        dr.text((i * w + 4, 4), name, fill=(255, 255, 0))
    out = os.path.join(args.dir, f"{args.prefix}_{clip.replace('.mp4', '')}_SHEET.png")
    sheet.save(out)
    print(f"   wrote {out}")
    for name, arr in (("rest", a), ("peak", b)):
        Image.fromarray(arr.astype(np.uint8)).save(
            os.path.join(args.dir, f"{args.prefix}_{clip.replace('.mp4', '')}_{name}.png"))

print("SHIP_VIDEO_OK")
