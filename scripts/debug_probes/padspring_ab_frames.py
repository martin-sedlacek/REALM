"""Side-by-side contact sheet of two gripper_squeeze_compliance.py close-up videos.

The pad-spring question is a *visual* one -- "do the pads visibly deflect" -- so the deliverable is a
frame from each variant at the same phase, next to each other. This pairs frames by the phase tag
recorded in each run's npz rather than by frame index, because the runs have different step counts
whenever a phase length or a --cal-steps differs.

    python scripts/debug_probes/padspring_ab_frames.py /logs/gripper_squeeze \
        --runs DROID_robolab_v2:default padspring_kp40_e20:pad-spring --out-prefix padspring_AB

Each --runs entry is <npz/mp4 stem>:<label>. The stem is whatever the file is called on disk, so it
works for both the probe's own `<ROBOT>_squeeze*` names and renamed rungs. Run it in the container:
imageio's pyav plugin is what decodes the mp4s.
"""
import argparse
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ap = argparse.ArgumentParser()
ap.add_argument("out_dir")
ap.add_argument("--runs", nargs="+", required=True, metavar="STEM:LABEL")
ap.add_argument("--video-suffix", default="_squeeze_closeup_ZOOM.mp4")
ap.add_argument("--npz-suffix", default="_squeeze.npz")
ap.add_argument("--out-prefix", default="padspring_AB")
ap.add_argument("--phases", nargs="+",
                default=["open", "free_close", "settle_B", "squeeze_B"],
                help="phase tags to sample; the LAST frame of each phase is taken")
args = ap.parse_args()


def load(stem):
    npz = os.path.join(args.out_dir, stem + args.npz_suffix)
    mp4 = os.path.join(args.out_dir, stem + args.video_suffix)
    # renamed rungs drop the "_squeeze" infix
    if not os.path.exists(npz):
        npz = os.path.join(args.out_dir, stem + ".npz")
    if not os.path.exists(mp4):
        mp4 = os.path.join(args.out_dir, stem + "__closeup_ZOOM.mp4")
    assert os.path.exists(npz), f"no npz for {stem}"
    assert os.path.exists(mp4), f"no video for {stem}"
    z = np.load(npz, allow_pickle=True)
    import imageio.v3 as iio
    frames = list(iio.imiter(mp4, plugin="pyav", format="rgb24"))
    tags = [str(t) for t in z["tag"]]
    print(f"  {stem}: {len(frames)} frames, {len(tags)} logged steps")
    return dict(stem=stem, tags=tags, frames=frames, z=z)


def pad_angles(z, i):
    names = [str(n) for n in z["joint_names"]]
    q = z["q"][i]
    out = {}
    for n in ("left_inner_finger_joint", "right_inner_finger_joint", "finger_joint"):
        if n in names:
            out[n] = float(q[names.index(n)])
    return out


runs = []
for spec in args.runs:
    stem, _, label = spec.partition(":")
    r = load(stem)
    r["label"] = label or stem
    runs.append(r)

FONT = None
for size in (34,):
    try:
        FONT = ImageFont.load_default(size=size)
    except TypeError:
        FONT = ImageFont.load_default()

for phase in args.phases:
    tiles = []
    for r in runs:
        idx = [i for i, t in enumerate(r["tags"]) if t == phase]
        if not idx:
            print(f"  [skip] {r['stem']} has no phase '{phase}'")
            tiles = []
            break
        i = min(idx[-1], len(r["frames"]) - 1)
        pa = pad_angles(r["z"], min(idx[-1], len(r["z"]["q"]) - 1))
        im = Image.fromarray(np.ascontiguousarray(r["frames"][i]))
        d = ImageDraw.Draw(im)
        txt = (f"{r['label']}   {phase}\n"
               f"finger {np.degrees(pa.get('finger_joint', np.nan)):+6.2f} deg   "
               f"pads {np.degrees(pa.get('left_inner_finger_joint', np.nan)):+6.2f} / "
               f"{np.degrees(pa.get('right_inner_finger_joint', np.nan)):+6.2f} deg")
        d.rectangle([0, im.size[1] - 92, im.size[0], im.size[1]], fill=(0, 0, 0))
        d.multiline_text((12, im.size[1] - 86), txt, fill=(120, 255, 140), font=FONT)
        tiles.append(im)
    if not tiles:
        continue
    h = min(t.size[1] for t in tiles)
    tiles = [t.resize((int(t.size[0] * h / t.size[1]), h)) for t in tiles]
    W = sum(t.size[0] for t in tiles)
    sheet = Image.new("RGB", (W, h), (12, 12, 12))
    x = 0
    for t in tiles:
        sheet.paste(t, (x, 0))
        x += t.size[0]
    dst = os.path.join(args.out_dir, f"{args.out_prefix}_{phase}.png")
    sheet.save(dst)
    print(f"wrote {dst}  ({sheet.size[0]}x{sheet.size[1]})")

print("AB_FRAMES_OK")
