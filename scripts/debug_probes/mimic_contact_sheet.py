"""Build ONE png that puts every sweep rung's clamped-jaw close-up side by side.

The compliance question is now "can you SEE the pads bend", and that is a judgement about pixels, not
about millimetres. A rung-per-row contact sheet answers it in a single look, and at a fixed crop and
scale so the comparison is fair -- which flipping between eight mp4s is not.

    python scripts/debug_probes/mimic_contact_sheet.py /logs/gripper_squeeze MIMIC_A \
        --frames -1 --zoom 2 --out /logs/gripper_squeeze/MIMIC_A_sheet.png

Reads the per-rung `<tag>_<rung>_closeup.mp4` written by gripper_squeeze_compliance.py in sweep mode
(NOT the _ZOOM ones -- this crops from the full frame itself so --zoom is honest), takes the requested
frame indices, centre-crops and tiles them.
"""
import argparse
import glob
import json
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ap = argparse.ArgumentParser()
ap.add_argument("root")
ap.add_argument("tag")
ap.add_argument("--frames", default="-1",
                help="comma-separated frame indices into each rung clip; negative counts from the end")
ap.add_argument("--zoom", type=float, default=2.0, help="centre crop factor (2 = keep the middle half)")
ap.add_argument("--width", type=int, default=520, help="pixels per tile after scaling")
ap.add_argument("--out", default=None)
ap.add_argument("--order", default=None, help="comma-separated rung names, to force the row order")
args = ap.parse_args()

import av  # noqa: E402

paths = sorted(glob.glob(os.path.join(args.root, f"{args.tag}_*_closeup.mp4")))
paths = [p for p in paths if "_ZOOM" not in p]
rungs = {}
for p in paths:
    name = os.path.basename(p)[len(args.tag) + 1:-len("_closeup.mp4")]
    rungs[name] = p
if not rungs:
    raise SystemExit(f"no {args.tag}_*_closeup.mp4 under {args.root}")

# Row order: the json's rung order (= the order they were measured), else alphabetical.
order = None
if args.order:
    order = [x.strip() for x in args.order.split(",")]
else:
    jf = os.path.join(args.root, f"{args.tag}_squeeze.json")
    if os.path.exists(jf):
        order = list(json.load(open(jf)).get("rungs", {}).keys())
order = [r for r in (order or []) if r in rungs] + [r for r in rungs if r not in (order or [])]

want = [int(x) for x in args.frames.split(",")]


def grab(path, idxs):
    with av.open(path) as c:
        fr = [f.to_ndarray(format="rgb24") for f in c.decode(video=0)]
    n = len(fr)
    return [fr[i if i >= 0 else n + i] for i in idxs], n


tiles, labels = [], []
for r in order:
    ims, n = grab(rungs[r], want)
    row = []
    for im in ims:
        h, w = im.shape[:2]
        ch, cw = int(h / (2 * args.zoom)), int(w / (2 * args.zoom))
        c = im[h // 2 - ch: h // 2 + ch, w // 2 - cw: w // 2 + cw]
        img = Image.fromarray(c)
        img = img.resize((args.width, int(args.width * c.shape[0] / c.shape[1])), Image.LANCZOS)
        row.append(np.asarray(img))
    tiles.append(np.concatenate(row, axis=1))
    labels.append(f"{r}   ({n} frames)")

TH = 26
W = max(t.shape[1] for t in tiles)
H = sum(t.shape[0] + TH for t in tiles)
sheet = Image.new("RGB", (W, H), (12, 12, 14))
d = ImageDraw.Draw(sheet)
try:
    font = ImageFont.load_default(size=18)
except TypeError:
    font = ImageFont.load_default()
y = 0
for t, lab in zip(tiles, labels):
    d.rectangle([0, y, W, y + TH], fill=(0, 0, 0))
    d.text((8, y + 4), lab, fill=(255, 235, 120), font=font)
    y += TH
    sheet.paste(Image.fromarray(t), (0, y))
    y += t.shape[0]

out = args.out or os.path.join(args.root, f"{args.tag}_sheet.png")
sheet.save(out)
print(f"wrote {out}  ({W}x{H}, {len(tiles)} rungs x {len(want)} frames, zoom {args.zoom})")
print("rows: " + ", ".join(order))
