"""Assemble the frames `link8_adapter_render.py` wrote into one contact sheet.

One row per camera azimuth, one column per condition, each panel auto-cropped to the region where
the conditions differ (i.e. to the adapter itself) so the 18 mm pad is not a handful of pixels in a
1280x720 wrist shot. The crop box is computed ONCE, from the union of all differences, so every
panel in the sheet shows the identical patch of the identical viewpoint -- a per-panel crop would
let two images be compared that are not actually the same framing.

Runs on the host (no container, no GPU):

    python scripts/debug_probes/link8_adapter_sheet.py \
        --panels before:/logs/link8_adapter/before/before_az{az}_base.png \
                 hidden:/logs/link8_adapter/before/before_az{az}_twinhidden.png \
        --out /logs/link8_adapter/sheet.png
"""
import argparse
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ap = argparse.ArgumentParser()
ap.add_argument("--panels", nargs="+", required=True, metavar="LABEL:PATH_TEMPLATE",
                help="PATH_TEMPLATE may contain {az}, filled with each --azimuths entry")
ap.add_argument("--azimuths", nargs="+", type=int, default=[0, 90, 180, 270])
ap.add_argument("--out", required=True)
ap.add_argument("--pad", type=int, default=60, help="pixels of context around the difference box")
ap.add_argument("--scale", type=float, default=2.0)
ap.add_argument("--crop", default=None, help="x0,y0,x1,y1 to force a crop instead of deriving one")
args = ap.parse_args()

PANELS = [p.split(":", 1) for p in args.panels]


def load(path):
    return np.asarray(Image.open(path).convert("RGB"))


rows = []
for az in args.azimuths:
    imgs = [(label, load(tpl.format(az=az))) for label, tpl in PANELS
            if os.path.exists(tpl.format(az=az))]
    if len(imgs) < 1:
        continue
    rows.append((az, imgs))

if not rows:
    raise SystemExit("no panels found")

# One crop box for the whole sheet: the union of every pairwise difference, padded.
if args.crop:
    x0, y0, x1, y1 = (int(v) for v in args.crop.split(","))
else:
    h, w = rows[0][1][0][1].shape[:2]
    acc = np.zeros((h, w), dtype=bool)
    for _, imgs in rows:
        ref = imgs[0][1].astype(np.int32)
        for _, im in imgs[1:]:
            acc |= (np.abs(ref - im.astype(np.int32)).sum(axis=2) > 8)
    if acc.any():
        ys, xs = np.nonzero(acc)
        x0, x1 = max(int(xs.min()) - args.pad, 0), min(int(xs.max()) + args.pad, w)
        y0, y1 = max(int(ys.min()) - args.pad, 0), min(int(ys.max()) + args.pad, h)
    else:
        # Nothing differs anywhere: centre crop, and say so on the sheet rather than silently
        # producing a full-frame montage that looks like it was framed on purpose.
        cx, cy, s = w // 2, h // 2, min(h, w) // 4
        x0, x1, y0, y1 = cx - s, cx + s, cy - s, cy + s
        print("[warn] no pixel differs between any pair of panels -- centre crop used")
print(f"crop = {x0},{y0},{x1},{y1}  ({x1 - x0}x{y1 - y0})")


def font_at(px):
    try:
        return ImageFont.load_default(size=px)
    except TypeError:
        return ImageFont.load_default()


cw, ch = int((x1 - x0) * args.scale), int((y1 - y0) * args.scale)
LABEL_H, GUT = 26, 6
ncol = max(len(imgs) for _, imgs in rows)
W = GUT + ncol * (cw + GUT)
H = GUT + len(rows) * (ch + LABEL_H + GUT)
sheet = Image.new("RGB", (W, H), (16, 16, 16))
d = ImageDraw.Draw(sheet)
f = font_at(16)

for r, (az, imgs) in enumerate(rows):
    y = GUT + r * (ch + LABEL_H + GUT)
    for c, (label, im) in enumerate(imgs):
        x = GUT + c * (cw + GUT)
        patch = Image.fromarray(np.ascontiguousarray(im[y0:y1, x0:x1])).resize((cw, ch), Image.LANCZOS)
        sheet.paste(patch, (x, y + LABEL_H))
        d.text((x + 4, y + 4), f"az={az} {label}", fill=(255, 235, 120), font=f)

sheet.save(args.out)
print(f"wrote {args.out}  ({W}x{H})")
