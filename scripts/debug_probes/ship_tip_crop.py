"""Crop the fingertip out of a curl close-up and strip rest -> peak, amplified.

The close-up frames the pad ORIGINS, so the tip sits low in frame; this crops to it and stacks the
press so a ~1.7 mm roll is visible at all. Frame labels are burned in by the probe, so the strip
also says which phase each frame is.
"""
import argparse
import os

import numpy as np
from PIL import Image, ImageDraw

ap = argparse.ArgumentParser()
ap.add_argument("--clip", default="/logs/gripper_squeeze/curl_B_nf100_L_open_closeup.mp4")
ap.add_argument("--crop", default="120,380,520,720", help="x0,y0,x1,y1 in the source frame")
ap.add_argument("--every", type=int, default=1)
ap.add_argument("--out", default="/logs/gripper_squeeze/ship_tipcrop.png")
ap.add_argument("--scale", type=int, default=2)
args = ap.parse_args()

import imageio.v3 as iio

fr = list(iio.imiter(args.clip, plugin="pyav", format="rgb24"))
x0, y0, x1, y1 = (int(v) for v in args.crop.split(","))
print(f"{args.clip}: {len(fr)} frames {fr[0].shape}; crop {x0},{y0}-{x1},{y1}")

# The probe burns the phase into the top-left of every frame; use the LABEL BAND to segment the
# clip, by finding where the text pixels change a lot.
band = np.stack([np.asarray(f)[0:90, 0:900].mean() for f in fr])
print("  label-band mean per frame:", np.round(band, 2))

sel = list(range(0, len(fr), args.every))
crops = [np.asarray(fr[i])[y0:y1, x0:x1] for i in sel]
h, w = crops[0].shape[:2]
s = args.scale
rest = crops[0].astype(float)

cols = len(sel)
sheet = Image.new("RGB", (w * s * cols, h * s * 2 + 36), (0, 0, 0))
dr = ImageDraw.Draw(sheet)
for j, (i, c) in enumerate(zip(sel, crops)):
    im = Image.fromarray(c).resize((w * s, h * s), Image.NEAREST)
    sheet.paste(im, (j * w * s, 18))
    d = np.clip(np.abs(c.astype(float) - rest) * 5.0, 0, 255).astype(np.uint8)
    sheet.paste(Image.fromarray(d).resize((w * s, h * s), Image.NEAREST), (j * w * s, 18 + h * s + 18))
    dr.text((j * w * s + 3, 3), f"f{i}", fill=(255, 255, 0))
dr.text((3, 18 + h * s + 3), "|frame - rest| x5", fill=(255, 255, 0))
sheet.save(args.out)
print(f"wrote {args.out}  ({sheet.size})")
print("SHIP_TIPCROP_OK")
