#!/usr/bin/env python
"""Tile one frame per perturbation into a single labelled sheet, for eyeballing a whole matrix.

sample_video_frames.py answers "what happened during THIS rollout"; this answers the other
question -- "does each perturbation actually look different from Default" -- which needs the cells
side by side in one image, because a lighting or colour-jitter change is invisible when you page
through the cells one at a time.

Reads each cell's videos/*.parquet directly (the mp4 bytes live in the `video` column), so it needs
only pandas + av + PIL, i.e. it runs on the login node -- no container, no allocation.

    python scripts/clara/interactive/perturbation_contact_sheet.py \
        /logs/vector_integrity/debug --repeat 0 --which first --out /logs/.../sheet.png

<root> is the directory holding one subdirectory per cell (t0_Default, t0_VAUG, ...).
"""
import argparse
import io
import os

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

FONT = "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf"


def cell_frames(run_dir, repeat):
    """Return (first, last, n_frames) RGB arrays for one cell's rollout, or None."""
    vdir = os.path.join(run_dir, "videos")
    if not os.path.isdir(vdir):
        return None
    for fn in sorted(os.listdir(vdir)):
        if not fn.endswith(".parquet"):
            continue
        df = pd.read_parquet(os.path.join(vdir, fn))
        rows = df[df["repeat"] == repeat]
        if rows.empty:
            continue
        import av
        blob = rows.iloc[-1]["video"]          # newest row wins; parquets are appended to
        container = av.open(io.BytesIO(bytes(blob)))
        frames = [f.to_ndarray(format="rgb24") for f in container.decode(video=0)]
        if frames:
            return frames[0], frames[-1], len(frames)
    return None


def label(img, text, height=28):
    """Stack a black caption strip above an image."""
    w = img.shape[1]
    strip = Image.new("RGB", (w, height), (16, 16, 16))
    d = ImageDraw.Draw(strip)
    d.text((6, 4), text, fill=(255, 235, 120), font=ImageFont.truetype(FONT, 19))
    return np.vstack([np.asarray(strip), img])


def grid(tiles, cols, pad=8, bg=32):
    h = max(t.shape[0] for t in tiles)
    w = max(t.shape[1] for t in tiles)
    rows = int(np.ceil(len(tiles) / cols))
    canvas = np.full((rows * h + (rows + 1) * pad, cols * w + (cols + 1) * pad, 3), bg, np.uint8)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        y, x = pad + r * (h + pad), pad + c * (w + pad)
        canvas[y:y + t.shape[0], x:x + t.shape[1]] = t[..., :3]
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="dir containing one subdir per cell")
    ap.add_argument("--cells", default="", help="comma-separated subdir names; default: all")
    ap.add_argument("--repeat", type=int, default=0)
    ap.add_argument("--which", choices=["first", "last", "both"], default="first")
    ap.add_argument("--panel", choices=["full", "left", "right"], default="full",
                    help="a REALM frame is the cameras concatenated side by side (external | "
                         "wrist); the external one carries the perturbation signature, and "
                         "cropping to it doubles the resolution the sheet can afford per cell")
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    names = ([c for c in a.cells.split(",") if c.strip()] or
             sorted(d for d in os.listdir(a.root) if os.path.isdir(os.path.join(a.root, d))))
    tiles = []
    for name in names:
        got = cell_frames(os.path.join(a.root, name), a.repeat)
        if got is None:
            print(f"  {name}: no frames")
            continue
        first, last, n = got
        picks = {"first": [("f0", first)], "last": [(f"f{n-1}", last)],
                 "both": [("f0", first), (f"f{n-1}", last)]}[a.which]
        for tag, im in picks:
            if a.panel != "full":
                half = im.shape[1] // 2
                im = im[:, :half] if a.panel == "left" else im[:, half:]
            if a.scale != 1.0:
                im = np.asarray(Image.fromarray(im).resize(
                    (int(im.shape[1] * a.scale), int(im.shape[0] * a.scale))))
            tiles.append(label(im, f"{name}  rep{a.repeat} {tag}/{n}"))
        print(f"  {name}: {n} frame(s) {first.shape[1]}x{first.shape[0]}")
    if not tiles:
        raise SystemExit("nothing to tile")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    Image.fromarray(grid(tiles, a.cols)).save(a.out)
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
