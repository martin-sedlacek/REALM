"""Sample evenly spaced frames out of a rollout mp4 into a contact sheet.

Reviewing a rollout normally means watching the mp4. When that is not practical -- headless node, or
you want a single image you can put in a doc -- this pulls N frames spread across the clip and tiles
them left-to-right, top-to-bottom, so the arc of the rollout (reach, grasp, lift, place) reads at a
glance.

    python scripts/clara/interactive/sample_video_frames.py <run_dir> --repeats 0,7 --frames 8

<run_dir> is a results directory containing videos/*.parquet. Writes
<run_dir>/frames_sheet/run<i>_sheet.png. Reads the parquet directly, so it does not care whether the
rollout came from the single-env or the vectorized path.
"""
import argparse
import io
import os

import numpy as np
import pandas as pd
from PIL import Image


def frames_from_mp4_bytes(blob, n, tail=0):
    """Decode and return n frames: evenly spaced, or the last n when tail is set."""
    import imageio.v3 as iio
    frames = list(iio.imiter(io.BytesIO(blob), plugin="pyav", format="rgb24"))
    if not frames:
        return []
    if tail:
        return frames[-min(tail, len(frames)):], len(frames)
    idx = np.linspace(0, len(frames) - 1, min(n, len(frames))).astype(int)
    return [frames[i] for i in idx], len(frames)


def sheet(images, cols=4, pad=6, bg=24):
    h, w = images[0].shape[:2]
    rows = int(np.ceil(len(images) / cols))
    canvas = np.full((rows * h + (rows + 1) * pad, cols * w + (cols + 1) * pad, 3), bg, np.uint8)
    for i, im in enumerate(images):
        r, c = divmod(i, cols)
        y, x = pad + r * (h + pad), pad + c * (w + pad)
        canvas[y:y + h, x:x + w] = im[..., :3]
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--repeats", default="", help="comma-separated repeat indices; default: all")
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--scale", type=float, default=0.5, help="downscale so the sheet stays readable")
    ap.add_argument("--tail", type=int, default=0,
                    help="instead of sampling across the clip, take the last N frames. Under "
                         "render_on_demand a clip is ~1 frame per action chunk, so an evenly "
                         "spaced sample can miss the moment that decides success; the end state "
                         "is what tells you whether the object was actually placed.")
    a = ap.parse_args()

    vdir = os.path.join(a.run_dir, "videos")
    out = os.path.join(a.run_dir, "frames_sheet")
    os.makedirs(out, exist_ok=True)
    wanted = {int(x) for x in a.repeats.split(",") if x.strip()} if a.repeats else None

    for f in sorted(os.listdir(vdir)):
        if not f.endswith(".parquet"):
            continue
        df = pd.read_parquet(os.path.join(vdir, f))
        for _, row in df.iterrows():
            rep = int(row["repeat"])
            if wanted is not None and rep not in wanted:
                continue
            got = frames_from_mp4_bytes(row["video"], a.frames, tail=a.tail)
            if not got:
                print(f"  run {rep}: no frames decoded")
                continue
            frames, total = got
            if a.scale != 1.0:
                frames = [np.asarray(Image.fromarray(x).resize(
                    (int(x.shape[1] * a.scale), int(x.shape[0] * a.scale)))) for x in frames]
            path = os.path.join(out, f"run{rep:03d}_sheet.png")
            Image.fromarray(sheet(frames, cols=a.cols)).save(path)
            print(f"  run {rep}: {total} frames in clip -> {path}")


if __name__ == "__main__":
    main()
