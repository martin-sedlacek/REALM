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
ap.add_argument("--sbs", default=None,
                help="instead of a sheet, write ONE mp4 with these rungs side by side (frame-synced), "
                     "which is the honest way to ask 'is this one visibly bendier than the default'")
ap.add_argument("--sbs-out", default=None)
ap.add_argument("--diff", default=None,
                help="ref,rungA[,rungB...] -- per rung a row [ref frame | rung frame | amplified "
                     "|difference|]. Always include a same-settings control rung so the diff has a "
                     "noise floor to be compared against.")
ap.add_argument("--diff-gain", type=float, default=3.0)
ap.add_argument("--fps", type=int, default=15)
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
if args.order:
    # An explicit --order also SELECTS: comparing two rungs means a sheet with two rows, not eight
    # sorted differently.
    order = [r for r in order if r in rungs]
else:
    order = [r for r in (order or []) if r in rungs] + [r for r in rungs if r not in (order or [])]

want = [int(x) for x in args.frames.split(",")]


def grab(path, idxs):
    with av.open(path) as c:
        fr = [f.to_ndarray(format="rgb24") for f in c.decode(video=0)]
    n = len(fr)
    return [fr[i if i >= 0 else n + i] for i in idxs], n


if args.sbs:
    names = [x.strip() for x in args.sbs.split(",")]
    missing = [n for n in names if n not in rungs]
    if missing:
        raise SystemExit(f"--sbs names {missing}; have {sorted(rungs)}")
    from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
    seqs = []
    for n in names:
        with av.open(rungs[n]) as c:
            seqs.append([f.to_ndarray(format="rgb24") for f in c.decode(video=0)])
    N = min(len(s) for s in seqs)
    out_frames = []
    for i in range(N):
        row = []
        for n, s in zip(names, seqs):
            im = s[i]
            h, w = im.shape[:2]
            ch, cw = int(h / (2 * args.zoom)), int(w / (2 * args.zoom))
            c = im[h // 2 - ch: h // 2 + ch, w // 2 - cw: w // 2 + cw]
            img = Image.fromarray(c).resize(
                (args.width, int(args.width * c.shape[0] / c.shape[1])), Image.LANCZOS)
            d2 = ImageDraw.Draw(img)
            try:
                f2 = ImageFont.load_default(size=20)
            except TypeError:
                f2 = ImageFont.load_default()
            d2.rectangle([0, 0, img.size[0], 26], fill=(0, 0, 0))
            d2.text((6, 3), n, fill=(255, 235, 120), font=f2)
            row.append(np.asarray(img))
        out_frames.append(np.concatenate(row, axis=1))
    out = args.sbs_out or os.path.join(args.root, f"{args.tag}_SBS_{'_vs_'.join(names)}.mp4")
    ImageSequenceClip(out_frames, fps=args.fps).write_videofile(out, codec="libx264", audio=False,
                                                               logger=None)
    print(f"wrote {out}  ({N} frames, {' | '.join(names)})")
    raise SystemExit(0)

if args.diff:
    # Objective visibility: |ref - rung| at the SAME frame, next to |ref - control| where the control
    # is a rung with identical settings. If the two difference images look the same, the rung is not
    # visibly different from the default -- eyeballing two stills cannot separate that honestly.
    names = [x.strip() for x in args.diff.split(",")]
    ref = names[0]
    rows_out, labs, stats = [], [], []
    for other in names[1:]:
        a, _ = grab(rungs[ref], want)
        b, _ = grab(rungs[other], want)
        row = []
        for fa, fb in zip(a, b):
            dif = np.abs(fa.astype(np.int16) - fb.astype(np.int16)).sum(axis=2)
            dif = np.clip(dif * args.diff_gain, 0, 255).astype(np.uint8)
            trio = [fa, fb, np.stack([dif] * 3, axis=2)]
            for im in trio:
                h, w = im.shape[:2]
                ch, cw = int(h / (2 * args.zoom)), int(w / (2 * args.zoom))
                c = im[h // 2 - ch: h // 2 + ch, w // 2 - cw: w // 2 + cw]
                img = Image.fromarray(c).resize(
                    (args.width, int(args.width * c.shape[0] / c.shape[1])), Image.LANCZOS)
                row.append(np.asarray(img))
        rows_out.append(np.concatenate(row, axis=1))
        # Quantify it. Eyeballing two stills cannot separate "bendier" from "the cube landed 2 mm
        # lower", and a diff image amplified 3x looks dramatic either way. The mean absolute pixel
        # difference against the SAME-SETTINGS control rung is the noise floor; a rung only counts as
        # visibly different if its number is well above that.
        mad = float(np.mean([np.abs(fa.astype(np.int16) - fb.astype(np.int16)).mean()
                             for fa, fb in zip(a, b)]))
        stats.append((other, mad))
        labs.append(f"{ref} | {other} | |diff| x{args.diff_gain}   frames {args.frames}   "
                    f"mean|diff| = {mad:.2f}/255")
    TH = 26
    W = max(t.shape[1] for t in rows_out)
    H = sum(t.shape[0] + TH for t in rows_out)
    sheet = Image.new("RGB", (W, H), (12, 12, 14))
    d = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.load_default(size=18)
    except TypeError:
        font = ImageFont.load_default()
    y = 0
    for t, lab in zip(rows_out, labs):
        d.rectangle([0, y, W, y + TH], fill=(0, 0, 0))
        d.text((8, y + 4), lab, fill=(255, 235, 120), font=font)
        y += TH
        sheet.paste(Image.fromarray(t), (0, y))
        y += t.shape[0]
    out = args.out or os.path.join(args.root, f"{args.tag}_diff.png")
    sheet.save(out)
    print(f"wrote {out}  ({W}x{H})")
    print(f"\nmean |pixel difference| vs {ref}, frames {args.frames} (out of 255):")
    base = stats[0][1] if stats else None
    for name, mad in stats:
        rel = "" if not base else f"   = {mad / base:.2f}x the control"
        print(f"  {name:<12} {mad:7.3f}{rel}"
              + ("   <- CONTROL (same settings as the reference)" if name == stats[0][0] else ""))
    raise SystemExit(0)

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
