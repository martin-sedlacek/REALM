"""Is the restored fingertip curl VISIBLE? Control-referenced mean |pixel difference|.

The question
------------
Three independent routes restore an INWARD tip curl of ~+0.35 deg at the authored
`naturalFrequency = 1000`. That value sits BETWEEN two rungs a predecessor judged not visible
(nf=300 -> +0.221 deg, nf=200 -> +0.537 deg, both at the pixel noise floor; only nf=100 -> +1.780 deg
read as visible). So the fix can be *correct* without being *visible*, and saying which is the whole
point of this script.

The method, which is the predecessor's and is deliberately unchanged
-------------------------------------------------------------------
"Visible" is not a threshold anyone can pick honestly by eye, so it is defined RELATIVE TO A CONTROL:

    signal = mean |pixel difference| between the two builds, same frame, same viewpoint
    floor  = mean |pixel difference| between two runs of the SAME configuration
    visible <=> signal / floor is comfortably > 1

The floor is what re-running the identical thing costs -- ray-traced sampling noise, contact-solver
jitter, any non-determinism in the stack. A signal at 1.0-1.5x the floor is indistinguishable from
having changed nothing. TWO floors are computed and BOTH are reported, because they bound different
things:

  * WITHIN-process floor: two identical rungs inside one simulator process (`nf1000a` vs `nf1000b`).
    Cheap, but shares one boot, one scene build and one camera latch.
  * CROSS-process floor: the same build run twice in two separate processes. This is the floor that
    actually applies here, because the signal pair is itself cross-process -- one process per build.
    It is the conservative one and the verdict is taken against it.

Inputs are the probe's `--raw-stills` pngs, which are UNANNOTATED. The annotated stills burn the
measured curl into the pixels, so a diff over them would partly be a diff of the text.

The viewpoint must be bit-identical across every image compared. `curl_press_direction.py --cam-freeze`
prints `CAM_LATCH`; sibling runs take that verbatim via `--cam-pose`. Without that, a camera that
re-aims off the pad origins follows the curling pad and cancels part of what is being measured.

Crops are reported at several scales rather than one, so the headline number cannot be crop-shopped:
the same crop is always applied to signal and to both floors, and every crop's ratio is printed.

    python scripts/debug_probes/val_pixel_diff.py \
        --img xflat=/logs/.../val_xflat_nf1000a_L_open_peakRAW.png \
        --img v2=/logs/.../val_v2_nf1000a_L_open_peakRAW.png \
        --cmp SIGNAL=xflat/v2 --crop-set full,z2,z4 --out /logs/.../val_diff.json
"""
import argparse
import json
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--img", action="append", default=[], metavar="LABEL=PATH",
                help="a named image. Repeat. Missing files are reported and skipped, never faked.")
ap.add_argument("--cmp", action="append", default=[], metavar="NAME=LABEL_A/LABEL_B",
                help="a comparison. Repeat. Name it SIGNAL... / FLOOR... -- any name containing "
                     "'FLOOR' is treated as a noise-floor pair when the ratios are formed.")
ap.add_argument("--crop-set", default="full,z2,z4",
                help="comma-separated crops applied to EVERY comparison. 'full' = whole frame, "
                     "'zN' = centre 1/N of each dimension, or 'x0:y0:x1:y1' for an explicit box.")
ap.add_argument("--sbs", default=None, help="write a side-by-side png here")
ap.add_argument("--sbs-order", default=None, help="comma-separated labels, left to right")
ap.add_argument("--sbs-crop", default="z4", help="crop used for the side-by-side panels")
ap.add_argument("--cap", action="append", default=[], metavar="LABEL=TEXT",
                help="caption burned under a side-by-side panel (put the curl in degrees here)")
ap.add_argument("--sbs-title", default=None)
ap.add_argument("--diff-gain", type=float, default=8.0,
                help="amplification for the |difference| panel; it is a LOOK-AT aid and the numbers "
                     "in the table are always computed on the un-amplified difference")
ap.add_argument("--out", default=None, help="write the result table here as json")
args = ap.parse_args()


def kv(items):
    out = {}
    for it in items:
        assert "=" in it, f"expected LABEL=VALUE, got {it!r}"
        k, v = it.split("=", 1)
        out[k] = v
    return out


IMGS = kv(args.img)
CAPS = kv(args.cap)

loaded, missing = {}, []
for lab, path in IMGS.items():
    if not os.path.exists(path):
        missing.append((lab, path))
        continue
    a = np.asarray(Image.open(path).convert("RGB"), dtype=np.float64)
    loaded[lab] = a
    print(f"  loaded {lab:22s} {a.shape[1]}x{a.shape[0]}  {path}")
for lab, path in missing:
    print(f"  *** MISSING {lab}: {path} -- every comparison using it is SKIPPED, not estimated ***")

assert loaded, "no images loaded"
SHAPES = {a.shape for a in loaded.values()}
assert len(SHAPES) == 1, f"images differ in size, cannot diff: {SHAPES}"
H, W = next(iter(SHAPES))[:2]


def crop_box(spec):
    if spec == "full":
        return (0, 0, W, H)
    if spec.startswith("z"):
        n = float(spec[1:])
        ch, cw = int(H / (2 * n)), int(W / (2 * n))
        return (W // 2 - cw, H // 2 - ch, W // 2 + cw, H // 2 + ch)
    x0, y0, x1, y1 = (int(v) for v in spec.split(":"))
    return (x0, y0, x1, y1)


CROPS = {s: crop_box(s) for s in args.crop_set.split(",")}


def sub(a, box):
    x0, y0, x1, y1 = box
    return a[y0:y1, x0:x1]


def stats(a, b, box):
    """mean |difference| in 0..255 units, plus how much of the frame moved at all."""
    d = np.abs(sub(a, box) - sub(b, box))
    per_px = d.mean(axis=2)                       # collapse RGB to one number per pixel
    return dict(mean_abs=float(per_px.mean()), p99=float(np.percentile(per_px, 99)),
                max=float(per_px.max()),
                frac_gt2=float((per_px > 2.0).mean()), frac_gt8=float((per_px > 8.0).mean()))


# ---------------------------------------------------------------- the table
results = {}
print(f"\n{'=' * 108}\nMEAN |PIXEL DIFFERENCE|, 0-255 units, per crop\n{'=' * 108}")
hdr = f"{'comparison':28s} {'A / B':34s}" + "".join(f"{c:>14s}" for c in CROPS)
print(hdr + "\n" + "-" * len(hdr))
for spec in args.cmp:
    name, pair = spec.split("=", 1)
    la, lb = pair.split("/", 1)
    if la not in loaded or lb not in loaded:
        print(f"{name:28s} {la + ' / ' + lb:34s}  SKIPPED (missing image)")
        results[name] = dict(a=la, b=lb, skipped=True)
        continue
    row = {}
    for cname, box in CROPS.items():
        row[cname] = stats(loaded[la], loaded[lb], box)
    results[name] = dict(a=la, b=lb, crops=row)
    print(f"{name:28s} {la + ' / ' + lb:34s}" +
          "".join(f"{row[c]['mean_abs']:14.4f}" for c in CROPS))

# ---------------------------------------------------------------- ratios against every floor
floors = [n for n in results if "FLOOR" in n.upper() and not results[n].get("skipped")]
signals = [n for n in results if "FLOOR" not in n.upper() and not results[n].get("skipped")]
if floors and signals:
    print(f"\n{'=' * 108}\nRATIO signal / floor -- this is the verdict. >1 by a margin = visible."
          f"\n{'=' * 108}")
    for f in floors:
        print(f"\n  against {f}:")
        h2 = f"    {'signal':26s}" + "".join(f"{c:>14s}" for c in CROPS)
        print(h2 + "\n    " + "-" * (len(h2) - 4))
        for s in signals:
            r = {}
            for c in CROPS:
                fl = results[f]["crops"][c]["mean_abs"]
                r[c] = float(results[s]["crops"][c]["mean_abs"] / fl) if fl > 0 else float("inf")
            results[s].setdefault("ratio", {})[f] = r
            print(f"    {s:26s}" + "".join(f"{r[c]:13.2f}x" for c in CROPS))

# ---------------------------------------------------------------- side by side
if args.sbs:
    order = (args.sbs_order.split(",") if args.sbs_order
             else [l for l in IMGS if l in loaded])
    order = [l for l in order if l in loaded]
    box = crop_box(args.sbs_crop)
    panels = [np.ascontiguousarray(sub(loaded[l], box)).astype(np.uint8) for l in order]
    labels = list(order)
    if len(order) >= 2:
        d = np.abs(sub(loaded[order[0]], box) - sub(loaded[order[1]], box))
        panels.append(np.clip(d * args.diff_gain, 0, 255).astype(np.uint8))
        labels.append(f"|{order[0]} - {order[1]}| x{args.diff_gain:g}")
    ph, pw = panels[0].shape[:2]
    scale = max(1, int(round(560 / max(pw, 1))))
    ph_s, pw_s = ph * scale, pw * scale
    top, cap_h = (34 if args.sbs_title else 0), 74
    sheet = Image.new("RGB", (pw_s * len(panels), top + ph_s + cap_h), (12, 12, 12))
    dr = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.load_default(size=max(15, pw_s // 34))
        tfont = ImageFont.load_default(size=max(17, pw_s // 28))
    except TypeError:
        font = tfont = ImageFont.load_default()
    if args.sbs_title:
        dr.text((10, 7), args.sbs_title, fill=(255, 255, 120), font=tfont)
    for j, (p, lab) in enumerate(zip(panels, labels)):
        sheet.paste(Image.fromarray(p).resize((pw_s, ph_s), Image.NEAREST), (j * pw_s, top))
        dr.multiline_text((j * pw_s + 8, top + ph_s + 5),
                          f"{lab}\n{CAPS.get(lab, '')}", fill=(255, 255, 120), font=font)
        dr.rectangle([j * pw_s, top, j * pw_s + pw_s - 1, top + ph_s - 1], outline=(70, 70, 70))
    sheet.save(args.sbs)
    print(f"\n  wrote {args.sbs}  ({sheet.size[0]}x{sheet.size[1]}, crop {args.sbs_crop} = {box})")

if args.out:
    with open(args.out, "w") as f:
        json.dump(dict(images=IMGS, crops={k: list(v) for k, v in CROPS.items()},
                       results=results, missing=dict(missing)), f, indent=2)
    print(f"  wrote {args.out}")
print("\nVAL_PIXEL_DIFF_COMPLETE")
