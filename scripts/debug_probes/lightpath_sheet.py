#!/usr/bin/env python
"""Host-side (no container, no GPU): tables + per-camera contact sheets for lightpath_sweep.py.

Separate from brightness_sheet.py on purpose. That one reads the `ladder` key and reports
mean/p50/p95/sat -- enough for a brightness question. This one reads `rows` and leads with **p5 and
%dark**, because the thing being chased is not brightness, it is lifted blacks: the 1.1.1 reference
on exterior cam1 is p5 12 / 20.4% of pixels below 60, og391 is p5 28 / 8.4%. A row that matches the
mean but leaves %dark at 8% has not reproduced the tone, and a mean-first table hides that.

The 1.1.1 reference frame is pulled in as the FIRST panel of every sheet when it can be found, so
each sheet shows the target next to the candidates rather than requiring two windows.

    /home/sedlam56/miniconda3/envs/behavior/bin/python scripts/debug_probes/lightpath_sheet.py \
        /mnt/home_lustre/sedlam56/projects/REALM_og391/logs/lightpath \
        --ref-dir /mnt/home_lustre/sedlam56/projects/REALM/logs/render_bright_ab
"""

import argparse
import glob
import json
import os

import numpy as np
from PIL import Image, ImageDraw

BAR = 46
PANEL_W = 440
PAD = 6

# The measured OG 1.1.1 reference, exterior cam1, rotate_mug / Default / DROID / rt.
REF_CAM1 = {"mean": 117.98, "p05": 12.0, "p50": 138.0, "p95": 171.0, "sat_pct": 0.0,
            "dark_pct": 20.44, "detail": 180.0}


def luma(a):
    a = np.asarray(a, dtype=np.float32)
    return 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]


def lap_var(l):
    lap = (l[:-2, 1:-1] + l[2:, 1:-1] + l[1:-1, :-2] + l[1:-1, 2:] - 4.0 * l[1:-1, 1:-1])
    return float(lap.var())


def stats_of_png(p):
    a = np.asarray(Image.open(p).convert("RGB"))
    l = luma(a)
    return {"mean": round(float(l.mean()), 2), "p05": round(float(np.percentile(l, 5)), 1),
            "p50": round(float(np.percentile(l, 50)), 1), "p95": round(float(np.percentile(l, 95)), 1),
            "sat_pct": round(100.0 * float((l >= 250).mean()), 4),
            "dark_pct": round(100.0 * float((l < 60).mean()), 2),
            "detail": round(lap_var(l), 1), "n_colors": -1, "gate_ok": True, "png": p}


def load_rows(out_dir):
    """(sort_key, label, variant, {cam: stats}) for every configuration in every report."""
    rows = []
    for jp in sorted(glob.glob(os.path.join(out_dir, "*.json"))):
        if os.path.basename(jp).startswith("carb_tree"):
            continue
        try:
            with open(jp) as f:
                r = json.load(f)
        except json.JSONDecodeError:
            print(f"  !! {jp}: truncated (run in flight?) -- skipped")
            continue
        lab = r.get("label", os.path.basename(jp))
        if "env_creation_error" in r:
            rows.append(((9, lab, ""), f"{lab}: ENV CREATION FAILED "
                         f"({r['env_creation_error']['type']})", "-", {}))
            continue
        for ent in r.get("rows", []):
            v = ent["variant"]
            pri = 0 if ent.get("kind") == "baseline" else (1 if ent.get("kind") == "applied_pre_first_render" else 5)
            rows.append(((pri, lab, v), f"{lab} / {v}", v, ent["cameras"]))
    rows.sort(key=lambda t: t[0])
    return rows


def resolve(st, out_dir):
    p = st.get("png")
    if not p:
        return None
    for c in (p, os.path.join(out_dir, os.path.basename(p))):
        if os.path.exists(c):
            return c
    return None


def fit_offset(cand_png, ref_png):
    """Least-squares `cand_luma ~= a * ref_luma + b` over pixel-aligned frames.

    `b` is the primary metric of this whole investigation: the gap between the stacks is ADDITIVE
    (slope ~1, offset ~+67 luma), so a change that lowers `b` is closing the gap and a change that
    only lowers the mean may just be scaling an image that still has the floor in it. Frames must be
    the same camera at the same framing, which is why the sheets are per-camera.
    """
    try:
        a_img = np.asarray(Image.open(cand_png).convert("RGB"), dtype=np.float32)
        r_img = np.asarray(Image.open(ref_png).convert("RGB"), dtype=np.float32)
    except Exception:
        return None
    if a_img.shape != r_img.shape:
        return None
    y, x = luma(a_img).ravel(), luma(r_img).ravel()
    A = np.stack([x, np.ones_like(x)], 1)
    (a, b), *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - (a * x + b)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float((resid ** 2).sum()) / ss_tot if ss_tot > 0 else float("nan")
    # What the mean would become if the flat offset were simply subtracted and clipped.
    corrected = float(np.clip(y - b, 0, 255).mean())
    return {"slope": round(float(a), 4), "offset": round(float(b), 2), "r2": round(r2, 4),
            "mean_minus_offset": round(corrected, 2)}


def find_ref(ref_dir, cam):
    """The settled OG 1.1.1 baseline PNG for this camera, if the earlier probe left one behind."""
    if not ref_dir:
        return None
    tag = cam.replace(".", "-")
    for pat in (f"og111_rt_clean__baseline__{tag}.png", f"og111*baseline*{tag}.png"):
        g = sorted(glob.glob(os.path.join(ref_dir, pat)))
        if g:
            return g[0]
    return None


def sheet(panels, path, title):
    if not panels:
        return None
    n = len(panels)
    ncol = 4 if n > 6 else (3 if n > 4 else min(n, 2))
    nrow = (n + ncol - 1) // ncol
    probe = Image.open(panels[0][1]["png"])
    pw = PANEL_W
    ph = max(1, int(round(probe.height * pw / probe.width)))
    W = ncol * pw + (ncol + 1) * PAD
    H = nrow * (ph + BAR) + (nrow + 1) * PAD + 32
    sh = Image.new("RGB", (W, H), (22, 22, 24))
    d = ImageDraw.Draw(sh)
    d.text((PAD, 9), title, fill=(238, 238, 238))
    for i, (lbl, st) in enumerate(panels):
        r, c = divmod(i, ncol)
        x = PAD + c * (pw + PAD)
        y = 32 + PAD + r * (ph + BAR + PAD)
        sh.paste(Image.open(st["png"]).convert("RGB").resize((pw, ph), Image.LANCZOS), (x, y))
        ok = st.get("gate_ok", False)
        d.rectangle([x, y + ph, x + pw, y + ph + BAR], fill=(38, 38, 42) if ok else (95, 28, 28))
        d.text((x + 4, y + ph + 3), lbl[:70], fill=(242, 242, 242))
        d.text((x + 4, y + ph + 17),
               f"mean {st['mean']:.1f}   p5 {st['p05']:.0f}   p50 {st['p50']:.0f}   p95 {st['p95']:.0f}",
               fill=(190, 200, 212) if ok else (255, 200, 200))
        f = st.get("_fit")
        d.text((x + 4, y + ph + 31),
               f"%dark {st['dark_pct']:.1f}   sat {st['sat_pct']:.2f}%   detail {st['detail']:.0f}"
               + (f"   offset {f['offset']:+.1f}" if f else "")
               + ("" if ok else f"  GATE FAIL {st.get('gate_fail')}"),
               fill=(190, 200, 212) if ok else (255, 200, 200))
    sh.save(path)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--ref-dir", default="/mnt/home_lustre/sedlam56/projects/REALM/logs/render_bright_ab")
    ap.add_argument("--sheet-dir", default=None)
    ap.add_argument("--sort", default="offset", choices=["offset", "dark", "mean", "name"],
                    help="table order. 'offset' (default) ranks by the fitted additive floor, which "
                         "is the metric that matters: the gap is additive, so the smallest offset "
                         "is the closest to 1.1.1 regardless of what the mean does")
    args = ap.parse_args()
    sheet_dir = args.sheet_dir or args.out_dir
    os.makedirs(sheet_dir, exist_ok=True)

    rows = load_rows(args.out_dir)
    if not rows:
        raise SystemExit(f"no lightpath reports under {args.out_dir}")
    cams = sorted({c for _k, _l, _v, st in rows for c in st})
    print(f"{len(rows)} configuration(s), cameras: {cams}\n")

    for cam in cams:
        print(f"===== {cam} " + "=" * max(0, 92 - len(cam)))
        ref_png = find_ref(args.ref_dir, cam)
        ref_st = stats_of_png(ref_png) if ref_png else None
        print(f"{'configuration':44s} {'mean':>8s} {'p5':>6s} {'p50':>6s} {'p95':>6s} "
              f"{'sat%':>7s} {'%dark':>7s} {'detail':>8s} {'offset':>8s} {'slope':>6s} {'R2':>5s}")
        if ref_st:
            print(f"{'OG 1.1.1 REFERENCE (measured)':44s} {ref_st['mean']:8.2f} {ref_st['p05']:6.1f} "
                  f"{ref_st['p50']:6.1f} {ref_st['p95']:6.1f} {ref_st['sat_pct']:7.3f} "
                  f"{ref_st['dark_pct']:7.2f} {ref_st['detail']:8.1f} {0.0:8.2f} {1.0:6.3f} "
                  f"{1.0:5.2f}   <== TARGET")
        table = []
        for _k, lbl, _v, st in rows:
            if cam not in st:
                continue
            s = st[cam]
            if not s.get("gate_ok", False):
                print(f"{lbl:44s} {'--':>8s} {'--':>6s} {'--':>6s} {'--':>6s} {'--':>7s} "
                      f"{'--':>7s} {'--':>8s}   GATE FAIL {s.get('gate_fail')}")
                continue
            p = resolve(s, args.out_dir)
            s = dict(s, _fit=(fit_offset(p, ref_png) if (p and ref_png) else None))
            table.append((lbl, s))
        if args.sort == "offset":
            table.sort(key=lambda t: (t[1]["_fit"] or {}).get("offset", 9e9))
        elif args.sort == "dark" and ref_st:
            table.sort(key=lambda t: abs(t[1]["dark_pct"] - ref_st["dark_pct"]))
        elif args.sort == "mean" and ref_st:
            table.sort(key=lambda t: abs(t[1]["mean"] - ref_st["mean"]))
        for lbl, s in table:
            f = s.get("_fit")
            tail = (f"{f['offset']:8.2f} {f['slope']:6.3f} {f['r2']:5.2f}" if f
                    else f"{'--':>8s} {'--':>6s} {'--':>5s}")
            print(f"{lbl:44s} {s['mean']:8.2f} {s['p05']:6.1f} {s['p50']:6.1f} {s['p95']:6.1f} "
                  f"{s['sat_pct']:7.3f} {s['dark_pct']:7.2f} {s['detail']:8.1f} {tail}")
        print()

        panels = []
        if ref_st:
            panels.append(("OG 1.1.1 REFERENCE (target)", ref_st))
        for lbl, s in table:
            p = resolve(s, args.out_dir)
            if p:
                panels.append((lbl, dict(s, png=p)))
        got = sheet(panels, os.path.join(sheet_dir, f"lightpath_sheet__{cam.replace('.', '-')}.png"),
                    f"light transport: OG 1.1.1 target vs og391 variants -- {cam}")
        if got:
            print(f"sheet: {got}\n")


if __name__ == "__main__":
    main()
