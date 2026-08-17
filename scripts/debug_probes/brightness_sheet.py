#!/usr/bin/env python
"""Host-side (no container, no GPU): tile render_brightness_ab.py's frames into contact sheets.

One sheet per camera, so panels in a column are always the same view at the same framing -- two
panels from different cameras are never put side by side. Rows are (stack, variant), ordered with
both stacks' baselines first so the gap being explained is the top two rows.

    /home/sedlam56/miniconda3/envs/behavior/bin/python scripts/debug_probes/brightness_sheet.py \
        /mnt/home_lustre/sedlam56/projects/REALM/logs/render_bright_ab

Also prints the luminance table across every JSON found, which is the actual deliverable; the sheet
is so a human can check that the numbers describe what they think they describe.
"""

import argparse
import glob
import json
import os

import numpy as np
from PIL import Image, ImageDraw

BAR = 34           # label strip height under each panel
PANEL_W = 480      # panels are downscaled to this width
PAD = 6


def load(out_dir):
    runs = []
    for jp in sorted(glob.glob(os.path.join(out_dir, "*.json"))):
        with open(jp) as f:
            try:
                r = json.load(f)
            except json.JSONDecodeError:
                print(f"  !! {jp}: truncated JSON (run still in flight?) -- skipped")
                continue
        r["_json"] = jp
        runs.append(r)
    return runs


def rows_from(runs):
    """(sort_key, row_label, stack, variant, {camera: stats}) per configuration."""
    rows = []
    for r in runs:
        stack = r.get("identity", {}).get("stack", "?")
        label = r.get("label", "?")
        if "env_creation_error" in r:
            rows.append(((0 if stack == "og111" else 1, 9, label, ""),
                         f"{label}  ENV CREATION FAILED: "
                         f"{r['env_creation_error']['type']}", stack, "-", {}))
            continue
        for ent in r.get("ladder", []):
            v = ent["variant"]
            # baseline rows first, 1.1.1 above og391
            pri = 0 if v == "baseline" else 5
            rows.append(((pri, 0 if stack == "og111" else 1, label, v),
                         f"{label} / {v}", stack, v, ent["cameras"]))
    rows.sort(key=lambda t: t[0])
    return rows


def sheet(rows, cam, path, title):
    have = [(lbl, st[cam]) for _k, lbl, _s, _v, st in rows if cam in st and st[cam].get("png")
            and os.path.exists(st[cam]["png"])]
    if not have:
        return None
    n = len(have)
    ncol = 3 if n > 4 else min(n, 2)
    nrow = (n + ncol - 1) // ncol

    probe = Image.open(have[0][1]["png"])
    pw = PANEL_W
    ph = max(1, int(round(probe.height * pw / probe.width)))
    W = ncol * pw + (ncol + 1) * PAD
    H = nrow * (ph + BAR) + (nrow + 1) * PAD + 30
    sh = Image.new("RGB", (W, H), (24, 24, 26))
    d = ImageDraw.Draw(sh)
    d.text((PAD, 8), title, fill=(235, 235, 235))

    for i, (lbl, st) in enumerate(have):
        r, c = divmod(i, ncol)
        x = PAD + c * (pw + PAD)
        y = 30 + PAD + r * (ph + BAR + PAD)
        im = Image.open(st["png"]).convert("RGB").resize((pw, ph), Image.LANCZOS)
        sh.paste(im, (x, y))
        ok = st.get("gate_ok", False)
        d.rectangle([x, y + ph, x + pw, y + ph + BAR], fill=(40, 40, 44) if ok else (90, 30, 30))
        d.text((x + 4, y + ph + 3), lbl[:74], fill=(240, 240, 240))
        d.text((x + 4, y + ph + 17),
               f"mean {st['mean']:.1f}  p50 {st['p50']:.0f}  p95 {st['p95']:.0f}  "
               f"sat {st['sat_pct']:.2f}%  colours {st['n_colors']}"
               + ("" if ok else f"   GATE FAIL: {st.get('gate_fail')}"),
               fill=(190, 200, 210) if ok else (255, 200, 200))
    sh.save(path)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--sheet-dir", default=None)
    args = ap.parse_args()
    sheet_dir = args.sheet_dir or args.out_dir
    os.makedirs(sheet_dir, exist_ok=True)

    runs = load(args.out_dir)
    if not runs:
        raise SystemExit(f"no JSON reports under {args.out_dir}")
    rows = rows_from(runs)

    cams = sorted({c for _k, _l, _s, _v, st in rows for c in st})
    print(f"\n{len(runs)} report(s), {len(rows)} configuration(s), cameras: {cams}\n")

    # ---- the table ----
    for cam in cams:
        print(f"===== {cam} " + "=" * max(0, 76 - len(cam)))
        print(f"{'configuration':46s} {'mean':>8s} {'p50':>7s} {'p95':>7s} {'sat%':>8s} "
              f"{'colours':>8s}  gate")
        # The reference is the og111 BASELINE -- but only if it passed the gate. Quoting a ratio
        # against a frame that failed is how a broken render turns into a confident number: the
        # first og111 run's baseline was 87% pure white with mean 225, and every og391 row was
        # duly reported as "x0.70 vs og111" against it.
        ref, ref_label = None, None
        for _k, lbl, stack, v, st in rows:
            if stack == "og111" and v == "baseline" and cam in st and st[cam].get("gate_ok"):
                ref, ref_label = st[cam]["mean"], lbl
                break
        for _k, lbl, stack, v, st in rows:
            if cam not in st:
                continue
            s = st[cam]
            if not s.get("gate_ok", False):
                print(f"{lbl:46s} {'--':>8s} {'--':>7s} {'--':>7s} {'--':>8s} {'--':>8s}  "
                      f"FAIL {s.get('gate_fail')}")
                continue
            rat = f"  x{s['mean']/ref:.2f} vs {ref_label}" if ref else ""
            print(f"{lbl:46s} {s['mean']:8.2f} {s['p50']:7.1f} {s['p95']:7.1f} "
                  f"{s['sat_pct']:8.3f} {s['n_colors']:8d}  ok{rat}")
        print()

    # ---- the sheets ----
    for cam in cams:
        p = os.path.join(sheet_dir, f"contact_sheet__{cam.replace('.', '-')}.png")
        got = sheet(rows, cam, p, f"REALM render brightness: 1.1.1 vs og391 -- {cam}")
        if got:
            print(f"sheet: {got}")

    # ---- carb readback, side by side: the mechanism ----
    print("\n===== carb readback after env creation (as-shipped, per stack) =====")
    snaps = {}
    for r in runs:
        st = r.get("identity", {}).get("stack")
        if st and "carb_readback_after_env_creation" in r and st not in snaps:
            snaps[st] = (r["label"], r["carb_readback_after_env_creation"])
    if len(snaps) >= 1:
        keys = sorted({k for _l, s in snaps.values() for k in s})
        order = list(snaps.keys())
        print(f"{'carb key':52s} " + " ".join(f"{snaps[o][0][:22]:>24s}" for o in order))
        for k in keys:
            vals = [snaps[o][1].get(k) for o in order]
            mark = "  <== DIFFERS" if len(set(map(str, vals))) > 1 else ""
            print(f"{k:52s} " + " ".join(f"{str(v)[:22]:>24s}" for v in vals) + mark)


if __name__ == "__main__":
    main()
