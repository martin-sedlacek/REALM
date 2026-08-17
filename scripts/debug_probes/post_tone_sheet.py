#!/usr/bin/env python
"""Host-side (no container, no GPU): the table and the contact sheets for post_tone_sweep.py.

Exists separately from brightness_sheet.py because that one prints mean / p50 / p95 / sat only, and
mean alone cannot distinguish an exposure change from a materials change. Everything here carries the
full set -- mean, p5, p50, p95, sat%, %dark (share below luma 60), detail (variance of the Laplacian)
-- for every row, and the reference frame is scored with the SAME metric implementation
(post_tone_sweep.frame_metrics) rather than quoted from another run's numbers.

ONE SHEET PER CAMERA, so a column is never two different views. The 1.1.1 reference panel for that
camera goes first when one is supplied, so the comparison a human is asked to eyeball is inside a
single image.

    python scripts/debug_probes/post_tone_sheet.py \
        --json  /mnt/.../logs/render_bright_ab/og391_post_t3.json \
        --ref-dir /mnt/.../logs/render_bright_ab --ref-prefix og111_rt_clean__baseline__ \
        --sheet-dir /mnt/.../logs/post_tone
"""

import argparse
import glob
import json
import os

import numpy as np
from PIL import Image, ImageDraw

from post_tone_sweep import frame_metrics          # ONE metric implementation, never two

BAR = 46
PANEL_W = 430
PAD = 6
HEAD = 34

# Rows that are always on the sheet regardless of how close they land: the canaries decide whether
# the table is data at all, and the named-diff rows are the hypothesis under test.
ALWAYS = ("baseline", "baseline_end", "canary_iso_x8", "canary_srgb_off", "invertToneMap_off",
          "invertColorCorrection_off", "invert_both_off", "post_match_ref")


def hostpath(p, ref_dir):
    """The probe records the path it wrote INSIDE the container (/logs/...), which does not exist on
    the host. Same basename, host directory."""
    if not p:
        return None
    for c in (p, os.path.join(ref_dir, os.path.basename(p))):
        if os.path.exists(c):
            return c
    return None


def load_rows(paths):
    """[(label, variant, note, {cam: stats})] over every report, baseline row first per report."""
    rows = []
    for jp in paths:
        try:
            r = json.load(open(jp))
        except json.JSONDecodeError:
            print(f"  !! {jp}: truncated JSON (run still in flight?) -- skipped")
            continue
        lab = r.get("label", "?")
        if "env_creation_error" in r:
            print(f"  !! {lab}: ENV CREATION FAILED {r['env_creation_error']['type']}")
            continue
        if r.get("baseline"):
            rows.append((lab, "baseline", "as-shipped for this stack", r["baseline"]))
        for e in r.get("ladder", []):
            rows.append((lab, e["variant"], e.get("note", ""), e["cameras"]))
        sol = r.get("solve", {})
        if sol.get("best_full"):
            rows.append((lab, f"SOLVED {sol['key'].rsplit('/', 1)[-1]}={sol['best']['value']}",
                         f"bisected to mean {sol['target']}", sol["best_full"]))
    return rows


def table(rows, cam, ref):
    hdr = (f"{'configuration':44s} {'mean':>8s} {'p5':>6s} {'p50':>6s} {'p95':>6s} {'sat%':>7s} "
           f"{'dark%':>7s} {'detail':>8s}  gate")
    print(f"===== {cam} " + "=" * max(0, 84 - len(cam)))
    if ref:
        print(f"{'*** 1.1.1 REFERENCE':44s} {ref['mean']:8.2f} {ref['p05']:6.1f} {ref['p50']:6.1f} "
              f"{ref['p95']:6.1f} {ref['sat_pct']:7.3f} {ref['dark_pct']:7.2f} {ref['detail']:8.1f}")
    print(hdr)
    for lab, var, _note, st in rows:
        if cam not in st:
            continue
        s = st[cam]
        name = f"{lab}/{var}"[:44]
        if not s.get("gate_ok", True):
            print(f"{name:44s} {'--':>8s} {'--':>6s} {'--':>6s} {'--':>6s} {'--':>7s} {'--':>7s} "
                  f"{'--':>8s}  FAIL {s.get('gate_fail')}")
            continue
        d = ""
        if ref:
            d = (f"  dmean {s['mean'] - ref['mean']:+7.2f}  ddark "
                 f"{s['dark_pct'] - ref['dark_pct']:+6.2f}")
        print(f"{name:44s} {s['mean']:8.2f} {s['p05']:6.1f} {s['p50']:6.1f} {s['p95']:6.1f} "
              f"{s['sat_pct']:7.3f} {s['dark_pct']:7.2f} {s['detail']:8.1f}  ok{d}")
    print()


def pick(rows, cam, ref, top):
    """Sheet panels: the always-on rows, then the closest to the reference by mean, in ladder order."""
    have = [(lab, v, st[cam]) for lab, v, _n, st in rows if cam in st and st[cam].get("png")]
    keep = [t for t in have if t[1] in ALWAYS or t[1].startswith("SOLVED")]
    rest = [t for t in have if t not in keep]
    if ref:
        rest.sort(key=lambda t: abs(t[2]["mean"] - ref["mean"]))
    keep += rest[:max(0, top)]
    order = {id(t): i for i, t in enumerate(have)}
    keep.sort(key=lambda t: order[id(t)])
    return keep


def sheet(panels, cam, path, title, ref_dir, ref=None, ref_png=None):
    items = []
    if ref_png:
        items.append(("*** OG 1.1.1 REFERENCE", ref, ref_png))
    for lab, v, st in panels:
        p = hostpath(st.get("png"), ref_dir)
        if p:
            items.append((f"{lab}/{v}", st, p))
    if not items:
        return None
    n = len(items)
    ncol = 3 if n > 4 else min(n, 2)
    nrow = (n + ncol - 1) // ncol
    probe = Image.open(items[0][2])
    pw = PANEL_W
    ph = max(1, int(round(probe.height * pw / probe.width)))
    W = ncol * pw + (ncol + 1) * PAD
    H = nrow * (ph + BAR) + (nrow + 1) * PAD + HEAD
    sh = Image.new("RGB", (W, H), (24, 24, 26))
    d = ImageDraw.Draw(sh)
    d.text((PAD, 6), title, fill=(235, 235, 235))
    d.text((PAD, 19), "%dark = share of pixels below luma 60   detail = variance of the Laplacian",
           fill=(150, 155, 165))
    for i, (lbl, st, p) in enumerate(items):
        r, c = divmod(i, ncol)
        x = PAD + c * (pw + PAD)
        y = HEAD + PAD + r * (ph + BAR + PAD)
        sh.paste(Image.open(p).convert("RGB").resize((pw, ph), Image.LANCZOS), (x, y))
        ok = st.get("gate_ok", True)
        isref = lbl.startswith("***")
        d.rectangle([x, y + ph, x + pw, y + ph + BAR],
                    fill=(30, 55, 40) if isref else ((40, 40, 44) if ok else (90, 30, 30)))
        d.text((x + 4, y + ph + 3), lbl[:66], fill=(190, 240, 200) if isref else (240, 240, 240))
        d.text((x + 4, y + ph + 17),
               f"mean {st['mean']:6.1f}  p5 {st['p05']:.0f}  p50 {st['p50']:.0f}  "
               f"p95 {st['p95']:.0f}", fill=(190, 200, 210))
        d.text((x + 4, y + ph + 31),
               f"sat {st['sat_pct']:.3f}%  dark {st['dark_pct']:.2f}%  detail {st['detail']:.0f}"
               + ("" if ok else f"   GATE FAIL {st.get('gate_fail')}"),
               fill=(190, 200, 210) if ok else (255, 200, 200))
    sh.save(path)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", nargs="+", required=True, help="probe report JSONs (globs allowed)")
    ap.add_argument("--ref-dir", required=True, help="host dir holding the probe's PNGs")
    ap.add_argument("--ref-prefix", default=None,
                    help="prefix of the 1.1.1 reference PNGs in --ref-dir, e.g. "
                         "og111_rt_clean__baseline__ . Scored here with the same metrics.")
    ap.add_argument("--sheet-dir", required=True)
    ap.add_argument("--top", type=int, default=11, help="closest-to-reference panels per sheet")
    ap.add_argument("--title", default="og391 /rtx/post/* tone sweep vs OG 1.1.1")
    args = ap.parse_args()

    paths = sorted({p for g in args.json for p in (glob.glob(g) or [g])})
    rows = load_rows(paths)
    if not rows:
        raise SystemExit(f"no usable rows in {paths}")
    os.makedirs(args.sheet_dir, exist_ok=True)
    cams = sorted({c for _l, _v, _n, st in rows for c in st})
    print(f"\n{len(paths)} report(s), {len(rows)} configuration(s), cameras: {cams}\n")

    refs, ref_pngs = {}, {}
    if args.ref_prefix:
        for cam in cams:
            p = os.path.join(args.ref_dir, f"{args.ref_prefix}{cam.replace('.', '-')}.png")
            if os.path.exists(p):
                refs[cam] = frame_metrics(np.asarray(Image.open(p).convert("RGB")))
                ref_pngs[cam] = p
            else:
                print(f"  (no reference PNG for {cam} at {p})")

    for cam in cams:
        table(rows, cam, refs.get(cam))

    for cam in cams:
        out = os.path.join(args.sheet_dir, f"contact_sheet__{cam.replace('.', '-')}.png")
        got = sheet(pick(rows, cam, refs.get(cam), args.top), cam, out,
                    f"{args.title} -- {cam}", args.ref_dir, refs.get(cam), ref_pngs.get(cam))
        print(f"sheet: {got}" if got else f"  (no panels for {cam})")


if __name__ == "__main__":
    main()
