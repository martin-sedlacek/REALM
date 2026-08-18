#!/usr/bin/env python
"""Host-side (no container, no GPU): the REALM_LIGHT_FIX off/on comparison across all ten tasks.

scene_sweep_sheet.py answers "on which tasks do the two STACKS disagree". This one answers "what
does the FLAG do to each task", so the sheet gains a third column: rows are tasks, columns are
flag-off / flag-on / the OG 1.1.1 reference, one sheet per camera.

The 1.1.1 column is REUSED, never re-rendered: logs/scene_sweep measured all ten tasks on 1.1.1 at
this exact schedule (Default / DROID / rt, 300 pre-renders, 5-frame median), and its frames_native/
holds every reference PNG. Metrics come from that sweep's own JSON where it has them and are
RE-SCORED from the PNG otherwise, always through post_tone_sweep.frame_metrics, so every number on a
sheet comes from one implementation.

A frame whose gate failed is never quoted as a datum -- red bar, no ratio, and it is counted as a
hard error in the summary rather than silently averaged in.

    python scripts/debug_probes/lightfix_sheet.py \
        --raw   /mnt/.../logs/lightfix_10task/frames \
        --ref   /mnt/.../logs/scene_sweep \
        --out   /mnt/.../logs/lightfix_10task
"""

import argparse
import json
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from post_tone_sweep import frame_metrics          # ONE metric implementation, never two

TASKS = ["put_green_block_into_bowl", "put_banana_into_box", "rotate_marker", "rotate_mug",
         "pick_spoon", "pick_water_bottle", "stack_cubes", "push_switch", "open_drawer",
         "close_drawer"]

# The 1.1.1 frame prefix per task inside the reference sweep's frames_native/. Tasks 0/3/7 were
# measured in an earlier session under different labels; the reference sweep copied those frames in
# under their original names, so they are named here rather than derived.
REF_PREFIX = {0: "og111_post_t0__baseline__", 3: "og111_rt_clean__baseline__",
              7: "og111_post_t7__baseline__"}

# Scenes where the two stacks are NOT photographing the same content, so a mean ratio measures
# layout and not tone. Carried over from scene_sweep_sheet.py, which established each one from the
# frames. Still reported; never counted as a lighting result.
CONFOUNDED = (2, 6, 9)
FLAGS = {2: "og391: objects off the surface", 6: "og391: objects off the surface",
         8: "1.1.1 has one extra object", 9: "og391: drawer not open at reset"}

ARMS = ("off", "on")
OFF_TOL = 0.05

PANEL_W = 430
PAD = 6
BAR = 50
HEAD = 80
BG = (24, 24, 26)


def fonts():
    for p in ("/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if os.path.exists(p):
            b = p.replace("DejaVuSans.ttf", "DejaVuSans-Bold.ttf")
            bold = b if os.path.exists(b) else p
            return (ImageFont.truetype(bold, 15), ImageFont.truetype(p, 11),
                    ImageFont.truetype(bold, 12), ImageFont.truetype(p, 11))
    d = ImageFont.load_default()
    return d, d, d, d


def score(png):
    return frame_metrics(np.asarray(Image.open(png).convert("RGB")))


def load_run(raw, label):
    jp = os.path.join(raw, f"{label}.json")
    if not os.path.exists(jp):
        return None
    try:
        rep = json.load(open(jp))
    except json.JSONDecodeError:
        print(f"  !! {jp}: truncated JSON (run still in flight?) -- skipped")
        return None
    if "env_creation_error" in rep:
        e = rep["env_creation_error"]
        print(f"  !! {label}: ENV CREATION FAILED {e['type']}: {e['msg'][:120]}")
        return None
    if not rep.get("baseline"):
        print(f"  !! {label}: no baseline yet -- skipped")
        return None
    return rep


def collect(raw, ref_dir, extra_arms=()):
    """{task_id: {cam: {arm: stats}}} plus the per-run light-prim readback."""
    ref_frames = os.path.join(ref_dir, "frames_native")
    ref_table = {}
    jp = os.path.join(ref_dir, "scene_sweep_table.json")
    if os.path.exists(jp):
        ref_table = json.load(open(jp)).get("tasks", {})

    data, lights = {}, {}
    for tid, task in enumerate(TASKS):
        per_cam = {}
        for arm in tuple(ARMS) + tuple(extra_arms):
            rep = load_run(raw, f"lf_t{tid}_{arm}")
            if not rep:
                continue
            lights[(tid, arm)] = rep.get("light_state", {})
            for cam, st in rep["baseline"].items():
                png = os.path.join(raw, f"lf_t{tid}_{arm}__baseline__{cam.replace('.', '-')}.png")
                st = dict(st)
                st["png"] = png if os.path.exists(png) else None
                per_cam.setdefault(cam, {})[arm] = st
        # --- the reused 1.1.1 reference -------------------------------------------------------
        pref = REF_PREFIX.get(tid, f"og111_ss_t{tid}__baseline__")
        if os.path.isdir(ref_frames):
            for png in sorted(f for f in os.listdir(ref_frames) if f.startswith(pref)):
                cam = png[len(pref):-len(".png")].replace("-", ".", 1)
                p = os.path.join(ref_frames, png)
                # Prefer the reference sweep's own recorded numbers; re-score only to fill gaps, so
                # this column is byte-for-byte the table already published in logs/scene_sweep.
                st = (ref_table.get(task, {}).get("cameras", {}).get(cam, {}) or {}).get("og111")
                st = dict(st) if st else score(p)
                st["png"] = p
                per_cam.setdefault(cam, {})["og111"] = st
        if per_cam:
            data[tid] = per_cam
    return data, lights


def ratio(byst, arm):
    a, b = byst.get("og111"), byst.get(arm)
    if not a or not b or not a.get("gate_ok", True) or not b.get("gate_ok", True):
        return None
    return b["mean"] / a["mean"] if a["mean"] else None


def spread(vals):
    vals = [v for v in vals if v is not None]
    return (max(vals) - min(vals)) if len(vals) > 1 else None


def table(data, cam, extra_arms=()):
    arms = ("og111",) + tuple(ARMS) + tuple(extra_arms)
    print(f"===== {cam} " + "=" * max(0, 92 - len(cam)))
    print(f"{'task':28s} {'arm':9s} {'mean':>8s} {'p5':>6s} {'p50':>6s} {'p95':>6s} "
          f"{'sat%':>7s} {'dark%':>7s} {'detail':>8s} {'ratio':>7s}  gate")
    for tid in sorted(data):
        byst = data[tid].get(cam, {})
        for arm in arms:
            s = byst.get(arm)
            if not s:
                continue
            r = ratio(byst, arm) if arm != "og111" else None
            if not s.get("gate_ok", True):
                print(f"{TASKS[tid][:28]:28s} {arm:9s} {'--':>8s} {'--':>6s} {'--':>6s} {'--':>6s} "
                      f"{'--':>7s} {'--':>7s} {'--':>8s} {'--':>7s}  FAIL {s.get('gate_fail')}")
                continue
            print(f"{TASKS[tid][:28]:28s} {arm:9s} {s['mean']:8.2f} {s['p05']:6.1f} {s['p50']:6.1f} "
                  f"{s['p95']:6.1f} {s['sat_pct']:7.3f} {s['dark_pct']:7.2f} {s['detail']:8.1f} "
                  f"{('x%.3f' % r) if r else '':>7s}  ok")
    print()


def sheet(data, cam, path, title, subtitle, bottom, bottom2=""):
    f_title, f_sub, f_lbl, f_num = fonts()
    rows = [t for t in sorted(data) if cam in data[t]]
    if not rows:
        return None
    probe = None
    for t in rows:
        for st in data[t][cam].values():
            if st.get("png") and os.path.exists(st["png"]):
                probe = Image.open(st["png"])
                break
        if probe:
            break
    if probe is None:
        return None
    cols = ("off", "on", "og111")
    pw = PANEL_W
    ph = max(1, int(round(probe.height * pw / probe.width)))
    W = len(cols) * pw + (len(cols) + 1) * PAD
    H = len(rows) * (ph + BAR + PAD) + PAD + HEAD
    sh = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(sh)
    d.text((PAD, 5), title, fill=(238, 238, 238), font=f_title)
    d.text((PAD, 24), subtitle, fill=(150, 155, 165), font=f_sub)
    d.text((PAD, 38), bottom, fill=(255, 214, 102), font=f_sub)
    if bottom2:
        d.text((PAD, 52), bottom2, fill=(214, 160, 110), font=f_sub)
    d.text((PAD, 66), "columns:  REALM_LIGHT_FIX=0 (stock 3.9.1)   |   REALM_LIGHT_FIX=1 "
                      "(1.1.1 lighting)   |   OG 1.1.1 reference, reused",
           fill=(130, 150, 170), font=f_sub)

    for r, tid in enumerate(rows):
        byst = data[tid][cam]
        y = HEAD + PAD + r * (ph + BAR + PAD)
        for c, arm in enumerate(cols):
            x = PAD + c * (pw + PAD)
            st = byst.get(arm)
            if st is None:
                d.rectangle([x, y, x + pw, y + ph + BAR], fill=(46, 30, 30))
                d.text((x + 6, y + ph // 2), f"{TASKS[tid]}  {arm}: NO DATA",
                       fill=(255, 180, 180), font=f_lbl)
                continue
            p = st.get("png")
            if p and os.path.exists(p):
                sh.paste(Image.open(p).convert("RGB").resize((pw, ph), Image.LANCZOS), (x, y))
            rat = ratio(byst, arm) if arm != "og111" else None
            ok = st.get("gate_ok", True)
            if not ok:
                bar = (110, 30, 30)
            elif arm == "og111":
                bar = (30, 55, 40)
            elif rat is not None and abs(rat - 1.0) > OFF_TOL:
                bar = (96, 52, 24)
            else:
                bar = (40, 40, 44)
            d.rectangle([x, y + ph, x + pw, y + ph + BAR], fill=bar)
            tag = {"off": "REALM_LIGHT_FIX=0  (as shipped)",
                   "on": "REALM_LIGHT_FIX=1  (1.1.1 lighting)",
                   "og111": "OG 1.1.1 reference"}[arm]
            if rat is not None:
                tag += f"   x{rat:.3f} vs 1.1.1"
                if abs(rat - 1.0) > OFF_TOL:
                    tag += "  <-- CONFOUNDED" if tid in CONFOUNDED else "  <-- OFF"
            head = f"{TASKS[tid]}  [{tid}]"
            if arm == "og111" and FLAGS.get(tid):
                head += f"   -- {FLAGS[tid]}"
            d.text((x + 5, y + ph + 3), head[:60],
                   fill=(200, 245, 210) if arm == "og111" else (240, 240, 240), font=f_lbl)
            d.text((x + 5, y + ph + 18), tag[:88],
                   fill=(170, 215, 185) if arm == "og111" else (235, 205, 160), font=f_num)
            if ok:
                d.text((x + 5, y + ph + 33),
                       f"mean {st['mean']:.1f}   p5 {st['p05']:.0f}  p50 {st['p50']:.0f}  "
                       f"p95 {st['p95']:.0f}   sat {st['sat_pct']:.3f}%  dark {st['dark_pct']:.2f}%"
                       f"   detail {st['detail']:.0f}", fill=(190, 200, 210), font=f_num)
            else:
                d.text((x + 5, y + ph + 33), f"GATE FAIL {st.get('gate_fail')}"[:106],
                       fill=(255, 190, 190), font=f_num)
    sh.save(path)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="dir holding this sweep's lf_t*.json and PNGs")
    ap.add_argument("--ref", required=True, help="logs/scene_sweep -- the reused 1.1.1 reference")
    ap.add_argument("--out", required=True)
    ap.add_argument("--extra-arms", nargs="*", default=["stockctl"],
                    help="further arms to TABULATE but not draw (e.g. the MODE=stock control)")
    ap.add_argument("--pre-renders", type=int, default=300)
    ap.add_argument("--frames", type=int, default=5)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    data, lights = collect(args.raw, args.ref, args.extra_arms)
    if not data:
        raise SystemExit(f"no usable reports in {args.raw}")
    cams = sorted({c for t in data for c in data[t]})
    print(f"\n{len(data)} task(s), cameras: {cams}\n")

    # ---------- the frame gate, as a hard error ----------
    bad = [(TASKS[t], cam, arm, s.get("gate_fail"))
           for t in sorted(data) for cam, byst in data[t].items()
           for arm, s in byst.items() if not s.get("gate_ok", True)]
    if bad:
        print("===== GATE FAILURES -- these are hard errors, not data points " + "=" * 30)
        for b in bad:
            print(f"  {b[0]:28s} {b[1]:38s} {b[2]:9s} {b[3]}")
        print()

    for cam in cams:
        table(data, cam, args.extra_arms)

    # ---------- the light-prim readback: direct proof of what the flag did ----------
    print("===== light prims as the stage actually carries them " + "=" * 40)
    print(f"{'task':28s} {'arm':9s} {'gm.FLI':>8s} {'gm.FIX':>7s} {'n':>5s}  combos")
    for (tid, arm) in sorted(lights, key=lambda k: (k[0], k[1])):
        L = lights[(tid, arm)]
        if not L or "error" in L:
            print(f"{TASKS[tid][:28]:28s} {arm:9s}  readback unavailable: {L.get('error', 'absent')}")
            continue
        combos = "; ".join(f"{v}x {k}" for k, v in sorted(L.get("combos", {}).items(),
                                                          key=lambda kv: -kv[1]))
        print(f"{TASKS[tid][:28]:28s} {arm:9s} {str(L.get('gm_FORCE_LIGHT_INTENSITY')):>8s} "
              f"{str(L.get('gm_REALM_LIGHT_FIX')):>7s} {str(L.get('n_lights')):>5s}  {combos[:120]}")
    print()

    # ---------- the machine-readable table ----------
    keep = ("mean", "p05", "p50", "p95", "sat_pct", "dark_pct", "black_pct", "detail",
            "n_colors", "dominant_frac", "gate_ok", "gate_fail")
    out = {"protocol": {"perturbation": "Default (pert-id 0)", "robot": "DROID",
                        "rendering_mode": "rt", "pre_renders": args.pre_renders,
                        "frames_median": args.frames, "multi_view": True, "mode": "oglite",
                        "flag": "REALM_LIGHT_FIX", "off_tolerance": OFF_TOL,
                        "gate": "n_colors >= 2000 and dominant_colour_fraction <= 0.50",
                        "dark_pct": "share of pixels with Rec.601 luma < 60",
                        "detail": "variance of the 4-neighbour Laplacian of luma",
                        "og111_reference": f"REUSED from {args.ref}, not re-rendered"},
           "cameras": cams, "gate_failures": bad, "tasks": {},
           "light_state": {f"{TASKS[t]}::{a}": L for (t, a), L in sorted(lights.items())}}
    for tid in sorted(data):
        e = {"task_id": tid, "flag": FLAGS.get(tid), "confounded": tid in CONFOUNDED, "cameras": {}}
        for cam, byst in data[tid].items():
            ent = {a: {k: byst[a][k] for k in keep if k in byst[a]} for a in byst}
            for a in tuple(ARMS) + tuple(args.extra_arms):
                r = ratio(byst, a)
                ent.setdefault(a, {})["ratio_mean_over_og111"] = round(r, 4) if r else None
            # off vs on within the SAME code state and the same node: the flag's own effect, with
            # nothing from the 1.1.1 side in it.
            o, n = byst.get("off"), byst.get("on")
            if o and n and o.get("gate_ok") and n.get("gate_ok") and o["mean"]:
                ent["on_over_off"] = round(n["mean"] / o["mean"], 4)
            # The control: MODE=stock (no patch bound at all) vs the patched path with the flag off.
            sc = byst.get("stockctl")
            if sc and o and sc.get("gate_ok") and o.get("gate_ok") and sc["mean"]:
                ent["off_vs_stockctl_pct"] = round(100.0 * (o["mean"] - sc["mean"]) / sc["mean"], 4)
            e["cameras"][cam] = ent
        out["tasks"][TASKS[tid]] = e

    # ---------- the headline: does the flag TIGHTEN the spread? ----------
    cam_ref = "external.external_sensor1"
    comparable = [t for t in sorted(data) if t not in CONFOUNDED]
    summary = {}
    for arm in ARMS:
        rs = {TASKS[t]: ratio(data[t].get(cam_ref, {}), arm) for t in comparable}
        vals = [v for v in rs.values() if v is not None]
        summary[arm] = {"ratios": {k: (round(v, 4) if v else None) for k, v in rs.items()},
                        "n": len(vals), "spread": round(spread(vals), 4) if vals else None,
                        "min": round(min(vals), 4) if vals else None,
                        "max": round(max(vals), 4) if vals else None,
                        "mean": round(float(np.mean(vals)), 4) if vals else None}
    out["spread_summary"] = {"camera": cam_ref, "comparable_tasks": [TASKS[t] for t in comparable],
                             "excluded_confounded": [TASKS[t] for t in CONFOUNDED], **summary}
    print(f"===== per-task ratio spread on {cam_ref} (confounded tasks excluded) " + "=" * 20)
    for arm in ARMS:
        s = summary[arm]
        print(f"  flag {arm:3s}: n={s['n']}  range x{s['min']}-x{s['max']}  "
              f"spread {s['spread']}  mean x{s['mean']}")
    if summary["off"]["spread"] and summary["on"]["spread"]:
        print(f"  -> spread {summary['off']['spread']:.3f} -> {summary['on']['spread']:.3f} "
              f"({summary['off']['spread'] / summary['on']['spread']:.1f}x tighter)"
              if summary["on"]["spread"] else "")
    print()

    jp = os.path.join(args.out, "lightfix_table.json")
    with open(jp, "w") as f:
        json.dump(out, f, indent=2)
    print(f"table: {jp}")

    off_s, on_s = summary["off"]["spread"], summary["on"]["spread"]
    bottom = (f"per-task ratio spread on {cam_ref}, {len(comparable)} comparable tasks:  "
              f"flag off {off_s}  ->  flag on {on_s}"
              + (f"   ({off_s / on_s:.1f}x tighter)" if off_s and on_s else ""))
    bottom2 = (f"{len(bad)} gate failure(s).  Confounded (different scene content, not tone): "
               f"{', '.join(TASKS[t] for t in CONFOUNDED)}")
    for cam in cams:
        p = os.path.join(args.out, f"SUMMARY_lightfix__{cam.replace('.', '-')}.png")
        got = sheet(data, cam, p,
                    "REALM_LIGHT_FIX off vs on, all ten tasks -- Default / DROID / rt / "
                    f"{args.pre_renders} pre-renders / {args.frames}-frame median / MODE=oglite",
                    f"camera {cam}   |   rows = tasks 0-9   |   1.1.1 column reused from "
                    "logs/scene_sweep, not re-rendered",
                    bottom, bottom2)
        if got:
            print(f"sheet: {got}")


if __name__ == "__main__":
    raise SystemExit(main())
