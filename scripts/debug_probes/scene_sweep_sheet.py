#!/usr/bin/env python
"""Host-side (no container, no GPU): the cross-task OG 1.1.1 vs og391 render comparison.

post_tone_sheet.py answers "which carb variant is closest to the reference on ONE task". This one
answers the orthogonal question -- "on which TASKS do the two stacks disagree at all" -- so the
sheet is transposed: rows are tasks, the two columns are the two stacks, one sheet per camera.

Metrics are read from the probe JSONs where they exist and RE-SCORED from the PNG with the same
post_tone_sweep.frame_metrics otherwise, so every number on a sheet comes from one implementation.
A frame whose gate failed is never quoted as a datum -- it is drawn with a red bar and no ratio.

    python scripts/debug_probes/scene_sweep_sheet.py \
        --raw   /mnt/.../logs/render_bright_ab/scene_sweep \
        --prior /mnt/.../logs/post_tone/frames_native \
        --out   /mnt/.../logs/scene_sweep
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

# Tasks whose scene has a defect already ruled known. CONFOUNDED means the two stacks are not
# rendering the same content, so the mean ratio is measuring scene layout and not tone -- it is
# still reported, but it must not be counted as a render difference.
FLAGS = {
    2: "og391: objects off the surface",
    6: "og391: objects off the surface",
    8: "1.1.1 has one extra object",
    9: "og391: cabinet not in frame -- but the drawer IS open (measured)",
}
# RETRACTION 2026-08-18. This said "og391: drawer not open at reset", read off the frames: 1.1.1
# starts `close_drawer` with the drawer OPEN and the exterior camera looking into its slide rails,
# og391's frame from the same viewpoint shows the room unobstructed, so the drawer was inferred to
# be shut. MEASURED FALSE. t13_drawer_stop.py --task_id 9, MODE=oglite, DROID_robolab_v2:
#
#   init_openness_fraction  1.0000 (want 1.0000) on all 4 vector members AND at num_envs=1,
#                           at construction and again after a re-driven reset_joints()
#   target drawer_joint_00  0.3000 of limits [0.0000, 0.3000], residual 0.0000
#   link displacement       closed -> open moves drawer_blender_cut_00 by (-0.052, +0.295, 0.000),
#                           |d| = 0.3000 = the full joint range, dominant axis y -- a horizontal
#                           slide of the right magnitude, so the link tracks the joint
#   200 free sim steps      openness 1.0000 throughout, drift 0.0000 -- it does not creep shut
#                           before the frame is taken
#
# So task 9 stays CONFOUNDED -- the two stacks genuinely are not photographing the same content --
# but the START STATE is not the reason and must not be cited as one. Why og391's exterior camera
# does not show a cabinet that is measurably present, upright, at the config pose and holding an
# open drawer is UNEXPLAINED; the open lead is that og391 logs a material fallback on exactly this
# asset ("Material prim at .../drawer/Materials/Material_Cabinet_{Body,Drawer} ... does not have a
# known shader file associated with it"). Do not turn that lead into a cause without measuring it.
CONFOUNDED = (2, 6, 9)

# The three tasks measured in the previous session, and the label each stack wrote there.
PRIOR = {0: ("og111_post_t0__baseline__", "og391_post_t0__baseline__"),
         3: ("og111_rt_clean__baseline__", "og391_post_t3__baseline__"),
         7: ("og111_post_t7__baseline__", "og391_post_t7__baseline__")}

OFF_TOL = 0.05          # |ratio - 1| above this is "off"

PANEL_W = 470
PAD = 6
BAR = 50
HEAD = 66
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
    """Metrics for one native frame, from the same implementation the probe used."""
    return frame_metrics(np.asarray(Image.open(png).convert("RGB")))


def load_run(raws, label):
    """(report, dir) for the first --raw dir holding a COMPLETE report under this label."""
    for d in raws:
        jp = os.path.join(d, f"{label}.json")
        if not os.path.exists(jp):
            continue
        try:
            rep = json.load(open(jp))
        except json.JSONDecodeError:
            print(f"  !! {jp}: truncated JSON (run still in flight?) -- skipped")
            continue
        if "env_creation_error" in rep:
            e = rep["env_creation_error"]
            print(f"  !! {label}: ENV CREATION FAILED {e['type']}: {e['msg'][:120]}")
            continue
        if not rep.get("baseline"):
            print(f"  !! {label}: no baseline yet -- skipped")
            continue
        return rep, d
    return None, None


def collect(raw, prior, out_frames):
    """{task_id: {cam: {stack: stats}}} over every task both stacks have a gated report for.

    og391 was booted twice per task on two independent allocations. The FIRST label is the datum and
    the second is only a reproducibility check, so a task keeps its `_ss_` boot when it has one and
    falls back to `_ss2_` when that boot was lost -- never a silent mixture of the two."""
    raws = raw if isinstance(raw, (list, tuple)) else [raw]
    data, sources, repeats = {}, {}, {}
    for tid, task in enumerate(TASKS):
        per_cam = {}
        src = None
        # --- this sweep's own runs -------------------------------------------------------------
        for stack in ("og111", "og391"):
            labels = [f"{stack}_ss_t{tid}"] + ([f"{stack}_ss2_t{tid}"] if stack == "og391" else [])
            rep, d = None, None
            for lab in labels:
                rep, d = load_run(raws, lab)
                if rep:
                    break
            if not rep:
                continue
            used = rep["label"]
            for cam, st in rep["baseline"].items():
                png = os.path.join(d, f"{used}__baseline__{cam.replace('.', '-')}.png")
                st = dict(st)
                st["png"] = png if os.path.exists(png) else None
                st["run_label"] = used
                per_cam.setdefault(cam, {})[stack] = st
            # --- the second og391 boot, as a reproducibility check, never as a datum ------------
            if stack == "og391":
                other = [l for l in labels if l != used]
                rep2, _ = load_run(raws, other[0]) if other else (None, None)
                if rep2:
                    for cam, st2 in rep2["baseline"].items():
                        a = per_cam.get(cam, {}).get("og391")
                        if a and st2.get("gate_ok") and a.get("gate_ok"):
                            repeats.setdefault(tid, {})[cam] = {
                                "boot_a": {"label": used, "mean": a["mean"]},
                                "boot_b": {"label": rep2["label"], "mean": st2["mean"]},
                                "delta_pct": round(100.0 * (st2["mean"] - a["mean"]) / a["mean"], 4),
                            }
            src = "new"
        # --- the three measured before, re-scored here from their native frames ----------------
        if not per_cam and tid in PRIOR:
            for stack, pref in zip(("og111", "og391"), PRIOR[tid]):
                for png in sorted(f for f in os.listdir(prior) if f.startswith(pref)):
                    cam = png[len(pref):-len(".png")].replace("-", ".", 1)
                    st = score(os.path.join(prior, png))
                    st["png"] = os.path.join(prior, png)
                    per_cam.setdefault(cam, {})[stack] = st
            src = "prior" if per_cam else None
        if per_cam:
            data[tid] = per_cam
            sources[tid] = src
    # A task with only one stack cannot be compared; keep it out of the sheet rather than draw a
    # half row that reads as a result.
    for tid in [t for t in data if len(set(s for c in data[t].values() for s in c)) < 2]:
        have = sorted({s for c in data[tid].values() for s in c})
        print(f"  !! {TASKS[tid]} (t{tid}): only {have} -- dropped, nothing to compare")
        data.pop(tid)
        sources.pop(tid, None)
    if out_frames:
        os.makedirs(out_frames, exist_ok=True)
        for tid, per_cam in data.items():
            for cam, byst in per_cam.items():
                for stack, st in byst.items():
                    if st.get("png") and os.path.exists(st["png"]):
                        dst = os.path.join(out_frames, os.path.basename(st["png"]))
                        if os.path.abspath(dst) != os.path.abspath(st["png"]):
                            Image.open(st["png"]).save(dst)
    return data, sources, repeats


def ratio(byst):
    a, b = byst.get("og111"), byst.get("og391")
    if not a or not b or not a.get("gate_ok", True) or not b.get("gate_ok", True):
        return None
    return b["mean"] / a["mean"] if a["mean"] else None


def table(data, sources, cam):
    hdr = (f"{'task':28s} {'stack':6s} {'mean':>8s} {'p5':>6s} {'p50':>6s} {'p95':>6s} "
           f"{'sat%':>7s} {'dark%':>7s} {'detail':>8s} {'ratio':>7s}  gate")
    print(f"===== {cam} " + "=" * max(0, 88 - len(cam)))
    print(hdr)
    for tid in sorted(data):
        byst = data[tid].get(cam, {})
        r = ratio(byst)
        for stack in ("og111", "og391"):
            s = byst.get(stack)
            if not s:
                print(f"{TASKS[tid][:28]:28s} {stack:6s} {'MISSING':>8s}")
                continue
            if not s.get("gate_ok", True):
                print(f"{TASKS[tid][:28]:28s} {stack:6s} {'--':>8s} {'--':>6s} {'--':>6s} "
                      f"{'--':>6s} {'--':>7s} {'--':>7s} {'--':>8s} {'--':>7s}  "
                      f"FAIL {s.get('gate_fail')}")
                continue
            rr = f"x{r:.2f}" if (r and stack == "og391") else ""
            print(f"{TASKS[tid][:28]:28s} {stack:6s} {s['mean']:8.2f} {s['p05']:6.1f} "
                  f"{s['p50']:6.1f} {s['p95']:6.1f} {s['sat_pct']:7.3f} {s['dark_pct']:7.2f} "
                  f"{s['detail']:8.1f} {rr:>7s}  ok  ({sources.get(tid)})")
    print()


def sheet(data, sources, cam, path, title, subtitle, bottom, bottom2=""):
    f_title, f_sub, f_lbl, f_num = fonts()
    tids = sorted(data)
    rows = [t for t in tids if cam in data[t]]
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
    pw = PANEL_W
    ph = max(1, int(round(probe.height * pw / probe.width)))
    W = 2 * pw + 3 * PAD
    H = len(rows) * (ph + BAR + PAD) + PAD + HEAD
    sh = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(sh)
    d.text((PAD, 5), title, fill=(238, 238, 238), font=f_title)
    d.text((PAD, 24), subtitle, fill=(150, 155, 165), font=f_sub)
    d.text((PAD, 37), bottom, fill=(255, 214, 102), font=f_sub)
    if bottom2:
        d.text((PAD, 50), bottom2, fill=(214, 160, 110), font=f_sub)

    for r, tid in enumerate(rows):
        byst = data[tid][cam]
        rat = ratio(byst)
        y = HEAD + PAD + r * (ph + BAR + PAD)
        for c, stack in enumerate(("og111", "og391")):
            x = PAD + c * (pw + PAD)
            st = byst.get(stack)
            if st is None:
                d.rectangle([x, y, x + pw, y + ph + BAR], fill=(46, 30, 30))
                d.text((x + 6, y + ph // 2), f"{TASKS[tid]}  {stack}: NO DATA",
                       fill=(255, 180, 180), font=f_lbl)
                continue
            p = st.get("png")
            if p and os.path.exists(p):
                sh.paste(Image.open(p).convert("RGB").resize((pw, ph), Image.LANCZOS), (x, y))
            ok = st.get("gate_ok", True)
            if not ok:
                bar = (110, 30, 30)
            elif stack == "og111":
                bar = (30, 55, 40)
            elif rat is not None and abs(rat - 1.0) > OFF_TOL:
                bar = (96, 52, 24)
            else:
                bar = (40, 40, 44)
            d.rectangle([x, y + ph, x + pw, y + ph + BAR], fill=bar)
            flag = FLAGS.get(tid, "")
            tag = "OG 1.1.1 reference" if stack == "og111" else "og391 as shipped"
            if stack == "og391" and rat is not None:
                tag += f"   x{rat:.2f} vs 1.1.1"
                if abs(rat - 1.0) > OFF_TOL:
                    tag += "   <-- CONFOUNDED" if tid in CONFOUNDED else "   <-- OFF"
            head = f"{TASKS[tid]}  [{tid}]" + (f"   -- {flag}" if flag and stack == "og111" else "")
            d.text((x + 5, y + ph + 3), head[:62],
                   fill=(200, 245, 210) if stack == "og111" else (240, 240, 240), font=f_lbl)
            d.text((x + 5, y + ph + 18), tag[:96],
                   fill=(170, 215, 185) if stack == "og111" else (235, 205, 160), font=f_num)
            if ok:
                d.text((x + 5, y + ph + 33),
                       f"mean {st['mean']:.1f}   p5 {st['p05']:.0f}  p50 {st['p50']:.0f}  "
                       f"p95 {st['p95']:.0f}   sat {st['sat_pct']:.3f}%  dark {st['dark_pct']:.2f}%"
                       f"   detail {st['detail']:.0f}", fill=(190, 200, 210), font=f_num)
            else:
                d.text((x + 5, y + ph + 33), f"GATE FAIL {st.get('gate_fail')}"[:110],
                       fill=(255, 190, 190), font=f_num)
    sh.save(path)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, nargs="+",
                    help="dirs holding this sweep's *_ss_t*.json and PNGs, in priority order")
    ap.add_argument("--prior", default=None, help="dir holding the previously measured frames")
    ap.add_argument("--out", required=True, help="deliverable dir")
    ap.add_argument("--pre-renders", type=int, default=300)
    ap.add_argument("--frames", type=int, default=5)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    frames_out = os.path.join(args.out, "frames_native")
    data, sources, repeats = collect(args.raw, args.prior, frames_out)
    if not data:
        raise SystemExit(f"no usable reports in {args.raw}")
    cams = sorted({c for t in data for c in data[t]})
    print(f"\n{len(data)} task(s), cameras: {cams}\n")
    for cam in cams:
        table(data, sources, cam)

    if repeats:
        print("===== og391 boot-to-boot reproducibility (two boots, two allocations) " + "=" * 20)
        worst = 0.0
        for tid in sorted(repeats):
            for cam, r in sorted(repeats[tid].items()):
                worst = max(worst, abs(r["delta_pct"]))
                print(f"{TASKS[tid][:28]:28s} {cam[-34:]:34s} {r['boot_a']['mean']:8.2f} "
                      f"{r['boot_b']['mean']:8.2f} {r['delta_pct']:+8.4f}%")
        print(f"worst |delta| over {sum(len(v) for v in repeats.values())} pairs: {worst:.4f}%\n")

    # ---------- the machine-readable table ----------
    keep = ("mean", "p05", "p50", "p95", "sat_pct", "dark_pct", "black_pct", "detail",
            "n_colors", "dominant_frac", "gate_ok", "gate_fail")
    out = {"protocol": {"perturbation": "Default (pert-id 0)", "robot": "DROID",
                        "rendering_mode": "rt", "pre_renders": args.pre_renders,
                        "frames_median": args.frames, "multi_view": True,
                        "off_tolerance": OFF_TOL,
                        "gate": "n_colors >= 2000 and dominant_colour_fraction <= 0.50",
                        "dark_pct": "share of pixels with Rec.601 luma < 60",
                        "detail": "variance of the 4-neighbour Laplacian of luma"},
           "cameras": cams, "tasks": {},
           "og391_boot_to_boot": {TASKS[t]: repeats[t] for t in sorted(repeats)}}
    for tid in sorted(data):
        e = {"task_id": tid, "source": sources.get(tid), "flag": FLAGS.get(tid),
             "run_labels": {s: st.get("run_label") for c in data[tid].values()
                            for s, st in c.items()}, "cameras": {}}
        for cam, byst in data[tid].items():
            r = ratio(byst)
            e["cameras"][cam] = {
                "og111": {k: byst["og111"][k] for k in keep if k in byst.get("og111", {})},
                "og391": {k: byst["og391"][k] for k in keep if k in byst.get("og391", {})},
                "ratio_mean_391_over_111": round(r, 4) if r else None,
                "off": bool(r and abs(r - 1.0) > OFF_TOL),
            }
        out["tasks"][TASKS[tid]] = e
    jp = os.path.join(args.out, "scene_sweep_table.json")
    with open(jp, "w") as f:
        json.dump(out, f, indent=2)
    print(f"table: {jp}")

    # ---------- the sheets ----------
    cam_ref = "external.external_sensor1"
    off = [t for t in sorted(data)
           if (r := ratio(data[t].get(cam_ref, {}))) and abs(r - 1.0) > OFF_TOL]
    clean = [TASKS[t] for t in off if t not in CONFOUNDED]
    conf = [TASKS[t] for t in off if t in CONFOUNDED]
    n_meas = sum(1 for t in sorted(data) if ratio(data[t].get(cam_ref, {})))
    n_comparable = n_meas - sum(1 for t in data if t in CONFOUNDED)
    bottom = (f"{len(clean)} of {n_comparable} comparable tasks are OFF by more than {OFF_TOL:.0%} "
              f"on {cam_ref}:  {', '.join(clean) if clean else 'none'}")
    bottom2 = (f"{len(conf)} further task(s) CONFOUNDED -- og391 is not rendering the same content, "
               f"so the ratio is scene layout, not tone:  {', '.join(conf)}") if conf else ""
    sub = (f"Default / DROID / rt / {args.pre_renders} pre-renders / {args.frames}-frame median, "
           f"every frame gated. Same schedule on both stacks. og391 reproducible to 0.004% "
           f"boot-to-boot.")
    for cam in cams:
        p = os.path.join(args.out, f"SUMMARY_{len(data)}tasks__{cam.replace('.', '-')}.png")
        got = sheet(data, sources, cam, p, f"OG 1.1.1 vs og391 across {len(data)} tasks -- {cam}",
                    sub, bottom, bottom2)
        print(f"sheet: {got}" if got else f"  (no panels for {cam})")
    print(f"\nBOTTOM LINE: {bottom}")
    if bottom2:
        print(f"             {bottom2}")


if __name__ == "__main__":
    main()
