#!/usr/bin/env python
"""`/rtx/post/*` -- the tonemap / auto-exposure / colour-grade family -- swept and SOLVED on og391.

WHY THIS FAMILY. Neither OG 1.1.1 nor OG 3.9.1 writes anything under `/rtx/post/` except
`dlss/execMode`, so both inherit KIT defaults, and the Kit versions differ (Isaac 4.x under 1.1.1,
5.1 under 3.9.1). A default that changed between Isaac versions is invisible to every hand-written
key list, and the tone curve is exactly what would have to change to turn og391's render into the
histogram-matched one (a post-hoc match of og391 onto 1.1.1 lands mean 117.69 / p50 138 / %dark
20.57, i.e. a pure monotone remap CAN reproduce 1.1.1's statistics).

THREE THINGS IN ONE BOOT, because an Isaac boot is ~5 min and a carb setting is runtime-mutable:

  1. DUMP the entire carb tree (gated), so the `/rtx/post/*` diff against 1.1.1 is enumerated rather
     than guessed. Same payload shape as render_brightness_ab.py --dump-carb, so carb_tree_diff.py
     reads it.
  2. LADDER. Rows auto-generated from the diff (each differing `/rtx/post` key set to 1.1.1's value,
     then all of them at once), followed by a hand-written sweep of the exposure and tone knobs.
  3. SOLVE. Bisect one key against a target mean luminance on one camera, so the answer is a NUMBER
     rather than the nearest rung of a ladder.

THE CANARY IS NOT OPTIONAL. Post-creation carb writes were MEASURED inert on this stack for
ambientLightIntensity / rendermode / dlss.execMode / sampledLighting -- every such ladder variant
landed within 0.5% of baseline. So the second row of every ladder is `canary_iso_x8`, which raises
filmIso by three stops and MUST blow the frame out. If the canary does not move, runtime writes are
inert for this family too, every later row is meaningless, and the run says so instead of reporting
a table of noise. A second canary (`canary_srgb_off`) runs at the END, so a pipeline that died
half-way through is also caught, and `baseline_end` proves the restore path returned the renderer to
where it started.

GATING. A blank or near-blank frame is a hard failure, not a data point: `og.sim.render()` +
`get_obs()` has returned a 99.99%-white buffer with four unique colours on this stack and a
confident verdict was once read off exactly that. Every frame must clear a minimum unique-colour
count and a maximum single-colour share.

RESTORE. Only keys a variant actually dirtied are restored, and a write whose value equals the
current value is SKIPPED ENTIRELY -- rewriting a carb key with its own value black-framed every
subsequent variant on the 1.1.1 stack.

    # og391: enumerate + sweep + solve, on rotate_mug
    STACK=og391 PROBE=post_tone_sweep.py scripts/debug_probes/run_brightness_ab.sh \
        --label og391_post --dump-carb /logs/render_bright_ab/carb_tree_og391.json \
        --ref-dump /logs/render_bright_ab/carb_tree_og111.json --solve

    # validate a found setting on another task, no ladder
    ... --label og391_post_t0 --task-id 0 --only baseline --set /rtx/post/tonemap/filmIso=43.0

    # host-side, no container, no GPU: score existing PNGs with the SAME metric implementation
    python scripts/debug_probes/post_tone_sweep.py --score-pngs a.png b.png
"""

import argparse
import inspect
import json
import os
import sys
import traceback

import numpy as np

# ==================================================================================================
# metrics -- ONE implementation, used by the probe and by --score-pngs, so a reference number and a
# swept number are never produced by two different formulas.
# ==================================================================================================
DARK_THRESH = 60.0      # "%dark" = share of pixels below this luma
SAT_THRESH = 250.0


def luma(a):
    a = np.asarray(a).astype(np.float64)
    return 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]


def lap_var(lum):
    """Variance of the 4-neighbour Laplacian of the luma plane -- "detail".

    High-frequency content. A denoiser destroys it, so it separates a soft render from a dark one:
    mean luminance cannot tell those apart and this can.
    """
    c = lum[1:-1, 1:-1]
    lap = 4.0 * c - lum[:-2, 1:-1] - lum[2:, 1:-1] - lum[1:-1, :-2] - lum[1:-1, 2:]
    return float(lap.var())


def frame_metrics(img, gate_min_colors=2000, gate_max_dominant=0.50):
    """Every number this probe reports, plus the blank-frame gate, for one HxWx3 uint8 frame."""
    a = np.asarray(img)
    lum = luma(a)
    flat = a.reshape(-1, 3)
    # uint32 pack: ~30x faster than np.unique over rows, and exact for uint8 RGB.
    packed = (flat[:, 0].astype(np.uint32) << 16) | (flat[:, 1].astype(np.uint32) << 8) | flat[:, 2]
    vals, counts = np.unique(packed, return_counts=True)
    n_colors = int(vals.size)
    dominant = float(counts.max()) / float(packed.size)
    st = {
        "shape": [int(a.shape[0]), int(a.shape[1])],
        "mean": round(float(lum.mean()), 3),
        "p05": round(float(np.percentile(lum, 5)), 2),
        "p50": round(float(np.percentile(lum, 50)), 2),
        "p95": round(float(np.percentile(lum, 95)), 2),
        "sat_pct": round(100.0 * float((lum >= SAT_THRESH).mean()), 4),
        "dark_pct": round(100.0 * float((lum < DARK_THRESH).mean()), 3),
        "black_pct": round(100.0 * float((lum <= 5).mean()), 4),
        "detail": round(lap_var(lum), 1),
        "n_colors": n_colors,
        "dominant_frac": round(dominant, 5),
    }
    fails = []
    if n_colors < gate_min_colors:
        fails.append(f"only {n_colors} unique colours (< {gate_min_colors})")
    if dominant > gate_max_dominant:
        fails.append(f"single colour is {dominant:.4%} of pixels (> {gate_max_dominant:.0%})")
    st["gate_ok"] = not fails
    st["gate_fail"] = fails
    return st


ROW_FMT = ("{name:30s} {cam:26s} mean={mean:7.2f} p5={p05:6.1f} p50={p50:6.1f} p95={p95:6.1f} "
           "sat={sat_pct:6.3f}% dark={dark_pct:6.2f}% detail={detail:7.1f}{flag}")


def print_row(name, cam, st):
    print(ROW_FMT.format(name=name[:30], cam=cam[-26:], flag="" if st["gate_ok"] else
                         f"  !! GATE FAIL {st['gate_fail']}", **st))


# ==================================================================================================
# the ladder
# ==================================================================================================
TM = "/rtx/post/tonemap/"
CC = "/rtx/post/colorcorr/"
CG = "/rtx/post/colorgrad/"
HI = "/rtx/post/histogram/"
RC = "/rtx/post/registeredCompositing/"
OC = "/rtx/post/tonemap/ocio/"


# Keys whose 1.1.1 value is worth forcing even when the trees agree, because they are the exposure
# and tone knobs and the sweep needs a named handle on each.
def hand_ladder():
    """(name, {key: value}, note). Values are plain Python; the writer infers the carb type."""
    L = []
    # ---- CANARY. Three stops of extra exposure. If this does not blow the frame out, runtime
    # writes are inert for /rtx/post and NOTHING below it means anything.
    L.append(("canary_iso_x8", {TM + "filmIso": 800.0},
              "CANARY: +3 stops. MUST brighten hard, else runtime post writes are inert."))
    # ---- THE MEASURED DIFF, hand-named so it is in the table even without --ref-dump.
    #
    # The exhaustive dumps put exactly FOUR shared /rtx/post keys apart, and two of them are these:
    # `invertToneMap` and `invertColorCorrection` are False on 1.1.1 (Isaac 4.x Kit) and True on
    # og391 (Isaac 5.1 Kit). They belong to Kit's registered-compositing path, whose job is to hand a
    # compositor LINEAR pixels -- i.e. to UNDO the tone curve on the way out. A flat, bright,
    # washed-out LdrColor with half the deep-shadow coverage is exactly what an inverted tone curve
    # looks like, so this is the first thing to measure and the reason this probe exists.
    L.append(("invertToneMap_off", {RC + "invertToneMap": False}, "1.1.1's value. THE prime suspect."))
    L.append(("invertColorCorrection_off", {RC + "invertColorCorrection": False}, "1.1.1's value"))
    L.append(("invert_both_off", {RC + "invertToneMap": False, RC + "invertColorCorrection": False},
              "both registeredCompositing inversions back to 1.1.1's False"))
    L.append(("regComposite_off", {RC + "enabled": False}, "registered compositing off entirely"))
    # ---- og391-ONLY tonemap keys. Absent from the 1.1.1 tree, so no "reference value" exists for
    # them and the auto-generated rows cannot cover them -- they are new Kit surface area.
    #
    # `exposureTime` = 0.02 s = 1/50 s, i.e. the SAME shutter `cameraShutter` names at 50. Two keys
    # for one physical quantity means one of them is the live one and the other is vestigial; the
    # only way to find out which is to move each and see which moves the frame.
    for et in (0.01, 0.005, 0.0025, 0.04):
        L.append((f"exposureTime_{et:g}", {TM + "exposureTime": et},
                  f"og391-only shutter, {np.log2(et / 0.02):+.2f} stops"))
    L.append(("ocio_useRtxTonemapping_on", {OC + "useRtxTonemapping": True},
              "og391-only: BOTH ocio/enabled and useRtxTonemapping ship False, which may mean no "
              "tone curve is applied at all"))
    L.append(("ocio_on", {OC + "enabled": True}, "og391-only: OpenColorIO managed output on"))
    L.append(("ocio_on_gamma2.2", {OC + "enabled": True, OC + "gamma": 2.2}, "og391-only: OCIO + gamma"))
    # ---- the physical-camera exposure triangle. Exposure ~ ISO / (fNumber^2 * shutter), so each of
    # these three is a stops knob and they must agree with each other; measuring two of them
    # cross-validates that the tonemapper's camera model is actually live.
    for iso in (70.0, 50.0, 43.0, 35.0, 25.0, 12.5):
        L.append((f"iso_{iso:g}", {TM + "filmIso": iso},
                  f"filmIso 100 -> {iso:g} = {np.log2(iso / 100.0):+.2f} stops"))
    for fn in (6.0, 7.1, 8.5, 10.0):
        L.append((f"fnumber_{fn:g}", {TM + "fNumber": fn},
                  f"fNumber 5 -> {fn:g} = {-2 * np.log2(fn / 5.0):+.2f} stops"))
    for sh in (70.0, 100.0, 141.0, 200.0):
        L.append((f"shutter_{sh:g}", {TM + "cameraShutter": sh},
                  f"cameraShutter 50 -> {sh:g} = {-np.log2(sh / 50.0):+.2f} stops"))
    # ---- the white point / scale family. Not stops: these reshape the curve's shoulder, so they
    # move p95 and sat% differently from an exposure change, which is how they are told apart.
    for ws in (10.0, 20.0, 40.2, 80.0, 160.0):
        L.append((f"whiteScale_{ws:g}", {TM + "whiteScale": ws}, "tonemap whiteScale"))
    for mw in (2.5, 5.0, 20.0, 40.0):
        L.append((f"maxWhiteLum_{mw:g}", {TM + "maxWhiteLuminance": mw}, "tonemap maxWhiteLuminance"))
    for cm in (0.25, 0.5, 2.0):
        L.append((f"cm2Factor_{cm:g}", {TM + "cm2Factor": cm}, "cd/m^2 factor -- a linear pre-scale"))
    for ek in (0.125, 0.5):
        L.append((f"exposureKey_{ek:g}", {TM + "exposureKey": ek}, "tonemap exposureKey"))
    # ---- the operator itself. Enumerated rather than guessed: the enum's meaning is Kit-internal
    # and has gained entries between versions.
    for op in (0, 1, 2, 3, 4, 5, 7, 8, 9):
        L.append((f"tonemap_op_{op}", {TM + "op": op}, "tonemap operator enum"))
    # ---- explicit colour correction / grade. Off by default in BOTH stacks, so these are not a
    # revert -- they are a corrective tone curve, available if no default turns out to differ.
    for g in (1.2, 1.4, 1.6):
        L.append((f"colorcorr_gamma_{g:g}", {CC + "enabled": True, CC + "gamma": [g, g, g]},
                  "colour-correction gamma > 1 darkens midtones"))
    for gn in (0.7, 0.5):
        L.append((f"colorcorr_gain_{gn:g}", {CC + "enabled": True, CC + "gain": [gn, gn, gn]},
                  "colour-correction gain -- a linear scale after the tonemap"))
    L.append(("colorgrad_gamma_1.5", {CG + "enabled": True, CG + "gamma": [1.5, 1.5, 1.5]},
              "colour-grade gamma"))
    # ---- histogram-driven auto exposure. OFF in both stacks; on the record as measured, not assumed.
    L.append(("histogram_on", {HI + "enabled": True}, "auto-exposure ON at Kit's default EV clamps"))
    L.append(("histogram_on_ws1", {HI + "enabled": True, HI + "whiteScale": 1.0},
              "auto-exposure ON with a 10x tighter white scale"))
    # ---- CANARY 2, at the end: catches a pipeline that died part-way through the ladder.
    L.append(("canary_srgb_off", {TM + "enableSrgbToGamma": False},
              "CANARY: sRGB encode OFF. MUST darken hard."))
    L.append(("baseline_end", {}, "restore proof: must land back on the baseline row"))
    return L


# ==================================================================================================
# main
# ==================================================================================================
def parse_set(s):
    """`--set /a/b=1.5,/c/d=true` -> {key: value}, values coerced from their text form."""
    out = {}
    for part in filter(None, (p.strip() for p in (s or "").split(","))):
        k, _, v = part.partition("=")
        v = v.strip()
        if v.lower() in ("true", "false"):
            out[k.strip()] = v.lower() == "true"
        elif v.startswith("["):
            out[k.strip()] = json.loads(v)
        else:
            try:
                out[k.strip()] = int(v) if v.lstrip("-").isdigit() else float(v)
            except ValueError:
                out[k.strip()] = v
    return out


def score_pngs(paths):
    """Host-side: the same metrics over existing PNGs, so reference numbers and swept numbers come
    from one formula. No container, no GPU, no Isaac."""
    from PIL import Image
    print(f"{'file':58s} {'mean':>8s} {'p5':>6s} {'p50':>6s} {'p95':>6s} {'sat%':>7s} "
          f"{'dark%':>7s} {'detail':>8s} {'colours':>8s}")
    rows = []
    for p in paths:
        st = frame_metrics(np.asarray(Image.open(p).convert("RGB")))
        rows.append((p, st))
        print(f"{os.path.basename(p)[:58]:58s} {st['mean']:8.2f} {st['p05']:6.1f} {st['p50']:6.1f} "
              f"{st['p95']:6.1f} {st['sat_pct']:7.3f} {st['dark_pct']:7.2f} {st['detail']:8.1f} "
              f"{st['n_colors']:8d}" + ("" if st["gate_ok"] else "  GATE FAIL"))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--score-pngs", nargs="+", default=None,
                    help="host-side mode: score these PNGs and exit (no Isaac, no GPU)")
    ap.add_argument("--out", help="output dir (Lustre, never /tmp)")
    ap.add_argument("--label")
    ap.add_argument("--task-id", type=int, default=3,
                    help="3 = rotate_mug, 0 = put_green_block_into_bowl, 7 = push_switch")
    ap.add_argument("--pert-id", type=int, default=0,
                    help="0 = Default -- randomises nothing, so the framing is identical across runs")
    ap.add_argument("--robot", default="DROID")
    ap.add_argument("--rendering-mode", default="rt")
    ap.add_argument("--frames", type=int, default=5, help="frames median-combined per row")
    ap.add_argument("--settle", type=int, default=6, help="frames discarded after a carb change")
    ap.add_argument("--pre-renders", type=int, default=0,
                    help="render ticks before the first capture. 1.1.1 needs ~250-300 for material "
                         "streaming; og391 does not.")
    ap.add_argument("--ref-dump", default=None,
                    help="a carb_tree_*.json from the OTHER stack. Every differing /rtx/post key "
                         "becomes its own ladder row set to THAT stack's value, plus one row that "
                         "sets all of them at once.")
    ap.add_argument("--dump-carb", default=None, help="write this run's full carb tree here (gated)")
    ap.add_argument("--only", default=None, help="comma-separated ladder subset; 'baseline' alone "
                                                "measures as-shipped and nothing else")
    ap.add_argument("--set", dest="set_", default=None,
                    help="apply these key=value pairs BEFORE the first render and leave them set for "
                         "the whole run -- how a found winner is validated on another task")
    ap.add_argument("--solve", action="store_true",
                    help="after the ladder, bisect --solve-key against --solve-target")
    ap.add_argument("--solve-key", default=TM + "filmIso")
    ap.add_argument("--solve-target", type=float, default=117.98,
                    help="1.1.1's measured cam1 mean on rotate_mug/Default/DROID/rt")
    ap.add_argument("--solve-cam", default="external_sensor1",
                    help="substring of the camera key the target refers to")
    ap.add_argument("--solve-lo", type=float, default=4.0)
    ap.add_argument("--solve-hi", type=float, default=100.0)
    ap.add_argument("--solve-iters", type=int, default=7)
    ap.add_argument("--gate-min-colors", type=int, default=2000)
    ap.add_argument("--gate-max-dominant", type=float, default=0.50)
    args = ap.parse_args()

    if args.score_pngs:
        score_pngs(args.score_pngs)
        return 0

    assert args.out and args.label, "--out and --label are required outside --score-pngs"
    assert not args.out.startswith("/tmp"), "/tmp is node-local and wiped -- artifacts go on Lustre"
    os.makedirs(args.out, exist_ok=True)
    report = {"label": args.label, "argv": sys.argv, "args": vars(args)}
    json_path = os.path.join(args.out, f"{args.label}.json")

    def flush():
        with open(json_path, "w") as f:
            json.dump(report, f, indent=2, sort_keys=False, default=str)

    # ---------- stack detection ----------
    sys.path.insert(0, "/app")
    import omnigibson as og
    from omnigibson.macros import gm
    import omnigibson.lazy as lazy

    try:                                     # og391 moved it out of eval.py
        from realm.sim_config import set_sim_config
    except ImportError:
        from realm.eval import set_sim_config
    from realm.eval import SUPPORTED_TASKS, SUPPORTED_PERTURBATIONS
    from realm.environments.env_dynamic import RealmEnvironmentDynamic

    sig = inspect.signature(set_sim_config)
    is_111 = "rendering_mode" in sig.parameters
    stack = "og111" if is_111 else "og391"
    report["identity"] = {"stack": stack, "omnigibson_file": og.__file__,
                          "omnigibson_version": getattr(og, "__version__", None),
                          "set_sim_config_signature": str(sig)}
    print(f"[identity] stack={stack} og={og.__file__} v={getattr(og, '__version__', '?')}")
    flush()

    if is_111:
        set_sim_config(rendering_mode=args.rendering_mode, robot=args.robot)
    else:
        set_sim_config(robot=args.robot)

    task = SUPPORTED_TASKS[args.task_id]
    pert = SUPPORTED_PERTURBATIONS[args.pert_id]
    report["scene"] = {"task": task, "perturbation": pert, "robot": args.robot,
                       "rendering_mode": args.rendering_mode}
    print(f"[scene] task={task} pert={pert} robot={args.robot} mode={args.rendering_mode}")
    report["macros"] = {k: str(getattr(gm, k, "<absent>")) for k in
                        ("ENABLE_HQ_RENDERING", "DEFAULT_RENDERING_FREQ", "RENDER_ON_STEP")}
    flush()

    try:
        env = RealmEnvironmentDynamic(
            config_path="/app/realm/config",
            task_cfg_path=f"REALM_DROID10/{task}/default.yaml",
            perturbations=[pert], multi_view=True, no_rendering=False,
            rendering_mode=args.rendering_mode, robot=args.robot)
    except Exception as e:
        report["env_creation_error"] = {"type": type(e).__name__, "msg": str(e),
                                       "traceback": traceback.format_exc()[-4000:]}
        print(f"[env] CREATION FAILED: {type(e).__name__}: {e}")
        flush()
        return 3

    cs = lazy.carb.settings.get_settings()

    # ---------- typed carb access ----------
    # carb is typed and the setter must match the stored type: set_float on an int key, or
    # get_as_float on an array key, silently returns garbage rather than raising. The stored value's
    # own Python type is the only reliable discriminator, so every write reads first.
    def cget(key):
        try:
            return cs.get(key)
        except Exception as e:
            return f"<err {type(e).__name__}>"

    def cset(key, val):
        """Write, typed off the CURRENT value. Returns (ok, old). A write whose value already
        matches is SKIPPED: rewriting a carb key with its own value black-framed every subsequent
        variant on the 1.1.1 stack, so it is never a no-op and must never be issued."""
        old = cget(key)
        if isinstance(old, str) and old.startswith("<err"):
            return False, old
        try:
            if isinstance(old, (list, tuple)):
                if list(old) == list(val):
                    return True, old
                cs.set(key, list(val))
            elif isinstance(old, bool):
                if bool(old) == bool(val):
                    return True, old
                cs.set_bool(key, bool(val))
            elif isinstance(old, int):
                if int(old) == int(val):
                    return True, old
                cs.set_int(key, int(val))
            elif isinstance(old, float):
                if abs(float(old) - float(val)) < 1e-12:
                    return True, old
                cs.set_float(key, float(val))
            elif isinstance(old, str):
                if str(old) == str(val):
                    return True, old
                cs.set_string(key, str(val))
            else:                                    # key absent -> create with python's own type
                cs.set(key, val)
        except Exception as e:
            print(f"  !! cset {key}={val!r} failed: {type(e).__name__}: {e}")
            return False, old
        return True, old

    # ---------- --set: applied pre-first-render and never restored ----------
    forced = parse_set(args.set_)
    if forced:
        report["forced"] = {}
        for k, v in forced.items():
            ok, old = cset(k, v)
            report["forced"][k] = {"requested": v, "was": old, "now": cget(k), "ok": ok}
        print(f"[set] pre-first-render: {json.dumps(report['forced'], default=str)}")
        flush()

    # ---------- reset + warmup ----------
    obs, _ = env.reset()
    obs, _r, _t, _tr, _i = env.warmup(obs)
    hold = np.concatenate((np.asarray(env.reset_qpos)[:7], np.atleast_1d(-1.0)))

    def collect_rgb(o):
        out = {}

        def walk(node, path):
            if isinstance(node, dict):
                for k, v in node.items():
                    if k == "rgb":
                        a = v.cpu().numpy() if hasattr(v, "cpu") else v
                        a = np.asarray(a)
                        if a.ndim == 3 and a.shape[-1] >= 3:
                            out[path] = a[..., :3].astype(np.uint8)
                    else:
                        walk(v, f"{path}.{k}" if path else str(k))

        walk(o, "")
        return out

    def step_n(n):
        nonlocal obs
        got = []
        for _ in range(n):
            obs, _r, _t, _tr, _i = env.step(hold)
            got.append(collect_rgb(obs))
        return got

    if args.pre_renders:
        report["material_settle"] = []
        prev = None
        for i in range(0, args.pre_renders, 25):
            for _ in range(25):
                og.sim.render()
            g = step_n(1)[0]
            k = sorted(g)[0]
            m = float(g[k].mean())
            report["material_settle"].append({"renders": i + 25, "mean": round(m, 3)})
            print(f"  [pre-render {i+25}/{args.pre_renders}] {k} raw-mean={m:.2f}"
                  + ("" if prev is None else f"  d={m - prev:+.3f}"))
            prev = m
        flush()

    step_n(args.settle)
    cams = sorted(collect_rgb(obs).keys())
    report["cameras"] = cams
    print(f"[cams] {cams}")
    flush()

    # ---------- the gated carb dump ----------
    def measure(name, save_png=True):
        """settle, then a frames-median per camera. rt re-renders an UNCHANGED scene differently on
        ~25% of pixels here, so a single frame is not evidence and the median is."""
        step_n(args.settle)
        seq = step_n(args.frames)
        out = {}
        for cam in cams:
            imgs = [s[cam] for s in seq if cam in s]
            if not imgs:
                out[cam] = {"gate_ok": False, "gate_fail": ["camera absent from obs"]}
                continue
            med = np.median(np.stack(imgs, 0), axis=0).astype(np.uint8)
            st = frame_metrics(med, args.gate_min_colors, args.gate_max_dominant)
            st["per_frame_mean"] = [round(float(luma(im).mean()), 3) for im in imgs]
            if save_png:
                png = os.path.join(args.out, f"{args.label}__{name}__{cam.replace('.', '-')}.png")
                try:
                    from PIL import Image
                    Image.fromarray(med).save(png)
                    st["png"] = png
                except Exception as e:
                    st["png_error"] = f"{type(e).__name__}: {e}"
            out[cam] = st
            print_row(name, cam, st)
        return out

    base_cams = measure("baseline")
    report["baseline"] = base_cams
    flush()

    if args.dump_carb:
        ok = all(s.get("gate_ok") for s in base_cams.values())
        gate = {c: {k: s.get(k) for k in ("mean", "n_colors", "dominant_frac", "gate_ok",
                                         "gate_fail")} for c, s in base_cams.items()}
        if not ok:
            report["carb_dump_error"] = "frame gate FAILED -- dump not written"
            print("[carb-dump] REFUSING: frame gate failed")
        else:
            flat, method, err = {}, "carb.dictionary.get_dict_copy", None
            try:
                import carb.dictionary

                def flatten(node, prefix, out):
                    if isinstance(node, dict):
                        for k, v in node.items():
                            flatten(v, f"{prefix}/{k}", out)
                    elif isinstance(node, (list, tuple)):
                        out[prefix] = list(node)
                    else:
                        out[prefix] = node

                di = carb.dictionary.get_dictionary()
                flatten(di.get_dict_copy(cs.get_settings_dictionary("/")), "", flat)
            except Exception as e:
                method, err = "FAILED", f"{type(e).__name__}: {e}"
            payload = {"stack": stack, "label": args.label, "method": method, "error": err,
                       "n_keys": len(flat), "gate": gate,
                       "settings": {k: (v if isinstance(v, (int, float, bool, str, list,
                                                            type(None))) else str(v))
                                    for k, v in flat.items()}}
            os.makedirs(os.path.dirname(args.dump_carb) or ".", exist_ok=True)
            with open(args.dump_carb, "w") as f:
                json.dump(payload, f, indent=1, sort_keys=True)
            report["carb_dump"] = {"path": args.dump_carb, "n_keys": len(flat), "method": method,
                                   "error": err}
            print(f"[carb-dump] {len(flat)} keys via {method} -> {args.dump_carb}")
        flush()

    # ---------- the /rtx/post diff, and the ladder rows it generates ----------
    ladder = []
    if args.ref_dump and os.path.exists(args.ref_dump):
        ref = json.load(open(args.ref_dump))
        rs = ref["settings"]
        mine = {k: cget(k) for k in rs if k.startswith("/rtx/post/")}
        differ, only_ref = {}, []
        for k, rv in sorted(rs.items()):
            if not k.startswith("/rtx/post/"):
                continue
            mv = mine.get(k)
            if isinstance(mv, str) and mv.startswith("<err"):
                only_ref.append(k)
                continue
            if mv is None and rv is not None:
                only_ref.append(k)
                continue
            a, b = (list(rv), list(mv)) if isinstance(rv, (list, tuple)) else (rv, mv)
            if isinstance(a, float) and isinstance(b, float):
                if abs(a - b) > 1e-9:
                    differ[k] = (rv, mv)
            elif repr(a) != repr(b):
                differ[k] = (rv, mv)
        report["post_diff"] = {"ref": args.ref_dump, "ref_stack": ref.get("stack"),
                               "n_ref_post_keys": sum(1 for k in rs if k.startswith("/rtx/post/")),
                               "differ": {k: {"ref": v[0], "this": v[1]} for k, v in differ.items()},
                               "absent_here": only_ref}
        print(f"\n===== /rtx/post/* DIFF vs {ref.get('stack')} ({args.ref_dump}) =====")
        print(f"{len(differ)} differing key(s), {len(only_ref)} present only in the reference")
        for k, (rv, mv) in differ.items():
            print(f"  {k:60s} ref={rv!r:>24}  this={mv!r:>24}")
        for k in only_ref:
            print(f"  ONLY IN REF  {k:56s} = {rs[k]!r}")
        # One row per differing key set to the reference's value, then all of them together.
        for k, (rv, _mv) in differ.items():
            ladder.append((f"ref__{k.rsplit('/', 1)[-1]}", {k: rv}, f"{k} -> reference value"))
        if differ:
            ladder.append(("post_match_ref", {k: v[0] for k, v in differ.items()},
                           "EVERY differing /rtx/post key set to the reference stack's value"))
        flush()

    ladder += hand_ladder()
    if args.only:
        want = set(args.only.split(","))
        ladder = [r for r in ladder if r[0] in want]

    # ---------- walk it ----------
    report["ladder"] = []
    dirty = {}          # key -> original value, for keys some variant has written
    gate_failures = 0
    for name, deltas, note in ladder:
        for k, old in list(dirty.items()):
            if k in deltas:
                continue
            cset(k, old)
            dirty.pop(k)
        for k, v in deltas.items():
            ok, old = cset(k, v)
            if ok and k not in dirty:
                dirty[k] = old
        got = measure(name)
        gate_failures += sum(1 for s in got.values() if not s.get("gate_ok"))
        report["ladder"].append({"variant": name, "note": note, "deltas": deltas,
                                 "effective": {k: cget(k) for k in deltas},
                                 "cameras": got})
        flush()

    # ---------- restore, then solve ----------
    for k, old in list(dirty.items()):
        cset(k, old)
        dirty.pop(k)

    if args.solve:
        cam = next((c for c in cams if args.solve_cam in c), None)
        if cam is None:
            report["solve_error"] = f"no camera matching {args.solve_cam!r} in {cams}"
            print(f"[solve] {report['solve_error']}")
        else:
            key, lo, hi = args.solve_key, args.solve_lo, args.solve_hi
            orig = cget(key)
            trace = []
            print(f"\n===== SOLVE {key} in [{lo}, {hi}] for {cam} mean == {args.solve_target} =====")
            for it in range(args.solve_iters):
                mid = 0.5 * (lo + hi)
                cset(key, mid)
                got = measure(f"solve{it}_{mid:.3f}")
                st = got.get(cam, {})
                m = st.get("mean")
                trace.append({"iter": it, "value": round(mid, 4), "mean": m,
                              "gate_ok": st.get("gate_ok"), "stats": st})
                if m is None or not st.get("gate_ok"):
                    print(f"  [solve {it}] {key}={mid:.3f} -> GATE FAIL, aborting")
                    break
                # Monotone increasing in the exposure knobs, so a plain bisection is enough and is
                # robust to the ~0.3 luma frame-to-frame noise a secant step would chase.
                if m > args.solve_target:
                    hi = mid
                else:
                    lo = mid
                print(f"  [solve {it}] {key}={mid:.4f} -> mean {m:.2f} "
                      f"(target {args.solve_target}, err {m - args.solve_target:+.2f})")
            best = min((t for t in trace if t["mean"] is not None),
                       key=lambda t: abs(t["mean"] - args.solve_target), default=None)
            report["solve"] = {"key": key, "cam": cam, "target": args.solve_target,
                               "original_value": orig, "trace": trace, "best": best}
            if best:
                print(f"[solve] best {key}={best['value']} -> mean {best['mean']:.2f}")
                cset(key, best["value"])
                report["solve"]["best_full"] = measure(f"solved_{best['value']:.3f}")
            cset(key, orig)
        flush()

    report["gate_failures"] = gate_failures
    # The canary decides whether the table is data or noise, so the verdict is COMPUTED here rather
    # than left to a reader: if +3 stops of film ISO did not move the frame, runtime /rtx/post writes
    # are inert on this stack and every row below baseline is meaningless.
    can = next((e for e in report["ladder"] if e["variant"] == "canary_iso_x8"), None)
    if can and base_cams:
        c0 = cams[0]
        b = base_cams[c0].get("mean")
        c = can["cameras"].get(c0, {}).get("mean")
        if b and c:
            report["canary"] = {"camera": c0, "baseline_mean": b, "canary_mean": c,
                                "delta_pct": round(100.0 * (c - b) / b, 2),
                                "runtime_writes_live": abs(c - b) / b > 0.02}
            print(f"\n[canary] {c0}: baseline {b:.2f} -> filmIso x8 {c:.2f} "
                  f"({100 * (c - b) / b:+.1f}%)  runtime_writes_live="
                  f"{report['canary']['runtime_writes_live']}")
    report["ok"] = gate_failures == 0
    flush()
    print(f"\n[done] {json_path}  gate_failures={gate_failures}")
    return 0 if gate_failures == 0 else 4


if __name__ == "__main__":
    # Isaac exits 0 even on an unhandled exception, so the failure has to be recorded in-band.
    try:
        rc = main()
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        rc = 9
    print(f"PROBE_RC={rc}")
    sys.exit(rc)
