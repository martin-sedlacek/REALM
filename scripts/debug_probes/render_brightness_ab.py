#!/usr/bin/env python
"""Why is og391 BRIGHTER than 1.1.1? A carb-setting ladder, measured, in ONE boot per stack.

Runs on BOTH stacks from this one file (the 1.1.1 checkout is read-only, so this is bound in at
/dbg rather than copied into it). It detects which stack it is on by introspecting
`set_sim_config`'s signature -- 1.1.1 takes `rendering_mode`, og391 does not.

What it does, per boot:

  1. Prints a CODE IDENTITY block: OmniGibson version + file, the `gm.*` macros that gate the
     renderer, and a READBACK of every carb RTX setting either stack's `_set_renderer_settings`
     touches. The readback is the cheapest decisive measurement here -- it says what each stack
     ACTUALLY configured, rather than what its source suggests.
  2. Builds one env (task/perturbation/robot fixed), resets, warms up.
  3. Walks a LADDER of carb variants. carb settings are runtime-mutable, so one boot covers the
     whole matrix instead of ~5 min of Isaac startup per cell. Every variant is applied from the
     SNAPSHOT taken in step 1, so variants never compound.
  4. Per variant: re-render, discard the settle frames, median-combine the rest per camera, write a
     PNG, and report mean / p50 / p95 luminance plus % saturated.

`--hq {0,1}` and `--render-freq` override `gm.ENABLE_HQ_RENDERING` / `gm.DEFAULT_RENDERING_FREQ`
AFTER set_sim_config() and BEFORE env creation -- both are macros read at env construction, so they
cannot be laddered and need their own boot. This is deliberately done here and NOT by editing
realm/sim_config.py.

GATING. A blank or near-blank frame is a hard failure, not a data point: every frame must clear a
minimum unique-colour count and a maximum single-colour share before it is allowed to produce a
luminance number. `og.sim.render()` + `CAM.get_obs()` has returned a 99.99%-white buffer with four
unique colours on this stack, and a confident verdict was once read off exactly that.

    python /dbg/render_brightness_ab.py --out /logs/bright_ab/og391_rt --label og391_rt
    python /dbg/render_brightness_ab.py --out ... --hq 1                  # does the >=60 FPS assert fire?
    python /dbg/render_brightness_ab.py --out ... --hq 1 --render-freq 60 # does raising the freq clear it?
"""

import argparse
import hashlib
import inspect
import json
import os
import sys
import traceback

import numpy as np

# ==================================================================================================
# The carb keys that matter. Everything either stack's _set_renderer_settings() writes, plus the
# handful REALM's own set_rendering_mode() writes for "r"/"pt".
#
#   type: "f" float, "i" int, "b" bool, "s" string -- carb is typed and get_as_float on a string key
#   returns garbage rather than raising.
# ==================================================================================================
CARB_KEYS = [
    ("/rtx/rendermode", "s"),
    ("/rtx/sceneDb/ambientLightIntensity", "f"),
    ("/rtx/post/dlss/execMode", "i"),
    ("/rtx/reflections/enabled", "b"),
    ("/rtx/indirectDiffuse/enabled", "b"),
    ("/rtx/ambientOcclusion/enabled", "b"),
    ("/rtx/directLighting/sampledLighting/enabled", "b"),
    ("/rtx/directLighting/sampledLighting/samplesPerPixel", "i"),
    ("/rtx/raytracing/showLights", "i"),
    ("/rtx/shadows/enabled", "b"),
    ("/rtx/translucency/enabled", "b"),
    ("/rtx/rtx/modes/rt/enabled", "b"),
    ("/rtx/rtx/modes/rt2/enabled", "b"),
    ("/rtx/raytracing/fractionalCutoutOpacity", "b"),
    ("/rtx/flow/enabled", "b"),
    ("/rtx/pathtracing/spp", "i"),
    ("/rtx/pathtracing/totalSpp", "i"),
    ("/rtx/post/tonemap/op", "i"),
    ("/rtx/post/tonemap/filmIso", "f"),
    ("/rtx/post/tonemap/cameraShutter", "f"),
    ("/rtx/post/tonemap/fNumber", "f"),
    ("/rtx/post/histogram/enabled", "b"),
    ("/rtx/post/histogram/whiteScale", "f"),
    ("/rtx/sceneDb/ambientLightColor", "s"),
    ("/app/renderer/skipMaterialLoading", "b"),
    ("/isaaclab/rendering/rendering_mode", "s"),
]

# ==================================================================================================
# THE LADDER. (name, {carb key: (type, value)}, note)
#
# Every variant starts from the snapshot, so `og391_rt_baseline` and `og111_rt_baseline` are the
# as-shipped state of each stack and the rest are single-setting deltas off it.
# ==================================================================================================
def build_ladder(only=None):
    L = [
        ("baseline", {}, "as-shipped for this stack at this rendering_mode"),
        # The two candidate causes, one at a time.
        ("amb_0.1", {"/rtx/sceneDb/ambientLightIntensity": ("f", 0.1)},
         "OG 1.1.1's value. 3.9.1 hardcodes 1.0 -- a 10x flat ambient lift."),
        ("amb_1.0", {"/rtx/sceneDb/ambientLightIntensity": ("f", 1.0)},
         "OG 3.9.1's value, forced onto whichever stack is running."),
        ("rendermode_raytraced", {"/rtx/rendermode": ("s", "RaytracedLighting")},
         "1.1.1 never sets rendermode in rt, so Kit's default applies; 3.9.1 forces RealTimePathTracing."),
        ("rendermode_rtpt", {"/rtx/rendermode": ("s", "RealTimePathTracing")},
         "3.9.1's renderer, forced onto whichever stack is running."),
        ("sampledLighting_off", {"/rtx/directLighting/sampledLighting/enabled": ("b", False)},
         "1.1.1 turns this OFF under HQ; 3.9.1 leaves it ON unconditionally."),
        # The combination: 1.1.1's rt+HQ carb state, reproduced exactly.
        ("og111_hq_full", {
            "/rtx/rendermode": ("s", "RaytracedLighting"),
            "/rtx/sceneDb/ambientLightIntensity": ("f", 0.1),
            "/rtx/reflections/enabled": ("b", True),
            "/rtx/indirectDiffuse/enabled": ("b", True),
            "/rtx/post/dlss/execMode": ("i", 3),
            "/rtx/ambientOcclusion/enabled": ("b", True),
            "/rtx/directLighting/sampledLighting/enabled": ("b", False),
            "/rtx/raytracing/showLights": ("i", 1),
        }, "every carb setting 1.1.1's _set_renderer_settings writes under ENABLE_HQ_RENDERING=True"),
        # ... and its ambient-only counterpart, to separate the 10x ambient from the renderer swap.
        ("og111_hq_full_but_amb1.0", {
            "/rtx/rendermode": ("s", "RaytracedLighting"),
            "/rtx/sceneDb/ambientLightIntensity": ("f", 1.0),
            "/rtx/reflections/enabled": ("b", True),
            "/rtx/indirectDiffuse/enabled": ("b", True),
            "/rtx/post/dlss/execMode": ("i", 3),
            "/rtx/ambientOcclusion/enabled": ("b", True),
            "/rtx/directLighting/sampledLighting/enabled": ("b", False),
            "/rtx/raytracing/showLights": ("i", 1),
        }, "og111_hq_full with ONLY the ambient term left at 3.9.1's value"),
        # dlss execMode is the ONLY thing gm.ENABLE_HQ_RENDERING changes in 3.9.1's renderer block
        # (the rest of that block is isosurface-only). Test it directly rather than paying a boot.
        ("dlss_perf", {"/rtx/post/dlss/execMode": ("i", 0)}, "Performance -- 3.9.1 with HQ off"),
        ("dlss_realism", {"/rtx/post/dlss/execMode": ("i", 1)}, "Realism -- 3.9.1 with HQ on"),
        ("dlss_auto", {"/rtx/post/dlss/execMode": ("i", 3)}, "Auto -- 1.1.1 with HQ on"),
        # REALM's own "r" mode, applied as carb settings. NOT what evals use (rendering_mode defaults
        # to "rt"); here so the mode the tree supports is measured rather than assumed.
        ("realm_mode_r", {
            "/rtx/rendermode": ("s", "RaytracedLighting"),
            "/rtx/translucency/enabled": ("b", True),
            "/rtx/reflections/enabled": ("b", False),
            "/rtx/indirectDiffuse/enabled": ("b", False),
            "/rtx/directLighting/sampledLighting/enabled": ("b", True),
            "/rtx/directLighting/sampledLighting/samplesPerPixel": ("i", 1),
            "/rtx/shadows/enabled": ("b", False),
            "/rtx/post/dlss/execMode": ("i", 0),
            "/rtx/ambientOcclusion/enabled": ("b", False),
            "/rtx/sceneDb/ambientLightIntensity": ("f", 1.0),
            "/isaaclab/rendering/rendering_mode": ("s", "performance"),
        }, "REALM set_rendering_mode('r') -- byte-identical between the two trees"),
        ("realm_mode_pt", {
            "/rtx/rendermode": ("s", "PathTracing"),
            "/rtx/pathtracing/spp": ("i", 8),
            "/rtx/pathtracing/totalSpp": ("i", 8),
        }, "REALM set_rendering_mode('pt') at spp=8"),
    ]
    if only:
        want = set(only.split(","))
        L = [v for v in L if v[0] in want]
    return L


# ==================================================================================================
# frames
# ==================================================================================================
def collect_rgb(obs):
    """Every RGB leaf in an obs dict, keyed by its dotted path. Stack-agnostic on purpose: the two
    trees disagree about extract_from_obs' return arity, but both nest {...: {'rgb': HxWx(3|4)}}."""
    out = {}

    def walk(node, path):
        if isinstance(node, dict):
            for k, v in node.items():
                if k == "rgb":
                    a = v
                    if hasattr(a, "cpu"):
                        a = a.cpu().numpy()
                    a = np.asarray(a)
                    if a.ndim == 3 and a.shape[-1] >= 3:
                        out[path] = a[..., :3].astype(np.uint8)
                else:
                    walk(v, f"{path}.{k}" if path else str(k))

    walk(obs, "")
    return out


def frame_stats(img, gate_min_colors, gate_max_dominant):
    """Luminance stats plus the blank-frame gate. Rec.601 luma, computed identically everywhere."""
    a = np.asarray(img)
    lum = 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]
    flat = a.reshape(-1, 3)
    # uint32 pack: ~30x faster than np.unique on rows and exact for uint8 RGB.
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
        "sat_pct": round(100.0 * float((lum >= 250).mean()), 4),
        "dark_pct": round(100.0 * float((lum <= 5).mean()), 4),
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


# ==================================================================================================
# main
# ==================================================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="output dir (must be on Lustre, not /tmp)")
    ap.add_argument("--label", required=True)
    ap.add_argument("--task-id", type=int, default=3, help="3 = rotate_mug")
    ap.add_argument("--pert-id", type=int, default=0,
                    help="0 = Default. Default is the right choice for a BRIGHTNESS A/B: it "
                         "randomises nothing, so the camera framing is identical across stacks. "
                         "VB-POSE moves the robot base and therefore the exterior view.")
    ap.add_argument("--robot", default="DROID",
                    help="DROID on BOTH stacks -- the 1.1.1 checkout has no robolab config at all.")
    ap.add_argument("--rendering-mode", default="rt")
    ap.add_argument("--multi-view", action="store_true", default=True)
    ap.add_argument("--no-multi-view", dest="multi_view", action="store_false")
    ap.add_argument("--hq", type=int, default=None,
                    help="override gm.ENABLE_HQ_RENDERING (a macro; needs its own boot)")
    ap.add_argument("--render-freq", type=int, default=None,
                    help="override gm.DEFAULT_RENDERING_FREQ (a macro; needs its own boot)")
    ap.add_argument("--sim-freq", type=int, default=None,
                    help="override gm.DEFAULT_SIM_STEP_FREQ. Needed alongside --render-freq: OG "
                         "3.9.1 asserts sim_step_dt == rendering_dt under gm.HEADLESS, so raising "
                         "the render rate alone fails a SECOND assert. Raising both moves the "
                         "policy's control rate off REALM's 15 Hz, which is a behaviour change, "
                         "not just a rendering one.")
    ap.add_argument("--frames", type=int, default=5, help="frames median-combined per variant")
    ap.add_argument("--settle", type=int, default=6, help="frames discarded after a carb change")
    ap.add_argument("--only", default=None, help="comma-separated ladder subset")
    ap.add_argument("--read-path", choices=["step", "render_obs"], default="step",
                    help="How a frame is obtained. 'step' reads the obs env.step() returns. "
                         "'render_obs' does n_pre_obs_renders explicit og.sim.render() calls and "
                         "then env.omnigibson_env.render_obs(), which is the ON-DEMAND path the "
                         "1.1.1 eval loop uses under --og_lite -- and the recorded 1.1.1 videos "
                         "this is being compared against were produced by exactly that path "
                         "(scripts/dreamzero_oglite_watchdog.sh passes --og_lite). Use it on 1.1.1.")
    ap.add_argument("--og-lite", action="store_true",
                    help="1.1.1 only: pass og_lite=True to set_sim_config, i.e. RENDER_ON_STEP off "
                         "plus the state whitelist. Pair with --read-path render_obs. On og391 the "
                         "equivalent macros are already set unconditionally, so this is a no-op.")
    ap.add_argument("--n-pre-obs-renders", type=int, default=3,
                    help="render() ticks before each render_obs(); 1.1.1's eval default is 3")
    ap.add_argument("--pre-renders", type=int, default=0,
                    help="extra og.sim.render() ticks before the first capture. OmniGibson streams "
                         "materials asynchronously, and on the 1.1.1 stack 6 settle steps was not "
                         "enough: the frame came back 85%% pure white with the geometry as unlit "
                         "silhouettes. Use a few hundred there.")
    ap.add_argument("--patch-launch", default=None,
                    help="apply a ladder variant's carb deltas from INSIDE "
                         "Simulator._set_renderer_settings, i.e. at simulator-launch time, which is "
                         "where OmniGibson itself applies them. Needed because writes made AFTER "
                         "env creation were MEASURED to be inert -- every post-creation ladder "
                         "variant landed within 0.5%% of baseline. One variant per boot.")
    ap.add_argument("--gate-min-colors", type=int, default=2000)
    ap.add_argument("--gate-max-dominant", type=float, default=0.50)
    args = ap.parse_args()

    assert not args.out.startswith("/tmp"), "/tmp is node-local and wiped -- artifacts must be on Lustre"
    os.makedirs(args.out, exist_ok=True)
    report = {"label": args.label, "argv": sys.argv, "args": vars(args)}
    json_path = os.path.join(args.out, f"{args.label}.json")

    def flush():
        with open(json_path, "w") as f:
            json.dump(report, f, indent=2, sort_keys=False)

    # ---------- stack detection, before anything heavy ----------
    sys.path.insert(0, "/app")
    import omnigibson as og
    from omnigibson.macros import gm
    import omnigibson.lazy as lazy

    try:                                    # og391 moved it out of eval.py
        from realm.sim_config import set_sim_config
    except ImportError:
        from realm.eval import set_sim_config
    from realm.eval import SUPPORTED_TASKS, SUPPORTED_PERTURBATIONS
    from realm.environments.env_dynamic import RealmEnvironmentDynamic

    sig = inspect.signature(set_sim_config)
    is_111 = "rendering_mode" in sig.parameters
    stack = "og111" if is_111 else "og391"

    og_file = og.__file__
    with open(os.path.join(os.path.dirname(og_file), "simulator.py"), "rb") as f:
        sim_md5 = hashlib.md5(f.read()).hexdigest()

    report["identity"] = {
        "stack": stack,
        "omnigibson_file": og_file,
        "omnigibson_version": getattr(og, "__version__", None),
        "simulator_py_md5": sim_md5,
        "set_sim_config_signature": str(sig),
        "set_sim_config_file": inspect.getsourcefile(set_sim_config),
    }
    print(f"[identity] stack={stack}  og={og_file}  simulator.py md5={sim_md5}")
    print(f"[identity] set_sim_config{sig}")
    flush()

    # ---------- macros ----------
    if is_111:
        set_sim_config(rendering_mode=args.rendering_mode, robot=args.robot, og_lite=args.og_lite)
    else:
        set_sim_config(robot=args.robot)

    hq_forced = None
    if args.hq is not None:
        hq_forced = bool(args.hq)
        gm.ENABLE_HQ_RENDERING = hq_forced
    if args.render_freq is not None:
        gm.DEFAULT_RENDERING_FREQ = args.render_freq
    if args.sim_freq is not None:
        gm.DEFAULT_SIM_STEP_FREQ = args.sim_freq

    # ---------- launch-time renderer-settings patch ----------
    # Post-creation carb writes were MEASURED inert for ambientLightIntensity / rendermode /
    # dlss.execMode / sampledLighting (every such ladder variant landed within 0.5% of baseline,
    # while the realm_mode_r bundle -- which also turns shadows/reflections/indirectDiffuse/AO off
    # -- moved the exterior mean by +13%). So for the settings that do not respond after the fact,
    # apply them where OmniGibson does: inside Simulator._set_renderer_settings, during launch.
    if args.patch_launch:
        deltas = {}
        for vname, dd, _note in build_ladder(args.patch_launch):
            deltas.update(dd)
        if not deltas:
            raise SystemExit(f"--patch-launch: no ladder variant named '{args.patch_launch}'")
        Sim = type(og.sim) if getattr(og, "sim", None) is not None else None
        if Sim is None:
            import omnigibson.simulator as _simmod
            Sim = _simmod.Simulator
        orig = Sim._set_renderer_settings

        def patched(self, _orig=orig, _deltas=deltas):
            _orig(self)
            cs2 = lazy.carb.settings.get_settings()
            for k, (t, v) in _deltas.items():
                if t == "f":
                    cs2.set_float(k, float(v))
                elif t == "i":
                    cs2.set_int(k, int(v))
                elif t == "b":
                    cs2.set_bool(k, bool(v))
                else:
                    cs2.set_string(k, str(v))
            print(f"[patch-launch] applied {len(_deltas)} delta(s) inside "
                  f"_set_renderer_settings: {sorted(_deltas)}")

        Sim._set_renderer_settings = patched
        report["patch_launch"] = {"variant": args.patch_launch,
                                 "deltas": {k: v[1] for k, v in deltas.items()}}
        print(f"[patch-launch] {args.patch_launch} -> {report['patch_launch']['deltas']}")

    macros = {k: getattr(gm, k, "<absent>") for k in [
        "ENABLE_HQ_RENDERING", "DEFAULT_RENDERING_FREQ", "DEFAULT_SIM_STEP_FREQ",
        "DEFAULT_PHYSICS_FREQ", "ENABLE_VISUAL_UPDATES", "RENDER_ON_STEP",
        "RENDER_VIEWER_CAMERA", "ENABLE_OBJECT_STATES", "ENABLE_TRANSITION_RULES",
        "USE_GPU_DYNAMICS",
    ]}
    macros = {k: (v if isinstance(v, (int, float, bool, str)) else str(v)) for k, v in macros.items()}
    report["macros_at_env_creation"] = macros
    report["hq_forced"] = hq_forced
    print(f"[macros] {json.dumps(macros)}")
    flush()

    # ---------- env ----------
    task = SUPPORTED_TASKS[args.task_id]
    pert = SUPPORTED_PERTURBATIONS[args.pert_id]
    report["scene"] = {"task": task, "perturbation": pert, "robot": args.robot,
                       "rendering_mode": args.rendering_mode, "multi_view": args.multi_view}
    print(f"[scene] task={task} pert={pert} robot={args.robot} mode={args.rendering_mode} "
          f"multi_view={args.multi_view}")

    kw = dict(
        config_path="/app/realm/config",
        task_cfg_path=f"REALM_DROID10/{task}/default.yaml",
        perturbations=[pert],
        multi_view=args.multi_view,
        no_rendering=False,
        rendering_mode=args.rendering_mode,
        robot=args.robot,
    )
    try:
        env = RealmEnvironmentDynamic(**kw)
    except Exception as e:
        # This is the expected outcome of --hq 1 on 3.9.1 at 15 Hz. Record it as a RESULT, not a
        # crash: whether the assert fires is one of the questions being measured.
        report["env_creation_error"] = {"type": type(e).__name__, "msg": str(e),
                                       "traceback": traceback.format_exc()[-4000:]}
        print(f"[env] CREATION FAILED: {type(e).__name__}: {e}")
        flush()
        return 3

    # The carb SNAPSHOT: taken after env creation, so it is what the stack actually configured.
    cs = lazy.carb.settings.get_settings()

    def carb_get(key, ty):
        try:
            if ty == "f":
                return round(float(cs.get_as_float(key)), 6)
            if ty == "i":
                return int(cs.get_as_int(key))
            if ty == "b":
                return bool(cs.get_as_bool(key))
            return cs.get_as_string(key)
        except Exception as e:
            return f"<err {type(e).__name__}>"

    snapshot = {k: carb_get(k, t) for k, t in CARB_KEYS}
    report["carb_readback_after_env_creation"] = snapshot
    print("[carb] " + json.dumps(snapshot, indent=2))
    flush()

    # ---------- reset + warmup ----------
    obs, _ = env.reset()
    # Under og_lite RENDER_ON_STEP is False, so warmup would settle the arm without ever refreshing
    # the renderer. 1.1.1's eval loop re-enables it for the duration of warmup; do the same.
    _flip = args.og_lite and hasattr(og.sim, "_render_on_step")
    if _flip:
        og.sim._render_on_step = True
    obs, rew, term, trunc, info = env.warmup(obs)
    if _flip:
        og.sim._render_on_step = False
    hold = np.concatenate((np.asarray(env.reset_qpos)[:7], np.atleast_1d(-1.0)))

    def step_n(n):
        """n frames, via whichever read path was selected."""
        nonlocal obs
        got = []
        for _ in range(n):
            obs, _r, _t, _tr, _i = env.step(hold)
            if args.read_path == "render_obs":
                # Flush the pipeline before reading the sensors, exactly as the 1.1.1 on-demand
                # eval loop does: N render() ticks, then render_obs().
                for _ in range(args.n_pre_obs_renders):
                    og.sim.render()
                obs, _i = env.omnigibson_env.render_obs()
            got.append(collect_rgb(obs))
        return got

    # Materials stream in asynchronously. On the 1.1.1 stack 6 settle steps produced an 85%-white
    # frame with the geometry as unlit silhouettes and exactly one texture resolved -- a partially
    # materialised scene, which the gate caught. Tick the renderer until it stops changing.
    if args.pre_renders:
        report["material_settle"] = []
        prev = None
        for i in range(0, args.pre_renders, 25):
            for _ in range(25):
                og.sim.render()
            got = step_n(1)[0]                     # a real obs read, same path as the measurement
            k = sorted(got)[0]
            m = float(got[k].mean())
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

    # ---------- the ladder ----------
    ladder = build_ladder(args.only)
    report["ladder"] = []
    gate_failures = 0

    for vname, deltas, note in ladder:
        # Restore the snapshot first so variants never compound.
        for k, t in CARB_KEYS:
            v = snapshot[k]
            if isinstance(v, str) and v.startswith("<err"):
                continue
            try:
                if t == "f":
                    cs.set_float(k, float(v))
                elif t == "i":
                    cs.set_int(k, int(v))
                elif t == "b":
                    cs.set_bool(k, bool(v))
                else:
                    cs.set_string(k, str(v))
            except Exception:
                pass
        for k, (t, v) in deltas.items():
            if t == "f":
                cs.set_float(k, float(v))
            elif t == "i":
                cs.set_int(k, int(v))
            elif t == "b":
                cs.set_bool(k, bool(v))
            else:
                cs.set_string(k, str(v))

        step_n(args.settle)
        seq = step_n(args.frames)

        entry = {"variant": vname, "note": note,
                 "deltas": {k: v[1] for k, v in deltas.items()},
                 "carb_effective": {k: carb_get(k, t) for k, t in CARB_KEYS},
                 "cameras": {}}
        for cam in cams:
            stack_imgs = [s[cam] for s in seq if cam in s]
            if not stack_imgs:
                entry["cameras"][cam] = {"gate_ok": False, "gate_fail": ["camera absent from obs"]}
                gate_failures += 1
                continue
            # Median over frames: at rendering_mode="rt" two renders of an UNCHANGED scene differ on
            # ~25% of pixels on this stack, so a single frame is not evidence.
            med = np.median(np.stack(stack_imgs, 0), axis=0).astype(np.uint8)
            st = frame_stats(med, args.gate_min_colors, args.gate_max_dominant)
            st["per_frame_mean"] = [round(float((0.299 * im[..., 0] + 0.587 * im[..., 1]
                                                + 0.114 * im[..., 2]).mean()), 3)
                                    for im in stack_imgs]
            png = os.path.join(args.out, f"{args.label}__{vname}__{cam.replace('.', '-')}.png")
            try:
                from PIL import Image
                Image.fromarray(med).save(png)
                st["png"] = png
            except Exception as e:
                st["png_error"] = f"{type(e).__name__}: {e}"
            if not st["gate_ok"]:
                gate_failures += 1
                print(f"  !! GATE FAIL {vname} {cam}: {st['gate_fail']}")
            entry["cameras"][cam] = st
            print(f"  {vname:26s} {cam:44s} mean={st['mean']:7.2f} p50={st['p50']:6.1f} "
                  f"p95={st['p95']:6.1f} sat={st['sat_pct']:6.3f}% colours={st['n_colors']}")
        report["ladder"].append(entry)
        flush()

    report["gate_failures"] = gate_failures
    report["ok"] = gate_failures == 0
    flush()
    print(f"\n[done] {json_path}  gate_failures={gate_failures}")
    return 0 if gate_failures == 0 else 4


if __name__ == "__main__":
    # Isaac exits 0 even on an unhandled exception, so record the failure in the JSON too.
    try:
        rc = main()
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        rc = 9
    print(f"PROBE_RC={rc}")
    sys.exit(rc)
