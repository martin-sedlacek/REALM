#!/usr/bin/env python
"""lightpath_sweep.py -- the LIGHT-TRANSPORT half of the og391-vs-OG-1.1.1 render-tone gap.

Scope, deliberately narrow: GI / indirect-diffuse, sampled direct lighting, ambient occlusion,
shadows, reflections, the `sceneDb` ambient term, and the RTX Real-Time mode flags. It refuses to
write anything under `/rtx/post/` -- tonemapping and auto-exposure are a different family and are
owned by a different probe; two probes writing the same keys in the same session would make both
sets of numbers unattributable.

WHY THIS EXISTS SEPARATELY FROM render_brightness_ab.py
-------------------------------------------------------
That probe swept the ~13 keys OmniGibson *explicitly sets*, found them inert, and concluded "no
setting closes the gap". But OmniGibson sets almost nothing under these prefixes: og111's live tree
carries 24 keys under /rtx/indirectDiffuse, 38 under /rtx/directLighting, 15 under /rtx/reflections
and 7 under /rtx/ambientOcclusion, and every one of those comes from a KIT DEFAULT. Kit differs
between Isaac 4.x (OG 1.1.1) and Isaac 5.1 (OG 3.9.1), so a default that moved between Isaac
versions is invisible to any hand-written key list. This probe therefore does not use a hand-written
key list: it DIFFS the two live trees and generates its sweep from the diff.

It also chases a specific signature rather than "brightness". og391's blacks are LIFTED: on exterior
cam1 the 1.1.1 reference is p5 12 with 20.4% of pixels below 60, og391 is p5 28 with 8.4% below 60.
That is what too much indirect/ambient fill looks like, and it is not something a mean can see. So
every row reports p5 and %dark alongside the mean, plus variance-of-Laplacian as a texture/detail
proxy.

HOW IT RUNS
-----------
Two-pass, adaptive, inside ONE boot:

  pass 1  one variant per differing PREFIX -- set every key that differs under, say,
          /rtx/indirectDiffuse to og111's value at once. Cheap, and it localises the effect.
  pass 2  every prefix whose |dmean| or |d%dark| cleared a threshold in pass 1 is expanded into
          one variant PER KEY, so the responsible key is named rather than guessed.

plus a curated set of INTENSITY knobs that exist on both stacks and therefore never show up in a
diff -- GI scaling factor, GI bounce count, AO ray length, reflection roughness cutoff, sampled-
lighting SPP -- because "same default on both stacks" does not mean "cannot be used to restore the
look", and the two RTX mode combinations (rt on/rt2 off, rt off/rt2 on) that the earlier ablation
never separated.

TIMING. Runtime carb writes on og391 are ATTENUATED but not inert: the earlier probe measured
REALM's own "r" bundle at +9.8% when written after warmup and +20.0% when written before the first
render. So a post-warmup sweep SCREENS (direction is trustworthy, magnitude is a lower bound) and
`--apply NAME` re-measures one variant at the pre-first-render timing that `set_rendering_mode`
itself uses. Screen wide, then confirm narrow.

SETTLING. /rtx/indirectDiffuse/denoiser/temporal/maxHistory is 100 frames, so a GI change needs far
more than a handful of ticks to converge. Settling is done with og.sim.render() ticks (cheap, no
physics) and defaults to 60; `--settle-trace` prints the convergence so a too-short settle is
visible rather than silent.

GATING. Every frame must clear a unique-colour floor and a single-colour ceiling before it is
allowed to produce a number. og.sim.render() + a sensor read has returned a 99.99%-white buffer on
these stacks and a confident verdict was once read off exactly that.

    # og391: read back this family, diff it against 1.1.1's tree, sweep what differs
    STACK=og391 PROBE=lightpath_sweep.py ./scripts/debug_probes/run_brightness_ab.sh \
        --label og391_screen --ref-tree /logs/render_bright_ab/carb_tree_og111.json \
        --dump-tree /logs/lightpath/carb_tree_og391.json --auto

    # confirm one winner at set_rendering_mode's own timing
    STACK=og391 PROBE=lightpath_sweep.py ./scripts/debug_probes/run_brightness_ab.sh \
        --label og391_apply_x --apply amb_both_ref
"""

import argparse
import inspect
import json
import os
import sys
import traceback

import numpy as np

# The prefixes this probe owns -- read back, diffed, AND swept. /rtx/post/* is deliberately absent.
#
# The list is not guesswork: it is every /rtx subtree in OG 1.1.1's live tree that carries a light-
# transport control. Four of these were missed on the first pass and matter --
#   /rtx/fog          an atmospheric term lifts blacks exactly the way this gap looks
#   /rtx/domeLight    a scene-wide fill neither OmniGibson sets explicitly
#   /rtx/scenedb      a SECOND subtree, distinct from /rtx/sceneDb; carb paths are case-sensitive
#                     and both exist, so matching only the camel-case one reads half the ambient
#                     configuration
#   /rtx/realtime     the real-time renderer's own block
MY_PREFIXES = (
    "/rtx/indirectDiffuse",
    "/rtx/directLighting",
    "/rtx/ambientOcclusion",
    "/rtx/shadows",
    "/rtx/reflections",
    "/rtx/sceneDb",
    "/rtx/scenedb",
    "/rtx/domeLight",
    "/rtx/fog",
    "/rtx/realtime",
    "/rtx/sampledLighting",
    "/rtx/lightcache",
    "/rtx/caustics",
    "/rtx/newDenoiser",
    "/rtx/matteObject",
    "/rtx/forwardLitMode",
    "/rtx/useViewLightingMode",
    "/rtx/normalMapRoughness",
    "/rtx/ecoMode",
    "/rtx/rtx/modes",
    "/rtx/translucency",
    "/rtx/raytracing",
    "/rtx/lightspeed",
)

# Read back and diffed so a difference is on the record, but NEVER written: these are the material /
# asset-pipeline side, which is a different investigation, and /rtx/pathtracing only drives the "pt"
# renderer that evals do not use.
REPORT_ONLY_PREFIXES = (
    "/rtx/materialDb",
    "/rtx/materialdb",
    "/rtx/material",
    "/rtx/textures",
    "/rtx/pathtracing",
    "/rtx/hydra",
    "/rtx/flow",
    "/rtx/shaderDb",
)
FORBIDDEN_PREFIX = "/rtx/post/"

REF = "__REF__"  # sentinel: "whatever OG 1.1.1's live tree has for this key"

# Keys whose value is an implementation detail rather than a light-transport control: writing them
# either cannot matter or actively breaks the renderer. Excluded from the auto-generated sweep.
AUTO_SKIP_SUBSTR = (
    "reservedInstances", "skipCmdListMangerDestroy", "gpuSkinning",
    "tlasInstanceList", "Debug", "debug", "constantSeed", "profil",
)


def is_mine(key):
    return key.startswith(MY_PREFIXES) and not key.startswith(FORBIDDEN_PREFIX)


# ==================================================================================================
# curated variants: intensity knobs that are the SAME on both stacks (so a diff never surfaces them)
# plus the two RTX-mode combinations the earlier ablation collapsed into one.
# ==================================================================================================
def curated_variants():
    return [
        # ---- the ambient term, properly. The earlier probe read ambientLightColor with
        # get_as_string() and got "" back, so it never actually compared the ARRAY. If og391's
        # colour is [1,1,1] where 1.1.1's is [0.1,0.1,0.1], then intensity alone (1.0 -> 0.1) leaves
        # a 10x ambient floor standing, which would explain why that lever measured inert.
        ("amb_int_ref", {"/rtx/sceneDb/ambientLightIntensity": REF},
         "intensity -> 1.1.1's value only (the lever already tested: -0.3%)"),
        ("amb_col_ref", {"/rtx/sceneDb/ambientLightColor": REF},
         "COLOUR -> 1.1.1's value only -- never tested, the old readback garbled the array"),
        ("amb_both_ref", {"/rtx/sceneDb/ambientLightIntensity": REF,
                          "/rtx/sceneDb/ambientLightColor": REF},
         "intensity AND colour -> 1.1.1 -- the product is what feeds the ambient floor"),
        ("amb_zero", {"/rtx/sceneDb/ambientLightIntensity": 0.0,
                      "/rtx/sceneDb/ambientLightColor": [0.0, 0.0, 0.0]},
         "ambient fully off -- an upper bound on how much of the lifted black is ambient"),

        # ---- indirect diffuse (GI): intensity, not just enable
        ("gi_off", {"/rtx/indirectDiffuse/enabled": False}, "GI off entirely"),
        ("gi_scale_0", {"/rtx/indirectDiffuse/scalingFactor": 0.0}, "GI contribution scaled to 0"),
        ("gi_scale_0.5", {"/rtx/indirectDiffuse/scalingFactor": 0.5}, "GI at half strength"),
        ("gi_bounce_0", {"/rtx/indirectDiffuse/maxBounces": 0}, "no diffuse bounces"),
        ("gi_bounce_1", {"/rtx/indirectDiffuse/maxBounces": 1}, "one diffuse bounce (1.1.1 has 2)"),
        ("gi_maxray_100", {"/rtx/indirectDiffuse/maxRayIntensity": 100.0},
         "clamp indirect ray intensity hard (default 6400)"),

        # ---- ambient occlusion: strength/radius, which is what sets contact-shadow depth
        ("ao_off", {"/rtx/ambientOcclusion/enabled": False}, "AO off"),
        ("ao_ray_2", {"/rtx/ambientOcclusion/rayLength": 2.0},
         "short AO radius -- tight contact shadows (default 35)"),
        ("ao_ray_200", {"/rtx/ambientOcclusion/rayLength": 200.0}, "very long AO radius"),
        ("ao_samples_32", {"/rtx/ambientOcclusion/maxSamples": 32,
                           "/rtx/ambientOcclusion/minSamples": 16}, "more AO samples"),

        # ---- shadows
        ("shadow_spp_8", {"/rtx/shadows/sampleCount": 8}, "8 shadow samples (default 1)"),
        ("shadow_off", {"/rtx/shadows/enabled": False},
         "shadows off -- a control: it must make things BRIGHTER, and if it does not the key is dead"),

        # ---- reflections: the mug reads reflective on og391 and matte on 1.1.1
        ("refl_off", {"/rtx/reflections/enabled": False}, "reflections off"),
        ("refl_maxrough_0", {"/rtx/reflections/maxRoughness": 0.0},
         "only mirror-smooth surfaces reflect (default 0.3)"),
        ("refl_nogi", {"/rtx/reflections/giInReflections": False}, "no GI inside reflections"),

        # ---- sampled direct lighting
        ("dl_spp_1", {"/rtx/directLighting/sampledLighting/samplesPerPixel": 1},
         "1 sample/pixel (og391 readback says 2)"),
        ("dl_spp_8", {"/rtx/directLighting/sampledLighting/samplesPerPixel": 8}, "8 samples/pixel"),
        ("dl_dome_off", {"/rtx/directLighting/domeLight/enabled": False},
         "dome light off -- a scene-wide fill term neither stack sets explicitly"),
        ("dl_dome_noibl", {"/rtx/directLighting/domeLight/approxIbl/enabled": False},
         "no approximate IBL from the dome"),

        # ---- the RTX mode flags, SEPARATED. The earlier ablation only ever tried both off, which
        # went the wrong way (+6.5%). 1.1.1 predates Real-Time 2.0 entirely, so rt on / rt2 off is
        # the faithful analogue and has never been measured on its own.
        ("rt_on_rt2_off", {"/rtx/rtx/modes/rt/enabled": True,
                           "/rtx/rtx/modes/rt2/enabled": False},
         "Real-Time 1.0 only -- the faithful 1.1.1 analogue, never measured"),
        ("rt_off_rt2_on", {"/rtx/rtx/modes/rt/enabled": False,
                           "/rtx/rtx/modes/rt2/enabled": True},
         "Real-Time 2.0 only"),
        ("rt_both_off", {"/rtx/rtx/modes/rt/enabled": False,
                         "/rtx/rtx/modes/rt2/enabled": False},
         "both off -- re-measured here only as the control the earlier run established (+6.5%)"),
        # Not a change at all: it rewrites the values OG 3.9.1's own _set_renderer_settings already
        # wrote (simulator.py:664-665), at the pre-first-render timing. It exists because the
        # as-shipped render is NOT reproducible boot to boot -- some boots come back matching the
        # earlier `rt/rt2 OFF` measurement on all three cameras while the readback still says both
        # flags are True. If re-asserting True changes the image, the launch-time write is being
        # lost and this is a port bug, not a tuning question.
        ("rt_both_on", {"/rtx/rtx/modes/rt/enabled": True,
                        "/rtx/rtx/modes/rt2/enabled": True},
         "re-assert 3.9.1's OWN rt/rt2 values at pre-first-render -- a no-op unless the launch write is lost"),
    ]


# ==================================================================================================
# frames + stats
# ==================================================================================================
def luma(a):
    a = np.asarray(a, dtype=np.float32)
    return 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]


def lap_var(lum):
    """Variance of a 4-neighbour Laplacian -- the standard focus/detail proxy.

    Absolute scale depends on the kernel, so compare rows produced by THIS function, not against a
    number from cv2.Laplacian with a different ksize.
    """
    lap = (lum[:-2, 1:-1] + lum[2:, 1:-1] + lum[1:-1, :-2] + lum[1:-1, 2:]
           - 4.0 * lum[1:-1, 1:-1])
    return float(lap.var())


def frame_stats(img, gate_min_colors, gate_max_dominant, dark_thresh=60.0):
    a = np.asarray(img)
    lum = luma(a)
    flat = a.reshape(-1, 3)
    packed = (flat[:, 0].astype(np.uint32) << 16) | (flat[:, 1].astype(np.uint32) << 8) | flat[:, 2]
    vals, counts = np.unique(packed, return_counts=True)
    n_colors = int(vals.size)
    dominant = float(counts.max()) / float(packed.size)
    st = {
        "shape": [int(a.shape[0]), int(a.shape[1])],
        "mean": round(float(lum.mean()), 2),
        "p05": round(float(np.percentile(lum, 5)), 1),
        "p50": round(float(np.percentile(lum, 50)), 1),
        "p95": round(float(np.percentile(lum, 95)), 1),
        "sat_pct": round(100.0 * float((lum >= 250).mean()), 4),
        "dark_pct": round(100.0 * float((lum < dark_thresh).mean()), 2),
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


def collect_rgb(obs):
    out = {}

    def walk(node, path):
        if isinstance(node, dict):
            for k, v in node.items():
                if k == "rgb":
                    x = v
                    if hasattr(x, "cpu"):
                        x = x.cpu().numpy()
                    x = np.asarray(x)
                    if x.ndim == 3 and x.shape[-1] >= 3:
                        out[path] = x[..., :3].astype(np.uint8)
                else:
                    walk(v, f"{path}.{k}" if path else str(k))

    walk(obs, "")
    return out


# ==================================================================================================
# carb tree
# ==================================================================================================
def dump_tree(lazy, prefixes=MY_PREFIXES):
    """Flat {path: value} for THIS PROBE'S PREFIXES ONLY. Never raises, never walks the root.

    Deliberately NOT a whole-tree dump. `carb.dictionary.get_dict_copy` on the root settings item
    HARD-CRASHES the og391 container -- a C-layer abort, not a Python exception, so it takes the whole
    Isaac boot with it and leaves a JSON with no dump in it. (That is what happened to the earlier
    `--dump-carb` attempt on og391: the report has `carb_readback_after_env_creation` and then
    nothing.) It works on og111/Isaac 4.x, which is why the 1.1.1 reference tree exists at all.

    So: one `settings.get(prefix)` per prefix. That returns a plain nested dict for a small subtree
    and is flattened in Python, with no recursion inside carb and no visit to the root. A per-prefix
    bounded BFS is kept as a fallback for a prefix whose `get()` returns something unexpected -- it
    uses only get_item_child_count / get_item_child_by_index, never get_dict_copy.
    """
    import collections

    def flatten(node, prefix, out):
        if isinstance(node, dict):
            for k, v in node.items():
                flatten(v, f"{prefix}/{k}", out)
        elif isinstance(node, (list, tuple)):
            out[prefix] = list(node)
        else:
            out[prefix] = node

    cs = lazy.carb.settings.get_settings()
    flat, how = {}, {}
    for p in prefixes:
        got = None
        try:
            got = cs.get(p)
        except Exception as e:
            how[p] = f"get() raised {type(e).__name__}: {e}"
        if isinstance(got, dict):
            n0 = len(flat)
            flatten(got, p, flat)
            how[p] = f"settings.get -> {len(flat) - n0} leaves"
            continue
        if got is not None and not isinstance(got, dict):
            flat[p] = got
            how[p] = "settings.get -> scalar leaf"
            continue
        # fallback: bounded BFS inside this prefix only
        try:
            import carb.dictionary
            di = carb.dictionary.get_dictionary()
            root = cs.get_settings_dictionary(p)
            if root is None:
                how[p] = "absent"
                continue
            n0, seen = len(flat), 0
            q = collections.deque([(root, p)])
            while q and seen < 20000:
                item, path = q.popleft()
                seen += 1
                n = di.get_item_child_count(item)
                if n == 0:
                    try:
                        flat[path] = cs.get(path)
                    except Exception:
                        flat[path] = "<unreadable>"
                    continue
                for i in range(n):
                    c = di.get_item_child_by_index(item, i)
                    q.append((c, f"{path}/{di.get_item_name(c)}"))
            how[p] = f"bounded BFS -> {len(flat) - n0} leaves"
        except Exception as e:
            how[p] = f"BFS failed: {type(e).__name__}: {e}"
    return flat, "per-prefix settings.get (root NEVER walked: get_dict_copy aborts og391)", how


def jsonable(v):
    if isinstance(v, (bool, int, float, str)) or v is None:
        return v
    if isinstance(v, (list, tuple)):
        return [jsonable(x) for x in v]
    return str(v)


def same(a, b, tol=1e-6):
    """Value equality that tolerates float noise and int/float mixing."""
    if isinstance(a, bool) or isinstance(b, bool):
        return bool(a) == bool(b)
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) <= tol * max(1.0, abs(float(a)), abs(float(b)))
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(same(x, y, tol) for x, y in zip(a, b))
    return a == b


def carb_set(cs, key, val):
    assert not key.startswith(FORBIDDEN_PREFIX), f"refusing to write {key}: /rtx/post/* is another probe's family"
    if isinstance(val, bool):
        cs.set_bool(key, val)
    elif isinstance(val, int):
        cs.set_int(key, int(val))
    elif isinstance(val, float):
        cs.set_float(key, float(val))
    elif isinstance(val, str):
        cs.set_string(key, val)
    else:
        cs.set(key, list(val) if isinstance(val, (list, tuple)) else val)


# ==================================================================================================
# main
# ==================================================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--task-id", type=int, default=3, help="3=rotate_mug 0=put_green_block_into_bowl 7=push_switch")
    ap.add_argument("--pert-id", type=int, default=0)
    ap.add_argument("--robot", default="DROID")
    ap.add_argument("--rendering-mode", default="rt")
    ap.add_argument("--ref-tree", default=None, help="OG 1.1.1 carb tree JSON to diff against")
    ap.add_argument("--dump-tree", default=None,
                    help="write this stack's LIGHT-TRANSPORT subtree here (this probe's prefixes "
                         "only -- a whole-tree dump aborts the og391 container in the C layer)")
    ap.add_argument("--auto", action="store_true",
                    help="generate the sweep from the tree diff (pass 1 prefix bundles, pass 2 the "
                         "per-key expansion of whichever bundles moved)")
    ap.add_argument("--curated", action="store_true", default=True)
    ap.add_argument("--no-curated", dest="curated", action="store_false")
    ap.add_argument("--only", default=None, help="comma-separated subset of variant names")
    ap.add_argument("--apply", default=None,
                    help="apply ONE variant's deltas pre-first-render (set_rendering_mode's own "
                         "timing, where writes are not attenuated), measure it, and stop")
    ap.add_argument("--apply-json", default=None,
                    help="like --apply but the deltas come from a JSON file {key: value}, so a "
                         "combination discovered by a screening run can be confirmed without "
                         "editing this file")
    ap.add_argument("--frames", type=int, default=5)
    ap.add_argument("--settle-renders", type=int, default=60,
                    help="og.sim.render() ticks after a carb change. The indirect-diffuse temporal "
                         "denoiser has a 100-frame history, so a small number here silently "
                         "measures a half-converged image.")
    ap.add_argument("--pre-renders", type=int, default=300, help="material-streaming settle")
    ap.add_argument("--settle-trace", action="store_true", default=True)
    ap.add_argument("--expand-thresh-mean", type=float, default=1.0)
    ap.add_argument("--expand-thresh-dark", type=float, default=0.5)
    ap.add_argument("--ref-cam", default="external.external_sensor1")
    ap.add_argument("--gate-min-colors", type=int, default=2000)
    ap.add_argument("--gate-max-dominant", type=float, default=0.50)
    args = ap.parse_args()

    assert not args.out.startswith("/tmp"), "/tmp is node-local and wiped; artifacts go on Lustre"
    os.makedirs(args.out, exist_ok=True)
    report = {"label": args.label, "argv": sys.argv}
    jpath = os.path.join(args.out, f"{args.label}.json")

    def flush():
        with open(jpath, "w") as f:
            json.dump(report, f, indent=1, sort_keys=False, default=str)

    sys.path.insert(0, "/app")
    import omnigibson as og
    from omnigibson.macros import gm
    import omnigibson.lazy as lazy

    try:
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
                          "set_sim_config": str(sig)}
    print(f"[identity] stack={stack} og={og.__file__} v={getattr(og, '__version__', None)}")
    flush()

    if is_111:
        set_sim_config(rendering_mode=args.rendering_mode, robot=args.robot)
    else:
        set_sim_config(robot=args.robot)

    # ---- reference tree, loaded BEFORE the boot so a bad path fails cheaply ----
    ref = {}
    if args.ref_tree:
        with open(args.ref_tree) as f:
            rt = json.load(f)
        ref = rt.get("settings") or rt.get("tree_after_settle") or {}
        assert ref, f"--ref-tree {args.ref_tree} has no settings/tree_after_settle"
        report["ref_tree"] = {"path": args.ref_tree, "n_keys": len(ref),
                              "stack": rt.get("stack") or rt.get("identity", {})}
        print(f"[ref-tree] {len(ref)} keys from {args.ref_tree}")

    task = SUPPORTED_TASKS[args.task_id]
    pert = SUPPORTED_PERTURBATIONS[args.pert_id]
    report["scene"] = {"task": task, "perturbation": pert, "robot": args.robot,
                       "rendering_mode": args.rendering_mode}
    report["macros"] = {k: jsonable(getattr(gm, k, "<absent>")) for k in
                        ("ENABLE_HQ_RENDERING", "DEFAULT_RENDERING_FREQ", "DEFAULT_SIM_STEP_FREQ",
                         "RENDER_ON_STEP", "HEADLESS")}
    print(f"[scene] task={task} pert={pert} robot={args.robot} mode={args.rendering_mode}")
    flush()

    try:
        env = RealmEnvironmentDynamic(
            config_path="/app/realm/config",
            task_cfg_path=f"REALM_DROID10/{task}/default.yaml",
            perturbations=[pert], multi_view=True, no_rendering=False,
            rendering_mode=args.rendering_mode, robot=args.robot,
        )
    except Exception as e:
        report["env_creation_error"] = {"type": type(e).__name__, "msg": str(e),
                                       "tb": traceback.format_exc()[-4000:]}
        print(f"[env] CREATION FAILED: {type(e).__name__}: {e}")
        flush()
        return 3

    cs = lazy.carb.settings.get_settings()

    # ---- this stack's light-transport subtree, right after env creation ----
    mine_tree, method, how = dump_tree(lazy)
    report["tree"] = {"n_keys": len(mine_tree), "method": method, "per_prefix": how,
                      "prefixes": list(MY_PREFIXES)}
    print(f"[tree] {len(mine_tree)} keys via {method}")
    for p, h in how.items():
        print(f"   {p:28s} {h}")
    if args.dump_tree:
        os.makedirs(os.path.dirname(args.dump_tree) or ".", exist_ok=True)
        with open(args.dump_tree, "w") as f:
            json.dump({"stack": stack, "label": args.label, "method": method, "per_prefix": how,
                       "prefixes": list(MY_PREFIXES), "n_keys": len(mine_tree),
                       "scope": "LIGHT-TRANSPORT PREFIXES ONLY -- not a whole-tree dump",
                       "settings": {k: jsonable(v) for k, v in mine_tree.items()}},
                      f, indent=1, sort_keys=True, default=str)
        print(f"[tree] -> {args.dump_tree}")

    # ---- the read-only neighbours: diffed so a difference is recorded, never written ----
    ro_tree, _m2, ro_how = dump_tree(lazy, REPORT_ONLY_PREFIXES)
    report["report_only_tree"] = {"n_keys": len(ro_tree), "per_prefix": ro_how}
    if ref:
        ro_diff = {k: {"og391": jsonable(v), "og111": jsonable(ref[k])}
                   for k, v in sorted(ro_tree.items())
                   if k in ref and not same(v, ref[k])}
        ro_only = {k: jsonable(v) for k, v in sorted(ro_tree.items()) if k not in ref}
        report["report_only_diff"] = {"differs": ro_diff, "only_here": ro_only}
        print(f"[report-only] {len(ro_diff)} differ / {len(ro_only)} new, over "
              f"{list(REPORT_ONLY_PREFIXES)} -- NOT swept")
        for k, d in ro_diff.items():
            print(f"  RO-DIFF {k}: {stack}={d['og391']!r}  ref={d['og111']!r}")
    flush()

    # ---- the diff, restricted to this probe's family ----
    diff = {"differs": {}, "only_here": {}, "only_ref": {}}
    if ref:
        for k, v in sorted(mine_tree.items()):
            if not is_mine(k):
                continue
            if k in ref:
                if not same(v, ref[k]):
                    diff["differs"][k] = {"og391": jsonable(v), "og111": jsonable(ref[k])}
            else:
                diff["only_here"][k] = jsonable(v)
        for k, v in sorted(ref.items()):
            if is_mine(k) and k not in mine_tree:
                diff["only_ref"][k] = jsonable(v)
        report["diff"] = diff
        print(f"\n[diff] over {len(MY_PREFIXES)} prefixes: {len(diff['differs'])} differ, "
              f"{len(diff['only_here'])} only on {stack}, {len(diff['only_ref'])} only on the ref")
        for k, d in diff["differs"].items():
            print(f"  DIFF {k}: {stack}={d['og391']!r}  ref={d['og111']!r}")
        for k, v in diff["only_here"].items():
            print(f"  ONLY-{stack} {k} = {v!r}")
        for k, v in diff["only_ref"].items():
            print(f"  ONLY-REF   {k} = {v!r}")
    flush()

    # ---- build the sweep ----
    def resolve(deltas):
        """REF -> the reference tree's value; drop anything unresolvable, and say so."""
        out, dropped = {}, {}
        for k, v in deltas.items():
            if v is REF or v == REF:
                if k in ref:
                    out[k] = ref[k]
                else:
                    dropped[k] = "absent from reference tree"
            else:
                out[k] = v
        return out, dropped

    variants = []
    if args.curated:
        variants += [(n, d, note, "curated") for n, d, note in curated_variants()]

    prefix_bundles = []
    if args.auto and ref:
        groups = {}
        for k in diff["differs"]:
            if any(s in k for s in AUTO_SKIP_SUBSTR):
                continue
            for p in MY_PREFIXES:
                if k.startswith(p):
                    groups.setdefault(p, []).append(k)
                    break
        # og391-only booleans that are True: 1.1.1's analogue is "off", exactly as r111 did for
        # rt/rt2/flow. Anything else og391-only is left alone -- there is no 1.1.1 value to target.
        onlyhere = {}
        for k, v in diff["only_here"].items():
            if any(s in k for s in AUTO_SKIP_SUBSTR):
                continue
            if isinstance(v, bool) and v:
                onlyhere[k] = False
        if onlyhere:
            groups.setdefault("__only_here_bools__", []).extend(sorted(onlyhere))
        for p, keys in sorted(groups.items()):
            if p == "__only_here_bools__":
                d = {k: False for k in keys}
            else:
                d = {k: REF for k in keys}
            prefix_bundles.append((f"bundle{p.replace('/', '_')}", d,
                                   f"all {len(keys)} differing key(s) under {p} -> 1.1.1", "bundle"))
        variants += prefix_bundles
        if not prefix_bundles:
            print("[auto] no differing keys in this family -- nothing to bundle")

    if args.only:
        want = set(args.only.split(","))
        variants = [v for v in variants if v[0] in want]

    # ---- apply-one-pre-first-render mode ----
    apply_deltas, apply_name = None, None
    if args.apply_json:
        with open(args.apply_json) as f:
            apply_deltas = json.load(f)
        apply_name = os.path.basename(args.apply_json).replace(".json", "")
    elif args.apply:
        cand = [v for v in variants if v[0] == args.apply]
        if not cand:
            cand = [(n, d, t, "curated") for n, d, t in curated_variants() if n == args.apply]
        if not cand:
            raise SystemExit(f"--apply: no variant named '{args.apply}' "
                             f"(known: {sorted(v[0] for v in variants)})")
        apply_deltas = cand[0][1]
        apply_name = args.apply

    if apply_deltas is not None:
        resolved, dropped = resolve(apply_deltas)
        for k in resolved:
            assert is_mine(k), f"--apply key {k} is outside this probe's family"
        before = {k: jsonable(cs.get(k)) for k in resolved}
        for k, v in resolved.items():
            carb_set(cs, k, v)
        after = {k: jsonable(cs.get(k)) for k in resolved}
        report["applied_pre_first_render"] = {
            "name": apply_name, "requested": {k: jsonable(v) for k, v in resolved.items()},
            "before": before, "after": after, "dropped": dropped,
            "write_verified": all(same(after[k], resolved[k]) for k in resolved),
        }
        print(f"[apply] {apply_name} pre-first-render: {len(resolved)} key(s), "
              f"write_verified={report['applied_pre_first_render']['write_verified']}")
        for k in resolved:
            print(f"   {k}: {before[k]!r} -> {after[k]!r}")
        if dropped:
            print(f"   DROPPED: {dropped}")
        variants = []          # measure exactly the applied config, nothing else
        flush()

    report["sweep_plan"] = [{"name": n, "kind": kind, "keys": sorted(d)} for n, d, _t, kind in variants]
    print(f"\n[plan] {len(variants)} variant(s): {[v[0] for v in variants]}")
    flush()

    # ---- reset, warm up, settle materials ----
    obs, _ = env.reset()
    obs, _r, _t, _tr, _i = env.warmup(obs)
    hold = np.concatenate((np.asarray(env.reset_qpos)[:7], np.atleast_1d(-1.0)))

    def read_obs():
        nonlocal obs
        obs, _a, _b, _c, _d = env.step(hold)
        return collect_rgb(obs)

    def settle(n, tag=""):
        """n render ticks, then one obs read. Traces convergence so a short settle is visible."""
        trace = []
        step = max(20, n // 4)
        done = 0
        while done < n:
            k = min(step, n - done)
            for _ in range(k):
                og.sim.render()
            done += k
            got = read_obs()
            if args.settle_trace and got:
                cam = args.ref_cam if args.ref_cam in got else sorted(got)[0]
                m = round(float(luma(got[cam]).mean()), 2)
                trace.append({"renders": done, "mean": m})
                d = "" if len(trace) < 2 else f" d={m - trace[-2]['mean']:+.2f}"
                print(f"    [settle{tag} {done}/{n}] {cam} mean={m}{d}")
        return trace

    report["material_settle"] = settle(args.pre_renders, " materials")
    cams = sorted(collect_rgb(obs).keys())
    report["cameras"] = cams
    print(f"[cams] {cams}")
    flush()

    def measure(name, note, kind, deltas_applied):
        seq = [read_obs() for _ in range(args.frames)]
        entry = {"variant": name, "kind": kind, "note": note,
                 "deltas": {k: jsonable(v) for k, v in (deltas_applied or {}).items()},
                 "cameras": {}}
        for cam in cams:
            imgs = [s[cam] for s in seq if cam in s]
            if not imgs:
                entry["cameras"][cam] = {"gate_ok": False, "gate_fail": ["camera absent"]}
                continue
            med = np.median(np.stack(imgs, 0), axis=0).astype(np.uint8)
            st = frame_stats(med, args.gate_min_colors, args.gate_max_dominant)
            st["per_frame_mean"] = [round(float(luma(im).mean()), 2) for im in imgs]
            png = os.path.join(args.out, f"{args.label}__{name}__{cam.replace('.', '-')}.png")
            try:
                from PIL import Image
                Image.fromarray(med).save(png)
                st["png"] = png
            except Exception as e:
                st["png_error"] = f"{type(e).__name__}: {e}"
            entry["cameras"][cam] = st
            flag = "" if st["gate_ok"] else "  !!GATE-FAIL " + "; ".join(st["gate_fail"])
            print(f"  {name:34s} {cam:44s} mean={st['mean']:7.2f} p5={st['p05']:6.1f} "
                  f"p50={st['p50']:6.1f} p95={st['p95']:6.1f} sat={st['sat_pct']:6.3f}% "
                  f"dark={st['dark_pct']:6.2f}% detail={st['detail']:7.1f}{flag}")
        return entry

    rows = []
    report["rows"] = rows

    # baseline: touches nothing at all
    base = measure("baseline", f"{stack} as shipped at rendering_mode={args.rendering_mode}",
                   "baseline", {})
    rows.append(base)
    flush()
    if apply_deltas is not None:
        # The "baseline" row of an --apply boot IS the applied config -- the deltas went in before
        # any render. Relabel so no later reader mistakes it for as-shipped.
        base["variant"] = f"APPLIED:{apply_name}"
        base["kind"] = "applied_pre_first_render"
        base["note"] = f"{apply_name} written before the first render"
        report["ok"] = all(c.get("gate_ok") for c in base["cameras"].values())
        flush()
        print(f"\n[done] {jpath}")
        return 0 if report["ok"] else 4

    def base_of(cam, key):
        return base["cameras"].get(cam, {}).get(key)

    dirty = {}   # key -> pristine value, so only what a variant touched is ever restored

    def run(name, deltas, note, kind):
        resolved, dropped = resolve(deltas)
        # restore anything a previous variant dirtied that this one does not set
        for k in list(dirty):
            if k in resolved:
                continue
            v = dirty.pop(k)
            try:
                carb_set(cs, k, v)
            except Exception as e:
                print(f"    (restore {k} failed: {type(e).__name__})")
        for k, v in resolved.items():
            if k not in dirty:
                dirty[k] = mine_tree.get(k, cs.get(k))
            try:
                carb_set(cs, k, v)
            except Exception as e:
                print(f"    (set {k} failed: {type(e).__name__}: {e})")
        eff = {k: jsonable(cs.get(k)) for k in resolved}
        verified = all(same(eff[k], resolved[k]) for k in resolved)
        settle(args.settle_renders, f" {name}")
        e = measure(name, note, kind, resolved)
        e["carb_effective"] = eff
        e["write_verified"] = verified
        e["dropped"] = dropped
        if not verified:
            bad = {k: (jsonable(resolved[k]), eff[k]) for k in resolved if not same(eff[k], resolved[k])}
            e["write_unverified"] = bad
            print(f"    WRITE NOT VERIFIED: {bad}")
        for cam, st in e["cameras"].items():
            if st.get("gate_ok") and base_of(cam, "mean") is not None:
                st["d_mean_pct"] = round(100.0 * (st["mean"] - base_of(cam, "mean"))
                                         / max(1e-6, base_of(cam, "mean")), 2)
                st["d_dark_pp"] = round(st["dark_pct"] - base_of(cam, "dark_pct"), 2)
                st["d_p05"] = round(st["p05"] - base_of(cam, "p05"), 1)
        rows.append(e)
        flush()
        return e

    # ---- pass 1 ----
    moved = []
    for name, deltas, note, kind in variants:
        e = run(name, deltas, note, kind)
        c = e["cameras"].get(args.ref_cam, {})
        if kind == "bundle" and c.get("gate_ok") and (
                abs(c.get("d_mean_pct", 0.0)) >= args.expand_thresh_mean
                or abs(c.get("d_dark_pp", 0.0)) >= args.expand_thresh_dark):
            moved.append((name, deltas, note))

    # ---- pass 2: expand only the bundles that moved ----
    report["expanded"] = [m[0] for m in moved]
    if moved:
        print(f"\n[pass2] expanding {len(moved)} bundle(s) key by key: {[m[0] for m in moved]}")
    for bname, deltas, _note in moved:
        for k, v in sorted(deltas.items()):
            run(f"{bname}|{k.split('/')[-1]}", {k: v}, f"single key {k} from {bname}", "single")

    # restore everything before shutdown so a later reader of the log sees a clean stack
    for k in list(dirty):
        try:
            carb_set(cs, k, dirty.pop(k))
        except Exception:
            pass

    gate_fails = sum(1 for r in rows for c in r["cameras"].values() if not c.get("gate_ok"))
    report["gate_failures"] = gate_fails
    report["ok"] = gate_fails == 0
    flush()
    print(f"\n[done] {jpath}  rows={len(rows)} gate_failures={gate_fails}")
    return 0 if gate_fails == 0 else 4


if __name__ == "__main__":
    try:
        rc = main()
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        rc = 9
    print(f"PROBE_RC={rc}")
    sys.exit(rc)
