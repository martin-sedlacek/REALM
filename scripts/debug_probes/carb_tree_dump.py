"""Dump the ENTIRE carb settings tree after env creation, so the two stacks can be diffed.

Every renderer test before this one only varied keys OmniGibson explicitly sets. But neither OG 1.1.1
nor 3.9.1 sets anything under `/rtx/post/` except `dlss/execMode` -- no tonemap, no auto-exposure, no
colour correction. Those therefore come from Kit defaults, and the Kit versions differ (Isaac 4.x
under OG 1.1.1 vs 5.1 under 3.9.1). A default that changed between Isaac versions is invisible to
every ablation run so far, and the tone curve is exactly what would have to change to turn og391's
render into the histogram-matched one.

So: enumerate rather than guess. Runs in either container -- the caller picks the stack.

    STACK=og391 PROBE=carb_tree_dump.py ./scripts/debug_probes/run_brightness_ab.sh \
        --label og391_tree --out /logs/render_bright_ab
    STACK=og111 PROBE=carb_tree_dump.py ./scripts/debug_probes/run_brightness_ab.sh \
        --label og111_tree --out /logs/render_bright_ab --pre-renders 300

Output is the payload shape `carb_tree_diff.py` reads, so the two dumps diff directly.

--------------------------------------------------------------------------------------------------
WHY THE WALK LOOKS LIKE THIS (2026-08-17)

The first version of this file walked the tree with `carb.dictionary`'s child-index API --
`get_item_child_count()` / `get_item_child_by_index()` -- and HARD-CRASHED in og391: the traceback
tail showed "no Python frame" after the extension-module subtree, i.e. it died inside the C layer.
It had a `try/except` fallback and the fallback never ran, because **a segfault in a C extension is
not a Python exception and no `except` can catch it.** Wrapping that recursion more carefully would
not have helped; the API had to stop being the primary route.

So route 1 is now `carb.dictionary.get_dictionary().get_dict_copy()`, ONE call into C that returns a
nested Python dict, measured working in BOTH stacks on the same scene (4964 leaves on og111, 5469 on
og391). Route 2 is `settings.get(prefix)`, also nested-dict-returning and also not the child-index
API. The crashing walk is gone rather than demoted -- there is no configuration in which it is the
best available option.
--------------------------------------------------------------------------------------------------
"""
import argparse
import json
import os

import numpy as np

# Prefixes route 2 probes when the whole-tree copy fails. Not exhaustive by design: it exists to
# return SOMETHING renderer-related rather than to be complete.
FALLBACK_PREFIXES = ("/rtx", "/rtx-defaults", "/rtx-flags", "/rtx-transient", "/app", "/persistent",
                     "/exts", "/physics", "/renderer", "/omni", "/isaaclab")


def _flatten(node, prefix, out):
    if isinstance(node, dict):
        for k, v in node.items():
            _flatten(v, f"{prefix}/{k}", out)
    elif isinstance(node, (list, tuple)):
        # Arrays are recorded whole; per-index keys would explode the diff for no gain.
        out[prefix] = list(node)
    else:
        out[prefix] = node


def walk(settings):
    """Every leaf as {path: value}, plus how it was obtained. Never raises."""
    # Route 1: one C call, a nested dict back. Proven on both stacks.
    try:
        import carb.dictionary
        di = carb.dictionary.get_dictionary()
        flat = {}
        _flatten(di.get_dict_copy(settings.get_settings_dictionary("/")), "", flat)
        if flat:
            return flat, "carb.dictionary.get_dict_copy", None
        route1 = "empty result"
    except Exception as e:
        route1 = f"{type(e).__name__}: {e}"

    # Route 2: settings.get() on each prefix, which also returns nested dicts. Still not the
    # child-index API that crashed og391.
    flat = {}
    errs = []
    for pref in FALLBACK_PREFIXES:
        try:
            v = settings.get(pref)
            if v is not None:
                _flatten(v, pref, flat)
        except Exception as e:
            errs.append(f"{pref}: {type(e).__name__}")
    method = f"settings.get() over {len(FALLBACK_PREFIXES)} prefixes (route1: {route1})"
    return flat, method, "; ".join(errs) or None


def frame_gate(obs, min_colors=2000, max_dominant=0.50):
    """A settings dump is only meaningful if the renderer was producing a REAL frame when it was
    taken. The 1.1.1 stack has returned 87%-pure-white buffers here, and a dump captured alongside
    one of those describes a half-initialised pipeline while looking perfectly well-formed."""
    frames = {}

    def rec(node, path):
        if isinstance(node, dict):
            for k, v in node.items():
                if k == "rgb":
                    a = v.cpu().numpy() if hasattr(v, "cpu") else v
                    a = np.asarray(a)
                    if a.ndim == 3 and a.shape[-1] >= 3:
                        frames[path] = a[..., :3].astype(np.uint8)
                else:
                    rec(v, f"{path}.{k}" if path else str(k))

    rec(obs, "")
    gate = {}
    for cam, a in frames.items():
        lum = 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]
        f = a.reshape(-1, 3)
        packed = (f[:, 0].astype(np.uint32) << 16) | (f[:, 1].astype(np.uint32) << 8) | f[:, 2]
        vals, counts = np.unique(packed, return_counts=True)
        dom = float(counts.max()) / float(packed.size)
        fails = []
        if vals.size < min_colors:
            fails.append(f"only {int(vals.size)} unique colours (< {min_colors})")
        if dom > max_dominant:
            fails.append(f"single colour is {dom:.4%} of pixels (> {max_dominant:.0%})")
        gate[cam] = {"mean": round(float(lum.mean()), 3), "n_colors": int(vals.size),
                     "dominant_frac": round(dom, 5), "gate_ok": not fails, "gate_fail": fails}
    return gate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="output dir (Lustre, not /tmp)")
    ap.add_argument("--label", required=True)
    ap.add_argument("--task-id", type=int, default=3, help="3 = rotate_mug")
    ap.add_argument("--pert-id", type=int, default=0, help="0 = Default -- randomises nothing")
    ap.add_argument("--robot", default="DROID")
    ap.add_argument("--rendering-mode", default="rt")
    ap.add_argument("--pre-renders", type=int, default=0,
                    help="1.1.1 needs ~300 render ticks for material streaming before it is "
                         "settled; og391 does not.")
    ap.add_argument("--allow-gate-fail", action="store_true",
                    help="write the dump even if no camera cleared the blank-frame gate. Off by "
                         "default: an ungated dump is not comparable and has already produced one "
                         "confident verdict read off an 85%%-white frame.")
    args = ap.parse_args()

    assert not args.out.startswith("/tmp"), "/tmp is node-local and wiped -- artifacts go on Lustre"

    import omnigibson as og
    from omnigibson.macros import gm  # noqa: F401 -- imported for its module-level side effects

    # Build the env the same way the brightness probe does, via whichever entry point this tree has.
    try:
        from realm.sim_config import set_sim_config
        set_sim_config(robot=args.robot)
        stack, set_rendering_mode = "og391", None
    except ImportError:
        from realm.eval import set_sim_config          # 1.1.1 keeps it in eval.py
        from realm.eval import set_rendering_mode
        set_sim_config(rendering_mode=args.rendering_mode, robot=args.robot)
        stack = "og111"

    from realm.eval import SUPPORTED_PERTURBATIONS, SUPPORTED_TASKS
    from realm.environments.env_dynamic import RealmEnvironmentDynamic

    task = SUPPORTED_TASKS[args.task_id]
    env = RealmEnvironmentDynamic(
        config_path="/app/realm/config",
        task_cfg_path=f"REALM_DROID10/{task}/default.yaml",
        perturbations=[SUPPORTED_PERTURBATIONS[args.pert_id]],
        robot=args.robot, multi_view=True, no_rendering=False,
        rendering_mode=args.rendering_mode)
    if set_rendering_mode is not None:
        set_rendering_mode(args.rendering_mode)

    import omnigibson.lazy as lazy
    settings = lazy.carb.settings.get_settings()

    # Reach a real frame before dumping: the tree has to describe a renderer that was actually
    # rendering, and on 1.1.1 material streaming needs a few hundred ticks to finish.
    obs, _ = env.reset()
    obs, _r, _t, _tr, _i = env.warmup(obs)
    hold = np.concatenate((np.asarray(env.reset_qpos)[:7], np.atleast_1d(-1.0)))
    prev = None
    for i in range(0, max(args.pre_renders, 1), 25):
        for _ in range(25 if args.pre_renders else 1):
            og.sim.render()
        obs, _r, _t, _tr, _i = env.step(hold)
        g = frame_gate(obs)
        if g:
            cam = sorted(g)[0]
            m = g[cam]["mean"]
            print(f"  [pre-render {i + 25}/{args.pre_renders}] {cam} mean={m:.2f}"
                  + ("" if prev is None else f"  d={m - prev:+.3f}"))
            prev = m

    gate = frame_gate(obs)
    ok = bool(gate) and all(g["gate_ok"] for g in gate.values())
    for cam, g in sorted(gate.items()):
        print(f"  [gate] {cam:46s} mean={g['mean']:7.2f} colours={g['n_colors']:6d} "
              f"dominant={g['dominant_frac']:.4f} {'ok' if g['gate_ok'] else g['gate_fail']}")

    flat, method, err = walk(settings)
    payload = {
        "stack": stack,
        "label": args.label,
        "method": method,
        "error": err,
        "n_keys": len(flat),
        "gate": gate,
        "identity": {"omnigibson_version": getattr(og, "__version__", "?"),
                     "omnigibson_file": getattr(og, "__file__", "?")},
        "scene": {"task": task, "perturbation": SUPPORTED_PERTURBATIONS[args.pert_id],
                  "robot": args.robot, "rendering_mode": args.rendering_mode},
        "args": vars(args),
        "settings": {k: (v if isinstance(v, (int, float, bool, str, list, type(None))) else str(v))
                     for k, v in flat.items()},
    }

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"carbtree_{args.label}.json")
    if not ok and not args.allow_gate_fail:
        payload["settings"] = {}
        payload["refused"] = "frame gate FAILED -- settings withheld (pass --allow-gate-fail to keep)"
        print(f"[carbtree] REFUSING to write {len(flat)} leaves: frame gate failed")
    with open(path, "w") as f:
        json.dump(payload, f, indent=1, sort_keys=True, default=str)
    print(f"[carbtree] {args.label}: {len(flat)} leaves via {method} -> {path}")
    print(f"[carbtree] /rtx/post keys: {sum(1 for k in flat if k.startswith('/rtx/post'))}")

    og.shutdown()
    return 0 if ok else 4


if __name__ == "__main__":
    # Isaac exits 0 even on an unhandled exception, so the rc has to be printed to be trusted.
    import sys
    import traceback
    try:
        rc = main()
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        rc = 9
    print(f"PROBE_RC={rc}")
    sys.exit(rc)
