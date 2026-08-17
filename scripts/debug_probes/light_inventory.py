#!/usr/bin/env python
"""Enumerate every LIGHT PRIM on the USD stage, in either stack, and test light-side fixes.

WHY LIGHTS AND NOT SETTINGS
---------------------------
The og391-vs-1.1.1 render gap is ADDITIVE, not multiplicative: a per-pixel regression over
pixel-aligned frames gives `og391 ~= 1.0 * og111 + 67`, i.e. slope one and a flat +67 luma floor.
Subtracting that one constant reconstructs the 1.1.1 image. A uniform additive radiance floor is
what a light reaching every surface does -- a dome light, an environment/IBL, or a raised ambient --
and the whole renderer-settings family has been exonerated by an exhaustive carb diff (zero real
differences under indirectDiffuse / reflections / ambientOcclusion / shadows / rtx modes). The
strongest remaining suspect is therefore the SCENE's own lights, which were re-exported between the
two dataset versions.

WHAT IT DOES
------------
  1. Traverses the stage INCLUDING instance proxies -- OmniGibson scenes are heavily instanced, and
     a plain Traverse() silently misses every light inside an instanced object.
  2. For each light prim records path, type, and both the modern `inputs:*` and the pre-21.02
     un-namespaced spellings of intensity / exposure / color / colorTemperature /
     enableColorTemperature / diffuse / specular / radius / width / height / length / angle /
     normalize / texture:file, plus IsActive, computed visibility, and whether it is an instance
     proxy (proxies cannot be authored).
  3. Computes a radiometric weight per light, `intensity * 2**exposure * luma(color)`, and totals it
     per type -- the number that actually matters for "did the lighting get brighter".
  4. Also dumps stage-level RenderSettings / Environment prims, which is where a dome or IBL would
     hide if it is not a light prim at all.
  5. `--apply` can then push a candidate change at PRE-FIRST-RENDER timing and measure it, with the
     same gated stats the rest of this investigation uses.

The inventory is written whether or not a frame was produced, but every MEASURED row is gated: a
near-uniform frame is rejected rather than turned into a number.

    # inventory only, both stacks
    STACK=og111 PROBE=light_inventory.py ./scripts/debug_probes/run_brightness_ab.sh \
        --out /logs/lightprims --label og111_lights --pre-renders 300 --no-measure
    STACK=og391 PROBE=light_inventory.py ./scripts/debug_probes/run_brightness_ab.sh \
        --out /logs/lightprims --label og391_lights

    # test a candidate: push 1.1.1's light values onto og391's matching prims
    STACK=og391 PROBE=light_inventory.py ./scripts/debug_probes/run_brightness_ab.sh \
        --out /logs/lightprims --label og391_lights_ref \
        --ref-lights /logs/lightprims/lights_og111_lights.json --apply ref
"""

import argparse
import inspect
import json
import os
import sys
import traceback

import numpy as np

# Shared with the carb probe so both report identically comparable numbers.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lightpath_sweep import collect_rgb, frame_stats, jsonable, luma  # noqa: E402

LIGHT_TYPES = {
    "DomeLight", "DistantLight", "SphereLight", "RectLight", "DiskLight", "CylinderLight",
    "GeometryLight", "PortalLight", "PluginLight", "MeshLight", "VolumeLight", "LightFilter",
}

# (canonical name, [spellings to try]). USD renamed light inputs to the `inputs:` namespace in
# 21.02; OG 1.1.1 runs Isaac 4.x and og391 runs Isaac 5.1, so both spellings must be probed or a
# whole stack reads back as "no attributes".
LIGHT_ATTRS = [
    ("intensity", ["inputs:intensity", "intensity"]),
    ("exposure", ["inputs:exposure", "exposure"]),
    ("color", ["inputs:color", "color"]),
    ("colorTemperature", ["inputs:colorTemperature", "colorTemperature"]),
    ("enableColorTemperature", ["inputs:enableColorTemperature", "enableColorTemperature"]),
    ("diffuse", ["inputs:diffuse", "diffuse"]),
    ("specular", ["inputs:specular", "specular"]),
    ("normalize", ["inputs:normalize", "normalize"]),
    ("radius", ["inputs:radius", "radius"]),
    ("width", ["inputs:width", "width"]),
    ("height", ["inputs:height", "height"]),
    ("length", ["inputs:length", "length"]),
    ("angle", ["inputs:angle", "angle"]),
    ("shaping:cone:angle", ["inputs:shaping:cone:angle", "shaping:cone:angle"]),
    ("texture:file", ["inputs:texture:file", "texture:file"]),
    ("texture:format", ["inputs:texture:format", "texture:format"]),
]

# The attributes a "make og391 look like 1.1.1" test is allowed to write. Geometry (radius/width)
# is deliberately excluded: changing a light's SIZE changes its shadow softness too, which would
# confound the additive-floor question this is meant to answer.
WRITABLE = ("intensity", "exposure", "color", "colorTemperature", "enableColorTemperature",
            "diffuse", "specular", "normalize")


def close_enough(a, b, tol=1e-4):
    """Float-tolerant equality, used to decide whether a written value survived env.reset()."""
    try:
        if isinstance(a, bool) or isinstance(b, bool):
            return bool(a) == bool(b)
        if hasattr(a, "__len__") and hasattr(b, "__len__"):
            return len(a) == len(b) and all(close_enough(x, y, tol) for x, y in zip(a, b))
        return abs(float(a) - float(b)) <= tol * max(1.0, abs(float(a)), abs(float(b)))
    except Exception:
        return a == b


def rgb_luma(c):
    if isinstance(c, (list, tuple)) and len(c) >= 3:
        return 0.299 * float(c[0]) + 0.587 * float(c[1]) + 0.114 * float(c[2])
    return 1.0


def radiometric_weight(rec):
    """intensity * 2**exposure * luma(color) -- a single comparable number per light."""
    i = rec["attrs"].get("intensity", {}).get("value")
    if i is None:
        return None
    e = rec["attrs"].get("exposure", {}).get("value") or 0.0
    c = rec["attrs"].get("color", {}).get("value")
    try:
        return float(i) * (2.0 ** float(e)) * rgb_luma(c)
    except Exception:
        return None


def inventory(lazy, stage):
    """Every light prim on the stage, instance proxies included."""
    Usd = lazy.pxr.Usd
    try:
        UsdGeom = lazy.pxr.UsdGeom
    except Exception:
        UsdGeom = None

    # Instance proxies are OFF by default in Traverse(); OmniGibson instances almost every object,
    # so without this predicate the scene's lights are largely invisible to the walk.
    try:
        pred = Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate)
        it = stage.Traverse(pred)
        how = "Traverse(TraverseInstanceProxies(PrimAllPrimsPredicate))"
    except Exception as e:
        it = stage.Traverse()
        how = f"Traverse() -- instance proxies UNAVAILABLE ({type(e).__name__}: {e})"

    lights, others, n_prims, type_counts = [], [], 0, {}
    for prim in it:
        n_prims += 1
        tn = str(prim.GetTypeName())
        type_counts[tn] = type_counts.get(tn, 0) + 1
        is_light = tn in LIGHT_TYPES or tn.endswith("Light")
        if not is_light:
            try:
                if lazy.pxr.UsdLux.LightAPI(prim):
                    is_light = True
            except Exception:
                pass
        path = str(prim.GetPath())
        if not is_light:
            if tn in ("RenderSettings", "RenderProduct", "RenderVar") or path.rstrip("/").endswith(
                    ("/Environment", "/environment", "/Render", "/Lights", "/lights")):
                others.append({"path": path, "type": tn, "active": bool(prim.IsActive())})
            continue

        rec = {"path": path, "type": tn, "active": bool(prim.IsActive()),
               "instance_proxy": bool(prim.IsInstanceProxy()), "attrs": {}}
        for canon, names in LIGHT_ATTRS:
            for n in names:
                a = prim.GetAttribute(n)
                if a and a.IsValid():
                    try:
                        v = a.Get()
                    except Exception as e:
                        rec["attrs"][canon] = {"spelling": n, "error": f"{type(e).__name__}"}
                        break
                    if v is None:
                        continue
                    if hasattr(v, "__len__") and not isinstance(v, str):
                        try:
                            v = [float(x) for x in v]
                        except Exception:
                            v = str(v)
                    rec["attrs"][canon] = {"spelling": n, "value": jsonable(v),
                                           "authored": bool(a.HasAuthoredValue())}
                    break
        if UsdGeom is not None:
            try:
                rec["visibility"] = str(UsdGeom.Imageable(prim).ComputeVisibility())
            except Exception:
                rec["visibility"] = "<n/a>"
        rec["weight"] = radiometric_weight(rec)
        lights.append(rec)

    lights.sort(key=lambda r: r["path"])
    by_type = {}
    for r in lights:
        b = by_type.setdefault(r["type"], {"count": 0, "weight_sum": 0.0, "weight_known": 0})
        b["count"] += 1
        if r["weight"] is not None:
            b["weight_sum"] += r["weight"]
            b["weight_known"] += 1
    for b in by_type.values():
        b["weight_sum"] = round(b["weight_sum"], 4)
    return {"traverse": how, "n_prims": n_prims, "n_lights": len(lights),
            "by_type": by_type, "lights": lights, "env_prims": others,
            "prim_type_counts": dict(sorted(type_counts.items(), key=lambda t: -t[1])[:40])}


def set_attr(prim, canon, value, lazy, edit_ctx=None):
    """Write one light attribute, preferring the spelling that already exists on the prim.

    `edit_ctx` is a zero-arg factory returning the simulator's USD-editing context. OG 3.9.1 guards
    the stage and raises `RuntimeError: USD edit detected outside of og.sim.editing_usd() context!`
    -- but it raises at the NEXT simulator operation, not at the write, so an unwrapped edit reads
    back as applied and then kills env.reset() several hundred frames later. Every write goes
    through here so the wrapping cannot be forgotten at one call site.
    """
    import contextlib
    Gf = lazy.pxr.Gf
    for n in dict(LIGHT_ATTRS)[canon]:
        a = prim.GetAttribute(n)
        if a and a.IsValid():
            v = value
            if isinstance(value, (list, tuple)) and len(value) == 3:
                v = Gf.Vec3f(float(value[0]), float(value[1]), float(value[2]))
            with (edit_ctx() if edit_ctx else contextlib.nullcontext()):
                a.Set(v)
            return n
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--task-id", type=int, default=3)
    ap.add_argument("--pert-id", type=int, default=0)
    ap.add_argument("--robot", default="DROID")
    ap.add_argument("--rendering-mode", default="rt")
    ap.add_argument("--ref-lights", default=None, help="the other stack's inventory JSON")
    ap.add_argument("--ref-values", default=None,
                    help="light_diff.py --emit-values output: {og391_path: {values: {...}}}. "
                         "Required by --apply refvals.")
    ap.add_argument("--apply", default=None,
                    help="ref | dome_off | scale:<f> | exposure_zero | dome_intensity:<f>")
    ap.add_argument("--frames", type=int, default=5)
    ap.add_argument("--pre-renders", type=int, default=300)
    ap.add_argument("--settle-renders", type=int, default=60)
    ap.add_argument("--no-measure", action="store_true", help="inventory only, no frames")
    ap.add_argument("--og-lite", action="store_true")
    ap.add_argument("--gate-min-colors", type=int, default=2000)
    ap.add_argument("--gate-max-dominant", type=float, default=0.50)
    args = ap.parse_args()

    assert not args.out.startswith("/tmp"), "artifacts go on Lustre"
    os.makedirs(args.out, exist_ok=True)
    report = {"label": args.label, "argv": sys.argv}
    jpath = os.path.join(args.out, f"{args.label}.json")

    def flush():
        with open(jpath, "w") as f:
            json.dump(report, f, indent=1, sort_keys=False, default=str)

    sys.path.insert(0, "/app")
    import omnigibson as og
    from omnigibson.macros import gm  # noqa: F401
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
                          "omnigibson_version": getattr(og, "__version__", None)}
    print(f"[identity] stack={stack} v={getattr(og, '__version__', None)}")
    flush()

    if is_111:
        set_sim_config(rendering_mode=args.rendering_mode, robot=args.robot, og_lite=args.og_lite)
    else:
        set_sim_config(robot=args.robot)

    task = SUPPORTED_TASKS[args.task_id]
    pert = SUPPORTED_PERTURBATIONS[args.pert_id]
    report["scene"] = {"task": task, "perturbation": pert, "robot": args.robot,
                       "rendering_mode": args.rendering_mode}
    print(f"[scene] {task} / {pert} / {args.robot} / {args.rendering_mode}")

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

    # ---- the stage ----
    stage = None
    for how, get in (("og.sim.stage", lambda: og.sim.stage),
                     ("omni.usd context", lambda: lazy.omni.usd.get_context().get_stage())):
        try:
            stage = get()
            if stage is not None:
                report["stage_via"] = how
                break
        except Exception:
            continue
    if stage is None:
        report["error"] = "could not obtain a USD stage"
        print("[stage] FAILED to obtain")
        flush()
        return 5
    try:
        report["stage_root_layer"] = str(stage.GetRootLayer().identifier)
    except Exception:
        pass
    print(f"[stage] via {report['stage_via']}")

    inv = inventory(lazy, stage)
    report["inventory"] = inv
    lp = os.path.join(args.out, f"lights_{args.label}.json")
    with open(lp, "w") as f:
        json.dump({"stack": stack, "label": args.label, "scene": report["scene"],
                   "inventory": inv}, f, indent=1, sort_keys=True, default=str)
    report["inventory_path"] = lp
    print(f"[lights] {inv['n_lights']} light prim(s) among {inv['n_prims']} prims "
          f"({inv['traverse']})")
    for t, b in sorted(inv["by_type"].items()):
        print(f"   {t:16s} n={b['count']:4d}  sum(intensity*2^exposure*luma(color))={b['weight_sum']:14.3f}"
              f"  ({b['weight_known']} with a readable intensity)")
    if inv["env_prims"]:
        print(f"   env/render prims: {[e['path'] for e in inv['env_prims']][:12]}")
    flush()

    # OG 3.9.1 guards the stage: any USD authoring outside this context raises
    # `RuntimeError: USD edit detected outside of og.sim.editing_usd() context!` at the NEXT
    # simulator operation, not at the write -- so an unwrapped light edit looks like it worked and
    # then kills env.reset(). OG 1.1.1 has no such guard, hence the nullcontext fallback.
    import contextlib

    def usd_edit():
        ctx = getattr(og.sim, "editing_usd", None)
        if ctx is None:
            return contextlib.nullcontext()
        try:
            return ctx()
        except Exception:
            return contextlib.nullcontext()

    # ---- optional runtime change, applied BEFORE the first render ----
    applied = []
    if args.apply:
        refvals = {}
        if args.ref_values:
            with open(args.ref_values) as f:
                refvals = json.load(f)["map"]
            print(f"[ref-values] {len(refvals)} transplant target(s) from {args.ref_values}")
        if args.apply == "refvals" and not refvals:
            raise SystemExit("--apply refvals needs --ref-values from light_diff.py --emit-values")
        ref_by_path = {}
        if args.ref_lights:
            with open(args.ref_lights) as f:
                rl = json.load(f)
            ref_by_path = {r["path"]: r for r in rl["inventory"]["lights"]}
            print(f"[ref-lights] {len(ref_by_path)} light(s) from {args.ref_lights}")

        def targets():
            for r in inv["lights"]:
                prim = stage.GetPrimAtPath(r["path"])
                if not prim or not prim.IsValid():
                    continue
                if r.get("instance_proxy"):
                    # Instance proxies are read-only in USD; authoring on one raises. Record the
                    # skip so a "no effect" result can never be confused with "no attempt".
                    yield r, prim, "SKIP: instance proxy (not authorable)"
                    continue
                yield r, prim, None

        mode = args.apply
        for r, prim, skip in targets():
            if skip:
                applied.append({"path": r["path"], "skipped": skip})
                continue
            try:
                if mode == "ref":
                    ref = ref_by_path.get(r["path"])
                    if ref is None:
                        applied.append({"path": r["path"], "skipped": "no matching path on ref stack"})
                        continue
                    wrote = {}
                    for canon in WRITABLE:
                        rv = ref["attrs"].get(canon, {}).get("value")
                        if rv is None:
                            continue
                        cur = r["attrs"].get(canon, {}).get("value")
                        if cur is not None and jsonable(cur) == jsonable(rv):
                            continue
                        n = set_attr(prim, canon, rv, lazy, usd_edit)
                        if n:
                            wrote[canon] = {"from": jsonable(cur), "to": jsonable(rv), "spelling": n}
                    if wrote:
                        applied.append({"path": r["path"], "wrote": wrote})
                elif mode == "dome_off":
                    if r["type"] == "DomeLight":
                        set_attr(prim, "intensity", 0.0, lazy, usd_edit)
                        applied.append({"path": r["path"], "wrote": {"intensity": {"to": 0.0}}})
                elif mode.startswith("dome_intensity:"):
                    if r["type"] == "DomeLight":
                        v = float(mode.split(":", 1)[1])
                        set_attr(prim, "intensity", v, lazy, usd_edit)
                        applied.append({"path": r["path"], "wrote": {"intensity": {"to": v}}})
                elif mode.startswith("scale:"):
                    s = float(mode.split(":", 1)[1])
                    cur = r["attrs"].get("intensity", {}).get("value")
                    if cur is not None:
                        set_attr(prim, "intensity", float(cur) * s, lazy, usd_edit)
                        applied.append({"path": r["path"],
                                        "wrote": {"intensity": {"from": cur, "to": float(cur) * s}}})
                elif mode == "refvals":
                    # The full 1.1.1 lighting, transplanted. The path map comes from
                    # light_diff.py --emit-values because the re-export renamed nearly every light
                    # prim (`room_light_widhrs_0` -> `downlight_hpkvem_0`), so only 1 of 108 lights
                    # joins on an exact path and the rest need the (type, basename) fallback.
                    ent = refvals.get(r["path"])
                    if ent is None:
                        applied.append({"path": r["path"], "skipped": "not in --ref-values map"})
                        continue
                    wrote = {}
                    for canon, rv in ent["values"].items():
                        cur = r["attrs"].get(canon, {}).get("value")
                        if cur is not None and close_enough(cur, rv):
                            continue
                        n = set_attr(prim, canon, rv, lazy, usd_edit)
                        if n:
                            wrote[canon] = {"from": jsonable(cur), "to": rv, "spelling": n}
                    if wrote:
                        applied.append({"path": r["path"], "wrote": wrote,
                                        "from_ref_path": ent["from_ref_path"]})
                elif mode.startswith("attr:"):
                    # attr:<canon>=<value> on EVERY light that has that attribute. The motivating
                    # case is `attr:normalize=False`: 62 of the 108 lights flipped `normalize`
                    # False -> True between the two exports, which changes what `intensity` MEANS
                    # (normalize divides emission by the light's area), and is the obvious suspect
                    # for og391 rendering brighter while its nominal intensity total is 15x lower.
                    canon, _, raw = mode[len("attr:"):].partition("=")
                    if raw.lower() in ("true", "false"):
                        v = raw.lower() == "true"
                    else:
                        try:
                            v = float(raw)
                        except ValueError:
                            v = raw
                    cur = r["attrs"].get(canon, {}).get("value")
                    if cur is not None and not close_enough(cur, v):
                        n = set_attr(prim, canon, v, lazy, usd_edit)
                        if n:
                            applied.append({"path": r["path"],
                                            "wrote": {canon: {"from": jsonable(cur), "to": v}}})
                elif mode == "exposure_zero":
                    cur = r["attrs"].get("exposure", {}).get("value")
                    if cur not in (None, 0.0):
                        set_attr(prim, "exposure", 0.0, lazy, usd_edit)
                        applied.append({"path": r["path"],
                                        "wrote": {"exposure": {"from": cur, "to": 0.0}}})
                else:
                    raise SystemExit(f"--apply: unknown mode '{mode}'")
            except Exception as e:
                applied.append({"path": r["path"], "error": f"{type(e).__name__}: {e}"})

        n_wrote = sum(1 for a in applied if "wrote" in a)
        n_skip = sum(1 for a in applied if "skipped" in a)
        n_err = sum(1 for a in applied if "error" in a)
        report["apply"] = {"mode": mode, "n_wrote": n_wrote, "n_skipped": n_skip,
                           "n_error": n_err, "detail": applied[:400]}
        print(f"[apply] {mode}: wrote {n_wrote}, skipped {n_skip}, errors {n_err}")
        for a in applied[:10]:
            print(f"   {a}")
        # Prove it took: re-inventory and report the totals before/after.
        inv2 = inventory(lazy, stage)
        report["inventory_after_apply"] = {"by_type": inv2["by_type"], "n_lights": inv2["n_lights"]}
        print("[apply] weight totals after:")
        for t, b in sorted(inv2["by_type"].items()):
            was = inv["by_type"].get(t, {}).get("weight_sum")
            print(f"   {t:16s} n={b['count']:4d}  weight {was} -> {b['weight_sum']}")
        flush()

    if args.no_measure:
        report["ok"] = True
        flush()
        print(f"\n[done] inventory only -> {jpath}")
        return 0

    # ---- measure ----
    obs, _ = env.reset()

    # env.reset() -> scene.restore(initial_file), which removes and re-adds objects. Re-assert the
    # light edits afterwards and RE-READ them, so the row records what was in force when the frames
    # were taken rather than what was requested several hundred frames earlier.
    if args.apply:
        reasserted, in_force = [], {}
        for a in applied:
            if "wrote" not in a:
                continue
            prim = stage.GetPrimAtPath(a["path"])
            if not prim or not prim.IsValid():
                in_force[a["path"]] = "<prim gone after reset>"
                continue
            for canon, w in a["wrote"].items():
                want = w["to"]
                cur = None
                for n in dict(LIGHT_ATTRS)[canon]:
                    at = prim.GetAttribute(n)
                    if at and at.IsValid():
                        cur = at.Get()
                        break
                if cur is None or not close_enough(cur, want):
                    set_attr(prim, canon, want, lazy, usd_edit)
                    reasserted.append({"path": a["path"], canon: {"was": jsonable(cur),
                                                                  "re-set to": want}})
                    for n in dict(LIGHT_ATTRS)[canon]:
                        at = prim.GetAttribute(n)
                        if at and at.IsValid():
                            cur = at.Get()
                            break
                in_force.setdefault(a["path"], {})[canon] = jsonable(cur)
        report["apply_in_force_after_reset"] = {"reasserted": reasserted, "values": in_force}
        print(f"[apply] after env.reset(): {len(reasserted)} value(s) had to be re-asserted; "
              f"in force now: {in_force}")
        flush()

    _flip = args.og_lite and hasattr(og.sim, "_render_on_step")
    if _flip:
        og.sim._render_on_step = True
    obs, _r, _t, _tr, _i = env.warmup(obs)
    if _flip:
        og.sim._render_on_step = False
    hold = np.concatenate((np.asarray(env.reset_qpos)[:7], np.atleast_1d(-1.0)))

    def read_obs():
        nonlocal obs
        obs, _a, _b, _c, _d = env.step(hold)
        return collect_rgb(obs)

    trace = []
    step = max(25, args.pre_renders // 4)
    done = 0
    while done < args.pre_renders:
        k = min(step, args.pre_renders - done)
        for _ in range(k):
            og.sim.render()
        done += k
        got = read_obs()
        if got:
            cam = "external.external_sensor1" if "external.external_sensor1" in got else sorted(got)[0]
            m = round(float(luma(got[cam]).mean()), 2)
            d = "" if not trace else f" d={m - trace[-1]['mean']:+.2f}"
            trace.append({"renders": done, "mean": m})
            print(f"    [settle {done}/{args.pre_renders}] {cam} mean={m}{d}")
    report["material_settle"] = trace

    cams = sorted(collect_rgb(obs).keys())
    seq = [read_obs() for _ in range(args.frames)]
    row = {"variant": args.apply or "baseline", "cameras": {}}
    for cam in cams:
        imgs = [s[cam] for s in seq if cam in s]
        if not imgs:
            row["cameras"][cam] = {"gate_ok": False, "gate_fail": ["camera absent"]}
            continue
        med = np.median(np.stack(imgs, 0), axis=0).astype(np.uint8)
        st = frame_stats(med, args.gate_min_colors, args.gate_max_dominant)
        png = os.path.join(args.out, f"{args.label}__{row['variant']}__{cam.replace('.', '-')}.png")
        try:
            from PIL import Image
            Image.fromarray(med).save(png)
            st["png"] = png
        except Exception as e:
            st["png_error"] = f"{type(e).__name__}: {e}"
        row["cameras"][cam] = st
        flag = "" if st["gate_ok"] else "  !!GATE-FAIL " + "; ".join(st["gate_fail"])
        print(f"  {row['variant']:22s} {cam:44s} mean={st['mean']:7.2f} p5={st['p05']:6.1f} "
              f"p50={st['p50']:6.1f} p95={st['p95']:6.1f} sat={st['sat_pct']:6.3f}% "
              f"dark={st['dark_pct']:6.2f}% detail={st['detail']:7.1f}{flag}")
    report["rows"] = [row]
    gate_fails = sum(1 for c in row["cameras"].values() if not c.get("gate_ok"))
    report["gate_failures"] = gate_fails
    report["ok"] = gate_fails == 0
    flush()
    print(f"\n[done] {jpath} gate_failures={gate_fails}")
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
