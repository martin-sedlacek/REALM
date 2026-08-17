#!/usr/bin/env python
"""Host-side (no container, no GPU): diff two exhaustive carb settings dumps.

Written because every ablation up to this point varied only keys OmniGibson *explicitly sets*.
Neither OG 1.1.1 nor OG 3.9.1 configures tonemapping, auto-exposure or colour correction -- both
mention nothing under `/rtx/post/` except `dlss/execMode` -- so both inherit KIT defaults, and Kit
differs between Isaac 4.x (OG 1.1.1) and Isaac 5.1 (OG 3.9.1). A default that changed between Isaac
versions would be invisible to any hand-written key list.

    .../behavior/bin/python scripts/debug_probes/carb_tree_diff.py \
        <dump_og111.json> <dump_og391.json>

Groups are ordered by how plausibly they affect exposure or tone, `/rtx/post/*` first, and the
EXPOSURE SUSPECTS section at the top pulls out anything matching tonemap / exposure / histogram /
gamma / white / colour-correction / film regardless of where it sits in the tree.
"""

import argparse
import json
import re

# Ordered: most plausibly exposure/tone-affecting first.
GROUPS = [
    ("/rtx/post/", "post-processing: tonemap, auto-exposure, DLSS, colour correction"),
    ("/rtx/sceneDb/", "scene database: ambient light term"),
    ("/rtx/directLighting/", "direct lighting"),
    ("/rtx/indirectDiffuse/", "indirect diffuse / GI"),
    ("/rtx/rtx/", "RTX mode selection (Real-Time 2.0)"),
    ("/rtx/pathtracing/", "path tracing"),
    ("/rtx/raytracing/", "ray tracing"),
    ("/rtx/reflections/", "reflections"),
    ("/rtx/ambientOcclusion/", "ambient occlusion"),
    ("/rtx/shadows/", "shadows"),
    ("/rtx/translucency/", "translucency"),
    ("/rtx/material", "material handling"),
    ("/rtx/", "other /rtx"),
    ("/app/renderer/", "renderer app settings"),
    ("/app/", "other /app"),
    ("/persistent/", "persistent settings"),
]

# Anything that could plausibly move exposure or tone, wherever it lives.
SUSPECT = re.compile(
    r"tonemap|exposure|histogram|gamma|whitepoint|white_point|whiteScale|whiteLevel|"
    r"colorcorrect|colorCorrection|colorGrad|colorTemp|filmIso|film|iso|shutter|fnumber|fNumber|"
    r"brightness|contrast|saturation|gain|luminance|ev100|aperture|toneCurve|srgb|sRGB",
    re.IGNORECASE)


def load(p):
    d = json.load(open(p))
    return d, d["settings"]


def fmt(v, w=26):
    s = repr(v)
    if len(s) > w:
        s = s[: w - 1] + "~"
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump_a", help="the 1.1.1 dump")
    ap.add_argument("dump_b", help="the og391 dump")
    ap.add_argument("--show-only-in-one", action="store_true",
                    help="also list keys present in only one tree (usually a large Kit-version list)")
    args = ap.parse_args()

    ma, a = load(args.dump_a)
    mb, b = load(args.dump_b)
    print(f"A = {ma['stack']:6s} {ma['label']:22s} {ma['n_keys']:6d} keys  via {ma['method']}")
    print(f"B = {mb['stack']:6s} {mb['label']:22s} {mb['n_keys']:6d} keys  via {mb['method']}")
    # Both dumps are gated: the probe refuses to write one unless every camera cleared the
    # blank-frame gate at dump time, so a dump can never describe a half-initialised pipeline.
    for m, nm in ((ma, "A"), (mb, "B")):
        g = m.get("gate", {})
        bad = [c for c, s in g.items() if not s.get("gate_ok")]
        verdict = "PASS" if g and not bad else f"FAIL/UNKNOWN {bad}"
        means = ", ".join(f"{c.split('.')[-1]}={s.get('mean')}" for c, s in sorted(g.items()))
        print(f"  {nm} frame gate: {verdict}  ({means})")

    keys_a, keys_b = set(a), set(b)
    both = keys_a & keys_b
    differ = sorted(k for k in both if repr(a[k]) != repr(b[k]))
    only_a = sorted(keys_a - keys_b)
    only_b = sorted(keys_b - keys_a)

    print(f"\nshared keys: {len(both)}   DIFFER: {len(differ)}   only in A: {len(only_a)}   "
          f"only in B: {len(only_b)}")

    # ---- exposure suspects first, wherever they live ----
    sus = [k for k in differ if SUSPECT.search(k)]
    print("\n" + "=" * 100)
    print(f"EXPOSURE / TONE SUSPECTS AMONG DIFFERING KEYS  ({len(sus)})")
    print("These are the ones that could plausibly produce a global brightness change.")
    print("=" * 100)
    if not sus:
        print("  NONE. No differing key matches tonemap/exposure/histogram/gamma/white-point/")
        print("  colour-correction/film/iso/shutter/aperture/contrast/saturation/gain.")
    else:
        print(f"{'key':64s} {'A (1.1.1)':>26s} {'B (og391)':>26s}")
        for k in sus:
            print(f"{k:64s} {fmt(a[k]):>26s} {fmt(b[k]):>26s}")

    sus_one = [k for k in only_a + only_b if SUSPECT.search(k)]
    if sus_one:
        print(f"\n  ...plus {len(sus_one)} exposure-ish keys present in only ONE tree:")
        for k in sus_one:
            side = "A only" if k in keys_a else "B only"
            v = a.get(k, b.get(k))
            print(f"    {side}  {k:60s} = {fmt(v)}")

    # ---- grouped full diff ----
    print("\n" + "=" * 100)
    print("FULL DIFF, GROUPED (order = plausibility of affecting exposure/tone)")
    print("=" * 100)
    claimed = set()
    for prefix, desc in GROUPS:
        ks = [k for k in differ if k.startswith(prefix) and k not in claimed]
        claimed.update(ks)
        if not ks:
            continue
        print(f"\n--- {prefix}*  ({len(ks)})  {desc}")
        print(f"    {'key':60s} {'A (1.1.1)':>26s} {'B (og391)':>26s}")
        for k in ks:
            print(f"    {k:60s} {fmt(a[k]):>26s} {fmt(b[k]):>26s}")
    rest = [k for k in differ if k not in claimed]
    if rest:
        print(f"\n--- everything else  ({len(rest)})")
        print(f"    {'key':60s} {'A (1.1.1)':>26s} {'B (og391)':>26s}")
        for k in rest:
            print(f"    {k:60s} {fmt(a[k]):>26s} {fmt(b[k]):>26s}")

    if args.show_only_in_one:
        for nm, ks, src in (("A (1.1.1) ONLY", only_a, a), ("B (og391) ONLY", only_b, b)):
            print(f"\n{'=' * 100}\n{nm}  ({len(ks)})\n{'=' * 100}")
            for k in ks:
                print(f"    {k:70s} = {fmt(src[k])}")


if __name__ == "__main__":
    main()
