#!/usr/bin/env python
"""Host-side (no container, no GPU): diff two light_inventory.py inventories.

The question this answers is narrow: does the scene's LIGHTING differ between OG 1.1.1 and og391 in
a way that could produce a uniform additive radiance floor of ~+67 luma? So the report leads with
the total radiometric weight per light type -- `sum(intensity * 2**exposure * luma(color))` -- and
only then goes key by key. A count that matches while the weight sum doubles is exactly the failure
mode a "102 lights in both" summary would hide.

Paths are matched exactly first. Whatever is left over is matched by (type, basename) so a scene
re-export that renamed `room_light_jnnhzm_0` to `rectangular_light_jnnhzm_0` still lines up instead
of showing as 102 removals and 102 additions.

    /home/sedlam56/miniconda3/envs/behavior/bin/python scripts/debug_probes/light_diff.py \
        <og111_inventory.json> <og391_inventory.json>
"""

import argparse
import json
import os
import re
from collections import defaultdict

ATTR_ORDER = ["intensity", "exposure", "color", "colorTemperature", "enableColorTemperature",
              "diffuse", "specular", "normalize", "radius", "width", "height", "length", "angle",
              "texture:file"]


def load(p):
    with open(p) as f:
        d = json.load(f)
    inv = d.get("inventory", d)
    return d.get("stack", os.path.basename(p)), inv


def val(rec, k):
    return rec.get("attrs", {}).get(k, {}).get("value")


def close(a, b, tol=1e-4):
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(close(x, y, tol) for x, y in zip(a, b))
    if isinstance(a, bool) or isinstance(b, bool):
        return bool(a) == bool(b)
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) <= tol * max(1.0, abs(float(a)), abs(float(b)))
    return a == b


def basekey(rec):
    """(type, model-ish basename) -- survives an instance-index or category rename."""
    leaf = rec["path"].rstrip("/").split("/")[-1]
    leaf = re.sub(r"_\d+$", "", leaf)
    return (rec["type"], leaf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref", help="OG 1.1.1 inventory JSON")
    ap.add_argument("cand", help="og391 inventory JSON")
    ap.add_argument("--max-rows", type=int, default=40)
    ap.add_argument("--emit-values", default=None,
                    help="write {og391_path: {attr: og111_value}} for every matched light, so "
                         "light_inventory.py --apply refvals can transplant 1.1.1's lighting onto "
                         "og391 wholesale. The matching lives HERE and not in the probe because it "
                         "needs the (type, basename) fallback: the re-export renamed nearly every "
                         "light prim, so an exact-path join finds 1 of 108.")
    ap.add_argument("--emit-attrs", default="intensity,exposure,color,normalize,diffuse,specular",
                    help="which attributes --emit-values carries")
    args = ap.parse_args()

    s_ref, ref = load(args.ref)
    s_cand, cand = load(args.cand)

    print(f"REF  = {s_ref:6s} {args.ref}")
    print(f"CAND = {s_cand:6s} {args.cand}\n")
    for nm, inv in ((s_ref, ref), (s_cand, cand)):
        print(f"{nm}: {inv['n_lights']} light prim(s) / {inv['n_prims']} prims  "
              f"[{inv.get('traverse', '?')}]")
    print()

    # ---- the headline: type totals ----
    types = sorted(set(ref["by_type"]) | set(cand["by_type"]))
    print(f"{'light type':18s} {s_ref+' n':>10s} {s_ref+' weight':>16s} "
          f"{s_cand+' n':>10s} {s_cand+' weight':>16s} {'weight ratio':>14s}")
    tot_r = tot_c = 0.0
    for t in types:
        r = ref["by_type"].get(t, {"count": 0, "weight_sum": 0.0})
        c = cand["by_type"].get(t, {"count": 0, "weight_sum": 0.0})
        tot_r += r["weight_sum"]
        tot_c += c["weight_sum"]
        rat = (c["weight_sum"] / r["weight_sum"]) if r["weight_sum"] else float("inf")
        print(f"{t:18s} {r['count']:10d} {r['weight_sum']:16.3f} {c['count']:10d} "
              f"{c['weight_sum']:16.3f} {rat:14.4f}")
    rat = (tot_c / tot_r) if tot_r else float("inf")
    print(f"{'TOTAL':18s} {ref['n_lights']:10d} {tot_r:16.3f} {cand['n_lights']:10d} "
          f"{tot_c:16.3f} {rat:14.4f}\n")

    # A DomeLight / environment light on one side only is the single most likely cause of a flat
    # additive floor, so it gets called out whether or not it moves the totals.
    for nm, inv in ((s_ref, ref), (s_cand, cand)):
        domes = [r for r in inv["lights"] if r["type"] in ("DomeLight", "PortalLight")]
        print(f"{nm}: {len(domes)} DomeLight/PortalLight prim(s)"
              + ("" if not domes else ":"))
        for r in domes:
            print(f"    {r['path']}  intensity={val(r,'intensity')} exposure={val(r,'exposure')} "
                  f"color={val(r,'color')} texture={val(r,'texture:file')} "
                  f"active={r['active']} vis={r.get('visibility')}")
        if inv.get("env_prims"):
            print(f"    env/render prims: {[e['path'] for e in inv['env_prims']][:10]}")
    print()

    # ---- per-light matching ----
    rmap = {r["path"]: r for r in ref["lights"]}
    cmap = {r["path"]: r for r in cand["lights"]}
    common = sorted(set(rmap) & set(cmap))
    only_r = sorted(set(rmap) - set(cmap))
    only_c = sorted(set(cmap) - set(rmap))
    print(f"matched by exact path: {len(common)};  {s_ref}-only {len(only_r)};  "
          f"{s_cand}-only {len(only_c)}")

    if only_r and only_c:
        rb, cb = defaultdict(list), defaultdict(list)
        for p in only_r:
            rb[basekey(rmap[p])].append(p)
        for p in only_c:
            cb[basekey(cmap[p])].append(p)
        shared = set(rb) & set(cb)
        n_pair = 0
        for k in shared:
            for a, b in zip(sorted(rb[k]), sorted(cb[k])):
                common.append((a, b))
                n_pair += 1
        print(f"matched by (type, basename) after stripping instance suffixes: {n_pair} more")
        left_r = [p for k in set(rb) - shared for p in rb[k]]
        left_c = [p for k in set(cb) - shared for p in cb[k]]
        if left_r:
            print(f"  unmatched {s_ref}-only ({len(left_r)}): {left_r[:12]}")
        if left_c:
            print(f"  unmatched {s_cand}-only ({len(left_c)}): {left_c[:12]}")
    print()

    # ---- attribute differences ----
    diffs = defaultdict(list)
    for item in common:
        pr, pc = (item, item) if isinstance(item, str) else item
        r, c = rmap[pr], cmap[pc]
        for k in ATTR_ORDER:
            a, b = val(r, k), val(c, k)
            if a is None and b is None:
                continue
            if not close(a, b):
                diffs[k].append((pr, pc, a, b))
    if not diffs:
        print("NO per-light attribute differences among matched lights.")
    for k, rows in sorted(diffs.items(), key=lambda t: -len(t[1])):
        print(f"--- {k}: {len(rows)} light(s) differ")
        for pr, pc, a, b in rows[:args.max_rows]:
            same_path = "" if pr == pc else f"  (vs {pc})"
            print(f"    {pr}{same_path}\n        {s_ref}={a!r}   {s_cand}={b!r}")
        if len(rows) > args.max_rows:
            print(f"    ... {len(rows) - args.max_rows} more")
    print()

    # ---- optional: the transplant table ----
    if args.emit_values:
        want = [a.strip() for a in args.emit_attrs.split(",") if a.strip()]
        out = {}
        for item in common:
            pr, pc = (item, item) if isinstance(item, str) else item
            r = rmap[pr]
            vals = {k: val(r, k) for k in want if val(r, k) is not None}
            if vals:
                out[pc] = {"from_ref_path": pr, "values": vals}
        with open(args.emit_values, "w") as f:
            json.dump({"ref": args.ref, "cand": args.cand, "n": len(out), "map": out},
                      f, indent=1, sort_keys=True)
        print(f"emitted {len(out)} light transplant(s) -> {args.emit_values}\n")

    # ---- the actionable summary ----
    print("=== what could produce a UNIFORM ADDITIVE FLOOR ===")
    notes = []
    if abs(rat - 1.0) > 0.02:
        notes.append(f"total radiometric weight ratio {s_cand}/{s_ref} = {rat:.4f}")
    nd_r = ref["by_type"].get("DomeLight", {}).get("count", 0)
    nd_c = cand["by_type"].get("DomeLight", {}).get("count", 0)
    if nd_r != nd_c:
        notes.append(f"DomeLight count differs: {s_ref}={nd_r} vs {s_cand}={nd_c}")
    if "exposure" in diffs:
        notes.append(f"{len(diffs['exposure'])} light(s) differ in EXPOSURE (a 2**x multiplier)")
    if "intensity" in diffs:
        notes.append(f"{len(diffs['intensity'])} light(s) differ in INTENSITY")
    if ref["n_lights"] != cand["n_lights"]:
        notes.append(f"light COUNT differs: {ref['n_lights']} vs {cand['n_lights']}")
    print("\n".join("  - " + n for n in notes) if notes
          else "  nothing: lights match in count, type, weight and every compared attribute.")


if __name__ == "__main__":
    main()
