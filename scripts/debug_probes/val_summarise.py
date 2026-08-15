"""Table the curl measurements out of several `*_curl.json` runs, paired build-vs-build.

Reads only what a run actually wrote. A press that did not complete has no record in `presses`, so
it appears here as `--` rather than as a number -- the point being that nothing in this table can be
a value someone expected rather than one a run produced.

    python scripts/debug_probes/val_summarise.py --root /logs/gripper_squeeze \
        --run xflat=val_xflat_a --run xflat_rep=val_xflat_b --run v2=val_v2_a
"""
import argparse
import json
import os

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="/logs/gripper_squeeze")
ap.add_argument("--run", action="append", default=[], metavar="LABEL=TAG",
                help="a run: LABEL is what to call it, TAG is the probe's --tag. Repeat.")
ap.add_argument("--out", default=None)
args = ap.parse_args()

runs = {}
for spec in args.run:
    lab, tag = spec.split("=", 1)
    p = os.path.join(args.root, f"{tag}_curl.json")
    if not os.path.exists(p):
        print(f"  *** MISSING {lab}: {p} -- reported as absent, never estimated ***")
        runs[lab] = None
        continue
    runs[lab] = json.load(open(p))
    n = len(runs[lab].get("presses", []))
    print(f"  loaded {lab:14s} tag={tag:18s} robot={runs[lab].get('robot')}  {n} presses")

KEYS = ("curl_in_deg", "d_pad_sep_mm", "d_tipg_sep_mm", "force_N")
CELLS = {}
for lab, d in runs.items():
    if not d:
        continue
    for e in d.get("presses", []):
        CELLS[(lab, e["rung"], e.get("finger", "?"))] = e

rungs = sorted({k[1] for k in CELLS})
fingers = sorted({k[2] for k in CELLS})
labels = list(runs)


def fmt(v, w=9, p=4):
    return f"{v:{w}.{p}f}" if isinstance(v, (int, float)) else f"{'--':>{w}}"


out = {}
for key, title, prec in (("curl_in_deg", "CURL (deg, + = INWARD)", 4),
                         ("d_pad_sep_mm", "d_pad_sep (mm, - = inward)", 4),
                         ("d_tipg_sep_mm", "d_tipg_sep (mm, - = inward)", 4),
                         ("force_N", "contact force (N)", 2)):
    print(f"\n{'=' * (30 + 12 * len(labels))}\n{title}\n{'=' * (30 + 12 * len(labels))}")
    head = f"{'rung':10s} {'finger':8s}" + "".join(f"{l:>12s}" for l in labels)
    print(head + "\n" + "-" * len(head))
    for rg in rungs:
        for fg in fingers:
            row = [CELLS.get((l, rg, fg), {}).get(key) for l in labels]
            if all(v is None for v in row):
                continue
            print(f"{rg:10s} {fg:8s}" + "".join(fmt(v, 12, prec) for v in row))
            out.setdefault(key, {})[f"{rg}/{fg}"] = dict(zip(labels, row))

# The direction verdict and the two-observable agreement, which is what makes a curl number stand.
print(f"\n{'=' * 96}\nPER-PRESS VERDICTS -- direction, and whether the two independent observables agree"
      f"\n{'=' * 96}")
print(f"{'run':14s} {'rung':10s} {'finger':8s} {'direction':16s} {'agree':7s} {'curl':>9s} {'F(N)':>8s}")
for l in labels:
    for rg in rungs:
        for fg in fingers:
            e = CELLS.get((l, rg, fg))
            if not e:
                continue
            print(f"{l:14s} {rg:10s} {fg:8s} {str(e.get('direction')):16s} "
                  f"{str(e.get('observables_agree')):7s} {fmt(e.get('curl_in_deg'), 9)} "
                  f"{fmt(e.get('force_N'), 8, 2)}")

# Replicate spread within each run: the measurement's OWN error bar, from identical rungs.
print(f"\n{'=' * 72}\nWITHIN-RUN REPLICATE SPREAD of curl (identical rungs) -- the error bar"
      f"\n{'=' * 72}")
for l in labels:
    for fg in fingers:
        vals = [CELLS[(l, rg, fg)]["curl_in_deg"] for rg in rungs
                if (l, rg, fg) in CELLS and CELLS[(l, rg, fg)].get("curl_in_deg") is not None]
        if len(vals) >= 2:
            print(f"  {l:14s} finger {fg:6s} n={len(vals)}  mean {sum(vals) / len(vals):+.4f} deg  "
                  f"spread {max(vals) - min(vals):.4f} deg")
            out.setdefault("spread", {})[f"{l}/{fg}"] = dict(
                n=len(vals), mean=sum(vals) / len(vals), spread=max(vals) - min(vals))

if args.out:
    with open(args.out, "w") as f:
        json.dump(dict(runs={k: (v or {}).get("robot") for k, v in runs.items()}, table=out), f,
                  indent=2)
    print(f"\n  wrote {args.out}")
print("\nVAL_SUMMARISE_COMPLETE")
