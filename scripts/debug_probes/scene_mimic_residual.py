"""Recompute RoboLab's mimic residual table from `wrapdiff_robolab_squeeze.py`'s JSON, correctly.

Two reasons this is a script and not a one-liner.

1. **The probe's own `res` field has the sign wrong.** PhysxMimicJointAPI's documented constraint is

       jointPosition + (gearing * referenceJointPosition) + offset = 0

   so the residual is `q_follower + gearing*q_ref + offset`. The probe prints
   `q_follower - (gearing*q_ref + offset)`, which puts a constant `2*q_lead` (= 1.5708 rad, i.e. an
   apparent "90 deg") into every follower whose gearing is -1 and makes its `max_abs_res` column
   meaningless. The raw joint angles in the JSON are fine, so the table is recoverable -- but only
   by recomputing. Anyone reusing that probe has to do this.

2. It diffs two runs, which is the actual question. `--device cpu` vs `--device cuda:0` on the
   RoboLab side is the cheap half of the GPU-vs-CPU-dynamics test: turning GPU dynamics ON in
   OmniGibson is a ~30-site device port, turning it OFF here is one flag on a stack that already
   produces the reference numbers. If RoboLab's soft constraint survives on CPU, GPU dynamics is
   not what makes it soft.

    python scripts/debug_probes/scene_mimic_residual.py \
        --gpu /logs/gripper_squeeze/wrapdiff_robolab_squeeze.json \
        --cpu /logs/gripper_squeeze/scene_rl_cpu_squeeze.json

No Isaac, no GPU -- runs anywhere the JSONs are readable.
"""
import argparse
import json

ap = argparse.ArgumentParser()
ap.add_argument("--gpu", required=True, help="JSON from the GPU-dynamics run")
ap.add_argument("--cpu", default=None, help="JSON from the --device cpu run")
ap.add_argument("--out", default=None, help="write the comparison here as JSON")
args = ap.parse_args()


def residuals(path):
    """{force_N: {joint: residual_deg}} plus the pipeline block, with the sign fixed."""
    d = json.load(open(path))
    mim = d["mimic"]
    rows = {}
    for s in d["sweep"]:
        q_lead = s["q_lead"]
        r = {}
        for j, q in s["q"].items():
            m = mim.get(j)
            if m is None:
                continue
            # THE constraint: q + gearing*q_ref + offset = 0. Not q - (gearing*q_ref + offset).
            r[j] = (q + m["gearing"] * q_lead + m["offset"]) * 180.0 / 3.141592653589793
        rows[float(s["force_N"])] = dict(res=r, gap_mm=s["gap"] * 1e3, q_lead=q_lead)
    return rows, d.get("pipeline"), mim


gpu, gpu_pipe, mim = residuals(args.gpu)
cpu, cpu_pipe, _ = (residuals(args.cpu)[:2] + (None,)) if args.cpu else (None, None, None)

print("mimic naturalFrequency / gearing as authored:")
for j, m in sorted(mim.items()):
    print(f"  {j:<36} nf={m['naturalFrequency']:<10} dr={m['dampingRatio']:.4f} "
          f"gearing={m['gearing']:+.1f} offset={m['offset']}")

print(f"\nGPU-run pipeline: {json.dumps(gpu_pipe) if gpu_pipe else '(not recorded -- older probe)'}")
if args.cpu:
    print(f"CPU-run pipeline: {json.dumps(cpu_pipe) if cpu_pipe else '(not recorded)'}")
    if cpu_pipe and str(cpu_pipe.get("is_gpu_dynamics_enabled")).lower() not in ("false", "0"):
        print("  *** THE 'CPU' RUN STILL HAS GPU DYNAMICS ENABLED -- its numbers prove nothing ***")

# The four INNER mimic joints are the soft ones; right_outer_knuckle_joint runs nf=1e6 and is the
# rigidity control, so it is reported separately rather than folded into the max.
INNER = [j for j in mim if j != "right_outer_knuckle_joint"]


def summarise(rows, label):
    print(f"\n{label}: max |residual| over the four INNER mimic joints "
          f"(right_outer_knuckle_joint, nf=1e6, in the last column as the rigidity control)")
    print(f"  {'F per pad (N)':>13} {'gap (mm)':>10} {'max|res| inner (deg)':>22} {'outer (deg)':>12}")
    for f in sorted(rows):
        r = rows[f]["res"]
        inner = max(abs(r[j]) for j in INNER if j in r)
        outer = abs(r.get("right_outer_knuckle_joint", float("nan")))
        print(f"  {f:>13.1f} {rows[f]['gap_mm']:>10.3f} {inner:>22.4f} {outer:>12.4f}")
    return {f: max(abs(rows[f]["res"][j]) for j in INNER if j in rows[f]["res"]) for f in rows}


g = summarise(gpu, "GPU dynamics")
result = dict(gpu=g, gpu_pipeline=gpu_pipe)
if cpu is not None:
    c = summarise(cpu, "CPU dynamics")
    result.update(cpu=c, cpu_pipeline=cpu_pipe)
    print("\nRATIO -- if GPU dynamics is what makes RoboLab's mimic constraint soft, CPU collapses")
    print(f"  {'F per pad (N)':>13} {'GPU (deg)':>12} {'CPU (deg)':>12} {'GPU/CPU':>10}")
    ratios = {}
    for f in sorted(set(g) & set(c)):
        ratio = (g[f] / c[f]) if c[f] > 1e-9 else float("inf")
        ratios[f] = ratio
        print(f"  {f:>13.1f} {g[f]:>12.4f} {c[f]:>12.4f} {ratio:>10.2f}x")
    result["ratio"] = ratios
    loaded = [ratios[f] for f in ratios if f > 0]
    verdict = ("GPU_DYNAMICS_IS_THE_MECHANISM" if loaded and min(loaded) > 3.0
               else "GPU_DYNAMICS_IS_NOT_THE_MECHANISM" if loaded and max(loaded) < 1.5
               else "INCONCLUSIVE")
    result["verdict"] = verdict
    print(f"\nMIMIC_RESIDUAL_VERDICT {verdict}  "
          f"(>3x at every loaded rung = the mechanism; <1.5x everywhere = not the mechanism)")

if args.out:
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nwrote {args.out}")
print("MIMIC_RESIDUAL_OK")
