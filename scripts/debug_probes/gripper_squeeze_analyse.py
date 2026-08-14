"""Cross-run table for gripper_squeeze_compliance.py -- one uniform re-analysis of every run's npz.

Each probe process prints its own analysis, so a change to the probe between two launches could move
a definition without anyone noticing. This recomputes every quantity for every run from the logged
per-step arrays with a single copy of the code, which is what makes the columns comparable.

Two traps it exists to encode:

  * The jaw gap has to be self-calibrated per asset -- raw measure minus its own value at full
    unloaded closure, where the pads touch and the physical gap is zero. Link-origin separations are
    not comparable across assets (robolab reads 33.0 mm shut, stock 7.1 mm), and the finger's convex
    collision hull fills in its own concavity, so that measure carries a constant offset too
    (-24.0 mm on robolab v2). After calibration the two agree to 0.01 mm on robolab (83.18 mm open,
    against the 2F-85's 85 mm nominal stroke) -- which is the cross-check that the zero is right. On
    the stock asset they do NOT agree (73.0 vs 87.2 mm); only the hull measure is validated there,
    by reading 29.989 mm against a 30.000 mm object.
  * The "flex vs the unloaded linkage at the same driven angle" estimator interpolates a curve that
    is only sampled at 6-13 driven angles, because the binary drive slews the whole jaw in one or two
    15 Hz control steps. Its two estimators disagreed by up to 0.5 mm on the soft-gain runs, so read
    that as its error bar and prefer the geometric numbers (jaw gap against the object's known width,
    and the follower deviation from the mimic gearing) when the signal is under a millimetre.

    python scripts/debug_probes/gripper_squeeze_analyse.py /logs/gripper_squeeze DROID DROID_robolab_v2
"""
import json
import os
import sys

import numpy as np

np.set_printoptions(precision=5, suppress=True, linewidth=200)
OUT = sys.argv[1]
ROBOTS = [r for r in sys.argv[2:] if os.path.exists(os.path.join(OUT, f"{r}_squeeze.npz"))]
RES = {}

for r in ROBOTS:
    z = np.load(os.path.join(OUT, f"{r}_squeeze.npz"), allow_pickle=True)
    with open(os.path.join(OUT, f"{r}_squeeze.json")) as f:
        S = json.load(f)
    tag = np.array([str(t) for t in z["tag"]])
    q = z["q"]
    names = [str(n) for n in z["joint_names"]]
    ctrl = S["ctrl_dof"]
    REF = int(ctrl[0])
    grip = [i for i, n in enumerate(names) if n not in
            ("panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4",
             "panda_joint5", "panda_joint6", "panda_joint7")]
    sep = z["sep_origin"]
    nc, fl, fr = z["n_contact"], z["f_l"], z["f_r"]

    # Jaw gap: raw measure minus its own value at full unloaded closure (pads touching => 0 mm).
    # NOTE: the npz's "gap_hull" key is NOT the raw hull gap -- the probe overwrote its `gaps`
    # variable with the CALIBRATED origin measure before saving, so that key duplicates jaw_sep.
    # The independent hull estimator is "jaw_hull", already calibrated.
    shut = tag == "free_close"
    z_sep = sep[shut][-1]
    jaw = sep - z_sep
    jaw_h = z["jaw_hull"] - z["jaw_hull"][shut][-1]

    # Unloaded reference curve = every step with no object anywhere near the jaws.
    unl = np.isin(tag, ["open", "cal_close", "cal_open", "reopen_cal", "free_close"])
    xr = q[unl, REF]
    o = np.argsort(xr)
    ndist = len(np.unique(np.round(xr, 5)))

    def at(vals, x):
        return np.interp(x, xr[o], np.asarray(vals)[unl][o])

    # Linear (gearing) model of each gripper joint against the driven one, fitted unloaded.
    A = np.stack([xr, np.ones_like(xr)], axis=1)
    gear = {}
    for i in grip:
        solq, *_ = np.linalg.lstsq(A, q[unl, i], rcond=None)
        gear[i] = (float(solq[0]), float(solq[1]),
                   float(np.abs(q[unl, i] - A @ solq).max()))

    out = dict(robot=r, ref=names[REF], names=names, grip=grip, ndist=ndist,
               n_unl=int(unl.sum()), gear=gear,
               jaw_open=float(jaw[tag == "open"][-1]),
               jaw_open_hull=float(jaw_h[tag == "open"][-1]),
               sep_open=float(sep[tag == "open"][-1]), sep_shut=float(z_sep),
               q_open=q[tag == "open"][-1], q_shut=q[shut][-1],
               close_target=S["close_target"], params=S["grip_joint_params"],
               isaac_kp=S.get("isaac_kp"), isaac_kd=S.get("isaac_kd"), sq={})
    OBJW = S["squeezes"][list(S["squeezes"])[0]]["obj_width_mm"] / 1000.0 if S["squeezes"] else 0.03
    out["objw"] = OBJW

    for lbl in ("A", "B"):
        m = tag == f"squeeze_{lbl}"
        if not m.any():
            continue
        idx = np.where(m)[0]
        touch = idx[nc[idx] > 0]
        if len(touch) == 0:
            out["sq"][lbl] = dict(no_contact=True)
            continue
        i0, iL = int(touch[0]), int(idx[-1])
        xl = q[iL, REF]
        jflex_i = {names[i]: float(np.abs(q[idx, i] - at(q[:, i], q[idx, REF])).max()) for i in grip}
        jflex_g = {names[i]: float(np.abs(q[idx, i] - (gear[i][0] * q[idx, REF] + gear[i][1])).max())
                   for i in grip}
        out["sq"][lbl] = dict(
            first=int(np.where(idx == i0)[0][0]),
            jaw_contact=float(jaw[i0]), jaw_final=float(jaw[iL]),
            past=float(OBJW - jaw[iL]), zero_err=float(jaw[i0] - OBJW),
            q_stall=float(xl), unresolved=float(S["close_target"][0] - xl),
            travel=float(q[shut][-1][REF] - q[tag == "open"][-1][REF]),
            f_l=float(fl[iL]), f_r=float(fr[iL]),
            f_max=float(np.nanmax(np.concatenate([fl[idx], fr[idx]]))),
            jaw_flex=float(jaw[iL] - at(jaw, xl)), jaw_flex_h=float(jaw_h[iL] - at(jaw_h, xl)),
            jaw_flex_max=float(np.abs(jaw[idx] - at(jaw, q[idx, REF])).max()),
            jflex_i=jflex_i, jflex_g=jflex_g,
            drift=float(z["cube_off"][iL]) if "cube_off" in z else float("nan"),
            arm=float(z["arm_dev"][idx].max()),
            q_final=q[iL],
        )
    RES[r] = out

W = 23


def line(label, fn):
    cells = []
    for r in ROBOTS:
        try:
            cells.append(fn(RES[r]))
        except Exception as e:
            cells.append(f"n/a({type(e).__name__})")
    print(f"  {label:<47}" + "".join(f"{str(c):>{W}}" for c in cells))


def head(t):
    print(f"\n{t}\n  {'':<47}" + "".join(f"{r[6:][:W - 1]:>{W}}" for r in ROBOTS))


print("=" * 120)
print("UNIFORM RE-ANALYSIS OF ALL RUNS (identical code for every column)")
print("=" * 120)
print(f"  columns are robot configs, names shortened by dropping the 'DROID_' prefix: {ROBOTS}")

head("CONTROL PATH (drive gains read back from the live articulation, not from the config)")
line("driven joint", lambda d: d["ref"])
line("gripper DOFs", lambda d: len(d["grip"]))
line("controller isaac_kp / isaac_kd",
     lambda d: f"{d['isaac_kp']} / {d['isaac_kd']}".replace("None", "dflt")[:W - 1])
line("driven joint stiffness / damping",
     lambda d: f"{d['params'][d['ref']]['stiffness']:.2g} / {d['params'][d['ref']]['damping']:.2g}")
line("driven joint max_effort", lambda d: f"{d['params'][d['ref']]['max_effort']:.4g}")
line("follower stiffness (all)", lambda d: ",".join(
    f"{v['stiffness']:.0f}" for k, v in d["params"].items() if k != d["ref"]) or "-")
line("follower max_effort (all)", lambda d: ",".join(
    f"{v['max_effort']:.0f}" for k, v in d["params"].items() if k != d["ref"]) or "-")
line("mimic-joint followers", lambda d: sum(1 for v in d["params"].values() if v["is_mimic"]))

head("UNLOADED JAW (gap zeroed where the pads meet at full closure)")
line("jaw gap OPEN, origin measure (mm)", lambda d: f"{d['jaw_open'] * 1e3:.2f}")
line("jaw gap OPEN, hull measure (mm)", lambda d: f"{d['jaw_open_hull'] * 1e3:.2f}")
line("raw origin separation OPEN / SHUT (mm)",
     lambda d: f"{d['sep_open'] * 1e3:.1f} / {d['sep_shut'] * 1e3:.1f}")
line("driven joint OPEN -> SHUT",
     lambda d: f"{d['q_open'][int(d['names'].index(d['ref']))]:+.3f} -> "
               f"{d['q_shut'][int(d['names'].index(d['ref']))]:+.3f}")
line("unloaded samples / distinct driven angles",
     lambda d: f"{d['n_unl']} / {d['ndist']}")
line("worst UNLOADED follower residual (rad|m)", lambda d: "%.6f" % max(
    [v[2] for k, v in d["gear"].items() if k != d["names"].index(d["ref"])] or [0]))

for lbl, nm in (("A", "SQUEEZE A -- 30 mm cube, 27 g, free, gravity off"),
                ("B", "SQUEEZE B -- same cube pinned heavy (immovable obstacle)")):
    if not any(lbl in RES[r]["sq"] for r in ROBOTS):
        continue
    head(nm)
    line("object width (mm)", lambda d: f"{d['objw'] * 1e3:.2f}")
    line("first contact at squeeze step", lambda d: d["sq"][lbl]["first"])
    line("jaw gap at first contact (mm)", lambda d: f"{d['sq'][lbl]['jaw_contact'] * 1e3:.3f}")
    line("  vs object width => zero check (mm)", lambda d: f"{d['sq'][lbl]['zero_err'] * 1e3:+.3f}")
    line("jaw gap at the end (mm)", lambda d: f"{d['sq'][lbl]['jaw_final'] * 1e3:.3f}")
    line("PADS CLOSE PAST THE OBJECT WIDTH (mm)", lambda d: f"{d['sq'][lbl]['past'] * 1e3:+.3f}")
    line("driven joint stalls at", lambda d: f"{d['sq'][lbl]['q_stall']:+.5f}")
    line("  commanded", lambda d: f"{d['close_target'][0]:+.5f}")
    line("  unresolved command (rad|m)", lambda d: f"{d['sq'][lbl]['unresolved']:+.5f}")
    line("  = % of full jaw travel", lambda d: "%.1f%%" % (
        100 * d["sq"][lbl]["unresolved"] / d["sq"][lbl]["travel"]))
    line("contact force L / R (N)",
         lambda d: f"{d['sq'][lbl]['f_l']:.2f} / {d['sq'][lbl]['f_r']:.2f}")
    line("max contact force over squeeze (N)", lambda d: f"{d['sq'][lbl]['f_max']:.2f}")
    print("  " + "-" * 47 + " COMPLIANCE: deviation from the unloaded linkage at the same driven angle")
    print("  (a difference at equal driven angle, so any constant error in the jaw-gap zero cancels)")
    line("jaw-gap FLEX, origin measure (mm)", lambda d: f"{d['sq'][lbl]['jaw_flex'] * 1e3:+.3f}")
    line("jaw-gap FLEX, hull measure (mm)", lambda d: f"{d['sq'][lbl]['jaw_flex_h'] * 1e3:+.3f}")
    # The two grippers do NOT squeeze with the same force (different authored effort limits), so the
    # raw flex is not comparable on its own -- flex per newton is.
    line("=> jaw compliance (um per N)", lambda d: "%.2f" % (
        1e6 * abs(d["sq"][lbl]["jaw_flex_h"]) / max(d["sq"][lbl]["f_max"], 1e-9)))
    line("=> joint compliance (mrad per N)", lambda d: "%.3f" % (
        1e3 * max(d["sq"][lbl]["jflex_g"].values()) / max(d["sq"][lbl]["f_max"], 1e-9)))
    line("max |joint FLEX|, interp (rad|m)",
         lambda d: "%.6f" % max(d["sq"][lbl]["jflex_i"].values()))
    line("max |joint FLEX|, gearing (rad|m)",
         lambda d: "%.6f" % max(d["sq"][lbl]["jflex_g"].values()))
    line("  worst joint", lambda d: max(d["sq"][lbl]["jflex_g"],
                                        key=d["sq"][lbl]["jflex_g"].get).replace("_joint", "")[:W - 1])
    line("object drift from pad midpoint (mm)", lambda d: f"{d['sq'][lbl]['drift'] * 1e3:.2f}")
    line("max arm joint deviation (rad)", lambda d: f"{d['sq'][lbl]['arm']:.1e}")

print("\nPER-JOINT FLEX (gearing estimator, max over the squeeze; rad, or m for prismatic)")
for lbl in ("A", "B"):
    for r in ROBOTS:
        if lbl not in RES[r]["sq"] or "jflex_g" not in RES[r]["sq"][lbl]:
            continue
        print(f"  [{lbl}] {r}")
        for k, v in RES[r]["sq"][lbl]["jflex_g"].items():
            print(f"        {k:<38} {v:.6f}")
print("\nANALYSE_OK")
