"""PHASE 1: which direction observable is wrong, and why. Rigid-body audit, no GPU, no Isaac.

`curl_A.log` reported `direction=DISAGREE` on every rung: the pad ROTATION about the hinge said
INWARD and the tip-to-tip SEPARATION said OUTWARD. This decides between them from the recorded
numbers alone, by testing the two competing stories about
`collision_boundary_points_world` against the measured tip displacement.

The two stories
---------------
The identity block of every run prints, in the `panda_link8` frame, both the pad LINK ORIGIN and
the tracked hull tip point. On robolab v2 they disagree by a constant (du, dv) = (-56.2, -116.1) mm,
identical for both pads.

  (A) the hull points are material points of the link that SIT AT THE OFFSET POSITION. Then their
      lever arm about the link origin is r = (-48.9, -98.3) mm -- pointing the WRONG WAY along the
      finger -- and the rotation term in their displacement carries the wrong sign.
  (B) the hull points are the TRUE material points, shifted by a constant world vector. Then a
      constant cancels out of any displacement and `d_tip_sep` is correct after all.

They make different, well separated predictions for the one number we measured directly:
`finger_tip_in_mm`, how far the loaded pad's tracked tip moved inboard. Run this and read which
prediction lands.

Then the same rigid-body model is evaluated at the TRUE tip -- and, because the exact tip location
is itself an estimate, over the whole range of tip locations that are geometrically possible on a
31.3 mm pad, to show the answer does not depend on getting it exactly right.
"""
import argparse
import json
import os

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--src", default="/logs/gripper_squeeze/curl_B_curl.json")
ap.add_argument("--also", default="/logs/gripper_squeeze/curl_C_curl.json")
ap.add_argument("--out", default="/logs/gripper_squeeze/ship_sign_audit.json")
args = ap.parse_args()

# ---------------------------------------------------------------- geometry, from the identity block
# panda_link8 frame, at the unloaded OPEN pose. u = along LONG (flange -> pads), v = along AXIS
# (left -> right). Printed by curl_press_direction.py; see curl_B.log lines 756-763.
GEOM = {
    "left_inner_finger":  dict(origin=(138.8, -58.1), hull_tip=(89.9, -156.4), hull_base=(70.6, -149.5), inboard=+1.0),
    "right_inner_finger": dict(origin=(138.8, +58.1), hull_tip=(90.8,  -90.3), hull_base=(65.4,  -42.3), inboard=-1.0),
}
HULL_OFFSET = (-56.2, -116.1)   # du, dv: hull centroid minus link origin, IDENTICAL for both pads
FINGER_LONG_EXTENT = 31.3       # mm, the pad's own long extent -- also identical for both pads


def dv_of_rotation(r_u, r_v, psi):
    """AXIS-component displacement of a material point at lever arm (r_u along LONG, r_v along AXIS)
    from the link origin, when the link rotates about H by psi.

    Rotating about H = AXIS x LONG by psi sends AXIS -> AXIS cos psi + LONG sin psi and
    LONG -> LONG cos psi - AXIS sin psi, so a point v*AXIS + u*LONG lands with AXIS component
    v cos psi - u sin psi.
    """
    return r_v * (np.cos(psi) - 1.0) - r_u * np.sin(psi)


def audit(rec, tag):
    out = dict(tag=tag, rung=rec["rung"], reported_direction=rec["direction"],
               reported_agree=rec["observables_agree"])
    print("=" * 100)
    print(f"{tag}  rung={rec['rung']}  probe verdict={rec['direction']} agree={rec['observables_agree']}")
    print(f"  reported: curl {rec['curl_in_deg']:+.4f} deg | d_pad_sep (origins, hull-free) "
          f"{rec['d_pad_sep_mm']:+.4f} mm | d_tip_sep (HULL) {rec['d_tip_sep_mm']:+.4f} mm")

    rot = rec["finger_rot_in_deg"]
    tip = rec["finger_tip_in_mm"]
    # which pad actually took the load
    loaded = max(rot, key=lambda k: abs(rot[k]))
    other = [k for k in rot if k != loaded][0]
    print(f"  loaded pad = {loaded} (rot {rot[loaded]:+.4f} deg); null control {other} "
          f"(rot {rot[other]:+.4f} deg, tip {tip[other]:+.4f} mm)")
    out.update(loaded_pad=loaded, rot_in_deg=rot[loaded], tip_in_mm_hull=tip[loaded],
               null_rot_in_deg=rot[other], null_tip_in_mm=tip[other])

    g = GEOM[loaded]
    s_in = g["inboard"]
    # rot_in = -s_in * psi  =>  psi = -rot_in / s_in
    psi = -np.radians(rot[loaded]) / s_in
    # only the loaded pad moves, so the whole pad-origin separation change is its own origin.
    # d_pad_sep = (pos1 - pos0) . AXIS, so for FL[0] (left) a fall in pad_sep is +v motion.
    dv_origin = -rec["d_pad_sep_mm"] * (1.0 if loaded == "left_inner_finger" else -1.0)
    print(f"  psi (rotation about H) = {np.degrees(psi):+.5f} deg; loaded pad ORIGIN moved "
          f"dv = {dv_origin:+.4f} mm along AXIS ({s_in * dv_origin:+.4f} mm inboard)")
    out.update(psi_deg=float(np.degrees(psi)), origin_dv_mm=float(dv_origin))

    # ---- story (A): hull points are material points AT THE OFFSET POSITION
    ru_A = g["hull_tip"][0] - g["origin"][0]
    rv_A = g["hull_tip"][1] - g["origin"][1]
    pred_A = s_in * (dv_origin + dv_of_rotation(ru_A, rv_A, psi))
    # ---- story (B): hull points are TRUE material points offset by a constant -> the constant
    #      cancels, and the correct lever arm is the offset-corrected one
    ru_B = ru_A - HULL_OFFSET[0]
    rv_B = rv_A - HULL_OFFSET[1]
    pred_B = s_in * (dv_origin + dv_of_rotation(ru_B, rv_B, psi))
    meas = tip[loaded]
    print(f"\n  MEASURED tip_in                                        {meas:+8.4f} mm")
    print(f"  (A) hull points sit at the offset, lever arm ({ru_A:+6.1f},{rv_A:+7.1f}) mm  -> {pred_A:+8.4f} mm"
          f"   residual {pred_A - meas:+.4f} mm")
    print(f"  (B) hull = true points + a constant, lever arm ({ru_B:+6.1f},{rv_B:+7.1f}) mm -> {pred_B:+8.4f} mm"
          f"   residual {pred_B - meas:+.4f} mm")
    win = "A" if abs(pred_A - meas) < abs(pred_B - meas) else "B"
    print(f"  ==> story ({win}) fits.")
    out.update(pred_A_mm=float(pred_A), pred_B_mm=float(pred_B), lever_A_mm=[ru_A, rv_A],
               lever_B_mm=[float(ru_B), float(rv_B)], story=win,
               resid_A_mm=float(pred_A - meas), resid_B_mm=float(pred_B - meas))

    # ---- the honest tip motion, and how sensitive the SIGN is to where the tip really is.
    # The pad link origin was put on the pad centroid by fix_robolab_link_origins.py and the pad's
    # own long extent is 31.3 mm, so the tip is somewhere in r_u in [-15.6, +15.6] mm.
    half = FINGER_LONG_EXTENT / 2.0
    print(f"\n  TRUE tip motion. Sweeping the tip's lever arm over every position that is possible on a"
          f" {FINGER_LONG_EXTENT:.1f} mm pad whose origin is on its own centroid:")
    rows = []
    for ru in np.linspace(-half, half, 7):
        for rv in (-half, 0.0, half):
            d = s_in * (dv_origin + dv_of_rotation(ru, rv, psi))
            rows.append((ru, rv, d))
    worst = min(r[2] for r in rows)
    best = max(r[2] for r in rows)
    for ru in np.linspace(-half, half, 7):
        vals = [s_in * (dv_origin + dv_of_rotation(ru, rv, psi)) for rv in (-half, 0.0, half)]
        print(f"    r_u = {ru:+6.2f} mm  ->  tip moved inboard by "
              f"{min(vals):+7.4f} .. {max(vals):+7.4f} mm")
    # the r_u at which the sign would flip
    flip = dv_origin / np.sin(psi) if abs(np.sin(psi)) > 1e-12 else np.inf
    print(f"  over the WHOLE physically possible range: {worst:+.4f} .. {best:+.4f} mm inboard")
    print(f"  the sign would only flip if the tip sat at r_u = {flip:+.1f} mm, i.e. "
          f"{abs(flip) / half:.1f}x the pad's own half-length on the FLANGE side of its origin.")
    best_guess = s_in * (dv_origin + dv_of_rotation(ru_B, rv_B, psi))
    print(f"  best estimate (offset-corrected hull tip, r_u={ru_B:+.1f} mm): {best_guess:+.4f} mm INBOARD")
    out.update(true_tip_in_mm_range=[float(worst), float(best)], sign_flip_r_u_mm=float(flip),
               true_tip_in_mm_best=float(best_guess),
               sign_robust=bool(worst > 0 or best < 0),
               verdict=("INWARD" if worst > 0 else "OUTWARD" if best < 0 else "AMBIGUOUS"))
    print(f"\n  PHASE1_ROW tag={tag} rung={rec['rung']} story={win} verdict={out['verdict']} "
          f"rot_deg={rot[loaded]:+.4f} origin_in_mm={s_in * dv_origin:+.4f} "
          f"hull_tip_in_mm={meas:+.4f} true_tip_in_mm={best_guess:+.4f} robust={out['sign_robust']}")
    return out


recs = []
for path in (args.src, args.also):
    if not os.path.exists(path):
        print(f"[skip] {path} missing")
        continue
    d = json.load(open(path))
    for p in d["presses"]:
        recs.append(audit(p, os.path.basename(path).replace("_curl.json", "")))

print("\n" + "=" * 100)
print("SUMMARY")
print(f"{'run':<10} {'rung':<10} {'story':<6} {'rot_deg':>9} {'origin_in':>10} {'hull_tip_in':>12} "
      f"{'TRUE_tip_in':>12} {'verdict':>10} {'robust':>7}")
for r in recs:
    print(f"{r['tag']:<10} {r['rung']:<10} {r['story']:<6} {r['rot_in_deg']:>+9.4f} "
          f"{r['origin_dv_mm'] * (1 if r['loaded_pad'] == 'left_inner_finger' else -1):>+10.4f} "
          f"{r['tip_in_mm_hull']:>+12.4f} {r['true_tip_in_mm_best']:>+12.4f} "
          f"{r['verdict']:>10} {str(r['sign_robust']):>7}")

os.makedirs(os.path.dirname(args.out), exist_ok=True)
json.dump(dict(geom=GEOM, hull_offset_mm=HULL_OFFSET, rows=recs), open(args.out, "w"), indent=1)
print(f"\nwrote {args.out}")
print("SHIP_SIGN_AUDIT_OK")
