"""One table across every rung of a `--rungs` sweep, in the units the conclusion is stated in.

The probe prints its own table at the end of a run; this reprints it from the json, adds the columns
that matter for the visibility question, and can merge several runs. Specifically it reports follower
deflection in DEGREES rather than radians, and it carries the unloaded slop next to the loaded flex,
because at soft rungs most of the "compliance" is slop that is present with no load at all.

*** Do not quote the millimetre columns below nf ~= 1000. *** Once the fingers rotate, the hull
extreme along the closing axis stops being the pad face, so the jaw-gap estimator's own validation
(jaw at first contact against the 30.000 mm cube) drifts to +2 mm at nf=100 and +17 mm at nf=10, and
`past obj` goes negative -- the jaw reads WIDER than the object it is holding. The follower-deflection
columns need no hull geometry and are the ones to use.

    python scripts/debug_probes/mimic_table.py /logs/gripper_squeeze MIMIC_A MIMIC_C MIMIC_D
"""
import json
import math
import os
import sys

R2D = 180.0 / math.pi
root = sys.argv[1]
tags = sys.argv[2:] or ["MIMIC_A"]

COLS = ("run", "rung", "nf", "dr", "onf", "me", "spi", "flexB", "slop", "jawB", "pastB",
        "F_l", "F_r", "FmaxB", "pads", "held", "dropA", "openjaw", "zeroerr")
W = (8, 12, 7, 5, 9, 7, 4, 7, 6, 8, 8, 7, 7, 7, 5, 5, 8, 8, 8)
print("".join(f"{c:>{w}}" for c, w in zip(COLS, W)))


def row(vals):
    print("".join(f"{'-' if v is None else v:>{w}}" for v, w in zip(vals, W)))


for tag in tags:
    path = os.path.join(root, f"{tag}_squeeze.json")
    if not os.path.exists(path):
        print(f"  [skip] no {path}")
        continue
    d = json.load(open(path))
    if not d.get("rungs"):
        print(f"  [skip] {tag} is not a sweep run (no rungs)")
        continue
    print(f"# {tag}: softened joints = {d.get('inner_mimic')}   "
          f"nf_in_schema={d.get('mimic_nf_in_schema')}")
    for name, rec in d["rungs"].items():
        sp, B, A = rec["spec"], rec["squeezes"].get("B"), rec["squeezes"].get("A")
        slop = max(rec["gear_resid"].values()) * R2D
        f = lambda x, k, p=3: ("n/a" if x is None else f"{x[k]:.{p}f}")  # noqa: E731
        flex = "n/a" if B is None else f"{max(B['max_joint_flex_gear'].values()) * R2D:.2f}"
        row((tag[-1], name, sp["nf"], sp["dr"], sp["onf"], sp["me"], sp["spi"],
             flex, f"{slop:.2f}", f(B, "jaw_final_mm"), f(B, "past_object_width_mm"),
             f(B, "force_l_N", 1), f(B, "force_r_N", 1), f(B, "max_force_N", 1),
             "n/a" if B is None else B["n_contact_final"],
             "n/a" if A is None else ("YES" if A.get("held") else "NO"),
             "n/a" if A is None or A.get("drop_mm") is None else f"{A['drop_mm']:.2f}",
             f"{rec['open_jaw_mm']:.2f}",
             f(B, "zero_validation_mm", 2)))
print("\nflexB  = max |follower deviation from the fitted mimic gearing| over squeeze B, DEGREES")
print("slop   = the same residual on the UNLOADED calibration sweep -- how much of flexB is not load")
print("pads   = pad links in contact at the end; is_grasping needs 2 (finger_link_names)")
print("dropA  = mm the free 27 g cube fell over the gravity-restored hold, HELD if < 10 mm")
print("openjaw= calibrated jaw gap in the OPEN pose: it moving means the resting pose changed too")
print("zeroerr= jaw-gap estimator's own error vs the 30.000 mm cube at first contact. Past ~1 mm the")
print("         millimetre columns are not trustworthy; use flexB.")
