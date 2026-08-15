"""Fit the fingertip-curl response to a two-parameter spring-in-series model and report what
effective inertia the SUCCESS CRITERION would require.

Why this exists
---------------
Three independent routes (loader patch, Xform flatten, authored mass + re-anchor) all land on
+0.33..+0.40 deg of curl at the AUTHORED naturalFrequency = 1000, against a target of "what nf=200
(+0.537) or nf=100 (+1.780) produced on the broken build". That reads as a shortfall. It is not.

The tip press is DISPLACEMENT controlled, not force controlled: the object is driven into the pad by
a commanded overtravel and the contact force is an OUTPUT. So the pad's rotation is set by a
compliance divider, not by 1/k:

    delta = delta_contact + L*theta,   F = K_c*delta_contact,   F*L = k*theta

    =>  theta = D / (X + k),     D = delta*L*K_c,   X = L^2*K_c,   k = omega^2 * I_eff

`D` and `X` are properties of the PRESS (commanded depth, contact stiffness, lever arm); `k` is the
mimic constraint. Two rungs at the same mass properties and different omega pin D and X, and the
model then PREDICTS every other rung with no free parameters left.

`I_eff` is the pad's inertia about its OWN PIVOT -- I_zz about the CoM plus the parallel-axis term
m*d^2 -- and not I about the CoM. That distinction is the whole reason the "77x inertia error should
buy 50x" expectation was too big: about the pivot the broken/fixed ratio is ~32x, not 77x, because
the broken CoM's m*d^2 already dominates both sides.

Run on the host, no GPU:  python scripts/debug_probes/resid_curl_model.py
"""

# --------------------------------------------------------------------------------------------
# Measured curl, left pad, jaws open, --load tip, same probe and flags throughout.
#   inertia_comfix_curl.json / mass_authored_anchor_curl.json / xflat_curl_curl.json
# I_eff = pad inertia about its own pivot, kg m^2, read back from the live articulation
#   (inertia_runtime_comfix.json "realm_I_eff", and I_zz_authored + m*d^2 for the anchored run).
# --------------------------------------------------------------------------------------------
I_EFF_PATCH = 8.3888e-06     # loader patch and Xform flatten -- identical mass properties
I_EFF_ANCHOR = 7.3777e-06    # authored diagonalInertia (RoboLab's tensor) + same m*d^2
I_EFF_ROBOLAB = 7.65295e-06  # RoboLab's own, same frame

OBS = [
    # label                      omega   I_eff          curl_L   curl_R
    ("loader patch  nf1000",     1000.0, I_EFF_PATCH,   0.3280,  0.3887),
    ("loader patch  nf100",       100.0, I_EFF_PATCH,   5.1488,  5.1712),
    ("Xform flatten nf1000",     1000.0, I_EFF_PATCH,   0.3380,  0.3822),
    ("mass+anchor   nf1000",     1000.0, I_EFF_ANCHOR,  0.3594,  0.4034),
    ("mass+anchor   nf100",       100.0, I_EFF_ANCHOR,  5.2184,  5.2039),
]

# The two rungs used to PIN D and X. Everything else is a prediction.
FIT = ("loader patch  nf1000", "loader patch  nf100")

TARGETS = [("nf=200 rung on the broken build", 0.537),
           ("nf=100 rung on the broken build", 1.780)]


def solve_DX(o1, o2, col):
    """theta = D/(X + omega^2 I) at two rungs -> D, X."""
    (_, w1, i1, *c1), (_, w2, i2, *c2) = o1, o2
    t1, t2 = c1[col], c2[col]
    k1, k2 = w1 * w1 * i1, w2 * w2 * i2
    #  D = t1*(X+k1) = t2*(X+k2)  ->  X*(t1-t2) = t2*k2 - t1*k1
    X = (t2 * k2 - t1 * k1) / (t1 - t2)
    D = t1 * (X + k1)
    return D, X


def main():
    by = {o[0]: o for o in OBS}
    for col, side in ((0, "LEFT"), (1, "RIGHT")):
        D, X = solve_DX(by[FIT[0]], by[FIT[1]], col)
        print(f"\n=== {side} pad ===")
        print(f"  fitted on {FIT[0]!r} + {FIT[1]!r}")
        print(f"  D = {D:.4f} deg N m/rad     X = {X:.4f} N m/rad")
        print(f"  ceiling as omega->0 (the press cannot express more) = D/X = {D / X:.2f} deg")
        print(f"\n  {'rung':<24} {'I_eff':>11} {'measured':>9} {'model':>9} {'err':>7}")
        for lab, w, i, *c in OBS:
            pred = D / (X + w * w * i)
            mark = "  <- fitted" if lab in FIT else ""
            print(f"  {lab:<24} {i:11.4e} {c[col]:9.4f} {pred:9.4f} "
                  f"{100 * (pred - c[col]) / c[col]:6.1f}%{mark}")

        print(f"\n  What the SUCCESS CRITERION would require at the authored omega=1000:")
        for lab, tgt in TARGETS:
            need = (D / tgt - X) / 1e6
            print(f"    {tgt:5.3f} deg ({lab}) -> I_eff = {need:.4e} "
                  f"= {I_EFF_ROBOLAB / need:5.2f}x BELOW RoboLab's physical {I_EFF_ROBOLAB:.4e}")
        best = D / (X + 1e6 * I_EFF_ROBOLAB)
        print(f"\n  Max curl at omega=1000 with RoboLab's exact I_eff = {best:.4f} deg")
    print("\n  => the three routes already sit at the physical ceiling for the authored nf.\n"
          "     Reaching the criterion needs I_eff BELOW the true pad inertia, i.e. it is not\n"
          "     an inertia problem any more. The nf=100-200 rungs were over-compensation.")


if __name__ == "__main__":
    main()
