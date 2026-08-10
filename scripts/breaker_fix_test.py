"""Quick test: make base_link kinematic (so it can't fall) and reset any
purpose=render to default. Run inside the OmniGibson docker:

    python /app/scripts/breaker_fix_test.py
"""
import omnigibson as og
og.launch()  # boots the sim so lazy.pxr resolves
import omnigibson.lazy as lazy

USD = "/app/custom_assets/breaker/breaker_box.usd"

Usd = lazy.pxr.Usd
UsdGeom = lazy.pxr.UsdGeom
UsdPhysics = lazy.pxr.UsdPhysics

s = Usd.Stage.Open(USD)
link = s.GetPrimAtPath("/breaker_box/base_link")
UsdPhysics.RigidBodyAPI(link).CreateKinematicEnabledAttr(True)
print("set kinematicEnabled=True on", link.GetPath())

for p in s.Traverse():
    if p.IsA(UsdGeom.Imageable):
        a = UsdGeom.Imageable(p).GetPurposeAttr()
        if a.HasAuthoredValue() and a.Get() == "render":
            a.Set("default")
            print("reset purpose -> default on", p.GetPath())

s.GetRootLayer().Save()
print("done")

og.shutdown()
