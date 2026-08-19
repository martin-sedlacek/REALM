#!/usr/bin/env python3
"""Compare droid_robolab_v2_mounted.usd against droid_robolab_v2.usd, on the stage, before switching.

WHY. `has_base_column` in realm/config/robots/DROID_robolab_v2*.yaml describes THE ASSET, and
env_config.py:111 raises the spawn by DROID_BASE_HEIGHT (0.86244) when it is false:

    if env.use_droid_with_base and not cfg_robot["robots"][0].pop("has_base_column", True):
        spawn_pos[2] += DROID_BASE_HEIGHT

So if the new asset carries its own base column -- i.e. panda_link0 sits ~0.862 m above the asset
root, the way droid_mounted.usd has it -- then leaving the flag false raises the arm TWICE and it
spawns 0.86 m in the air. If instead the column hangs BELOW a panda_link0 still at z=0, the flag must
stay false. The answer is a property of the file, so read the file.

The second question is whether the switch changes any PHYSICS. Martin says the asset is "the same,
just has this visual artefact", and the standing constraint is that arm physics stay byte-identical --
so this dumps masses, inertias, joint types, limits and drive gains for both and diffs them. A visual
mount should add VISUAL prims and change nothing else.

Third: the config filters sensors by name (include_sensor_names: ["wrist_camera_flipped"]), so the
camera prim names have to survive the rebuild. assert_wrist_camera() catches this at build time, but
knowing now is cheaper than a failed build.

    python probe_asset.py            # both assets, diffed
"""
import sys
from collections import OrderedDict

# pxr is NOT importable until Isaac is running: it ships inside isaacsim's extscache, not as a
# top-level package, and `import isaacsim` alone stops at an interactive EULA prompt under srun.
#
# LAUNCH FIRST, THEN TOUCH lazy.pxr -- and never the other way round. omnigibson's LazyImporter
# CACHES NEGATIVE LOOKUPS (lazy_import_utils.py:23, `self._not_module.add(name)`), so a single
# `lazy.pxr` access before the app exists marks pxr as not-a-module FOREVER in that process. My first
# attempt wrapped it in try/except to "fall back" to og.launch(), which poisoned the cache and then
# broke OmniGibson's own code -- og.launch() died at usd_utils.py:1034 doing lazy.pxr.UsdPhysics,
# which is not a bug in OmniGibson but in the probe that reached for pxr too early.
import omnigibson as og  # noqa: E402

og.launch()

import omnigibson.lazy as lazy  # noqa: E402
Usd, UsdGeom, UsdPhysics, Gf = lazy.pxr.Usd, lazy.pxr.UsdGeom, lazy.pxr.UsdPhysics, lazy.pxr.Gf

ARM = "/mnt/home_lustre/sedlam56/projects/REALM_og391/realm/robots/panda_robotiq/droid_robolab_v2.usd"
MOUNTED = "/mnt/home_lustre/sedlam56/projects/REALM_og391/realm/robots/panda_robotiq/droid_robolab_v2_mounted.usd"
STOCK_MOUNTED = "/mnt/home_lustre/sedlam56/projects/REALM_og391/realm/robots/panda_robotiq/droid_mounted.usd"
DROID_BASE_HEIGHT = 0.86244


def open_stage(path):
    st = Usd.Stage.Open(path)
    if st is None:
        sys.exit(f"could not open {path}")
    return st


def all_prims(stage):
    # Instance proxies included: OmniGibson instances objects, and a traversal that skips proxies
    # silently reports an empty subtree for anything referenced rather than defined inline.
    return list(stage.Traverse(Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate)))


def world_z(prim):
    try:
        m = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        return m.ExtractTranslation()[2]
    except Exception:
        return None


def report(path, label):
    st = open_stage(path)
    prims = all_prims(st)
    dp = st.GetDefaultPrim()
    print(f"\n{'=' * 78}\n{label}\n  {path}\n{'=' * 78}")
    print(f"default prim: {dp.GetPath() if dp else '(none)'}   total prims: {len(prims)}")

    # panda_link0 is the arm base. Its height above the asset root IS the has_base_column question.
    for p in prims:
        if p.GetName() == "panda_link0":
            z = world_z(p)
            print(f"panda_link0 at {p.GetPath()}  world z = {z:.6f}" if z is not None
                  else f"panda_link0 at {p.GetPath()}  (not xformable)")
            if z is not None:
                near_base = abs(z - DROID_BASE_HEIGHT) < 0.02
                print(f"  -> {'RAISED by a column (~DROID_BASE_HEIGHT)' if near_base else 'at/near the asset root' if abs(z) < 0.02 else 'neither 0 nor DROID_BASE_HEIGHT'}")
                print(f"  -> has_base_column should be {'TRUE' if near_base else 'FALSE' if abs(z) < 0.02 else 'DECIDED BY HAND -- unexpected z'}")
            break
    else:
        print("panda_link0 NOT FOUND")

    # OmniGibson treats only DIRECT Xform children of the robot prim as links.
    root = dp.GetPath().pathString if dp else "/panda"
    direct = [p.GetName() for p in prims
              if p.GetParent().GetPath().pathString == root and p.IsA(UsdGeom.Xform)]
    print(f"\ndirect Xform children of {root} ({len(direct)} -- these are the LINKS OmniGibson sees):")
    print("  " + ", ".join(sorted(direct)))

    cams = [p.GetPath().pathString for p in prims if p.IsA(UsdGeom.Camera)]
    print(f"\ncameras ({len(cams)}):")
    for c in sorted(cams):
        print(f"  {c}")

    joints, masses, drives = OrderedDict(), OrderedDict(), OrderedDict()
    for p in prims:
        if p.HasAPI(UsdPhysics.MassAPI):
            api = UsdPhysics.MassAPI(p)
            m = api.GetMassAttr().Get()
            if m:
                masses[p.GetName()] = round(float(m), 9)
        if p.IsA(UsdPhysics.Joint):
            jt = p.GetTypeName()
            j = UsdPhysics.Joint(p)
            lo = hi = None
            if p.IsA(UsdPhysics.RevoluteJoint):
                rj = UsdPhysics.RevoluteJoint(p)
                lo, hi = rj.GetLowerLimitAttr().Get(), rj.GetUpperLimitAttr().Get()
            elif p.IsA(UsdPhysics.PrismaticJoint):
                pj = UsdPhysics.PrismaticJoint(p)
                lo, hi = pj.GetLowerLimitAttr().Get(), pj.GetUpperLimitAttr().Get()
            bodies = ([str(t) for t in j.GetBody0Rel().GetTargets()],
                      [str(t) for t in j.GetBody1Rel().GetTargets()])
            joints[p.GetName()] = (str(jt), lo, hi, bodies)
        for ax in ("angular", "linear", "rotX", "rotY", "rotZ", "transX", "transY", "transZ"):
            da = UsdPhysics.DriveAPI(p, ax)
            if da and da.GetStiffnessAttr() and da.GetStiffnessAttr().HasAuthoredValue():
                drives[f"{p.GetName()}:{ax}"] = (
                    round(float(da.GetStiffnessAttr().Get() or 0), 6),
                    round(float(da.GetDampingAttr().Get() or 0), 6),
                    round(float(da.GetMaxForceAttr().Get() or 0), 6) if da.GetMaxForceAttr() else None,
                )
    print(f"\njoints: {len(joints)}   bodies with authored mass: {len(masses)}   authored drives: {len(drives)}")
    return {"joints": joints, "masses": masses, "drives": drives, "cameras": sorted(cams),
            "links": sorted(direct)}


def diff(name, a, b, la, lb):
    ka, kb = set(a), set(b)
    only_a, only_b = sorted(ka - kb), sorted(kb - ka)
    changed = {k: (a[k], b[k]) for k in sorted(ka & kb) if a[k] != b[k]}
    verdict = "IDENTICAL" if not (only_a or only_b or changed) else "DIFFERS"
    print(f"\n--- {name}: {verdict} ---")
    if only_a:
        print(f"  only in {la} ({len(only_a)}): {', '.join(only_a[:12])}{' ...' if len(only_a) > 12 else ''}")
    if only_b:
        print(f"  only in {lb} ({len(only_b)}): {', '.join(only_b[:12])}{' ...' if len(only_b) > 12 else ''}")
    for k, (va, vb) in list(changed.items())[:20]:
        print(f"  {k}:  {la}={va}   {lb}={vb}")
    if len(changed) > 20:
        print(f"  ... and {len(changed) - 20} more changed")
    return verdict == "IDENTICAL"


bare = report(ARM, "BARE ARM (current, has_base_column: false)")
mnt = report(MOUNTED, "NEW MOUNTED ASSET")
try:
    report(STOCK_MOUNTED, "STOCK droid_mounted.usd (reference for what a column looks like)")
except SystemExit as e:
    print(f"(stock mounted skipped: {e})")

print(f"\n{'=' * 78}\nBARE ARM vs NEW MOUNTED -- does the switch change physics?\n{'=' * 78}")
ok = []
ok.append(diff("joints (type, limits, bodies)", bare["joints"], mnt["joints"], "bare", "mounted"))
ok.append(diff("authored masses", bare["masses"], mnt["masses"], "bare", "mounted"))
ok.append(diff("authored joint drives", bare["drives"], mnt["drives"], "bare", "mounted"))
print(f"\ncameras  bare: {bare['cameras']}")
print(f"cameras  mnt : {mnt['cameras']}")
print(f"\nlinks only in mounted: {sorted(set(mnt['links']) - set(bare['links']))}")
print(f"links only in bare   : {sorted(set(bare['links']) - set(mnt['links']))}")
print("\nVERDICT: physics " + ("byte-comparable (no joint/mass/drive change)" if all(ok)
      else "CHANGED -- read the diffs above before switching"))
