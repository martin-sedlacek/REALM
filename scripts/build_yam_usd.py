#!/usr/bin/env python
"""Rebuild realm/robots/yam/yam.usd from YAMLab's arm USD so OmniGibson 3.9.1 can load it.

YAMLab's ``robot/yam/arm/yam.usd`` is an Isaac Sim 5.1 export written for IsaacLab, and three things
about it do not fit OmniGibson's robot loader:

1. **Links are nested one level too deep.** The default prim is ``/yam_new_gripper`` and the rigid
   links are children of ``/yam_new_gripper/arm``. ``EntityPrim.update_links`` only treats DIRECT
   Xform children of the loaded prim as links, so as shipped OmniGibson would see a single link
   called ``arm`` and no joints. Every link is reparented to the new root ``/yam``; the base link
   ``arm`` is renamed ``base_link`` (it collided with the articulation prim's name, and
   ``base_link`` is the fallback OmniGibson's root-link inference looks for).
2. **There is no camera.** IsaacLab spawned the wrist cameras at runtime from ``configs/robot/
   yam.yaml``. OmniGibson only discovers Camera prims that are direct children of a link, so the LEFT
   arm's wrist camera is authored at ``/yam/link_6/wrist_camera`` with YAMLab's offset (the
   ``quaternion_opengl`` is the USD Camera convention and is written verbatim as ``xformOp:orient``).
3. **The finger drives have maxForce 0.** IsaacLab set ``effort_limit_sim=100`` at runtime; under
   OmniGibson the USD value is the effort limit, and 0 means the gripper cannot move. The value is
   authored. The arm drives already carry YAMLab's 28/10 N m.
4. **There is no end-effector frame.** OmniGibson makes the eef link of every manipulation robot
   invisible (``Robot._initialize``), on the convention that it is a geometry-free tool frame like the
   Franka's ``panda_link8``. YAMLab's IK is Jacobian-based and has no such frame; pointing the
   definition at ``link_6`` hid the whole gripper housing. A massless, mesh-free ``eef_link`` is
   authored, fixed to ``link_6`` at the midpoint of YAMLab's two fingertip keypoints.
5. **The root prim has no xform ops.** IsaacLab writes its own; OmniGibson reads
   ``xformOp:translate/orient/scale`` off the root at load with no fallback. Identity ops are authored.

Also removed: the export's own fixed-to-world joint ``rootJoint_arm`` (OmniGibson creates its own
``rootJoint`` for fixed-base robots and a second one would double-constrain the base), the
``physicsScene``, and Kit viewport cameras/lights/render settings that have no business inside a
robot asset. Drive stiffness/damping stay 0 -- OmniGibson writes the controller's ``isaac_kp`` /
``isaac_kd`` onto the drives at load, which is how YAMLab's ImplicitActuator gains are applied.

Every number comes from ``realm.robots.yam.YamRobot``; this script has no constants of its own.

Host-side; needs ``pxr`` (``pip install usd-core``). Not needed at runtime -- the output is committed.

    python scripts/build_yam_usd.py --source /path/to/yamlab/yamlab/robot/yam/arm/yam.usd

Writes ``realm/robots/yam/yam.usd`` and ``realm/robots/yam/PROVENANCE``, then re-opens the result
and checks the structure OmniGibson relies on.
"""

import argparse
import datetime
import hashlib
import os
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from realm.robots.yam import YamRobot  # noqa: E402

try:
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics  # noqa: E402
except ImportError as exc:  # pragma: no cover - host tooling
    sys.exit(f"pxr is required (pip install usd-core): {exc}")

OUT_DIR = os.path.join(REPO_ROOT, "realm", "robots", "yam")
OUT_USD = os.path.join(OUT_DIR, "yam.usd")
OUT_PROVENANCE = os.path.join(OUT_DIR, "PROVENANCE")

SRC_ROOT = "/yam_new_gripper"
SRC_ARTICULATION = f"{SRC_ROOT}/arm"
SRC_BASE_LINK = f"{SRC_ARTICULATION}/arm"
SRC_JOINTS = f"{SRC_ROOT}/joints"
SRC_ROOT_JOINT = f"{SRC_JOINTS}/rootJoint_arm"
SRC_PHYSICS_MATERIAL = f"{SRC_ROOT}/PhysicsMaterial"
SRC_DROP = (f"{SRC_ROOT}/worldBody", "/physicsScene", "/Render", "/OmniverseKit_Persp",
            "/OmniverseKit_Front", "/OmniverseKit_Top", "/OmniverseKit_Right", "/OmniKit_Viewport_LightRig")

DST_ROOT = f"/{YamRobot.MODEL}"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _remap_path(path, mapping):
    """Longest-prefix remap of an Sdf.Path through {old_prefix: new_prefix}."""
    s = path.pathString
    for old in sorted(mapping, key=len, reverse=True):
        if s == old or s.startswith(old + "/") or s.startswith(old + "."):
            return Sdf.Path(mapping[old] + s[len(old):])
    return path


def _remap_listop(listop_proxy, mapping):
    """Rewrite every item of an Sdf list-op proxy (relationship targets / connections) in place."""
    for field in ("explicitItems", "addedItems", "prependedItems", "appendedItems", "deletedItems"):
        items = list(getattr(listop_proxy, field))
        if items:
            setattr(listop_proxy, field, [_remap_path(p, mapping) for p in items])


def _walk_prim_specs(layer):
    stack = list(layer.rootPrims)
    while stack:
        spec = stack.pop()
        yield spec
        stack.extend(spec.nameChildren)


def remap_dependents(layer, mapping):
    """Fix relationship targets and attribute connections after namespace edits.

    Sdf namespace edits move prim specs but leave the paths stored in relationships (physics:body0/1,
    physics:filteredPairs, material:binding) and shader connections pointing at the old locations.
    """
    for spec in _walk_prim_specs(layer):
        for rel in spec.relationships:
            _remap_listop(rel.targetPathList, mapping)
        for attr in spec.attributes:
            _remap_listop(attr.connectionPathList, mapping)


def stale_paths(layer, prefix):
    """Every relationship target / connection still under `prefix` (should be none after the remap)."""
    stale = []
    for spec in _walk_prim_specs(layer):
        for rel in spec.relationships:
            for p in rel.targetPathList.GetAddedOrExplicitItems():
                if p.pathString.startswith(prefix):
                    stale.append((spec.path.pathString, rel.name, p.pathString))
        for attr in spec.attributes:
            for p in attr.connectionPathList.GetAddedOrExplicitItems():
                if p.pathString.startswith(prefix):
                    stale.append((spec.path.pathString, attr.name, p.pathString))
    return stale


def build(source, output):
    src_stage = Usd.Stage.Open(source)
    assert src_stage, f"cannot open {source}"
    assert src_stage.GetDefaultPrim().GetPath() == Sdf.Path(SRC_ROOT), (
        f"unexpected default prim {src_stage.GetDefaultPrim().GetPath()}; is this YAMLab's yam.usd?")
    for path in (SRC_ARTICULATION, SRC_BASE_LINK, SRC_JOINTS, SRC_ROOT_JOINT, SRC_PHYSICS_MATERIAL):
        assert src_stage.GetPrimAtPath(path), f"missing {path} in {source}"
    assert not src_stage.GetPrimAtPath(f"{SRC_ARTICULATION}/link_6/{YamRobot.WRIST_CAMERA_PRIM}"), \
        "source already has a wrist camera; this script expects the bare YAMLab export"

    # The YAMLab file is a single self-contained layer; copy it into an anonymous layer to edit.
    # (Stage.Flatten would also work but spews warnings about Kit-only metadata fields.)
    non_session = [l for l in src_stage.GetUsedLayers() if not l.anonymous]
    assert len(non_session) == 1, f"expected a single-layer USD, got {[l.identifier for l in non_session]}"
    layer = Sdf.Layer.CreateAnonymous("yam-realm.usd")
    layer.TransferContent(src_stage.GetRootLayer())

    # --- 1. namespace edits -------------------------------------------------------------------
    # Root: a plain Xform. OmniGibson strips and re-applies ArticulationRootAPI itself
    # (USDObject._preapply_articulation_root), but authoring it on the root documents intent.
    root_spec = Sdf.CreatePrimInLayer(layer, DST_ROOT)
    root_spec.typeName = "Xform"
    root_spec.specifier = Sdf.SpecifierDef

    link_children = [c.name for c in layer.GetPrimAtPath(SRC_ARTICULATION).nameChildren]
    expected_links = {"arm", *YamRobot.ARM_LINKS[1:], *YamRobot.FINGER_LINKS, *YamRobot.FIXED_CAMERA_LINKS}
    assert set(link_children) == expected_links, f"unexpected link set in source: {sorted(link_children)}"

    edit = Sdf.BatchNamespaceEdit()
    mapping = {}
    for name in link_children:
        new_name = YamRobot.BASE_LINK if name == "arm" else name
        src = f"{SRC_ARTICULATION}/{name}"
        dst = f"{DST_ROOT}/{new_name}"
        edit.Add(Sdf.NamespaceEdit.ReparentAndRename(Sdf.Path(src), Sdf.Path(DST_ROOT), new_name,
                                                     Sdf.NamespaceEdit.atEnd))
        mapping[src] = dst
    edit.Add(Sdf.NamespaceEdit.Remove(Sdf.Path(SRC_ROOT_JOINT)))
    edit.Add(Sdf.NamespaceEdit.Reparent(Sdf.Path(SRC_JOINTS), Sdf.Path(DST_ROOT), Sdf.NamespaceEdit.atEnd))
    mapping[SRC_JOINTS] = f"{DST_ROOT}/joints"
    edit.Add(Sdf.NamespaceEdit.Reparent(Sdf.Path(SRC_PHYSICS_MATERIAL), Sdf.Path(DST_ROOT), Sdf.NamespaceEdit.atEnd))
    mapping[SRC_PHYSICS_MATERIAL] = f"{DST_ROOT}/PhysicsMaterial"
    for path in SRC_DROP:
        if layer.GetPrimAtPath(path):
            edit.Add(Sdf.NamespaceEdit.Remove(Sdf.Path(path)))
    ok = layer.Apply(edit)
    assert ok, "namespace edit batch was rejected"
    assert layer.GetPrimAtPath(SRC_ROOT) is not None
    remaining = [c.name for c in layer.GetPrimAtPath(SRC_ROOT).nameChildren]
    assert remaining == ["arm"], f"unexpected leftovers under {SRC_ROOT}: {remaining}"
    assert not list(layer.GetPrimAtPath(SRC_ARTICULATION).nameChildren)
    layer.Apply(Sdf.BatchNamespaceEdit([Sdf.NamespaceEdit.Remove(Sdf.Path(SRC_ROOT))]))

    # Relationship targets / connections still name the old paths (also the base link's rename).
    mapping[SRC_BASE_LINK] = f"{DST_ROOT}/{YamRobot.BASE_LINK}"
    remap_dependents(layer, mapping)
    stale = stale_paths(layer, SRC_ROOT)
    assert not stale, f"paths still pointing into {SRC_ROOT}: {stale[:5]}"

    layer.defaultPrim = YamRobot.MODEL

    # --- 2. physics / camera authoring on a stage over the edited layer -----------------------
    stage = Usd.Stage.Open(layer)
    root = stage.GetPrimAtPath(DST_ROOT)
    UsdPhysics.ArticulationRootAPI.Apply(root)
    # OmniGibson's XFormPrim._post_load reads xformOp:scale (and translate/orient) off the loaded root
    # with no fallback; YAMLab's root authored none because IsaacLab writes its own pose. Identity, in
    # the same op order droid.usd uses.
    root_xf = UsdGeom.Xformable(root)
    root_xf.ClearXformOpOrder()
    root_xf.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.0))
    root_xf.AddOrientOp(UsdGeom.XformOp.PrecisionFloat).Set(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
    root_xf.AddScaleOp().Set(Gf.Vec3f(1.0, 1.0, 1.0))

    for joint_name in YamRobot.FINGER_JOINTS:
        joint = stage.GetPrimAtPath(f"{DST_ROOT}/joints/{joint_name}")
        assert joint and joint.GetTypeName() == "PhysicsPrismaticJoint", joint_name
        drive = UsdPhysics.DriveAPI.Get(joint, "linear")
        assert drive, f"{joint_name} has no linear drive"
        drive.GetMaxForceAttr().Set(YamRobot.EFFORT_LIMITS[joint_name])
    for joint_name in YamRobot.ARM_JOINTS:
        joint = stage.GetPrimAtPath(f"{DST_ROOT}/joints/{joint_name}")
        assert joint and joint.GetTypeName() == "PhysicsRevoluteJoint", joint_name
        max_force = UsdPhysics.DriveAPI.Get(joint, "angular").GetMaxForceAttr().Get()
        assert abs(max_force - YamRobot.EFFORT_LIMITS[joint_name]) < 1e-6, (
            f"{joint_name}: USD maxForce {max_force} != YamRobot.EFFORT_LIMITS {YamRobot.EFFORT_LIMITS[joint_name]}")

    cam_path = f"{DST_ROOT}/{YamRobot.WRIST_CAMERA_LINK}/{YamRobot.WRIST_CAMERA_PRIM}"
    cam = UsdGeom.Camera.Define(stage, cam_path)
    focal = YamRobot.wrist_camera_focal_length()
    h_ap = YamRobot.WRIST_CAMERA_HORIZONTAL_APERTURE
    w, h = YamRobot.RENDER_RESOLUTION
    cam.GetFocalLengthAttr().Set(focal)
    cam.GetHorizontalApertureAttr().Set(h_ap)
    cam.GetVerticalApertureAttr().Set(h_ap * h / w)
    cam.GetClippingRangeAttr().Set(Gf.Vec2f(*YamRobot.WRIST_CAMERA_CLIPPING_RANGE))
    xf = UsdGeom.Xformable(cam.GetPrim())
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(*YamRobot.WRIST_CAMERA_POSITION))
    qw, qx, qy, qz = YamRobot.WRIST_CAMERA_QUAT_WXYZ
    xf.AddOrientOp(UsdGeom.XformOp.PrecisionFloat).Set(Gf.Quatf(qw, Gf.Vec3f(qx, qy, qz)))
    xf.AddScaleOp().Set(Gf.Vec3f(1.0, 1.0, 1.0))

    # --- 3. virtual eef frame -----------------------------------------------------------------
    # OmniGibson hides every manipulation robot's eef link (Robot._initialize: "make eef link not
    # visible") because by its convention that link is a geometry-free tool frame. YAMLab has no such
    # frame, so one is authored here: a massless rigid body with no meshes, fixed to the flange at the
    # midpoint of YAMLab's two fingertip keypoints, oriented like the flange.
    cache = UsdGeom.XformCache()
    flange = stage.GetPrimAtPath(f"{DST_ROOT}/{YamRobot.FLANGE_LINK}")
    X_flange = cache.GetLocalToWorldTransform(flange)
    tips = []
    for finger, keypoint in YamRobot.FINGERTIP_KEYPOINTS.items():
        X_finger = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(f"{DST_ROOT}/{finger}"))
        tips.append(X_finger.Transform(Gf.Vec3d(*keypoint)))
    tcp_world = (tips[0] + tips[1]) / 2.0
    tcp_in_flange = X_flange.GetInverse().Transform(tcp_world)
    flange_quat = X_flange.ExtractRotationQuat()

    eef = UsdGeom.Xform.Define(stage, f"{DST_ROOT}/{YamRobot.EEF_LINK}")
    eef_prim = eef.GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(eef_prim)
    UsdPhysics.MassAPI.Apply(eef_prim).GetMassAttr().Set(0.001)
    eef.ClearXformOpOrder()
    eef.AddTranslateOp().Set(tcp_world)
    eef.AddOrientOp(UsdGeom.XformOp.PrecisionFloat).Set(Gf.Quatf(flange_quat.GetReal(), Gf.Vec3f(*flange_quat.GetImaginary())))
    eef.AddScaleOp().Set(Gf.Vec3f(1.0, 1.0, 1.0))

    joint = UsdPhysics.FixedJoint.Define(stage, f"{DST_ROOT}/joints/{YamRobot.EEF_LINK}")
    joint.CreateBody0Rel().SetTargets([flange.GetPath()])
    joint.CreateBody1Rel().SetTargets([eef_prim.GetPath()])
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*tcp_in_flange))
    joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))

    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    stage.SetDefaultPrim(root)
    layer.comment = ("REALM copy of YAMLab robot/yam/arm/yam.usd restructured for OmniGibson 3.9.1 by "
                     "scripts/build_yam_usd.py -- see realm/robots/yam/PROVENANCE")
    os.makedirs(os.path.dirname(output), exist_ok=True)
    assert layer.Export(output), f"failed to write {output}"
    return output


def verify(output):
    """Re-open the written file and check what OmniGibson's loader relies on. Returns a report."""
    from realm.robots.yam import YamRobot as Y

    stage = Usd.Stage.Open(output)
    root = stage.GetDefaultPrim()
    problems = []
    if root.GetPath() != Sdf.Path(DST_ROOT):
        problems.append(f"default prim is {root.GetPath()}, expected {DST_ROOT}")
    if UsdGeom.GetStageUpAxis(stage) != UsdGeom.Tokens.z:
        problems.append("up axis is not Z")
    for op in ("xformOp:translate", "xformOp:orient", "xformOp:scale"):
        if root.GetAttribute(op).Get() is None:
            problems.append(f"root has no {op} (OmniGibson's XFormPrim reads it at load with no fallback)")
    for name in (*Y.ARM_LINKS, *Y.FINGER_LINKS, *Y.FIXED_CAMERA_LINKS, *Y.VIRTUAL_LINKS):
        link = root.GetChild(name)
        if link and link.GetAttribute("xformOp:scale").Get() is None:
            problems.append(f"{name} has no xformOp:scale")

    xform_children = {c.GetName() for c in root.GetChildren() if c.GetTypeName() == "Xform"}
    expected_links = {*Y.ARM_LINKS, *Y.FINGER_LINKS, *Y.FIXED_CAMERA_LINKS, *Y.VIRTUAL_LINKS}
    if xform_children != expected_links:
        problems.append(f"root Xform children {sorted(xform_children)} != links {sorted(expected_links)}")
    for name in expected_links:
        prim = root.GetChild(name)
        if not prim or not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            problems.append(f"{name} is not a rigid body")

    joints = {}
    for prim in Usd.PrimRange(root):
        if "Joint" in prim.GetTypeName():
            joints[prim.GetName()] = prim
    if "rootJoint_arm" in joints:
        problems.append("rootJoint_arm still present (OmniGibson adds its own rootJoint)")
    for name in (*Y.ARM_JOINTS, *Y.FINGER_JOINTS):
        j = joints.get(name)
        if j is None:
            problems.append(f"joint {name} missing")
            continue
        for rel in ("physics:body0", "physics:body1"):
            targets = j.GetRelationship(rel).GetTargets()
            if len(targets) != 1 or not stage.GetPrimAtPath(targets[0]):
                problems.append(f"{name}.{rel} -> {[t.pathString for t in targets]} does not resolve")
            elif not targets[0].pathString.startswith(DST_ROOT + "/"):
                problems.append(f"{name}.{rel} -> {targets[0]} is outside {DST_ROOT}")
        drive_kind = "linear" if name in Y.FINGER_JOINTS else "angular"
        max_force = UsdPhysics.DriveAPI.Get(j, drive_kind).GetMaxForceAttr().Get()
        if max_force is None or abs(max_force - Y.EFFORT_LIMITS[name]) > 1e-6:
            problems.append(f"{name} drive maxForce {max_force} != {Y.EFFORT_LIMITS[name]}")
    finger_joint_children = set()
    for j in joints.values():
        b0 = j.GetRelationship("physics:body0").GetTargets()
        b1 = j.GetRelationship("physics:body1").GetTargets()
        if b0 and b1:
            finger_joint_children.add(b1[0].name)
    roots = xform_children - finger_joint_children
    if roots != {Y.BASE_LINK}:
        problems.append(f"root-link inference would find {sorted(roots)}, expected [{Y.BASE_LINK}]")

    # The eef frame: geometry-free (nothing for OmniGibson to hide), fixed to the flange, and NOT the
    # flange itself -- OmniGibson hides whatever link eef_link_names points at.
    eef = root.GetChild(Y.EEF_LINK)
    tcp_in_flange = None
    if not eef or not eef.HasAPI(UsdPhysics.RigidBodyAPI):
        problems.append(f"{Y.EEF_LINK} missing or not a rigid body")
    else:
        if any(p.GetTypeName() in ("Mesh", "Cube", "Cylinder", "Sphere") for p in Usd.PrimRange(eef)):
            problems.append(f"{Y.EEF_LINK} carries geometry; OmniGibson makes the eef link invisible")
        j = joints.get(Y.EEF_LINK)
        if j is None or j.GetTypeName() != "PhysicsFixedJoint":
            problems.append(f"no fixed joint named {Y.EEF_LINK}")
        else:
            b0 = j.GetRelationship("physics:body0").GetTargets()
            if not b0 or b0[0].name != Y.FLANGE_LINK:
                problems.append(f"{Y.EEF_LINK} joint body0 is {b0}, expected {Y.FLANGE_LINK}")
            tcp_in_flange = tuple(round(float(v), 4) for v in j.GetAttribute("physics:localPos0").Get())
    if Y.EEF_LINK == Y.FLANGE_LINK or Y.EEF_LINK in Y.ARM_LINKS:
        problems.append("EEF_LINK must be the virtual frame, not an arm link")

    cam = stage.GetPrimAtPath(f"{DST_ROOT}/{Y.WRIST_CAMERA_LINK}/{Y.WRIST_CAMERA_PRIM}")
    if not cam or cam.GetTypeName() != "Camera":
        problems.append("wrist camera prim missing")
    else:
        t = cam.GetAttribute("xformOp:translate").Get()
        if t is None or max(abs(a - b) for a, b in zip(t, Y.WRIST_CAMERA_POSITION)) > 1e-7:
            problems.append(f"wrist camera translate {t} != {Y.WRIST_CAMERA_POSITION}")

    leftovers = [p.GetPath().pathString for p in stage.Traverse()
                 if p.GetPath().pathString.startswith(SRC_ROOT) or p.GetName() == "physicsScene"]
    if leftovers:
        problems.append(f"leftover prims: {leftovers[:5]}")
    stale = stale_paths(stage.GetRootLayer(), SRC_ROOT)
    if stale:
        problems.append(f"stale paths: {stale[:5]}")

    summary = {
        "links": sorted(xform_children),
        "joints": sorted(joints),
        "wrist_camera": cam.GetPath().pathString if cam else None,
        "hfov_deg": round(Y.wrist_camera_hfov_deg(), 2),
        "eef_link": Y.EEF_LINK,
        "tcp_in_flange_frame_m": tcp_in_flange,
    }
    return problems, summary


def write_provenance(source, output):
    try:
        commit = subprocess.check_output(["git", "-C", os.path.dirname(source), "rev-parse", "HEAD"],
                                         text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        commit = "unknown"
    lines = [
        "realm/robots/yam/ -- YAM arm assets ported from YAMLab (https://github.com/ARISE-Initiative/yamlab)",
        "",
        f"generated: {datetime.date.today().isoformat()} by scripts/build_yam_usd.py",
        f"yamlab commit: {commit}",
        "",
        "yam.usd",
        f"  source: {source}",
        f"  source sha256: {sha256(source)}",
        f"  output sha256: {sha256(output)}",
        "  changes vs source: see the module docstring of scripts/build_yam_usd.py (links reparented under",
        "  /yam, base link renamed base_link, rootJoint_arm/physicsScene/Kit cameras removed, finger drive",
        "  maxForce authored, wrist camera authored under link_6, massless eef_link fixed to link_6 at the",
        "  fingertip midpoint, identity xform ops on the root -- all from YamRobot).",
        "",
        "workstation/ (workstation.usd, Props/instanceable_meshes.usd, mesh/*.usd)",
        "  verbatim copies of yamlab/robot/yam/workstation/ (STL sources not copied). Reference only;",
        "  no REALM config loads them.",
    ]
    ws_dir = os.path.join(OUT_DIR, "workstation")
    for dirpath, _, files in sorted(os.walk(ws_dir)):
        for f in sorted(files):
            p = os.path.join(dirpath, f)
            lines.append(f"  {os.path.relpath(p, OUT_DIR)} sha256: {sha256(p)}")
    with open(OUT_PROVENANCE, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", required=True, help="YAMLab yamlab/robot/yam/arm/yam.usd")
    parser.add_argument("--output", default=OUT_USD)
    parser.add_argument("--verify-only", action="store_true", help="only re-check an existing output")
    args = parser.parse_args()

    if not args.verify_only:
        build(args.source, args.output)
        write_provenance(args.source, args.output)
    problems, summary = verify(args.output)
    for k, v in summary.items():
        print(f"{k}: {v}")
    if problems:
        print("PROBLEMS:")
        for p in problems:
            print(f"  - {p}")
        sys.exit(1)
    print(f"OK: {args.output}")


if __name__ == "__main__":
    main()
