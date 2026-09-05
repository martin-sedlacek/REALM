#!/usr/bin/env python
"""Build realm/robots/yam/yam_bimanual.usd -- YAMLab's two-arm workstation as ONE OmniGibson robot.

YAMLab spawns two copies of the arm USD as separate IsaacLab articulations at the poses in
``configs/robot/yam.yaml`` (``arms.left`` / ``arms.right``: 0.61 m apart in y, both facing +x). REALM
wants one robot per environment (``len(env.robots) == 1`` is asserted everywhere), and OmniGibson's
multi-arm robots are single articulations with per-arm link sets (cf. its R1), so the two arms are
composed into one asset:

* a geometry-free root link ``base_link`` at the midpoint of the two arm bases (the robot frame);
* every link and joint of ``realm/robots/yam/yam.usd`` copied twice, prefixed ``left_`` / ``right_``,
  the links translated by YAMLab's arm offsets (links are direct children of the root, so the offset
  is added to each link's own ``xformOp:translate``);
* a ``PhysicsFixedJoint`` ``<arm>_mount`` from ``base_link`` to each ``<arm>_base_link``, so
  OmniGibson's root-link inference finds exactly ``base_link`` and fixes it to the world;
* the right arm's wrist camera moved to YAMLab's separately calibrated ``cameras.right_wrist`` offset;
* a visual-only ``frame`` link fixed to ``base_link`` carrying YAMLab's aluminium-extrusion gate (the
  structure the arms bolt onto, from ``workstation/workstation.usd``), so the arms do not float; the part
  below the arm plates is stretched in z to reach the floor at REALM's ``mount_height``.

Everything else (drive limits, the massless ``eef_link`` frames, the wrist cameras, the physics
material) is inherited from the single-arm file, which is the ONLY input -- rebuild that first
(``scripts/build_yam_usd.py``) if the arm changes. The top camera is deliberately not in the USD: it is
REALM's ``external_sensor0`` placed by the ``exterior_camera`` key of ``realm/config/robots/
YAM_bimanual.yaml`` (see ``YamBimanualRobot``), so it participates in V-VIEW and the recorder like
every other exterior view.

The mesh data is duplicated rather than referenced: USD references cannot rewrite the physics-material
and joint relationships that point outside a referenced link, and a flattened self-contained file is
what OmniGibson's loader is known to handle. Every number comes from ``realm.robots.yam``.

Host-side; needs ``pxr`` (``pip install usd-core``). The output is committed.

    python scripts/build_yam_bimanual_usd.py                  # yam.usd -> yam_bimanual.usd
    python scripts/build_yam_bimanual_usd.py --variant crank  # yam_crank.usd -> yam_crank_bimanual.usd
    python scripts/build_yam_bimanual_usd.py --verify-only

``--variant`` picks the spec (``YamBimanualRobot`` or ``YamCrankBimanualRobot``); every path, name and
number below comes from it, so the ABC crank-gripper workstation is the same script over a different
single-arm file.
"""

import argparse
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from realm.robots.yam import YamBimanualRobot, YamCrankBimanualRobot  # noqa: E402
from build_yam_usd import (  # noqa: E402
    OUT_DIR,
    OUT_PROVENANCE,
    remap_dependents,
    replace_provenance_section,
    sha256,
    stale_paths,
)

try:
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt  # noqa: E402
except ImportError as exc:  # pragma: no cover - host tooling
    sys.exit(f"pxr is required (pip install usd-core): {exc}")

#: --variant -> the bimanual spec it builds; the single-arm source and the output follow the spec.
VARIANTS = {"yamlab": YamBimanualRobot, "crank": YamCrankBimanualRobot}


def _roots(spec):
    return f"/{spec.ARM.MODEL}", f"/{spec.MODEL}"


def _single_links(spec):
    arm = spec.ARM
    return (*arm.ARM_LINKS, *arm.FINGER_LINKS, *arm.FIXED_CAMERA_LINKS, *arm.VIRTUAL_LINKS)


def source_usd(spec):
    return os.path.join(OUT_DIR, f"{spec.ARM.MODEL}.usd")


def output_usd(spec):
    return os.path.join(OUT_DIR, f"{spec.MODEL}.usd")


def _shift_translate(prim, offset):
    """Add `offset` to a prim's xformOp:translate, keeping the attribute's precision."""
    attr = prim.GetAttribute("xformOp:translate")
    value = attr.Get()
    assert value is not None, f"{prim.GetPath()} has no xformOp:translate"
    attr.Set(type(value)(value[0] + offset[0], value[1] + offset[1], value[2] + offset[2]))


def _identity_ops(prim):
    xf = UsdGeom.Xformable(prim)
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.0))
    xf.AddOrientOp(UsdGeom.XformOp.PrecisionFloat).Set(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
    xf.AddScaleOp().Set(Gf.Vec3f(1.0, 1.0, 1.0))


def load_frame_mesh(B, workstation_usd=None):
    """Read YAMLab's gate mesh (triangle soup, one vertex per corner) off the composed workstation stage.

    Returns (points (N, 3) float32 in YAMLab's world frame, faceVertexCounts, faceVertexIndices). The
    mesh lives in Props/instanceable_meshes.usd behind an instanceable reference, so it is read through
    the composed stage rather than the layer.
    """
    import numpy as np

    path = os.path.join(REPO_ROOT, B.FRAME_SOURCE_USD) if workstation_usd is None else workstation_usd
    stage = Usd.Stage.Open(path)
    assert stage, f"cannot open {path}"
    prim = stage.GetPrimAtPath(B.FRAME_SOURCE_PRIM)
    assert prim and prim.GetTypeName() == "Mesh", f"{B.FRAME_SOURCE_PRIM} is not a Mesh in {path}"
    mesh = UsdGeom.Mesh(prim)
    xf = UsdGeom.XformCache().GetLocalToWorldTransform(prim)
    pts = np.array([xf.Transform(Gf.Vec3d(p)) for p in mesh.GetPointsAttr().Get()], dtype=np.float64)
    counts = list(mesh.GetFaceVertexCountsAttr().Get())
    indices = list(mesh.GetFaceVertexIndicesAttr().Get())
    assert set(counts) == {3}, "expected a triangle mesh"
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    # YAMLab's frame: x +-0.30, y +-0.65, z 0..1.68 with the arm plates at 0.76 (arms.<side>.position).
    assert abs(lo[2]) < 0.01 and 1.6 < hi[2] < 1.8 and 0.55 < hi[1] < 0.75, f"unexpected gate extent {lo} .. {hi}"
    return pts, counts, indices


def author_frame_link(stage, base_prim, frame_source, B):
    """Author the visual-only workstation frame as a link fixed to base_link.

    The mesh is copied (not referenced) so the asset stays a single file; its points are moved into the
    mount frame with the part below the arm plates stretched by B.FRAME_STRETCH_BELOW_MOUNT so the
    frame's feet reach the floor at REALM's mount_height (B.frame_z_in_mount). Authored normals are
    dropped: the soup has its own vertices per face, so flat shading falls out for free and the file
    stays ~9 MB smaller. No CollisionAPI anywhere under the link.
    """
    import numpy as np

    _, DST_ROOT = _roots(B)
    pts, counts, indices = frame_source
    origin = np.array(B.frame_origin_in_mount(), dtype=np.float64)
    z = np.array([B.frame_z_in_mount(v) for v in pts[:, 2]], dtype=np.float64)
    local = np.column_stack([pts[:, 0] + origin[0], pts[:, 1] + origin[1], z])

    link = UsdGeom.Xform.Define(stage, f"{DST_ROOT}/{B.FRAME_LINK}")
    UsdPhysics.RigidBodyAPI.Apply(link.GetPrim())
    UsdPhysics.MassAPI.Apply(link.GetPrim()).GetMassAttr().Set(1.0)
    _identity_ops(link.GetPrim())
    visuals = UsdGeom.Xform.Define(stage, f"{DST_ROOT}/{B.FRAME_LINK}/visuals")
    _identity_ops(visuals.GetPrim())
    mesh = UsdGeom.Mesh.Define(stage, f"{DST_ROOT}/{B.FRAME_LINK}/visuals/gate")
    _identity_ops(mesh.GetPrim())
    mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(local.astype(np.float32)))
    mesh.GetFaceVertexCountsAttr().Set(Vt.IntArray(counts))
    mesh.GetFaceVertexIndicesAttr().Set(Vt.IntArray(indices))
    mesh.GetSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
    mesh.GetDoubleSidedAttr().Set(True)
    lo, hi = local.min(axis=0), local.max(axis=0)
    mesh.GetExtentAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*[float(v) for v in lo]), Gf.Vec3f(*[float(v) for v in hi])]))

    material = UsdShade.Material.Define(stage, f"{DST_ROOT}/{B.FRAME_LINK}/visuals/aluminium")
    shader = UsdShade.Shader.Define(stage, f"{DST_ROOT}/{B.FRAME_LINK}/visuals/aluminium/shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*B.FRAME_MATERIAL["diffuse"]))
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(B.FRAME_MATERIAL["metallic"])
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(B.FRAME_MATERIAL["roughness"])
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)

    joint = UsdPhysics.FixedJoint.Define(stage, f"{DST_ROOT}/joints/{B.FRAME_LINK}")
    joint.CreateBody0Rel().SetTargets([base_prim.GetPath()])
    joint.CreateBody1Rel().SetTargets([link.GetPrim().GetPath()])
    for attr, value in (("LocalPos0", Gf.Vec3f(0, 0, 0)), ("LocalPos1", Gf.Vec3f(0, 0, 0))):
        getattr(joint, f"Create{attr}Attr")().Set(value)
    for attr in ("LocalRot0", "LocalRot1"):
        getattr(joint, f"Create{attr}Attr")().Set(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
    return lo, hi


def build(source, output, B=YamBimanualRobot, workstation_usd=None):
    YamRobot = B.ARM
    SRC_ROOT, DST_ROOT = _roots(B)
    frame_source = load_frame_mesh(B, workstation_usd)
    src_stage = Usd.Stage.Open(source)
    assert src_stage, f"cannot open {source}"
    src_root = src_stage.GetDefaultPrim()
    assert src_root.GetPath() == Sdf.Path(SRC_ROOT), (
        f"default prim is {src_root.GetPath()}, expected {SRC_ROOT}; is this the built single-arm yam.usd?")
    src_layer = src_stage.GetRootLayer()

    link_names = [c.GetName() for c in src_root.GetChildren() if c.GetTypeName() == "Xform"]
    assert set(link_names) == set(_single_links(B)), f"unexpected link set in {source}: {sorted(link_names)}"
    joint_names = [c.GetName() for c in src_stage.GetPrimAtPath(f"{SRC_ROOT}/joints").GetChildren()]
    assert set(joint_names) >= {*YamRobot.ARM_JOINTS, *YamRobot.FINGER_JOINTS, YamRobot.EEF_LINK}, joint_names
    assert src_stage.GetPrimAtPath(f"{SRC_ROOT}/PhysicsMaterial"), "single-arm file has no PhysicsMaterial"
    # Root children that are neither links nor the joints scope (PhysicsMaterial, a Looks scope) are shared
    # by both arms: copied once, and every binding into them redirected.
    shared = [c.GetName() for c in src_root.GetChildren() if c.GetTypeName() != "Xform" and c.GetName() != "joints"]

    layer = Sdf.Layer.CreateAnonymous("yam-bimanual.usd")
    root_spec = Sdf.CreatePrimInLayer(layer, DST_ROOT)
    root_spec.typeName = "Xform"
    root_spec.specifier = Sdf.SpecifierDef
    joints_spec = Sdf.CreatePrimInLayer(layer, f"{DST_ROOT}/joints")
    joints_spec.typeName = "Scope"
    joints_spec.specifier = Sdf.SpecifierDef
    for name in shared:
        assert Sdf.CopySpec(src_layer, Sdf.Path(f"{SRC_ROOT}/{name}"), layer, Sdf.Path(f"{DST_ROOT}/{name}"))

    # Copy the arm twice. After each copy, every relationship target / connection that still names a
    # single-arm path (joint bodies, material bindings) is redirected to this arm's prims; the previous
    # arm's paths no longer match the /yam/ prefixes, so the sweep is idempotent per arm.
    for arm in B.ARMS:
        mapping = {f"{SRC_ROOT}/{name}": f"{DST_ROOT}/{name}" for name in shared}
        for name in link_names:
            assert Sdf.CopySpec(src_layer, Sdf.Path(f"{SRC_ROOT}/{name}"), layer, Sdf.Path(f"{DST_ROOT}/{B.link_name(arm, name)}"))
            mapping[f"{SRC_ROOT}/{name}"] = f"{DST_ROOT}/{B.link_name(arm, name)}"
        for name in joint_names:
            assert Sdf.CopySpec(src_layer, Sdf.Path(f"{SRC_ROOT}/joints/{name}"), layer,
                                Sdf.Path(f"{DST_ROOT}/joints/{B.joint_name(arm, name)}"))
        remap_dependents(layer, mapping)
    stale = stale_paths(layer, SRC_ROOT + "/")
    assert not stale, f"paths still pointing into {SRC_ROOT}/: {stale[:5]}"
    layer.defaultPrim = B.MODEL

    stage = Usd.Stage.Open(layer)
    root = stage.GetPrimAtPath(DST_ROOT)
    UsdPhysics.ArticulationRootAPI.Apply(root)
    _identity_ops(root)  # OmniGibson reads translate/orient/scale off the root with no fallback

    # The mount frame: a rigid body with no geometry. OmniGibson infers it as the root link (the only
    # link that is no joint's child body) and, for a fixed-base robot, fixes it to the world.
    base = UsdGeom.Xform.Define(stage, f"{DST_ROOT}/{B.BASE_LINK}")
    UsdPhysics.RigidBodyAPI.Apply(base.GetPrim())
    UsdPhysics.MassAPI.Apply(base.GetPrim()).GetMassAttr().Set(1.0)
    _identity_ops(base.GetPrim())

    for arm in B.ARMS:
        offset = B.ARM_OFFSETS[arm]
        for name in link_names:
            _shift_translate(stage.GetPrimAtPath(f"{DST_ROOT}/{B.link_name(arm, name)}"), offset)

        mount = UsdPhysics.FixedJoint.Define(stage, f"{DST_ROOT}/joints/{B.mount_joint(arm)}")
        mount.CreateBody0Rel().SetTargets([base.GetPath()])
        mount.CreateBody1Rel().SetTargets([Sdf.Path(f"{DST_ROOT}/{B.link_name(arm, YamRobot.BASE_LINK)}")])
        mount.CreateLocalPos0Attr().Set(Gf.Vec3f(*offset))
        mount.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
        mount.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        mount.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))

        cam = stage.GetPrimAtPath(f"{DST_ROOT}/{B.flange_link(arm)}/{YamRobot.WRIST_CAMERA_PRIM}")
        assert cam and cam.GetTypeName() == "Camera", f"{arm}: wrist camera missing after copy"
        cam.GetAttribute("xformOp:translate").Set(Gf.Vec3d(*B.WRIST_CAMERA_POSITIONS[arm]))

    author_frame_link(stage, base.GetPrim(), frame_source, B)

    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    stage.SetDefaultPrim(root)
    layer.comment = (f"Two copies of realm/robots/yam/{YamRobot.MODEL}.usd on a shared mount (YAMLab bimanual workstation), "
                     "built by scripts/build_yam_bimanual_usd.py -- see realm/robots/yam/PROVENANCE")
    os.makedirs(os.path.dirname(output), exist_ok=True)
    assert layer.Export(output), f"failed to write {output}"
    return output


def verify(output, B=YamBimanualRobot):
    """Re-open the written file and check what OmniGibson's loader relies on. Returns (problems, summary)."""
    YamRobot = B.ARM
    SRC_ROOT, DST_ROOT = _roots(B)
    stage = Usd.Stage.Open(output)
    root = stage.GetDefaultPrim()
    problems = []
    if root.GetPath() != Sdf.Path(DST_ROOT):
        problems.append(f"default prim is {root.GetPath()}, expected {DST_ROOT}")
    if [p.GetName() for p in stage.GetPseudoRoot().GetChildren()] != [B.MODEL]:
        problems.append(f"stage has stray root prims: {[p.GetName() for p in stage.GetPseudoRoot().GetChildren()]}")
    if UsdGeom.GetStageUpAxis(stage) != UsdGeom.Tokens.z:
        problems.append("up axis is not Z")
    for op in ("xformOp:translate", "xformOp:orient", "xformOp:scale"):
        if root.GetAttribute(op).Get() is None:
            problems.append(f"root has no {op}")

    expected_links = {B.BASE_LINK, B.FRAME_LINK}
    for arm in B.ARMS:
        expected_links.update(B.all_links(arm))
    xform_children = {c.GetName() for c in root.GetChildren() if c.GetTypeName() == "Xform"}
    if xform_children != expected_links:
        problems.append(f"root Xform children differ from the expected link set: "
                        f"missing {sorted(expected_links - xform_children)}, extra {sorted(xform_children - expected_links)}")
    for name in expected_links:
        prim = root.GetChild(name)
        if not prim or not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            problems.append(f"{name} is not a rigid body")
        elif prim.GetAttribute("xformOp:scale").Get() is None:
            problems.append(f"{name} has no xformOp:scale")

    joints = {p.GetName(): p for p in Usd.PrimRange(root) if "Joint" in p.GetTypeName()}
    child_links = set()
    for j in joints.values():
        for rel in ("physics:body0", "physics:body1"):
            targets = j.GetRelationship(rel).GetTargets()
            if len(targets) != 1 or not stage.GetPrimAtPath(targets[0]):
                problems.append(f"{j.GetName()}.{rel} -> {[t.pathString for t in targets]} does not resolve")
            elif not targets[0].pathString.startswith(DST_ROOT + "/"):
                problems.append(f"{j.GetName()}.{rel} -> {targets[0]} is outside {DST_ROOT}")
        b1 = j.GetRelationship("physics:body1").GetTargets()
        if b1:
            child_links.add(b1[0].name)
    roots = xform_children - child_links
    if roots != {B.BASE_LINK}:
        problems.append(f"root-link inference would find {sorted(roots)}, expected [{B.BASE_LINK}]")

    driven = []
    mounts = {}
    cameras = {}
    for arm in B.ARMS:
        offset = B.ARM_OFFSETS[arm]
        for joint_name, single in zip(B.arm_joints(arm) + B.finger_joints(arm), YamRobot.ARM_JOINTS + YamRobot.FINGER_JOINTS):
            j = joints.get(joint_name)
            if j is None:
                problems.append(f"joint {joint_name} missing")
                continue
            kind = "linear" if single in YamRobot.FINGER_JOINTS else "angular"
            want_type = "PhysicsPrismaticJoint" if kind == "linear" else "PhysicsRevoluteJoint"
            if j.GetTypeName() != want_type:
                problems.append(f"{joint_name} is {j.GetTypeName()}, expected {want_type}")
            max_force = UsdPhysics.DriveAPI.Get(j, kind).GetMaxForceAttr().Get()
            if max_force is None or abs(max_force - YamRobot.EFFORT_LIMITS[single]) > 1e-6:
                problems.append(f"{joint_name} drive maxForce {max_force} != {YamRobot.EFFORT_LIMITS[single]}")
            driven.append(joint_name)

        mount = joints.get(B.mount_joint(arm))
        if mount is None or mount.GetTypeName() != "PhysicsFixedJoint":
            problems.append(f"no fixed joint {B.mount_joint(arm)}")
        else:
            b0 = mount.GetRelationship("physics:body0").GetTargets()
            b1 = mount.GetRelationship("physics:body1").GetTargets()
            if not b0 or b0[0].name != B.BASE_LINK or not b1 or b1[0].name != B.link_name(arm, YamRobot.BASE_LINK):
                problems.append(f"{B.mount_joint(arm)} connects {b0} -> {b1}")
            pos0 = mount.GetAttribute("physics:localPos0").Get()
            if pos0 is None or max(abs(a - b) for a, b in zip(pos0, offset)) > 1e-6:
                problems.append(f"{B.mount_joint(arm)} localPos0 {pos0} != arm offset {offset}")
            mounts[arm] = tuple(round(float(v), 4) for v in pos0) if pos0 is not None else None

        arm_base = root.GetChild(B.link_name(arm, YamRobot.BASE_LINK))
        t = arm_base.GetAttribute("xformOp:translate").Get() if arm_base else None
        if t is None or max(abs(a - b) for a, b in zip(t, offset)) > 1e-6:
            problems.append(f"{arm} base link translate {t} != arm offset {offset}")

        eef = root.GetChild(B.eef_link(arm))
        if not eef or not eef.HasAPI(UsdPhysics.RigidBodyAPI):
            problems.append(f"{B.eef_link(arm)} missing or not a rigid body")
        elif any(p.GetTypeName() in ("Mesh", "Cube", "Cylinder", "Sphere") for p in Usd.PrimRange(eef)):
            problems.append(f"{B.eef_link(arm)} carries geometry; OmniGibson makes the eef link invisible")
        ej = joints.get(B.eef_link(arm))
        if ej is None or ej.GetTypeName() != "PhysicsFixedJoint":
            problems.append(f"no fixed joint {B.eef_link(arm)}")
        else:
            b0 = ej.GetRelationship("physics:body0").GetTargets()
            if not b0 or b0[0].name != B.flange_link(arm):
                problems.append(f"{B.eef_link(arm)} joint body0 is {b0}, expected {B.flange_link(arm)}")

        cam = stage.GetPrimAtPath(f"{DST_ROOT}/{B.flange_link(arm)}/{YamRobot.WRIST_CAMERA_PRIM}")
        if not cam or cam.GetTypeName() != "Camera":
            problems.append(f"{arm} wrist camera prim missing")
        else:
            t = cam.GetAttribute("xformOp:translate").Get()
            if t is None or max(abs(a - b) for a, b in zip(t, B.WRIST_CAMERA_POSITIONS[arm])) > 1e-7:
                problems.append(f"{arm} wrist camera translate {t} != {B.WRIST_CAMERA_POSITIONS[arm]}")
            cameras[arm] = cam.GetPath().pathString

    if root.GetChild(B.BASE_LINK) and any(
            p.GetTypeName() in ("Mesh", "Cube", "Cylinder", "Sphere") for p in Usd.PrimRange(root.GetChild(B.BASE_LINK))):
        problems.append(f"{B.BASE_LINK} carries geometry; it is meant to be the bare mount frame")
    if set(driven) != set(B.dof_order()):
        problems.append(f"driven joints {sorted(driven)} != YamBimanualRobot.dof_order() {sorted(B.dof_order())}")

    # The workstation frame: a visual-only link (no CollisionAPI anywhere -- check_collisions would count a
    # colliding frame as an environment collision every step) fixed to base_link, standing on the floor
    # (lowest point at -MOUNT_HEIGHT) with YAMLab's arm plates at the mount plane (z ~ 0).
    frame_bbox = None
    frame = root.GetChild(B.FRAME_LINK)
    if not frame or not frame.HasAPI(UsdPhysics.RigidBodyAPI):
        problems.append(f"{B.FRAME_LINK} missing or not a rigid body")
    else:
        meshes = [p for p in Usd.PrimRange(frame) if p.GetTypeName() == "Mesh"]
        if not meshes:
            problems.append(f"{B.FRAME_LINK} has no visual mesh")
        if any(p.HasAPI(UsdPhysics.CollisionAPI) for p in Usd.PrimRange(frame)):
            problems.append(f"{B.FRAME_LINK} carries collision geometry; it must be visual-only")
        fj = joints.get(B.FRAME_LINK)
        if fj is None or fj.GetTypeName() != "PhysicsFixedJoint" or \
                [t.name for t in fj.GetRelationship("physics:body0").GetTargets()] != [B.BASE_LINK]:
            problems.append(f"{B.FRAME_LINK} is not fixed to {B.BASE_LINK}")
        if meshes:
            ext = UsdGeom.Mesh(meshes[0]).GetExtentAttr().Get()
            if ext is None:
                problems.append(f"{B.FRAME_LINK} mesh has no extent")
            else:
                lo, hi = ext
                frame_bbox = (tuple(round(float(v), 3) for v in lo), tuple(round(float(v), 3) for v in hi))
                if abs(lo[2] + B.MOUNT_HEIGHT) > 0.005:
                    problems.append(f"{B.FRAME_LINK} foot at z={lo[2]:.3f}, expected -MOUNT_HEIGHT={-B.MOUNT_HEIGHT:.3f} (the floor)")
                if not (0.8 < hi[2] < 1.0):
                    problems.append(f"{B.FRAME_LINK} top at z={hi[2]:.3f}; YAMLab's top bar is ~0.92 above the plates")
                if not (abs(abs(lo[1]) - abs(hi[1])) < 0.01 and 0.6 < hi[1] < 0.7):
                    problems.append(f"{B.FRAME_LINK} is not centred on the mount frame in y: {lo[1]:.3f} .. {hi[1]:.3f}")
                if not any(p.GetTypeName() == "Material" for p in Usd.PrimRange(frame)):
                    problems.append(f"{B.FRAME_LINK} has no material")

    stale = stale_paths(stage.GetRootLayer(), SRC_ROOT + "/")
    if stale:
        problems.append(f"stale single-arm paths: {stale[:5]}")

    summary = {
        "links": len(xform_children),
        "joints": sorted(joints),
        "driven_joints": sorted(driven),
        "mount_offsets_m": mounts,
        "wrist_cameras": cameras,
        "exterior_camera": B.exterior_camera(),
        "frame_bbox_in_mount_m": frame_bbox,
    }
    return problems, summary


def write_provenance(source, output, B=YamBimanualRobot):
    lines = [
        f"{B.MODEL}.usd",
        f"  source: {os.path.relpath(source, REPO_ROOT)} (the built single-arm file above)",
        f"  source sha256: {sha256(source)}",
        f"  output sha256: {sha256(output)}",
        f"  built by scripts/build_yam_bimanual_usd.py: every link/joint of {B.ARM.MODEL}.usd copied twice with left_/right_",
        f"  prefixes, links translated by the arm offsets (+-{abs(B.ARM_OFFSETS['left'][1])} m in y), geometry-free root base_link at",
        "  the arm-base midpoint with a fixed <arm>_mount joint per arm, right wrist camera at YAMLab's",
        f"  wrist-camera offset. Numbers from realm/robots/yam.py::{B.__name__}.",
        f"  frame link: visual-only copy of {B.FRAME_SOURCE_PRIM} from workstation/workstation.usd (sha256 above),",
        f"  moved into the mount frame, the part below the arm plates stretched x{B.FRAME_STRETCH_BELOW_MOUNT:.4f} in z",
        "  so the feet reach the floor at REALM's mount_height; normals and the OmniPBR/emissive material dropped",
        "  for a UsdPreviewSurface with the same aluminium constants.",
    ]
    replace_provenance_section(OUT_PROVENANCE, f"{B.MODEL}.usd", lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", default="yamlab", choices=sorted(VARIANTS),
                        help="yamlab: YAMLab gripper (yam.usd -> yam_bimanual.usd); crank: ABC crank gripper "
                             "(yam_crank.usd -> yam_crank_bimanual.usd)")
    parser.add_argument("--source", default=None, help="the built single-arm file (default: the variant's)")
    parser.add_argument("--output", default=None)
    parser.add_argument("--verify-only", action="store_true", help="only re-check an existing output")
    args = parser.parse_args()
    B = VARIANTS[args.variant]
    source = args.source or source_usd(B)
    output = args.output or output_usd(B)

    if not args.verify_only:
        build(source, output, B)
        write_provenance(source, output, B)
    problems, summary = verify(output, B)
    for k, v in summary.items():
        print(f"{k}: {v}")
    if problems:
        print("PROBLEMS:")
        for p in problems:
            print(f"  - {p}")
        sys.exit(1)
    print(f"OK: {output}")


if __name__ == "__main__":
    main()
