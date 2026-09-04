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
* the right arm's wrist camera moved to YAMLab's separately calibrated ``cameras.right_wrist`` offset.

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

    python scripts/build_yam_bimanual_usd.py            # reads realm/robots/yam/yam.usd
    python scripts/build_yam_bimanual_usd.py --verify-only
"""

import argparse
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from realm.robots.yam import YamBimanualRobot as B, YamRobot  # noqa: E402
from build_yam_usd import (  # noqa: E402
    OUT_DIR,
    OUT_PROVENANCE,
    OUT_USD as SINGLE_USD,
    remap_dependents,
    replace_provenance_section,
    sha256,
    stale_paths,
)

try:
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics  # noqa: E402
except ImportError as exc:  # pragma: no cover - host tooling
    sys.exit(f"pxr is required (pip install usd-core): {exc}")

SRC_ROOT = f"/{YamRobot.MODEL}"
DST_ROOT = f"/{B.MODEL}"
OUT_USD = os.path.join(OUT_DIR, "yam_bimanual.usd")

SINGLE_LINKS = (*YamRobot.ARM_LINKS, *YamRobot.FINGER_LINKS, *YamRobot.FIXED_CAMERA_LINKS, *YamRobot.VIRTUAL_LINKS)


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


def build(source, output):
    src_stage = Usd.Stage.Open(source)
    assert src_stage, f"cannot open {source}"
    src_root = src_stage.GetDefaultPrim()
    assert src_root.GetPath() == Sdf.Path(SRC_ROOT), (
        f"default prim is {src_root.GetPath()}, expected {SRC_ROOT}; is this the built single-arm yam.usd?")
    src_layer = src_stage.GetRootLayer()

    link_names = [c.GetName() for c in src_root.GetChildren() if c.GetTypeName() == "Xform"]
    assert set(link_names) == set(SINGLE_LINKS), f"unexpected link set in {source}: {sorted(link_names)}"
    joint_names = [c.GetName() for c in src_stage.GetPrimAtPath(f"{SRC_ROOT}/joints").GetChildren()]
    assert set(joint_names) >= {*YamRobot.ARM_JOINTS, *YamRobot.FINGER_JOINTS, YamRobot.EEF_LINK}, joint_names
    assert src_stage.GetPrimAtPath(f"{SRC_ROOT}/PhysicsMaterial"), "single-arm file has no PhysicsMaterial"

    layer = Sdf.Layer.CreateAnonymous("yam-bimanual.usd")
    root_spec = Sdf.CreatePrimInLayer(layer, DST_ROOT)
    root_spec.typeName = "Xform"
    root_spec.specifier = Sdf.SpecifierDef
    joints_spec = Sdf.CreatePrimInLayer(layer, f"{DST_ROOT}/joints")
    joints_spec.typeName = "Scope"
    joints_spec.specifier = Sdf.SpecifierDef
    assert Sdf.CopySpec(src_layer, Sdf.Path(f"{SRC_ROOT}/PhysicsMaterial"), layer, Sdf.Path(f"{DST_ROOT}/PhysicsMaterial"))

    # Copy the arm twice. After each copy, every relationship target / connection that still names a
    # single-arm path (joint bodies, material bindings) is redirected to this arm's prims; the previous
    # arm's paths no longer match the /yam/ prefixes, so the sweep is idempotent per arm.
    for arm in B.ARMS:
        mapping = {f"{SRC_ROOT}/PhysicsMaterial": f"{DST_ROOT}/PhysicsMaterial"}
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

    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    stage.SetDefaultPrim(root)
    layer.comment = ("Two copies of realm/robots/yam/yam.usd on a shared mount (YAMLab bimanual workstation), "
                     "built by scripts/build_yam_bimanual_usd.py -- see realm/robots/yam/PROVENANCE")
    os.makedirs(os.path.dirname(output), exist_ok=True)
    assert layer.Export(output), f"failed to write {output}"
    return output


def verify(output):
    """Re-open the written file and check what OmniGibson's loader relies on. Returns (problems, summary)."""
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

    expected_links = {B.BASE_LINK}
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
    }
    return problems, summary


def write_provenance(source, output):
    lines = [
        "yam_bimanual.usd",
        f"  source: {os.path.relpath(source, REPO_ROOT)} (the built single-arm file above)",
        f"  source sha256: {sha256(source)}",
        f"  output sha256: {sha256(output)}",
        "  built by scripts/build_yam_bimanual_usd.py: every link/joint of yam.usd copied twice with left_/right_",
        "  prefixes, links translated by YAMLab's arm offsets (+-0.305 m in y), geometry-free root base_link at",
        "  the arm-base midpoint with a fixed <arm>_mount joint per arm, right wrist camera at YAMLab's",
        "  right_wrist offset. Numbers from realm/robots/yam.py::YamBimanualRobot.",
    ]
    replace_provenance_section(OUT_PROVENANCE, "yam_bimanual.usd", lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", default=SINGLE_USD, help="the built single-arm yam.usd")
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
