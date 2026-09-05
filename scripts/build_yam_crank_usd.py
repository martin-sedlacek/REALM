#!/usr/bin/env python
"""Build realm/robots/yam/yam_crank.usd: the YAM arm with I2RT's "crankshaft" gripper, from ABC's MJCF.

The ABC project (https://abc.bot) ships a MuJoCo model of the earlier YAM gripper
(``assets/put_bottles/assets/i2rt_yam/yam.xml``, model ``yam_v0``). Its six arm links are the parts
REALM already has -- the same ``model2*`` meshes at the same joint frames as YAMLab's export -- so this
script starts from the BUILT ``realm/robots/yam/yam.usd`` and replaces everything downstream of the
``link_6`` wrist motor with what the MJCF says:

* ``link_6`` itself keeps its frame and joint but gets the crank housing: the MJCF's visual meshes
  (``model2__12/13``), its three capsule collisions, and its inertial (mass, CoM, principal inertia);
* the two fingers are rebuilt as links ``left_finger`` / ``right_finger`` at the MJCF body poses, with
  the MJCF visual meshes (``model2__14..17``), the capsule/box collision pads of the ``lf_rot``/``lf_down``
  sub-bodies composed into the finger frame (the 0.6 mm contact spheres MuJoCo uses for contact
  modelling are dropped), the MJCF inertials, and prismatic joints along the finger body's z with the
  MJCF ranges (``left`` -0.00205..0.0475, ``right`` mirrored; 0 = CLOSED, +-0.0475 = OPEN);
* the wrist D405 becomes links ``camera_d405`` (housing mesh + ABC's 2 cm collision sphere) and
  ``camera_frame`` (geometry-free) at ABC's bracket pose, and the ``wrist_camera`` Camera prim is authored
  under ``link_6`` at the pose composed through the MJCF camera chain (asserted against
  ``YamCrankRobot.WRIST_CAMERA_*``);
* the massless ``eef_link`` frame is re-authored at ABC's ``grasp_site`` (13.47 cm along the flange).

The MJCF is read with ElementTree and its ``<default>`` classes resolved, so nothing about the gripper is
transcribed by hand; the spec (``realm.robots.yam.YamCrankRobot``) carries only what REALM needs at
runtime and the build asserts the MJCF agrees with it. STL meshes are read directly (binary STL) and
authored as flat-shaded triangle soups, each placed by the MJCF geom pose, and bound to copies of the
OmniPBR materials the YAMLab export uses on the arm so the gripper renders like the rest of the robot.

Host-side; needs ``pxr`` (``pip install usd-core``) and numpy. The output is committed.

    python scripts/build_yam_crank_usd.py --mjcf /path/to/abc/assets/put_bottles/assets/i2rt_yam/yam.xml
    python scripts/build_yam_crank_usd.py --verify-only
"""

import argparse
import os
import struct
import subprocess
import sys
import xml.etree.ElementTree as ET

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from realm.robots.yam import YamCrankRobot as C, YamRobot  # noqa: E402
from build_yam_usd import (  # noqa: E402
    OUT_DIR,
    OUT_PROVENANCE,
    OUT_USD as SOURCE_USD,
    remap_dependents,
    replace_provenance_section,
    sha256,
    stale_paths,
    verify_frame,
)

try:
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt  # noqa: E402
except ImportError as exc:  # pragma: no cover - host tooling
    sys.exit(f"pxr is required (pip install usd-core): {exc}")

SRC_ROOT = f"/{YamRobot.MODEL}"
DST_ROOT = f"/{C.MODEL}"
OUT_USD = os.path.join(OUT_DIR, "yam_crank.usd")

#: Links and joints of the YAMLab file that the MJCF replaces (everything hanging off link_6).
REPLACED_LINKS = (*YamRobot.FINGER_LINKS, *YamRobot.FIXED_CAMERA_LINKS, *YamRobot.VIRTUAL_LINKS)
#: MuJoCo contact-point spheres (class sphere_collision) are not collision geometry worth a PhysX shape.
MIN_COLLISION_RADIUS = 0.002
#: MJCF material name -> the OmniPBR material the YAMLab export binds to the SAME part family, copied out
#: of yam.usd so the crank parts render exactly like the arm (MJCF "black" 0.25 is the export's (0, 0, 0),
#: MJCF "white" 0.9 its 0.5 grey, the D405 housing its 0.2). A UsdPreviewSurface stand-in rendered visibly
#: lighter/flatter than the arm next to it.
MJCF_MATERIALS = {
    "black": f"{SRC_ROOT}/base_link/visuals/model2/DefaultMaterial",
    "white": f"{SRC_ROOT}/link_2/visuals/model2__5/DefaultMaterial",
    "camera_housing": f"{SRC_ROOT}/camera_d405/visuals/camera_d405/DefaultMaterial_0",
}


# --- MJCF -----------------------------------------------------------------------------------------

def _quat(text):
    w, x, y, z = (float(v) for v in text.split())
    n = (w * w + x * x + y * y + z * z) ** 0.5
    return Gf.Quatd(w / n, Gf.Vec3d(x / n, y / n, z / n))


def _vec(text):
    return Gf.Vec3d(*(float(v) for v in text.split()))


def _pose(pos_text, quat_text):
    """MuJoCo (pos, quat) -> Gf.Matrix4d in pxr's row-vector convention (point * M)."""
    m = Gf.Matrix4d(1.0)
    m.SetTransform(Gf.Rotation(_quat(quat_text or "1 0 0 0")), _vec(pos_text or "0 0 0"))
    return m


def _compose(local, parent_world):
    """World matrix of a child whose pose `local` is expressed in `parent_world`'s frame."""
    return local * parent_world


class Mjcf:
    """The parts of an MJCF this build needs: default-class resolution, mesh files, body tree."""

    def __init__(self, path):
        self.path = path
        self.tree = ET.parse(path)
        self.root = self.tree.getroot()
        compiler = self.root.find("compiler")
        assert compiler is not None and compiler.get("angle", "degree") == "radian", "expected angle=radian"
        self.meshdir = os.path.join(os.path.dirname(path), compiler.get("meshdir", "."))
        self.defaults = {}
        self._collect_defaults(self.root.find("default"), {}, None)
        self.mesh_files = {}
        for mesh in self.root.find("asset").findall("mesh"):
            name = mesh.get("name") or os.path.splitext(os.path.basename(mesh.get("file")))[0]
            self.mesh_files[name] = os.path.join(self.meshdir, mesh.get("file"))
        self.bodies = {}
        self.body_childclass = {}
        self._collect_bodies(self.root.find("worldbody"), None)

    def _collect_defaults(self, elem, inherited, name):
        if elem is None:
            return
        merged = {tag: dict(attrs) for tag, attrs in inherited.items()}
        for child in elem:
            if child.tag == "default":
                continue
            merged.setdefault(child.tag, {}).update(child.attrib)
        if name is not None:
            self.defaults[name] = merged
        for child in elem.findall("default"):
            self._collect_defaults(child, merged, child.get("class"))

    def _collect_bodies(self, elem, childclass):
        for body in elem.findall("body"):
            cc = body.get("childclass", childclass)
            self.bodies[body.get("name")] = body
            self.body_childclass[body.get("name")] = cc
            self._collect_bodies(body, cc)

    def attrs(self, elem, childclass):
        """Effective attributes of a geom/joint/etc.: its class (or the inherited childclass) defaults
        overlaid with what is written on the element."""
        cls = elem.get("class", childclass)
        out = dict(self.defaults.get(cls, {}).get(elem.tag, {})) if cls else {}
        out.update(elem.attrib)
        return out

    def body_pose(self, body):
        return _pose(body.get("pos"), body.get("quat"))

    def inertial(self, body):
        inertial = body.find("inertial")
        assert inertial is not None, f"{body.get('name')} has no <inertial>"
        return {
            "mass": float(inertial.get("mass")),
            "com": _vec(inertial.get("pos", "0 0 0")),
            "quat": _quat(inertial.get("quat", "1 0 0 0")),
            "diag": _vec(inertial.get("diaginertia")),
        }

    def geoms(self, body, childclass, kind):
        """Direct geoms of `body` of `kind` ("visual" | "collision"), with resolved attributes."""
        out = []
        for geom in body.findall("geom"):
            a = self.attrs(geom, childclass)
            is_visual = a.get("contype", "1") == "0" and a.get("conaffinity", "1") == "0"
            if (kind == "visual") == is_visual:
                out.append(a)
        return out

    def collision_geoms_recursive(self, body, childclass, world=None):
        """Collision geoms of `body` and of its jointless sub-bodies, each with its pose composed into
        `body`'s frame (MuJoCo lets a body carry pose-only children for convenient geom placement)."""
        world = Gf.Matrix4d(1.0) if world is None else world
        out = [(a, _compose(_pose(a.get("pos"), a.get("quat")), world)) for a in self.geoms(body, childclass, "collision")]
        for child in body.findall("body"):
            if child.find("joint") is not None:
                continue  # a jointed child is a link of its own (the fingers under link_6), not a geom holder
            out += self.collision_geoms_recursive(child, self.body_childclass.get(child.get("name"), childclass),
                                                  _compose(self.body_pose(child), world))
        return out


def read_stl(path):
    """Binary STL -> (points (3N, 3) float64 as a triangle soup, faceVertexCounts, faceVertexIndices)."""
    with open(path, "rb") as fh:
        data = fh.read()
    assert not (data[:5] == b"solid" and b"facet normal" in data[:200]), f"{path}: ASCII STL not supported"
    n = struct.unpack("<I", data[80:84])[0]
    rec = np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("attr", "<u2")])
    tris = np.frombuffer(data[84:84 + n * rec.itemsize], dtype=rec)["v"].astype(np.float64)
    pts = tris.reshape(-1, 3)
    return pts, [3] * n, list(range(3 * n))


# --- USD authoring --------------------------------------------------------------------------------

def _set_ops(prim, matrix=None, translate=None, orient=None):
    """Author translate/orient(float)/scale=1 ops, from a matrix or explicit values."""
    if matrix is not None:
        translate = matrix.ExtractTranslation()
        orient = matrix.ExtractRotationQuat()
    xf = UsdGeom.Xformable(prim)
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(translate))
    q = orient if orient is not None else Gf.Quatd(1.0)
    xf.AddOrientOp(UsdGeom.XformOp.PrecisionFloat).Set(Gf.Quatf(float(q.GetReal()), Gf.Vec3f(*[float(v) for v in q.GetImaginary()])))
    xf.AddScaleOp().Set(Gf.Vec3f(1.0, 1.0, 1.0))


def _link_world(stage, path):
    prim = stage.GetPrimAtPath(path)
    t = prim.GetAttribute("xformOp:translate").Get()
    q = prim.GetAttribute("xformOp:orient").Get()
    m = Gf.Matrix4d(1.0)
    m.SetTransform(Gf.Rotation(Gf.Quatd(float(q.GetReal()), Gf.Vec3d(*[float(v) for v in q.GetImaginary()]))), Gf.Vec3d(t))
    return m


def copy_materials(src_layer, layer, looks_path):
    """Copy the YAMLab export's OmniPBR materials (MJCF_MATERIALS) into `looks_path` and redirect the
    shader connections that the copies still aim at their old location. Returns {mjcf name: Sdf.Path}."""
    out = {}
    mapping = {}
    for name, src_path in MJCF_MATERIALS.items():
        assert src_layer.GetPrimAtPath(src_path), f"{src_path} not in the YAMLab file; the export changed"
        dst = f"{looks_path}/{name}"
        assert Sdf.CopySpec(src_layer, Sdf.Path(src_path), layer, Sdf.Path(dst))
        mapping[src_path] = dst
        out[name] = Sdf.Path(dst)
    remap_dependents(layer, mapping)
    return out


def author_visual_mesh(stage, link_path, name, stl_path, pose, material):
    visuals = f"{link_path}/visuals"
    if not stage.GetPrimAtPath(visuals):
        _set_ops(UsdGeom.Xform.Define(stage, visuals).GetPrim(), translate=Gf.Vec3d(0, 0, 0))
    mesh = UsdGeom.Mesh.Define(stage, f"{visuals}/{name}")
    pts, counts, indices = read_stl(stl_path)
    mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(pts.astype(np.float32)))
    mesh.GetFaceVertexCountsAttr().Set(Vt.IntArray(counts))
    mesh.GetFaceVertexIndicesAttr().Set(Vt.IntArray(indices))
    mesh.GetSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    mesh.GetExtentAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*[float(v) for v in lo]), Gf.Vec3f(*[float(v) for v in hi])]))
    _set_ops(mesh.GetPrim(), matrix=pose)
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    return mesh


def author_collision(stage, link_path, name, attrs, pose):
    """One MJCF collision geom as a PhysX primitive shape, a DIRECT child of the link (the CoM rule)."""
    size = [float(v) for v in attrs["size"].split()]
    kind = attrs.get("type", "sphere")
    if kind == "capsule":
        prim = UsdGeom.Capsule.Define(stage, f"{link_path}/{name}")
        prim.GetRadiusAttr().Set(size[0])
        prim.GetHeightAttr().Set(2.0 * size[1])
        prim.GetAxisAttr().Set(UsdGeom.Tokens.z)
        _set_ops(prim.GetPrim(), matrix=pose)
    elif kind == "box":
        prim = UsdGeom.Cube.Define(stage, f"{link_path}/{name}")
        prim.GetSizeAttr().Set(2.0)
        _set_ops(prim.GetPrim(), matrix=pose)
        prim.GetPrim().GetAttribute("xformOp:scale").Set(Gf.Vec3f(*size[:3]))
    elif kind == "sphere":
        prim = UsdGeom.Sphere.Define(stage, f"{link_path}/{name}")
        prim.GetRadiusAttr().Set(size[0])
        _set_ops(prim.GetPrim(), matrix=pose)
    else:
        raise ValueError(f"unsupported collision geom type {kind}")
    UsdPhysics.CollisionAPI.Apply(prim.GetPrim())
    # OmniGibson hides collision geometry by setting purpose "guide" at load, but only on the gprim types it
    # classifies ({Sphere, Cube, Cone, Cylinder, Mesh}); a Capsule slips through and renders as a grey blob
    # over the finger meshes (seen in the GUI, 2026-09-05). Author the purpose here for every shape.
    UsdGeom.Imageable(prim.GetPrim()).CreatePurposeAttr().Set(UsdGeom.Tokens.guide)
    return prim


def author_mass(prim, inertial):
    mass = UsdPhysics.MassAPI.Apply(prim)
    mass.GetMassAttr().Set(inertial["mass"])
    mass.GetCenterOfMassAttr().Set(Gf.Vec3f(*[float(v) for v in inertial["com"]]))
    mass.GetDiagonalInertiaAttr().Set(Gf.Vec3f(*[float(v) for v in inertial["diag"]]))
    q = inertial["quat"]
    mass.GetPrincipalAxesAttr().Set(Gf.Quatf(float(q.GetReal()), Gf.Vec3f(*[float(v) for v in q.GetImaginary()])))


def author_fixed_joint(stage, name, body0_path, body1_path, local_pose):
    joint = UsdPhysics.FixedJoint.Define(stage, f"{DST_ROOT}/joints/{name}")
    joint.CreateBody0Rel().SetTargets([Sdf.Path(body0_path)])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(body1_path)])
    q = local_pose.ExtractRotationQuat()
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*[float(v) for v in local_pose.ExtractTranslation()]))
    joint.CreateLocalRot0Attr().Set(Gf.Quatf(float(q.GetReal()), Gf.Vec3f(*[float(v) for v in q.GetImaginary()])))
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
    return joint


def drop_dangling_targets(layer):
    """Remove relationship targets (filtered collision pairs, material bindings) that name removed prims."""
    stack = list(layer.rootPrims)
    while stack:
        spec = stack.pop()
        for rel in spec.relationships:
            for field in ("explicitItems", "addedItems", "prependedItems", "appendedItems"):
                items = list(getattr(rel.targetPathList, field))
                kept = [p for p in items if layer.GetPrimAtPath(p.GetPrimPath()) is not None]
                if len(kept) != len(items):
                    setattr(rel.targetPathList, field, kept)
        stack.extend(spec.nameChildren)


def build(source, mjcf_path, output):
    mj = Mjcf(mjcf_path)
    flange_body = mj.bodies[C.MJCF_FLANGE_BODY]
    flange_class = mj.body_childclass[C.MJCF_FLANGE_BODY]
    for finger, body_name in C.MJCF_FINGER_BODIES.items():
        joint = mj.bodies[body_name].find("joint")
        lo, hi = (float(v) for v in joint.get("range").split())
        assert (lo, hi) == C.FINGER_LIMITS[finger], f"{finger}: MJCF range {(lo, hi)} != spec {C.FINGER_LIMITS[finger]}"
        assert mj.attrs(joint, flange_class).get("type") == "slide", f"{finger} joint is not a slide joint"

    src_stage = Usd.Stage.Open(source)
    assert src_stage and src_stage.GetDefaultPrim().GetPath() == Sdf.Path(SRC_ROOT), f"{source} is not the built yam.usd"
    layer = Sdf.Layer.CreateAnonymous("yam-crank.usd")
    layer.TransferContent(src_stage.GetRootLayer())

    edit = Sdf.BatchNamespaceEdit()
    for name in REPLACED_LINKS:
        edit.Add(Sdf.NamespaceEdit.Remove(Sdf.Path(f"{SRC_ROOT}/{name}")))
        edit.Add(Sdf.NamespaceEdit.Remove(Sdf.Path(f"{SRC_ROOT}/joints/{name}")))
    for child in list(layer.GetPrimAtPath(f"{SRC_ROOT}/{YamRobot.FLANGE_LINK}").nameChildren):
        edit.Add(Sdf.NamespaceEdit.Remove(child.path))
    edit.Add(Sdf.NamespaceEdit.Rename(Sdf.Path(SRC_ROOT), C.MODEL))
    assert layer.Apply(edit), "namespace edit batch was rejected"
    remap_dependents(layer, {SRC_ROOT: DST_ROOT})
    drop_dangling_targets(layer)
    assert not stale_paths(layer, SRC_ROOT + "/"), "paths still pointing into the YAMLab root"
    layer.defaultPrim = C.MODEL
    looks = f"{DST_ROOT}/Looks"
    looks_spec = Sdf.CreatePrimInLayer(layer, looks)
    looks_spec.typeName = "Scope"
    looks_spec.specifier = Sdf.SpecifierDef
    materials = copy_materials(src_stage.GetRootLayer(), layer, looks)

    stage = Usd.Stage.Open(layer)
    flange_path = f"{DST_ROOT}/{YamRobot.FLANGE_LINK}"
    flange = stage.GetPrimAtPath(flange_path)
    flange_world = _link_world(stage, flange_path)

    def place_visuals(link_path, body, childclass, pose_prefix=None):
        for i, a in enumerate(mj.geoms(body, childclass, "visual")):
            assert a.get("type") == "mesh", f"visual geom in {body.get('name')} is not a mesh"
            pose = _pose(a.get("pos"), a.get("quat"))
            if pose_prefix is not None:
                pose = _compose(pose, pose_prefix)
            author_visual_mesh(stage, link_path, a["mesh"], mj.mesh_files[a["mesh"]], pose,
                               UsdShade.Material(stage.GetPrimAtPath(materials[a.get("material", "black")])))

    def place_collisions(link_path, body, childclass):
        n = 0
        for a, pose in mj.collision_geoms_recursive(body, childclass):
            size = [float(v) for v in a["size"].split()]
            if a.get("type", "sphere") == "sphere" and size[0] < MIN_COLLISION_RADIUS:
                continue  # MuJoCo contact points, not shapes
            author_collision(stage, link_path, f"{os.path.basename(link_path)}_col_{n}", a, pose)
            n += 1
        return n

    # --- link_6: the crank housing --------------------------------------------------------------
    for attr in ("physics:centerOfMass", "physics:diagonalInertia", "physics:mass", "physics:principalAxes"):
        if flange.HasAttribute(attr):
            flange.RemoveProperty(attr)
    author_mass(flange, mj.inertial(flange_body))
    place_visuals(flange_path, flange_body, flange_class)
    n_flange_col = place_collisions(flange_path, flange_body, flange_class)
    assert n_flange_col >= 3, f"expected the MJCF's three link_6 capsules, authored {n_flange_col}"

    # --- fingers -----------------------------------------------------------------------------
    for finger, body_name in C.MJCF_FINGER_BODIES.items():
        body = mj.bodies[body_name]
        cc = mj.body_childclass[body_name]
        local = mj.body_pose(body)
        link_path = f"{DST_ROOT}/{finger}"
        link = UsdGeom.Xform.Define(stage, link_path).GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(link)
        _set_ops(link, matrix=_compose(local, flange_world))
        author_mass(link, mj.inertial(body))
        place_visuals(link_path, body, cc)
        n_col = place_collisions(link_path, body, cc)
        assert n_col >= 5, f"{finger}: expected the MJCF's capsule/box pads, authored {n_col}"

        joint = UsdPhysics.PrismaticJoint.Define(stage, f"{DST_ROOT}/joints/{finger}")
        joint.CreateBody0Rel().SetTargets([Sdf.Path(flange_path)])
        joint.CreateBody1Rel().SetTargets([Sdf.Path(link_path)])
        q = local.ExtractRotationQuat()
        joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*[float(v) for v in local.ExtractTranslation()]))
        joint.CreateLocalRot0Attr().Set(Gf.Quatf(float(q.GetReal()), Gf.Vec3f(*[float(v) for v in q.GetImaginary()])))
        joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
        joint.CreateAxisAttr().Set(UsdGeom.Tokens.z)  # the MJCF slide axis (0 0 1) in the finger body
        lo, hi = C.FINGER_LIMITS[finger]
        joint.CreateLowerLimitAttr().Set(lo)
        joint.CreateUpperLimitAttr().Set(hi)
        drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "linear")
        drive.CreateTypeAttr().Set("force")
        drive.CreateMaxForceAttr().Set(C.EFFORT_LIMITS[finger])
        drive.CreateStiffnessAttr().Set(0.0)  # OmniGibson writes the controller's isaac_kp/kd at load
        drive.CreateDampingAttr().Set(0.0)
        drive.CreateTargetPositionAttr().Set(0.0)
        armature = mj.attrs(body.find("joint"), cc).get("armature")
        if armature is not None:
            joint.GetPrim().CreateAttribute("physxJoint:armature", Sdf.ValueTypeNames.Float).Set(float(armature))

    # --- wrist D405 + optical frame ------------------------------------------------------------
    d405_local = Gf.Matrix4d(1.0)
    d405_local.SetTransform(Gf.Rotation(Gf.Quatd(*[C.D405_BODY_QUAT_WXYZ[0]], Gf.Vec3d(*C.D405_BODY_QUAT_WXYZ[1:])).GetNormalized()),
                            Gf.Vec3d(*C.D405_BODY_POSITION))
    d405_path = f"{DST_ROOT}/camera_d405"
    d405 = UsdGeom.Xform.Define(stage, d405_path).GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(d405)
    UsdPhysics.MassAPI.Apply(d405).GetMassAttr().Set(0.001)
    d405_world = _compose(d405_local, flange_world)
    _set_ops(d405, matrix=d405_world)
    author_visual_mesh(stage, d405_path, "d405", os.path.join(mj.meshdir, C.D405_MESH), Gf.Matrix4d(1.0),
                       UsdShade.Material(stage.GetPrimAtPath(materials["camera_housing"])))
    author_collision(stage, d405_path, "camera_d405_col_0", {"type": "sphere", "size": str(C.D405_COLLISION_RADIUS)}, Gf.Matrix4d(1.0))
    author_fixed_joint(stage, "camera_d405", flange_path, d405_path, d405_local)

    frame_local = Gf.Matrix4d(1.0)
    frame_local.SetTransform(Gf.Rotation(Gf.Quatd(C.CAMERA_FRAME_QUAT_WXYZ[0], Gf.Vec3d(*C.CAMERA_FRAME_QUAT_WXYZ[1:]))),
                             Gf.Vec3d(*C.CAMERA_FRAME_POSITION))
    frame_path = f"{DST_ROOT}/camera_frame"
    frame = UsdGeom.Xform.Define(stage, frame_path).GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(frame)
    UsdPhysics.MassAPI.Apply(frame).GetMassAttr().Set(0.001)
    _set_ops(frame, matrix=_compose(frame_local, d405_world))
    author_fixed_joint(stage, "camera_frame", d405_path, frame_path, frame_local)

    # The Camera prim must be a direct child of a link for OmniGibson to find it: compose the MJCF camera
    # chain into link_6 and check it against the spec's transcription.
    cam_local = Gf.Matrix4d(1.0)
    cam_local.SetTransform(Gf.Rotation(Gf.Quatd(C.MUJOCO_CAMERA_QUAT_WXYZ[0], Gf.Vec3d(*C.MUJOCO_CAMERA_QUAT_WXYZ[1:]))), Gf.Vec3d(0, 0, 0))
    cam_in_flange = _compose(cam_local, _compose(frame_local, d405_local))
    cam_t = cam_in_flange.ExtractTranslation()
    cam_q = cam_in_flange.ExtractRotationQuat()
    if cam_q.GetReal() < 0 or (abs(cam_q.GetReal()) < 1e-6 and cam_q.GetImaginary()[2] < 0):
        cam_q = -cam_q
    spec_q = Gf.Quatd(C.WRIST_CAMERA_QUAT_WXYZ[0], Gf.Vec3d(*C.WRIST_CAMERA_QUAT_WXYZ[1:]))
    assert Gf.IsClose(cam_t, Gf.Vec3d(*C.WRIST_CAMERA_POSITION), 1e-3), f"camera position {cam_t} != spec {C.WRIST_CAMERA_POSITION}"
    assert abs(abs(Gf.Dot(cam_q.GetImaginary(), spec_q.GetImaginary()) + cam_q.GetReal() * spec_q.GetReal()) - 1.0) < 1e-4, \
        f"camera orientation {cam_q} != spec {C.WRIST_CAMERA_QUAT_WXYZ}"
    cam = UsdGeom.Camera.Define(stage, f"{flange_path}/{C.WRIST_CAMERA_PRIM}")
    w, h = C.RENDER_RESOLUTION
    cam.GetFocalLengthAttr().Set(C.wrist_camera_focal_length())
    cam.GetHorizontalApertureAttr().Set(C.WRIST_CAMERA_HORIZONTAL_APERTURE)
    cam.GetVerticalApertureAttr().Set(C.WRIST_CAMERA_HORIZONTAL_APERTURE * h / w)
    cam.GetClippingRangeAttr().Set(Gf.Vec2f(*C.WRIST_CAMERA_CLIPPING_RANGE))
    _set_ops(cam.GetPrim(), translate=Gf.Vec3d(*C.WRIST_CAMERA_POSITION), orient=spec_q)

    # --- eef frame at ABC's grasp_site ---------------------------------------------------------------
    tcp_local = Gf.Matrix4d(1.0)
    tcp_local.SetTranslate(Gf.Vec3d(*C.TCP_IN_FLANGE))
    eef_path = f"{DST_ROOT}/{C.EEF_LINK}"
    eef = UsdGeom.Xform.Define(stage, eef_path).GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(eef)
    UsdPhysics.MassAPI.Apply(eef).GetMassAttr().Set(0.001)
    _set_ops(eef, matrix=_compose(tcp_local, flange_world))
    author_fixed_joint(stage, C.EEF_LINK, flange_path, eef_path, tcp_local)

    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    stage.SetDefaultPrim(stage.GetPrimAtPath(DST_ROOT))
    layer.comment = ("realm/robots/yam/yam.usd with the I2RT crank gripper, wrist D405 and TCP from ABC's yam.xml, "
                     "built by scripts/build_yam_crank_usd.py -- see realm/robots/yam/PROVENANCE")
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
    if [p.GetName() for p in stage.GetPseudoRoot().GetChildren()] != [C.MODEL]:
        problems.append(f"stray root prims: {[p.GetName() for p in stage.GetPseudoRoot().GetChildren()]}")
    for op in ("xformOp:translate", "xformOp:orient", "xformOp:scale"):
        if root.GetAttribute(op).Get() is None:
            problems.append(f"root has no {op}")

    expected_links = {*C.ARM_LINKS, *C.FINGER_LINKS, *C.FIXED_CAMERA_LINKS, *C.VIRTUAL_LINKS, C.FRAME_LINK}
    xform_children = {c.GetName() for c in root.GetChildren() if c.GetTypeName() == "Xform"}
    if xform_children != expected_links:
        problems.append(f"root Xform children: missing {sorted(expected_links - xform_children)}, "
                        f"extra {sorted(xform_children - expected_links)}")
    for name in expected_links:
        prim = root.GetChild(name)
        if not prim or not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            problems.append(f"{name} is not a rigid body")
            continue
        if prim.GetAttribute("xformOp:scale").Get() is None:
            problems.append(f"{name} has no xformOp:scale")
        # collision prims must be DIRECT children of their link (OmniGibson's CoM composition rule), and the
        # authored primitive shapes must carry purpose=guide (OmniGibson only hides the types it classifies)
        for p in Usd.PrimRange(prim):
            if p.HasAPI(UsdPhysics.CollisionAPI) and p.GetParent() != prim:
                problems.append(f"{p.GetPath()} is a nested collision prim")
            if p.HasAPI(UsdPhysics.CollisionAPI) and p.GetTypeName() in ("Capsule", "Cube", "Sphere") \
                    and UsdGeom.Imageable(p).GetPurposeAttr().Get() != UsdGeom.Tokens.guide:
                problems.append(f"{p.GetPath()} ({p.GetTypeName()}) would render: purpose is not guide")

    joints = {p.GetName(): p for p in Usd.PrimRange(root) if "Joint" in p.GetTypeName()}
    child_links = set()
    for j in joints.values():
        for rel in ("physics:body0", "physics:body1"):
            targets = j.GetRelationship(rel).GetTargets()
            if len(targets) != 1 or not stage.GetPrimAtPath(targets[0]):
                problems.append(f"{j.GetName()}.{rel} -> {[t.pathString for t in targets]} does not resolve")
        b1 = j.GetRelationship("physics:body1").GetTargets()
        if b1:
            child_links.add(b1[0].name)
    if xform_children - child_links != {C.BASE_LINK}:
        problems.append(f"root-link inference would find {sorted(xform_children - child_links)}")

    for name in (*C.ARM_JOINTS, *C.FINGER_JOINTS):
        j = joints.get(name)
        if j is None:
            problems.append(f"joint {name} missing")
            continue
        kind = "linear" if name in C.FINGER_JOINTS else "angular"
        max_force = UsdPhysics.DriveAPI.Get(j, kind).GetMaxForceAttr().Get()
        if max_force is None or abs(max_force - C.EFFORT_LIMITS[name]) > 1e-6:
            problems.append(f"{name} drive maxForce {max_force} != {C.EFFORT_LIMITS[name]}")
    finger_limits = {}
    for name in C.FINGER_JOINTS:
        j = joints.get(name)
        if j is None:
            continue
        lim = (float(j.GetAttribute("physics:lowerLimit").Get()), float(j.GetAttribute("physics:upperLimit").Get()))
        finger_limits[name] = tuple(round(v, 5) for v in lim)
        if max(abs(a - b) for a, b in zip(lim, C.FINGER_LIMITS[name])) > 1e-6:  # float32 in the file
            problems.append(f"{name} limits {lim} != spec {C.FINGER_LIMITS[name]}")
        i = C.FINGER_JOINTS.index(name)
        if not (lim[0] - 1e-6 <= C.finger_closed_qpos()[i] <= lim[1] + 1e-6
                and lim[0] - 1e-6 <= C.finger_open_qpos()[i] <= lim[1] + 1e-6):
            problems.append(f"{name}: open/closed positions outside the joint limits {lim}")
        if j.GetRelationship("physics:body0").GetTargets()[0].name != C.FLANGE_LINK:
            problems.append(f"{name} is not attached to {C.FLANGE_LINK}")
        if not [p for p in Usd.PrimRange(root.GetChild(name)) if p.GetTypeName() == "Mesh"]:
            problems.append(f"{name} has no visual mesh")
        if len([p for p in root.GetChild(name).GetChildren() if p.HasAPI(UsdPhysics.CollisionAPI)]) < 5:
            problems.append(f"{name} has fewer collision shapes than the MJCF pads")

    flange = root.GetChild(C.FLANGE_LINK)
    if flange:
        if not [p for p in Usd.PrimRange(flange) if p.GetTypeName() == "Mesh"]:
            problems.append(f"{C.FLANGE_LINK} lost its visual meshes")
        if len([p for p in flange.GetChildren() if p.HasAPI(UsdPhysics.CollisionAPI)]) < 3:
            problems.append(f"{C.FLANGE_LINK} is missing the MJCF capsule collisions")

    eef = root.GetChild(C.EEF_LINK)
    tcp = None
    if eef and any(p.GetTypeName() in ("Mesh", "Cube", "Cylinder", "Sphere", "Capsule") for p in Usd.PrimRange(eef)):
        problems.append(f"{C.EEF_LINK} carries geometry; OmniGibson makes the eef link invisible")
    ej = joints.get(C.EEF_LINK)
    if ej is None or ej.GetTypeName() != "PhysicsFixedJoint":
        problems.append(f"no fixed joint {C.EEF_LINK}")
    else:
        tcp = tuple(round(float(v), 4) for v in ej.GetAttribute("physics:localPos0").Get())
        if max(abs(a - b) for a, b in zip(tcp, C.TCP_IN_FLANGE)) > 1e-4:
            problems.append(f"{C.EEF_LINK} at {tcp}, spec TCP {C.TCP_IN_FLANGE}")

    cam = stage.GetPrimAtPath(f"{DST_ROOT}/{C.WRIST_CAMERA_LINK}/{C.WRIST_CAMERA_PRIM}")
    if not cam or cam.GetTypeName() != "Camera":
        problems.append("wrist camera prim missing")
    else:
        t = cam.GetAttribute("xformOp:translate").Get()
        if t is None or max(abs(a - b) for a, b in zip(t, C.WRIST_CAMERA_POSITION)) > 1e-6:
            problems.append(f"wrist camera translate {t} != {C.WRIST_CAMERA_POSITION}")
        if abs(cam.GetAttribute("clippingRange").Get()[0] - C.WRIST_CAMERA_CLIPPING_RANGE[0]) > 1e-6:
            problems.append("wrist camera near plane is not the spec's")
    for name in C.FIXED_CAMERA_LINKS:
        if name not in joints or joints[name].GetTypeName() != "PhysicsFixedJoint":
            problems.append(f"no fixed joint {name}")

    stale = stale_paths(stage.GetRootLayer(), SRC_ROOT + "/")
    if stale:
        problems.append(f"stale YAMLab paths: {stale[:5]}")
    frame_bbox = verify_frame(root, joints, C, problems)

    summary = {
        "frame_bbox_in_mount_m": frame_bbox,
        "links": sorted(xform_children),
        "joints": sorted(joints),
        "finger_limits": finger_limits,
        "finger_open_qpos": C.finger_open_qpos(),
        "finger_closed_qpos": C.finger_closed_qpos(),
        "tcp_in_flange_frame_m": tcp,
        "wrist_camera": cam.GetPath().pathString if cam else None,
    }
    return problems, summary


def write_provenance(source, mjcf_path, output):
    try:
        commit = subprocess.check_output(["git", "-C", os.path.dirname(mjcf_path), "rev-parse", "HEAD"],
                                         text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        commit = "unknown"
    mj = Mjcf(mjcf_path)
    lines = [
        "yam_crank.usd",
        f"  source: {os.path.relpath(source, REPO_ROOT)} (the built YAMLab arm above) + ABC's MJCF",
        f"  source sha256: {sha256(source)}",
        f"  mjcf: {mjcf_path}",
        f"  mjcf sha256: {sha256(mjcf_path)}",
        f"  abc commit: {commit}",
        f"  output sha256: {sha256(output)}",
        "  built by scripts/build_yam_crank_usd.py: link_6 housing, both fingers (visuals, capsule/box collision",
        "  pads, inertials, prismatic joints), the wrist D405 links and camera, and the eef frame re-authored from",
        "  the MJCF; the six arm links are YAMLab's. Numbers from realm/robots/yam.py::YamCrankRobot.",
    ]
    for name in sorted(mj.mesh_files):
        lines.append(f"  {os.path.basename(mj.mesh_files[name])} sha256: {sha256(mj.mesh_files[name])}")
    d405 = os.path.join(mj.meshdir, C.D405_MESH)
    lines.append(f"  {C.D405_MESH} sha256: {sha256(d405)}")
    replace_provenance_section(OUT_PROVENANCE, "yam_crank.usd", lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", default=SOURCE_USD, help="the built single-arm yam.usd")
    parser.add_argument("--mjcf", help="ABC's i2rt_yam/yam.xml (required unless --verify-only)")
    parser.add_argument("--output", default=OUT_USD)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    if not args.verify_only:
        assert args.mjcf, "--mjcf is required to build"
        build(args.source, args.mjcf, args.output)
        write_provenance(args.source, args.mjcf, args.output)
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
