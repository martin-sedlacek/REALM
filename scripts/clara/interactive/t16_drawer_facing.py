"""Which way do the cabinet's drawer fronts point, and is that toward the policy camera?

The residual complaint on the og391 port is that the impact_drawer cabinet "shows the viewer its
back". Every numeric check passes -- the cabinet sits at exactly the config quaternion, is upright,
and its drawer joint travels the full 0.300 m -- so the claim rests entirely on which way the drawer
FRONTS face relative to the camera that feeds the policy. That is a geometric question with a
readback answer, and this script is that readback.

The front direction comes from the **handle cylinders**: the three `Cylinder*` meshes under the target
drawer link are its handles, which sit on the front face, so the vector from the link's AABB centre to
their world centroid is the outward front direction. Pure geometry, no assumption about which local
axis means what -- which is the point, because the obvious alternative is wrong:

    A first version of this probe took the drawer link's local +Z, reasoning from the asset that the
    front panel is the `Cube` mesh at local (0.0007, 0.0677, +0.2348) with scale (0.39, 0.09, 0.005)
    -- a plate 0.005 thick along Z -- with the side and bottom walls (`Cube_06/07/08`, scale
    (*, 0.001, 0.48)) extending 0.48 along Z behind it. That reasoning is about the MESH frame, and
    `ObjectCapture` sits between the link and the meshes carrying `xformOp:rotateXYZ = (90, 0, 0)`.
    Measured, the link's own +Z is world (-0.013, -0.003, 0.9999) -- straight up. It reported "FACE
    this camera" both before and after the cabinet was turned round. The link axes are still printed
    as a diagnostic, but no verdict is drawn from them.

Then the actual question: `external_sensor0` is the camera whose RGB becomes the policy's `base_im`
(`realm/inference/utils.py:137-139`); `external_sensor1` exists only under --multi-view. Its pose is
NOT the raw yaml extrinsic -- `_apply_camera_cfg` passes the yaml pose through
`construct_ext_cam_pose_by_name(name, robot_pos, robot_rot)`, so it is robot-relative and the sign of
its world y cannot be read off `camera_extrinsics.yaml`. On task 8 that matters: `CP3`'s yaml
`y = +0.55` lands at world `y = -2.45` once the robot's 180 deg base yaw is applied. Hence: read the
PLACED camera, and report `front . (cam_pos - drawer_centre)`. Positive means the fronts face it.

Builds the members and reports; runs no policy. Does not move any joint, so it is safe to run before
t13's slide test (which does).

    ./scripts/clara/interactive/rr python -u scripts/clara/interactive/t16_drawer_facing.py \
        --num_envs 1 --task_id 8
"""
import argparse

import torch as th

import omnigibson as og
import omnigibson.lazy as lazy

from realm.environments.env_vector import RealmVectorEnvironment
from realm.eval import SUPPORTED_PERTURBATIONS, SUPPORTED_TASKS
from realm.sim_config import set_sim_config


def f3(v):
    return tuple(round(float(x), 4) for x in (v.cpu() if hasattr(v, "cpu") else v))


def quat_to_mat(q):
    """xyzw -> 3x3 rotation matrix. Written out rather than imported so the probe does not depend
    on which of the several quaternion conventions in this stack a helper happens to use."""
    x, y, z, w = (float(c) for c in q)
    return th.tensor([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ], dtype=th.float64)


def angle_between_quats_deg(q1, q2):
    """Geodesic angle between two xyzw quaternions, sign-insensitive (q and -q are one rotation)."""
    a = th.tensor([float(c) for c in q1], dtype=th.float64)
    b = th.tensor([float(c) for c in q2], dtype=th.float64)
    a = a / th.linalg.norm(a)
    b = b / th.linalg.norm(b)
    d = float(th.abs(th.dot(a, b)).clamp(max=1.0))
    return float(th.rad2deg(2 * th.acos(th.tensor(d))))


def world_xform(prim_path):
    """World transform of @prim_path straight off the USD stage (not Fabric).

    Deliberately USD and not `get_world_pose`: the whole family of bugs this asset has produced
    lives in the difference between what is authored and what the runtime believes, so the probe
    reads the authored composition and prints the Fabric value beside it for comparison.
    """
    stage = og.sim.stage
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return None
    cache = lazy.pxr.UsdGeom.XformCache(lazy.pxr.Usd.TimeCode.Default())
    return cache.GetLocalToWorldTransform(prim)


def mat_to_rows(m):
    return [tuple(round(float(v), 5) for v in m.GetRow(i)) for i in range(4)]


def dump_xform_chain(prim_path, label):
    """xformOpOrder, every xformOp, and the composed local/world transform for one prim."""
    stage = og.sim.stage
    prim = stage.GetPrimAtPath(prim_path)
    print(f"    ---- {label}: {prim_path}")
    if not prim.IsValid():
        print("         <invalid prim>")
        return
    xf = lazy.pxr.UsdGeom.Xformable(prim)
    order = xf.GetXformOpOrderAttr().Get()
    print(f"         xformOpOrder = {list(order) if order else None}")
    for name in sorted(n for n in prim.GetPropertyNames() if n.startswith("xformOp")):
        if name == "xformOpOrder":
            continue
        attr = prim.GetAttribute(name)
        print(f"         {name:<42s} = {attr.Get()}")
    local = xf.GetLocalTransformation()
    print(f"         local rows  = {mat_to_rows(local)[:3]}")
    w = world_xform(prim_path)
    if w is not None:
        print(f"         world rows  = {mat_to_rows(w)[:3]}")
        print(f"         world trans = {mat_to_rows(w)[3][:3]}")


def handle_centroid(cabinet, link_name):
    """World centroid of the drawer's handle cylinders, and how many were found."""
    stage = og.sim.stage
    link = cabinet.links[link_name]
    root = stage.GetPrimAtPath(link.prim_path)
    if not root.IsValid():
        return None, 0
    cache = lazy.pxr.UsdGeom.XformCache(lazy.pxr.Usd.TimeCode.Default())
    pts, n = th.zeros(3, dtype=th.float64), 0
    for prim in lazy.pxr.Usd.PrimRange(root):
        if not prim.IsA(lazy.pxr.UsdGeom.Mesh):
            continue
        if not prim.GetName().startswith("Cylinder"):
            continue
        m = cache.GetLocalToWorldTransform(prim)
        t = m.ExtractTranslation()
        pts += th.tensor([float(t[0]), float(t[1]), float(t[2])], dtype=th.float64)
        n += 1
    return (pts / n if n else None), n


def facing_block(env, i):
    cabinet = env.main_objects[0]
    j = env.mo_joint
    link_name = j.body1.split("/")[-1]
    link = cabinet.links[link_name]

    print(f"\n  ===== member {i}: cabinet={cabinet.name!r} prim={cabinet.prim_path} =====")
    print(f"        stage upAxis   = {lazy.pxr.UsdGeom.GetStageUpAxis(og.sim.stage)}")
    print(f"        metersPerUnit  = {lazy.pxr.UsdGeom.GetStageMetersPerUnit(og.sim.stage)}")
    print(f"        target joint   = {j.joint_name!r}  drawer link = {link_name!r}")

    # --- pose, and the config quaternion it is supposed to be at -------------------------------
    pos, ori = cabinet.get_position_orientation()
    cfg_quat = getattr(env, "_t16_cfg_quat", None)
    print(f"        cabinet world  pos={f3(pos)} ori(xyzw)={f3(ori)}")
    if cfg_quat is not None:
        print(f"        config quat    {f3(cfg_quat)}   "
              f"angle_to_config_quat_deg = {angle_between_quats_deg(ori, cfg_quat):.4f}")

    # --- the xformOp chain that produced it -----------------------------------------------------
    print("\n        --- authored transform chain (USD) ---")
    dump_xform_chain(cabinet.prim_path, "entity prim")
    dump_xform_chain(cabinet.root_link.prim_path, "root link")
    if link.prim_path != cabinet.root_link.prim_path:
        dump_xform_chain(link.prim_path, f"drawer link ({link_name})")

    # --- diagnostic only: the drawer LINK frame's axes ------------------------------------------
    # NOT a front-direction estimate. The docstring's reasoning about the front panel being the thin
    # face along local +Z is about the MESH frame, and `ObjectCapture` sits between the link and the
    # meshes carrying `xformOp:rotateXYZ = (90, 0, 0)`, so the link frame is a further Rx(90) away
    # from the mesh frame. Measured: the link's own +Z comes out as world (-0.013, -0.003, 0.9999),
    # i.e. straight up, which is not a front direction at all. Printed because the axes are useful
    # when reading the joint frame, but no FACE / FACE-AWAY verdict is derived from it -- an earlier
    # version did, and reported "FACE this camera" both before and after the cabinet was turned round.
    lw = world_xform(link.prim_path)
    R = th.tensor([[float(lw.GetRow(r)[c]) for c in range(3)] for r in range(3)], dtype=th.float64)
    lo, hi = link.aabb
    centre = ((lo + hi) / 2.0).to(th.float64)
    print(f"\n        --- drawer link frame axes (diagnostic, NOT the front direction) ---")
    for i, name in enumerate("XYZ"):
        print(f"        link local +{name} -> world {f3(R[i] / th.linalg.norm(R[i]))}")
    print(f"        drawer link aabb centre = {f3(centre)} extent={f3(hi - lo)}")

    # --- the front direction: the handle cylinders ----------------------------------------------
    # The three `Cylinder*` meshes under the drawer link are its handles, which are physically on the
    # front face, so the vector from the link's AABB centre to their world centroid IS the outward
    # front direction. This is pure geometry -- no assumption about which local axis means what --
    # which is why it is the estimate the verdict below is built on.
    hc, nh = handle_centroid(cabinet, link_name)
    print(f"\n        --- front direction (handle cylinders) ---")
    if hc is None:
        print("        no Cylinder* meshes found under the drawer link -- CANNOT JUDGE FACING")
        front_axis = None
    else:
        v = hc - centre
        front_axis = v / th.linalg.norm(v)
        print(f"        {nh} handle meshes, world centroid = {f3(hc)}")
        print(f"        centre -> handles = {f3(v)}  unit = {f3(front_axis)}  |v|={float(th.linalg.norm(v)):.4f}")

    # --- the camera that actually feeds the policy ------------------------------------------------
    print(f"\n        --- cameras ---")
    ext = getattr(env.omnigibson_env, "external_sensors", None) or {}
    if not ext:
        print("        env.omnigibson_env.external_sensors is empty -- no external camera placed")
    for name in sorted(ext.keys()):
        sensor = ext[name]
        cpos, cori = sensor.get_position_orientation()
        Rc = quat_to_mat(cori)
        # OmniGibson/USD cameras look down their own -Z.
        fwd = -Rc[:, 2]
        to_cab = centre - th.tensor([float(c) for c in cpos], dtype=th.float64)
        to_cab_u = to_cab / th.linalg.norm(to_cab)
        cam_to = th.tensor([float(c) for c in cpos], dtype=th.float64) - centre
        cam_to_u = cam_to / th.linalg.norm(cam_to)
        role = "POLICY base_im" if name == "external_sensor0" else "second view (--multi-view only)"
        print(f"        {name} [{role}]")
        print(f"            world pos = {f3(cpos)} ori(xyzw)={f3(cori)}")
        print(f"            view dir (-Z) = {f3(fwd)}   aim at cabinet = {f3(to_cab_u)}   "
              f"dot={float(th.dot(fwd, to_cab_u)):+.4f}")
        if front_axis is None:
            print("            front direction unavailable -- no verdict")
            continue
        faces = float(th.dot(front_axis, cam_to_u))
        print(f"            front . (cam - drawer_centre) = {faces:+.4f}  "
              f"=> fronts {'FACE this camera' if faces > 0 else '** FACE AWAY from this camera **'}")

    # --- robot base, since the camera pose is robot-relative -------------------------------------
    robot = env.omnigibson_env.robots[0]
    rpos, rori = robot.get_position_orientation()
    print(f"\n        robot base world pos={f3(rpos)} ori(xyzw)={f3(rori)}")

    # --- joint slide axis in world, read not moved -----------------------------------------------
    jp = og.sim.stage.GetPrimAtPath(j.prim_path) if hasattr(j, "prim_path") else None
    if jp is not None and jp.IsValid():
        ax = jp.GetAttribute("physics:axis")
        lr0 = jp.GetAttribute("physics:localRot0")
        print(f"        joint prim {j.prim_path}")
        print(f"            physics:axis = {ax.Get() if ax else None}  "
              f"localRot0 = {lr0.Get() if lr0 else None}")
        print(f"            lower={float(j.lower_limit):+.4f} upper={float(j.upper_limit):+.4f}")


def main(num_envs, task_id, robot, perturbation):
    set_sim_config(robot=robot)
    task_cfg_path = f"REALM_DROID10/{SUPPORTED_TASKS[task_id]}/default.yaml"
    vec_env = RealmVectorEnvironment(
        num_envs,
        task_cfg_path=task_cfg_path,
        perturbations=[perturbation],
        robot=robot,
    )

    # Stash the config quaternion so facing_block can report the angle to it. Read from the task
    # config the env actually loaded rather than re-parsing the yaml by hand.
    import yaml as _yaml
    import os as _os
    for env in vec_env.envs:
        cfg_file = _os.path.join(env.config_path, "tasks", task_cfg_path)
        try:
            tc = _yaml.load(open(cfg_file), Loader=_yaml.FullLoader)
            env._t16_cfg_quat = tc["main_objects"][0]["orientation"]
        except Exception as e:
            print(f"  (could not read config quat from {cfg_file}: {e})")
            env._t16_cfg_quat = None

    print(f"\n########## t16 drawer facing: task {task_id} "
          f"({SUPPORTED_TASKS[task_id]}), num_envs={num_envs}, pert={perturbation} ##########",
          flush=True)
    og.sim.render()  # flush physx->fabric so the AABB reads are the settled ones
    for i, env in enumerate(vec_env.envs):
        facing_block(env, i)
    print("\n########## t16 done ##########", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--num_envs", type=int, default=1)
    p.add_argument("--task_id", type=int, default=8)
    p.add_argument("--robot", type=str, default="DROID_robolab")
    p.add_argument("--perturbation", type=str, default=SUPPORTED_PERTURBATIONS[0])
    a = p.parse_args()
    main(a.num_envs, a.task_id, a.robot, a.perturbation)
