"""Why the `panda_link8` gripper-adapter twin does not render -- the whole ladder in ONE boot.

`link8_adapter_render.py` established that OmniGibson classifies
`/panda/panda_link8/gripper_adapter_01` as a **visual** mesh with `purpose = "default"` -- so the
dupe-and-strip-the-collider half of the trick is already correct on the shipped v2 asset -- and yet
every child of `panda_link8`, including the `contact_frame` Xform and the `panda_hand_joint` that no
mesh code ever touches, computes `visibility = invisible`. `panda_link8` is this robot's **eef
link** (`droid_robolab_v2.yaml`), and `Robot._load_controllers` hides it unconditionally
(`robot.py:1255`, `self._links[self.eef_link_names[arm]].visible = False`). USD visibility is
*pruning*: once an ancestor is `invisible` no descendant can override it, and `purpose` is
orthogonal to it. So `purpose = "render"`, an explicit `visibility`, a material, the `visual__`
rename and a Cylinder->Mesh conversion are all downstream of the prune and cannot work.

This probe boots once and proves that, by walking the ladder in process against a frozen camera:

    control            render again, change nothing          -> the noise floor
    purpose_render     authored `purpose = "render"`         -> measured: no-op
    purpose_default    authored `purpose = "default"`        -> measured: no-op
    purpose_cleared    `purpose` removed                     -> measured: no-op
    twin_makevisible   MakeVisible() on the twin itself      -> works, but only by un-hiding link8
    twin_invisible     MakeInvisible() on the twin           -> the A/B, once the twin is visible
    link8_visible      un-hide the eef LINK                  -> the pad appears
    clone_base         a copy of the twin under `base_link`  -> the pad appears, link8 still hidden

Three things this probe does that the previous one did not, each because getting it wrong produced
a confident wrong answer:

  * **It renders through `env.step()`**, not a bare `og.sim.render()` + `CAM.get_obs()`. The latter
    returned a 99.99%-pure-white buffer with four unique colours -- an unticked annotator, not a
    scene -- and the handful of speck differences between two such buffers read as "the twin
    renders".
  * **Every capture is gated on not being blank** (unique colours, background fraction). A blank
    frame is a hard failure, never a verdict.
  * **Differences are reported as a shape, not just a count**: bounding box and largest connected
    component. A 7.5 cm disc is thousands of contiguous pixels in one compact blob; sampling noise
    is a scatter across the whole frame. The count alone cannot tell those apart.

    ./scripts/clara/interactive/rr python -u scripts/debug_probes/link8_adapter_ladder.py \
        --out /logs/link8_adapter/ladder --tag ladder
"""
import argparse
import json
import os
import traceback

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--robot", default="DROID_robolab_v2")
ap.add_argument("--task-cfg", default="REALM_DROID10/put_green_block_into_bowl/default.yaml")
ap.add_argument("--out", default="/logs/link8_adapter/ladder")
ap.add_argument("--tag", default="ladder")
ap.add_argument("--variant-usd", default=None)
ap.add_argument("--link", default="panda_link8")
ap.add_argument("--twin", default="gripper_adapter_01")
ap.add_argument("--collider", default="gripper_adapter")
ap.add_argument("--gripper-base", default="base_link")
ap.add_argument("--azimuths", nargs="+", type=int, default=[0, 180])
ap.add_argument("--cam-dist", type=float, default=0.20)
ap.add_argument("--cam-elev", type=float, default=10.0)
ap.add_argument("--focal", type=float, default=30.0)
ap.add_argument("--warmup", type=int, default=25, help="hold steps before the first capture")
ap.add_argument("--frames", type=int, default=5, help="frames per capture, median-combined")
ap.add_argument("--twin-path", default=None,
                help="absolute-ish prim path of the render-only twin relative to the robot, e.g. "
                     "'base_link/visual__gripper_adapter' for the FIXED asset")
ap.add_argument("--only", default=None, help="comma-separated subset of condition names")
args = ap.parse_args()

os.makedirs(args.out, exist_ok=True)
JSON_PATH = os.path.join(args.out, f"{args.tag}_ladder.json")
RESULT = {"tag": args.tag, "robot": args.robot, "variant_usd": args.variant_usd,
          "camera": {}, "facts": {}, "conditions": {}, "errors": [], "blank_frames": []}


def dump():
    tmp = JSON_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(RESULT, f, indent=2, default=str)
    os.replace(tmp, JSON_PATH)


def hdr(s):
    print(f"\n{'=' * 100}\n{s}\n{'=' * 100}", flush=True)


def _np(x):
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    return np.asarray(x, dtype=np.float64)


import torch as th  # noqa: E402
from PIL import Image  # noqa: E402
from scipy import ndimage  # noqa: E402
from scipy.spatial.transform import Rotation as Rot  # noqa: E402

import omnigibson as og  # noqa: E402
import omnigibson.lazy as lazy  # noqa: E402

from realm.sim_config import set_sim_config  # noqa: E402
from realm.environments.env_dynamic import RealmEnvironmentDynamic  # noqa: E402

if args.variant_usd:
    assert os.path.exists(args.variant_usd), f"no variant USD at {args.variant_usd}"
    from omnigibson.robots.robot import Robot  # noqa: E402

    _orig = Robot.usd_path.fget
    Robot.usd_path = property(
        lambda self: args.variant_usd if "droid_robolab_v2" in str(_orig(self)) else _orig(self))
    print(f"[variant] -> {args.variant_usd}", flush=True)

hdr(f"BUILD  robot={args.robot} tag={args.tag} variant={args.variant_usd}")
set_sim_config(robot=args.robot)
env = RealmEnvironmentDynamic(
    config_path="/app/realm/config", task_cfg_path=args.task_cfg, perturbations=["Default"],
    multi_view=False, no_rendering=False, rendering_mode="rt", robot=args.robot,
)

# `omnigibson.lazy` only resolves `pxr` once the Kit app is up, which happens inside the env build.
# Binding these at import time raises "module lazy_ has no attribute pxr".
UsdGeom, Gf = lazy.pxr.UsdGeom, lazy.pxr.Gf

robot = env.omnigibson_env.robots[0]
stage = og.sim.stage
LINK_PATH = f"{robot.prim_path}/{args.link}"
TWIN_PATH = (f"{robot.prim_path}/{args.twin_path}" if args.twin_path
             else f"{LINK_PATH}/{args.twin}")
COLL_PATH = f"{LINK_PATH}/{args.collider}"
BASE_PATH = f"{robot.prim_path}/{args.gripper_base}"
CLONE_PATH = f"{BASE_PATH}/visual__{args.twin}_clone"
ARM_Q = np.asarray(env.reset_qpos[:7], dtype=np.float64)
HOLD = np.concatenate([ARM_Q, [-1.0]])          # hold reset_qpos, jaws open

# ---------------------------------------------------------------- the mechanism, stated as facts
hdr("FACTS -- who hid what")


def vis_of(path):
    p = stage.GetPrimAtPath(path)
    if not p:
        return {"exists": False}
    im = UsdGeom.Imageable(p)
    va, pa = p.GetAttribute("visibility"), p.GetAttribute("purpose")
    return {
        "exists": True,
        "authored_visibility": str(va.Get()) if va and va.HasAuthoredValue() else None,
        "computed_visibility": str(im.ComputeVisibility()),
        "authored_purpose": str(pa.Get()) if pa and pa.HasAuthoredValue() else None,
        "computed_purpose": str(im.ComputePurpose()),
        "schemas": list(p.GetAppliedSchemas()),
    }


RESULT["facts"] = {
    "usd_path_used": str(robot.usd_path),
    "arm_names": list(robot.arm_names),
    "eef_link_names": {str(k): str(v) for k, v in robot.eef_link_names.items()},
    "robot_prim": str(robot.prim_path),
    "vis": {p: vis_of(p) for p in (robot.prim_path, LINK_PATH, TWIN_PATH, COLL_PATH, BASE_PATH,
                                   f"{robot.prim_path}/panda_link7")},
}
print(f"  eef_link_names = {RESULT['facts']['eef_link_names']}")
for p, v in RESULT["facts"]["vis"].items():
    print(f"  {p.split('/')[-1]:26s} authored_vis={v.get('authored_visibility')!s:10s} "
          f"computed_vis={v.get('computed_visibility')!s:10s} purpose={v.get('computed_purpose')}")
dump()

# ---------------------------------------------------------------- camera, frozen
hdr("CAMERA")
CAM = env.omnigibson_env.external_sensors["external_sensor0"]
try:
    CAM.focal_length = args.focal
except Exception as e:
    print(f"  [warn] focal_length: {e!r}")
FOCAL = float(_np(CAM.focal_length))
APER = float(_np(CAM.horizontal_aperture))
W, H = int(CAM.image_width), int(CAM.image_height)
print(f"  focal={FOCAL} aperture={APER} image={W}x{H} clip={_np(CAM.clipping_range)}")

link8 = robot.links[args.link]
p8, R8 = _np(link8.get_position_orientation()[0]), Rot.from_quat(_np(link8.get_position_orientation()[1]))
target = p8 + R8.apply(np.array([0.0, 0.0, 0.009]))
CORNERS_W = np.array([p8 + R8.apply(np.array([sx * 0.0375, sy * 0.0375, z]))
                      for sx in (-1, 1) for sy in (-1, 1) for z in (0.0, 0.0184)])
RESULT["camera"] = {"focal": FOCAL, "aperture": APER, "w": W, "h": H,
                    "link8_pos": p8.tolist(), "target": target.tolist()}


def cam_basis(az_deg):
    a, e = np.deg2rad(az_deg), np.deg2rad(args.cam_elev)
    z_c = R8.apply(np.array([np.cos(a) * np.cos(e), np.sin(a) * np.cos(e), np.sin(e)]))
    z_c /= np.linalg.norm(z_c)
    up = R8.apply(np.array([0.0, 0.0, 1.0]))
    x_c = np.cross(up, z_c)
    x_c /= np.linalg.norm(x_c)
    y_c = np.cross(z_c, x_c)
    return target + z_c * args.cam_dist, np.stack([x_c, y_c, z_c], axis=1)


def project(P, C, Rm):
    v = Rm.T @ (P - C)
    if v[2] >= -1e-6:
        return None
    nx = (v[0] / -v[2]) * FOCAL / (APER / 2.0)
    ny = (v[1] / -v[2]) * FOCAL / ((APER * H / W) / 2.0)
    return np.array([(nx + 1.0) / 2.0 * W, (1.0 - ny) / 2.0 * H])


CROPS = {}
for az in args.azimuths:
    C, Rm = cam_basis(az)
    pts = np.array([p for p in (project(P, C, Rm) for P in CORNERS_W) if p is not None])
    m = 45
    CROPS[az] = ((int(max(pts[:, 0].min() - m, 0)), int(max(pts[:, 1].min() - m, 0)),
                  int(min(pts[:, 0].max() + m, W)), int(min(pts[:, 1].max() + m, H)))
                 if len(pts) else (0, 0, W, H))
    print(f"  az={az:3d} geometric adapter crop = {CROPS[az]}")
RESULT["camera"]["crops"] = {str(k): list(v) for k, v in CROPS.items()}
dump()


def blankness(rgb):
    """A real render has thousands of distinct colours. An unticked buffer has a handful."""
    u = len(np.unique(rgb.reshape(-1, 3), axis=0))
    white = float((rgb > 250).all(axis=2).mean())
    return {"unique_colours": int(u), "near_white_frac": round(white, 5),
           "std": round(float(rgb.std()), 3), "BLANK": bool(u < 500 or white > 0.90)}


def capture(az, name):
    """Aim, hold reset_qpos for k frames, return their per-pixel MEDIAN.

    One frame is not usable as evidence here: at `rendering_mode="rt"` two renders of an unchanged
    scene differ on 25% of pixels at a >12 threshold (measured: p50=6, p90=27, p99=97 of the summed
    channel delta). The median over k frames removes almost all of it, and what survives is bounded
    by the `control` condition, which is the same measurement with nothing changed.
    """
    C, Rm = cam_basis(az)
    CAM.set_position_orientation(th.tensor(C, dtype=th.float32),
                                 th.tensor(Rot.from_matrix(Rm).as_quat(), dtype=th.float32), "world")
    frames = []
    for _ in range(args.frames):
        obs = env.step(HOLD, n_render_iterations=2)[0]
        f = _np(obs["external"]["external_sensor0"]["rgb"])[..., :3]
        frames.append(f * 255.0 if f.max() <= 1.0 else f)
    rgb = np.median(np.stack(frames), axis=0).clip(0, 255).astype(np.uint8)
    b = blankness(rgb)
    path = os.path.join(args.out, f"{args.tag}_az{az:03d}_{name}.png")
    Image.fromarray(rgb).save(path)
    if b["BLANK"]:
        RESULT["blank_frames"].append({"name": name, "az": az, **b})
        print(f"  !! BLANK FRAME {name} az={az}: {b}")
    return rgb, path, b


#: Summed-channel delta above which a change is structural rather than sampling noise. The verdict
#: uses THRESH; the others are reported so the choice can be second-guessed from the JSON.
THRESH = 120
LEVELS = (12, 60, 120, 200)


def diff_stats(a, b, az, tag=""):
    d = np.abs(a.astype(np.int32) - b.astype(np.int32)).sum(axis=2)
    x0, y0, x1, y1 = CROPS[az]
    out = {"crop_total_px": int((y1 - y0) * (x1 - x0)),
           "by_threshold": {str(t): {"full": int((d > t).sum()),
                                     "crop": int((d[y0:y1, x0:x1] > t).sum())} for t in LEVELS}}
    mask = d > THRESH
    cm = mask[y0:y1, x0:x1]
    out.update({"thresh": THRESH, "full_changed_px": int(mask.sum()),
                "crop_changed_px": int(cm.sum()),
                "crop_changed_frac": round(float(cm.sum()) / max(cm.size, 1), 4)})
    if cm.any():
        ys, xs = np.nonzero(cm)
        out["crop_diff_bbox"] = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
        lab, n = ndimage.label(cm)
        sizes = ndimage.sum(cm, lab, range(1, n + 1))
        out["n_blobs"], out["largest_blob_px"] = int(n), int(sizes.max())
    else:
        out["crop_diff_bbox"], out["n_blobs"], out["largest_blob_px"] = None, 0, 0
    if tag:
        hm = (cm.astype(np.uint8) * 255)
        Image.fromarray(np.stack([hm, hm // 4, hm // 4], axis=2)).save(
            os.path.join(args.out, f"{args.tag}_az{az:03d}_{tag}_diffmask.png"))
    return out


# ---------------------------------------------------------------- ladder mutations
# editing_usd() refuses to nest and a guard raises on edits outside it, so each helper opens
# exactly one context -- except set_link_visible, which goes through a setter that opens its own.
def set_active(path, on):
    with og.sim.editing_usd():
        stage.GetPrimAtPath(path).SetActive(on)
    return f"{path.split('/')[-1]} SetActive({on})"


def set_purpose(path, value):
    with og.sim.editing_usd():
        UsdGeom.Imageable(stage.GetPrimAtPath(path)).CreatePurposeAttr().Set(value)
    return f"purpose={value} on {path.split('/')[-1]}"


def clear_purpose(path):
    with og.sim.editing_usd():
        p = stage.GetPrimAtPath(path)
        if "purpose" in p.GetPropertyNames():
            p.RemoveProperty("purpose")
    return f"purpose removed from {path.split('/')[-1]}"


def make_visible(path):
    with og.sim.editing_usd():
        UsdGeom.Imageable(stage.GetPrimAtPath(path)).MakeVisible()
    return f"MakeVisible() on {path.split('/')[-1]}"


def make_invisible(path):
    with og.sim.editing_usd():
        UsdGeom.Imageable(stage.GetPrimAtPath(path)).MakeInvisible()
    return f"MakeInvisible() on {path.split('/')[-1]}"


def set_link_visible(on):
    robot.links[args.link].visible = on
    return f"{args.link}.visible = {on}"


def clone_under_base():
    """A copy of the twin parented to the gripper base, at the identical WORLD pose.

    `base_link` is fixed to `panda_link8` by `panda_hand_joint`, so a visual-only prim under it
    tracks the flange exactly -- and `panda_link8` is not touched at all, which is the point.
    """
    xc = UsdGeom.XformCache()
    m_twin = Gf.Matrix4d(xc.GetLocalToWorldTransform(stage.GetPrimAtPath(TWIN_PATH)))
    m_base = Gf.Matrix4d(xc.GetLocalToWorldTransform(stage.GetPrimAtPath(BASE_PATH)))
    m_local = m_twin * m_base.GetInverse()
    t = m_local.ExtractTranslation()
    # Normalise the rows BEFORE extracting the quaternion. `ExtractRotationQuat()` on a matrix that
    # still carries the (0.075, 0.075, 0.018) scale does not return a unit rotation, and the clone
    # then renders as a tilted, faceted slab instead of a disc. Same bug, same fix, as
    # scripts/fix_link8_adapter_visual.py's decompose().
    rows = [Gf.Vec3d(m_local[i][0], m_local[i][1], m_local[i][2]) for i in range(3)]
    s = [r.GetLength() for r in rows]
    n = [rows[i] / s[i] for i in range(3)]
    q = Gf.Matrix4d(Gf.Matrix3d(n[0][0], n[0][1], n[0][2],
                                n[1][0], n[1][1], n[1][2],
                                n[2][0], n[2][1], n[2][2]),
                    Gf.Vec3d(0, 0, 0)).ExtractRotationQuat()
    with og.sim.editing_usd():
        src = stage.GetPrimAtPath(TWIN_PATH)
        dst = stage.DefinePrim(CLONE_PATH, "Cylinder")
        for a in ("radius", "height", "axis", "extent"):
            sa = src.GetAttribute(a)
            dst.CreateAttribute(a, sa.GetTypeName()).Set(sa.Get())
        x = UsdGeom.Xformable(dst)
        x.ClearXformOpOrder()
        x.AddTranslateOp().Set(Gf.Vec3d(t))
        x.AddOrientOp().Set(Gf.Quatf(q.GetReal(), Gf.Vec3f(q.GetImaginary())))
        x.AddScaleOp().Set(Gf.Vec3f(s[0], s[1], s[2]))
    # Prove the reparent is pose-preserving before any pixel is believed.
    xc2 = UsdGeom.XformCache()
    m_new = Gf.Matrix4d(xc2.GetLocalToWorldTransform(stage.GetPrimAtPath(CLONE_PATH)))
    err = max(abs(m_new[i][j] - m_twin[i][j]) for i in range(4) for j in range(4))
    RESULT.setdefault("clone_pose_error", {})["max_abs_matrix_delta"] = err
    return (f"clone at {CLONE_PATH} local_t={tuple(round(v, 6) for v in t)} "
            f"scale={tuple(round(v, 6) for v in s)} world_pose_err={err:.3e}")


# `SetActive(False)` on a prim inside a loaded robot is NOT a safe A/B toggle here: it made
# `og.sim.step()` die in `toggle.py:209` with "'NoneType' object is not subscriptable" on the very
# next step. `MakeInvisible()` is the equivalent test and is local to the prim.
LADDER = [
    ("control", lambda: "no change"),
    ("purpose_render", lambda: set_purpose(TWIN_PATH, "render")),
    ("purpose_default", lambda: set_purpose(TWIN_PATH, "default")),
    ("purpose_cleared", lambda: clear_purpose(TWIN_PATH)),
    ("twin_makevisible", lambda: make_visible(TWIN_PATH)),
    ("twin_invisible", lambda: make_invisible(TWIN_PATH)),
    ("twin_visible_again", lambda: make_visible(TWIN_PATH)),
    ("link8_visible", lambda: set_link_visible(True)),
    ("link8_hidden", lambda: set_link_visible(False)),
    ("clone_base", clone_under_base),
]
if args.only:
    keep = set(args.only.split(","))
    LADDER = [c for c in LADDER if c[0] in keep]

hdr("BASELINE")
for i in range(args.warmup):
    try:
        env.step(HOLD, n_render_iterations=0)
    except Exception as e:
        print(f"  [warn] n_render_iterations=0 rejected ({e!r}); warming up with 1")
        env.step(HOLD, n_render_iterations=1)
baseline, floor = {}, {}
for az in args.azimuths:
    baseline[az], p, b = capture(az, "00_baseline")
    print(f"  az={az} baseline -> {p}  {b}")
RESULT["baseline_blankness"] = {str(az): blankness(baseline[az]) for az in args.azimuths}
dump()
assert not any(RESULT["baseline_blankness"][str(az)]["BLANK"] for az in args.azimuths), (
    "baseline frames are blank -- the camera is not looking at the robot, so NO verdict is valid. "
    f"{RESULT['baseline_blankness']}")

hdr("LADDER")
for name, fn in LADDER:
    try:
        detail = fn()
    except Exception as e:
        RESULT["errors"].append({"condition": name, "error": repr(e), "tb": traceback.format_exc()})
        print(f"  !! {name}: {e!r}")
        dump()
        continue
    rec = {"detail": detail,
           "vis": {p: vis_of(p) for p in (LINK_PATH, TWIN_PATH, CLONE_PATH)}, "views": {}}
    print(f"  -- {name}: {detail}")
    for az in args.azimuths:
        rgb, path, b = capture(az, name)
        vs = diff_stats(baseline[az], rgb, az, tag=name)
        vs.update({"png": path, "blankness": b})
        if name == "control":
            floor[az] = {"crop": max(vs["crop_changed_px"], 1),
                         "blob": max(vs["largest_blob_px"], 1)}
        fl = floor.get(az, {"crop": 1, "blob": 1})
        vs["floor"] = fl
        # The pad is a 7.5 cm disc filling a ~940 px-wide crop: appearing or vanishing is tens of
        # thousands of contiguous pixels. Require BOTH a large crop count and one big connected
        # blob, each an order of magnitude over the measured control, so neither sampling noise nor
        # a thin edge-aliasing rim can pass.
        vs["RENDERS_PAD"] = bool(vs["crop_changed_px"] > max(5000, 10 * fl["crop"])
                                 and vs["largest_blob_px"] > max(3000, 10 * fl["blob"]))
        rec["views"][str(az)] = vs
        print(f"     az={az:3d} crop={vs['crop_changed_px']:7d}/{vs['crop_total_px']:7d} "
              f"full={vs['full_changed_px']:7d} blobs={vs['n_blobs']:5d} "
              f"largest={vs['largest_blob_px']:7d} bbox={vs['crop_diff_bbox']} "
              f"floor={fl} RENDERS_PAD={vs['RENDERS_PAD']}")
    rec["RENDERS_PAD"] = any(v["RENDERS_PAD"] for v in rec["views"].values())
    RESULT["conditions"][name] = rec
    dump()

hdr("SUMMARY")
for name, rec in RESULT["conditions"].items():
    print(f"LADDER_LINE {name:18s} RENDERS_PAD={rec['RENDERS_PAD']!s:5s} :: "
          + " ".join(f"az{az}: crop={v['crop_changed_px']} blob={v['largest_blob_px']}"
                     for az, v in rec["views"].items()))
print(f"BLANK_FRAMES {len(RESULT['blank_frames'])}")
print(f"JSON {JSON_PATH}")
dump()

try:
    og.shutdown()
except Exception as e:
    print(f"[shutdown] {e!r}")
os._exit(0)
