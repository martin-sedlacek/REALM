"""Does the render-only twin of the `panda_link8` gripper adapter actually SHOW UP?

The robolab asset carries a collider cylinder at `/panda/panda_link8/gripper_adapter` -- the pad
between the end of the arm and the start of the gripper. OmniGibson hides every collision geom by
setting `purpose = "guide"` (`RigidPrim.update_meshes`), so the standard trick is to keep a
collider-free duplicate alongside it for rendering. `gripper_adapter_01` is meant to be that twin.

This probe answers three questions in one boot, without believing any of them in advance:

  1. **How does OmniGibson classify each geom under every gripper link?** It reads the live
     `RigidPrim.visual_meshes` / `.collision_meshes` dicts and each `GeomPrim`'s `purpose` and
     `visible`, so "OmniGibson skipped it" and "OmniGibson kept it but you cannot see it" are
     distinguishable rather than conflated.
  2. **Does it contribute pixels?** It parks a camera on the wrist from four azimuths around
     `panda_link8`'s own axis and renders. Then it hides ONLY the twin, re-renders the same
     viewpoints, and reports the per-view pixel delta. A view whose delta is zero is a view in
     which the twin drew nothing -- which is the difference between "invisible" and "occluded",
     and it is not something an eyeball on a single frame can settle.
  3. **What does it weigh?** It records `mass` / `density` / `center_of_mass` / the collision-mesh
     set of every robot link, so a later run of the same probe against a modified asset is a
     physics diff rather than a promise.

Everything is written to JSON *before* anything is printed, because a crash in Isaac's teardown
can still exit 0 and a report assembled from stdout would then be silently truncated.

    ./scripts/clara/interactive/rr python -u scripts/debug_probes/link8_adapter_render.py \
        --robot DROID_robolab_v2 --out /logs/link8_adapter/before --tag before

`--variant-usd <path>` redirects the robolab v2 asset at load time (the shipped file is never
written to), which is how the before/after pair is produced from one checkout.
"""
import argparse
import json
import os

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--robot", default="DROID_robolab_v2")
ap.add_argument("--task-cfg", default="REALM_DROID10/put_green_block_into_bowl/default.yaml")
ap.add_argument("--out", default="/logs/link8_adapter")
ap.add_argument("--tag", default="run")
ap.add_argument("--variant-usd", default=None,
                help="Load this USD instead of droid_robolab_v2.usd (leaves the shipped file alone)")
ap.add_argument("--link", default="panda_link8")
ap.add_argument("--twin", default="gripper_adapter_01", help="prim name of the render-only twin")
ap.add_argument("--collider", default="gripper_adapter", help="prim name of the collider cylinder")
ap.add_argument("--cam-dist", type=float, default=0.16)
ap.add_argument("--cam-elev", type=float, default=12.0, help="degrees above the link8 xy-plane")
ap.add_argument("--focal", type=float, default=8.0, help="camera focal length, mm; wider = more context")
ap.add_argument("--no-ab", action="store_true", help="skip the hide-the-twin A/B pass")
args = ap.parse_args()

os.makedirs(args.out, exist_ok=True)
RESULT = {"tag": args.tag, "robot": args.robot, "variant_usd": args.variant_usd,
          "classification": {}, "links": {}, "views": {}, "ab": {}, "verdict": {}}
JSON_PATH = os.path.join(args.out, f"{args.tag}_link8_adapter.json")


def dump():
    """Write the JSON now. Called after every stage so a later hang still leaves evidence."""
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
from scipy.spatial.transform import Rotation as Rot  # noqa: E402

import omnigibson as og  # noqa: E402
import omnigibson.lazy as lazy  # noqa: E402

from realm.sim_config import set_sim_config  # noqa: E402
from realm.environments.env_dynamic import RealmEnvironmentDynamic  # noqa: E402

if args.variant_usd:
    assert os.path.exists(args.variant_usd), f"no variant USD at {args.variant_usd}"
    from omnigibson.robots.robot import Robot  # noqa: E402

    _orig_usd_path = Robot.usd_path.fget

    def _patched_usd_path(self):
        p = _orig_usd_path(self)
        if "droid_robolab_v2" in str(p):
            print(f"[variant] usd_path {p} -> {args.variant_usd}", flush=True)
            return args.variant_usd
        return p

    Robot.usd_path = property(_patched_usd_path)

hdr(f"BUILD  robot={args.robot}  tag={args.tag}  variant={args.variant_usd}")
set_sim_config(robot=args.robot)
env = RealmEnvironmentDynamic(
    config_path="/app/realm/config", task_cfg_path=args.task_cfg, perturbations=["Default"],
    multi_view=False, no_rendering=False, rendering_mode="rt", robot=args.robot,
)
robot = env.omnigibson_env.robots[0]
RESULT["usd_path_used"] = str(robot.usd_path)
dump()

# ------------------------------------------------------------------ 1. how OG classified each geom
hdr("CLASSIFICATION -- what OmniGibson made of every geom under every link")


def safe(fn, default="<err>"):
    """Never let one bad attribute cost a whole boot."""
    try:
        return fn()
    except Exception as e:
        return f"<{type(e).__name__}: {e}>" if default == "<err>" else default


for lname, link in robot.links.items():
    entry = {"visual_meshes": {}, "collision_meshes": {}}
    for kind in ("visual_meshes", "collision_meshes"):
        for mname, mesh in getattr(link, kind, {}).items():
            prim = mesh.prim
            entry[kind][mname] = {
                "prim_path": safe(lambda: str(prim.GetPrimPath())),
                "type": safe(lambda: str(prim.GetTypeName())),
                "schemas": safe(lambda: list(prim.GetAppliedSchemas())),
                "purpose": safe(lambda: str(mesh.purpose)),
                "visible": safe(lambda: bool(mesh.visible)),
            }
    RESULT["classification"][lname] = entry

l8 = RESULT["classification"].get(args.link, {})
print(f"{args.link}:")
for kind in ("visual_meshes", "collision_meshes"):
    for mname, m in l8.get(kind, {}).items():
        print(f"  [{kind:16s}] {mname:24s} type={str(m['type']):10s} purpose={str(m['purpose']):8s} "
              f"visible={m['visible']} schemas={m['schemas']}")

# The dicts above are OmniGibson's view. This is the STAGE's view of the same link: a geom missing
# from both dicts but present here would mean OG dropped it, which is a different bug from OG
# keeping it and the renderer not drawing it.
raw = {}
for child in og.sim.stage.GetPrimAtPath(f"{robot.prim_path}/{args.link}").GetChildren():
    im = lazy.pxr.UsdGeom.Imageable(child)
    raw[child.GetName()] = {
        "type": safe(lambda: str(child.GetTypeName())),
        "schemas": safe(lambda: list(child.GetAppliedSchemas())),
        "active": safe(lambda: bool(child.IsActive())),
        "computed_visibility": safe(lambda: str(im.ComputeVisibility())) if im else None,
        "computed_purpose": safe(lambda: str(im.ComputePurpose())) if im else None,
    }
RESULT["raw_stage_children"] = raw
print(f"\nstage children of {args.link}:")
for n, r in raw.items():
    print(f"  {n:24s} type={str(r['type']):20s} vis={r['computed_visibility']} "
          f"purpose={r['computed_purpose']} schemas={r['schemas']}")
dump()

# ------------------------------------------------------------------ 2. physics fingerprint per link
hdr("PHYSICS FINGERPRINT -- mass / com / collision set, per link")
for lname, link in robot.links.items():
    rec = {"collision_mesh_names": sorted(getattr(link, "collision_meshes", {}).keys()),
           "visual_mesh_names": sorted(getattr(link, "visual_meshes", {}).keys())}
    for attr in ("mass", "density"):
        try:
            rec[attr] = float(_np(getattr(link, attr)))
        except Exception as e:
            rec[attr] = f"<{type(e).__name__}>"
    try:
        rec["center_of_mass"] = [round(v, 12) for v in _np(link.center_of_mass).tolist()]
    except Exception as e:
        rec["center_of_mass"] = f"<{type(e).__name__}>"
    prim = link.prim
    for a in ("physics:mass", "physics:density", "physics:diagonalInertia",
              "physics:centerOfMass", "physics:principalAxes"):
        at = prim.GetAttribute(a)
        rec[a] = str(at.Get()) if at and at.HasAuthoredValue() else None
    rec["schemas"] = list(prim.GetAppliedSchemas())
    RESULT["links"][lname] = rec
    if lname in (args.link, "base_link"):
        print(f"  {lname}: mass={rec.get('mass')} com={rec.get('center_of_mass')} "
              f"cols={rec['collision_mesh_names']}")
dump()

# ------------------------------------------------------------------ 3. render the wrist
hdr("RENDER -- four azimuths around the link8 axis")
try:
    CAM = env.omnigibson_env.external_sensors["external_sensor0"]
except Exception as e:
    print(f"  [warn] no external_sensor0 ({e!r}); falling back to the viewer camera")
    CAM = og.sim.viewer_camera
try:
    CAM.focal_length = args.focal
    print(f"  focal_length set to {args.focal}")
except Exception as e:
    print(f"  [warn] could not set focal_length: {e!r}")

link8 = robot.links[args.link]
p8_t, q8_t = link8.get_position_orientation()
p8, R8 = _np(p8_t), Rot.from_quat(_np(q8_t))
# The adapter sits at +9 mm along link8's own z, which is the flange normal pointing at the gripper.
TARGET_LOCAL = np.array([0.0, 0.0, 0.009])
target = p8 + R8.apply(TARGET_LOCAL)
RESULT["link8_world"] = {"pos": p8.tolist(), "quat": _np(q8_t).tolist(), "target": target.tolist()}

AZIMUTHS = [0, 90, 180, 270]
elev = np.deg2rad(args.cam_elev)


def aim(az_deg):
    """Camera on a cone around link8's local z, looking in at `target`. USD cameras look along -z."""
    a = np.deg2rad(az_deg)
    d_local = np.array([np.cos(a) * np.cos(elev), np.sin(a) * np.cos(elev), np.sin(elev)])
    z_c = R8.apply(d_local)                     # camera +z points from target back to the camera
    z_c /= np.linalg.norm(z_c)
    up = R8.apply(np.array([0.0, 0.0, 1.0]))    # link8 +z (toward the gripper) points UP in frame
    x_c = np.cross(up, z_c)
    x_c /= np.linalg.norm(x_c)
    y_c = np.cross(z_c, x_c)
    quat = Rot.from_matrix(np.stack([x_c, y_c, z_c], axis=1)).as_quat()
    CAM.set_position_orientation(
        th.tensor(target + z_c * args.cam_dist, dtype=th.float32),
        th.tensor(quat, dtype=th.float32), "world")


def grab(az_deg, suffix):
    aim(az_deg)
    for _ in range(6):          # RT needs a few frames to converge before the buffer is worth reading
        og.sim.render()
    obs = CAM.get_obs()
    obs = obs[0] if isinstance(obs, tuple) else obs
    rgb = _np(obs["rgb"])[..., :3]
    if rgb.dtype.kind == "f" or rgb.max() <= 1.0:
        rgb = (rgb * 255.0).clip(0, 255)
    rgb = rgb.astype(np.uint8)
    path = os.path.join(args.out, f"{args.tag}_az{az_deg:03d}_{suffix}.png")
    Image.fromarray(rgb).save(path)
    return rgb, path


base_frames = {}
for az in AZIMUTHS:
    rgb, path = grab(az, "base")
    base_frames[az] = rgb
    RESULT["views"][str(az)] = {"base_png": path, "shape": list(rgb.shape)}
    print(f"  az={az:3d} -> {path}")
dump()

# ------------------------------------------------------------------ 4. A/B: hide ONLY the twin
if not args.no_ab:
    hdr("A/B -- hide the twin, re-render, diff. Nonzero delta == the twin was drawing those pixels.")
    stage = og.sim.stage
    twin_path = None
    for cand in (f"{robot.prim_path}/{args.link}/{args.twin}",):
        if stage.GetPrimAtPath(cand):
            twin_path = cand
    RESULT["ab"]["twin_prim_path"] = twin_path
    if twin_path is None:
        print(f"  [warn] no twin prim at {robot.prim_path}/{args.link}/{args.twin} -- skipping A/B")
    else:
        with og.sim.editing_usd():
            lazy.pxr.UsdGeom.Imageable(stage.GetPrimAtPath(twin_path)).MakeInvisible()
        for az in AZIMUTHS:
            rgb, path = grab(az, "twinhidden")
            d = np.abs(base_frames[az].astype(np.int32) - rgb.astype(np.int32))
            n_px = int((d.sum(axis=2) > 8).sum())
            # A heat map of WHERE the twin drew, so "the cylinder is visible" can be checked
            # against the twin's own footprint rather than against whatever else is in frame.
            dm = (d.sum(axis=2) > 8).astype(np.uint8) * 255
            diff_path = os.path.join(args.out, f"{args.tag}_az{az:03d}_twindiff.png")
            Image.fromarray(np.stack([dm, dm // 3, dm // 3], axis=2)).save(diff_path)
            RESULT["views"][str(az)].update({
                "twindiff_png": diff_path,
                "twinhidden_png": path,
                "changed_pixels": n_px,
                "changed_frac": round(n_px / float(d.shape[0] * d.shape[1]), 6),
                "max_abs_delta": int(d.max()),
                "mean_abs_delta": round(float(d.mean()), 4),
            })
            print(f"  az={az:3d}  changed_px={n_px:7d}  max|d|={int(d.max()):3d}  "
                  f"mean|d|={float(d.mean()):.4f}")
        with og.sim.editing_usd():
            lazy.pxr.UsdGeom.Imageable(stage.GetPrimAtPath(twin_path)).MakeVisible()
    dump()

# ------------------------------------------------------------------ 5. verdict
hdr("VERDICT")
cls8 = RESULT["classification"].get(args.link, {})
twin_cls = ("visual" if args.twin in cls8.get("visual_meshes", {}) else
            "collision" if args.twin in cls8.get("collision_meshes", {}) else "ABSENT")
coll_cls = ("visual" if args.collider in cls8.get("visual_meshes", {}) else
            "collision" if args.collider in cls8.get("collision_meshes", {}) else "ABSENT")
twin_rec = (cls8.get("visual_meshes", {}) or {}).get(args.twin) or \
           (cls8.get("collision_meshes", {}) or {}).get(args.twin)
total_changed = sum(v.get("changed_pixels", 0) for v in RESULT["views"].values())
RESULT["verdict"] = {
    "twin_classified_as": twin_cls,
    "collider_classified_as": coll_cls,
    "twin_purpose": (twin_rec or {}).get("purpose"),
    "twin_visible_flag": (twin_rec or {}).get("visible"),
    "twin_changed_pixels_total": total_changed,
    # NOT a render verdict, and deliberately not named like one. This probe's frames go through
    # og.sim.render() + CAM.get_obs(), which on 2026-08-17 returned a blank buffer whose speck noise
    # read as "it renders"; and with --no-ab there is no A/B at all, so a bare pixel count would say
    # "False" for a twin that renders perfectly well. Use link8_adapter_ladder.py for pixels. What
    # THIS probe is authoritative about is the classification and the physics fingerprint above.
    "PIXEL_EVIDENCE_VALID": False if args.no_ab else None,
    "NOTE": "render verdicts come from link8_adapter_ladder.py, not from this probe",
}
dump()
for k, v in RESULT["verdict"].items():
    print(f"  {k} = {v}")
print(f"\nVERDICT_LINE tag={args.tag} twin_class={twin_cls} twin_purpose={(twin_rec or {}).get('purpose')} "
      f"twin_visible={(twin_rec or {}).get('visible')} changed_px={total_changed} "
      f"(pixel counts here are NOT a render verdict -- see link8_adapter_ladder.py)")
print(f"JSON {JSON_PATH}")

# Isaac's teardown can hang for minutes and hold the GPU. The JSON and the PNGs are already on
# disk, so leave immediately rather than waiting on it.
try:
    og.shutdown()
except Exception as e:
    print(f"[shutdown] {e!r}")
os._exit(0)
