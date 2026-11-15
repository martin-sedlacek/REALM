import omnigibson as og
import omnigibson.lazy as lazy


def debug_world(max_depth=4):
    """List all scenes in /World; for each, print its bbox and every prim with an infinite/NaN bbox."""
    ctx = lazy.omni.usd.get_context()

    def bbox(path):
        bmin, bmax = ctx.compute_path_world_bounding_box(path)
        bmin, bmax = list(bmin), list(bmax)
        bad = any(abs(v) > 1e6 or v != v for v in bmin + bmax) # -- inf or NaN
        return bmin, bmax, bad

    print(f"[WORLD] {len(og.sim.scenes)} scenes", flush=True)
    for s in og.sim.scenes:
        path = s.prim_path
        pos = [round(float(v), 2) for v in s._scene_prim.get_position_orientation()[0]]
        bmin, bmax, bad = bbox(path)
        print(f"[WORLD] {path}  idx={s.idx}  type={type(s).__name__}  position={pos}  "
              f"bbox={'INF' if bad else [round(v, 2) for v in bmin] + [round(v, 2) for v in bmax]}",
              flush=True)

        root = og.sim.stage.GetPrimAtPath(path)
        if not root:
            continue
        n_bad = 0
        for prim in lazy.pxr.Usd.PrimRange(root):
            if prim.GetPath().pathElementCount > max_depth: # -- only the top levels
                continue
            if prim == root:
                continue
            *_, bad = bbox(str(prim.GetPath()))
            if bad:
                n_bad += 1
                print(f"[BBOX]   INF {prim.GetPath()}  type={prim.GetTypeName()}", flush=True)
        print(f"[BBOX] {path}: {n_bad} infinite prim(s) within depth {max_depth}", flush=True)


def inspect_scene_prim(scene, tag, max_depth=5):
    """Print everything under /World/scene_<idx> right now: type, visibility and own bbox of each prim."""
    ctx = lazy.omni.usd.get_context()
    path = f"/World/scene_{scene.idx}"
    root = og.sim.stage.GetPrimAtPath(path)

    def box(p):
        bmin, bmax = ctx.compute_path_world_bounding_box(p)
        bmin, bmax = [float(v) for v in bmin], [float(v) for v in bmax]
        if bmin[0] > bmax[0]:
            return "EMPTY"
        if any(abs(v) > 1e6 for v in bmin + bmax):
            return "HUGE"
        return f"[{bmin[0]:.2f},{bmin[1]:.2f},{bmin[2]:.2f}]..[{bmax[0]:.2f},{bmax[1]:.2f},{bmax[2]:.2f}]"

    print(f"[INSPECT] ===== scene_{scene.idx}: {tag} =====", flush=True)
    print(f"[INSPECT] type={type(scene).__name__}  scene_file={scene.scene_file}", flush=True)
    print(f"[INSPECT] _init_objs ({len(scene._init_objs)}): {list(scene._init_objs.keys())[:30]}", flush=True)
    print(f"[INSPECT] {path}  exists={bool(root)}  bbox={box(path) if root else '-'}", flush=True)
    if not root:
        return
    n = 0
    for p in lazy.pxr.Usd.PrimRange(root):
        if p == root or p.GetPath().pathElementCount > max_depth:
            continue
        n += 1
        vis = lazy.pxr.UsdGeom.Imageable(p).ComputeVisibility() if p.IsA(lazy.pxr.UsdGeom.Imageable) else "-"
        print(f"[INSPECT]   {str(p.GetPath()):70s} type={p.GetTypeName():20s} vis={str(vis):10s} "
              f"bbox={box(str(p.GetPath()))}", flush=True)
    print(f"[INSPECT] {n} prims under {path} (depth <= {max_depth})", flush=True)


def install_scene_load_inspector():
    """Observe OmniGibson's scene loading without changing it: after (or on crash inside)
    Scene._load_scene_prim_with_objects, print what /World/scene_<idx> contained at that moment,
    i.e. what OmniGibson measured to place the scene."""
    from omnigibson.scenes.scene_base import Scene

    if getattr(Scene, "_inspector_installed", False):
        return
    original = Scene._load_scene_prim_with_objects

    def wrapped(self, *args, **kwargs):
        try:
            out = original(self, *args, **kwargs) # -- OmniGibson's code, unchanged
        except Exception:
            inspect_scene_prim(self, "CRASHED inside load (state OmniGibson measured)")
            raise
        inspect_scene_prim(self, f"right after load (state OmniGibson measured), returned edge={out[0]}")
        return out

    Scene._load_scene_prim_with_objects = wrapped
    Scene._inspector_installed = True