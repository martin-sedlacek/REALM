#!/usr/bin/env python
"""Static (no-Isaac, no-GPU) structural dump of a USD asset's render-relevant attributes.

WHY STATIC FIRST. The question "why is `impact_drawer`'s cabinet physically present but not drawn"
has at least four candidate mechanisms, and two of them are decidable from the authored file alone,
on a login node, in seconds -- no allocation, no 5-minute Isaac boot:

  * geometry not authored / no points          -> readable here
  * authored `visibility = invisible`          -> readable here
  * authored `purpose` != default/render       -> readable here
  * material binding target missing or broken  -> partially readable here (does the Shader prim
                                                  exist, what is its `info:*:sourceAsset`)

What is NOT decidable here is what OmniGibson does to the prims at runtime (MakeInvisible, purpose
rewrites, material re-resolution). That needs the live stage. Running this first means the live
probe only has to test what the file cannot answer.

Usage:
    apptainer run <sif> python scripts/debug_probes/cabinet_usd_static.py <asset.usd> [--json out.json]
"""

import argparse
import json
import sys
from collections import Counter

from pxr import Usd, UsdGeom, UsdShade, Sdf


def describe(stage, prim):
    """Every render-relevant authored fact about one prim."""
    rec = {
        "path": str(prim.GetPath()),
        "type": prim.GetTypeName(),
        "active": bool(prim.IsActive()),
        "instance": bool(prim.IsInstance()),
        "instance_proxy": bool(prim.IsInstanceProxy()),
        "specifier": str(prim.GetSpecifier()),
    }
    img = UsdGeom.Imageable(prim)
    if img:
        # AUTHORED visibility, not computed: computed visibility on an unloaded/uncomposed stage
        # tells you about the file's defaults, which is not the interesting signal. The interesting
        # signal is "did the author write `invisible` anywhere on this subtree".
        vis_attr = img.GetVisibilityAttr()
        rec["vis_authored"] = bool(vis_attr and vis_attr.HasAuthoredValue())
        rec["vis"] = str(vis_attr.Get()) if vis_attr else None
        # ComputeVisibility walks ancestors, so it catches pruning by an ancestor.
        try:
            rec["vis_computed"] = str(img.ComputeVisibility())
        except Exception as e:  # noqa: BLE001
            rec["vis_computed"] = f"<err {type(e).__name__}>"
        p_attr = img.GetPurposeAttr()
        rec["purpose_authored"] = bool(p_attr and p_attr.HasAuthoredValue())
        rec["purpose"] = str(p_attr.Get()) if p_attr else None
        try:
            rec["purpose_computed"] = str(img.ComputePurpose())
        except Exception as e:  # noqa: BLE001
            rec["purpose_computed"] = f"<err {type(e).__name__}>"

    if prim.GetTypeName() == "Mesh":
        mesh = UsdGeom.Mesh(prim)
        pts = mesh.GetPointsAttr().Get()
        rec["n_points"] = len(pts) if pts is not None else 0
        fvc = mesh.GetFaceVertexCountsAttr().Get()
        rec["n_faces"] = len(fvc) if fvc is not None else 0
        # A mesh OmniGibson treats as collision-only is usually tagged; record whatever is there.
        for name in ("physics:collisionEnabled", "primvars:doNotCastShadows"):
            a = prim.GetAttribute(name)
            if a and a.HasAuthoredValue():
                rec[name] = a.Get()
        rec["api_schemas"] = [str(s) for s in prim.GetAppliedSchemas()]

        # Material binding -- the direct/collection binding and whether the target prim EXISTS.
        bapi = UsdShade.MaterialBindingAPI(prim)
        direct = bapi.GetDirectBinding()
        tgt = direct.GetMaterialPath()
        rec["bound_material"] = str(tgt) if tgt else None
        if tgt:
            mprim = stage.GetPrimAtPath(tgt)
            rec["bound_material_exists"] = bool(mprim and mprim.IsValid())
    return rec


def material_report(stage):
    """Every Material on the stage, its surface source, and whether that source is resolvable.

    This is the half of the material question that IS static: OmniGibson's
    "no known shader file ... using MaterialPrim as a fallback" is emitted by its own
    `MaterialPrim.get_material()` subclass dispatch, which keys off the shader prim's
    `info:mdl:sourceAsset` / `info:id`. If that attribute names an .mdl the file itself does not
    carry, the fallback is expected and the reason is right here.
    """
    out = []
    for prim in stage.Traverse(Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate)):
        if prim.GetTypeName() != "Material":
            continue
        mat = UsdShade.Material(prim)
        rec = {"path": str(prim.GetPath()), "shaders": []}
        surf = mat.GetSurfaceOutput()
        rec["has_surface_output"] = bool(surf)
        if surf:
            srcs = surf.GetConnectedSources()
            rec["surface_connected"] = bool(srcs and srcs[0])
        for child in Usd.PrimRange(prim, Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate)):
            if child.GetTypeName() != "Shader":
                continue
            sh = UsdShade.Shader(child)
            srec = {"path": str(child.GetPath())}
            iid = sh.GetIdAttr()
            srec["info:id"] = iid.Get() if iid and iid.HasAuthoredValue() else None
            for impl in ("mdl", "glslfx", "osl"):
                a = child.GetAttribute(f"info:{impl}:sourceAsset")
                if a and a.HasAuthoredValue():
                    v = a.Get()
                    srec[f"info:{impl}:sourceAsset"] = str(v.path) if v is not None else None
                    srec[f"info:{impl}:sourceAsset:resolved"] = (
                        str(v.resolvedPath) if v is not None else None
                    )
                sid = child.GetAttribute(f"info:{impl}:sourceAsset:subIdentifier")
                if sid and sid.HasAuthoredValue():
                    srec[f"info:{impl}:subIdentifier"] = sid.Get()
            srec["impl_source"] = str(sh.GetImplementationSource()) if sh else None
            # Texture inputs and whether each file resolves on THIS host.
            tex = []
            for inp in sh.GetInputs():
                val = inp.Get()
                if isinstance(val, Sdf.AssetPath):
                    tex.append({
                        "input": inp.GetBaseName(),
                        "path": val.path,
                        "resolved": val.resolvedPath or None,
                    })
            if tex:
                srec["textures"] = tex
            rec["shaders"].append(srec)
        out.append(rec)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("asset")
    ap.add_argument("--json")
    ap.add_argument("--max-print", type=int, default=400)
    args = ap.parse_args()

    stage = Usd.Stage.Open(args.asset)
    if stage is None:
        print(f"FAIL: could not open {args.asset}")
        return 2

    prims = list(stage.Traverse(Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate)))
    print(f"asset            : {args.asset}")
    print(f"default prim     : {stage.GetDefaultPrim().GetPath() if stage.GetDefaultPrim() else None}")
    print(f"up axis          : {UsdGeom.GetStageUpAxis(stage)}")
    print(f"metres per unit  : {UsdGeom.GetStageMetersPerUnit(stage)}")
    print(f"prims (w/ proxy) : {len(prims)}")

    recs = [describe(stage, p) for p in prims]

    by_type = Counter(r["type"] for r in recs)
    print("\n--- prim types ---")
    for t, n in by_type.most_common():
        print(f"  {n:5d}  {t or '<untyped>'}")

    # THE TWO STATIC VERDICTS.
    inv = [r for r in recs if r.get("vis") == "invisible"]
    print(f"\n--- authored visibility == invisible: {len(inv)} ---")
    for r in inv[: args.max_print]:
        print(f"  {r['path']}  ({r['type']})")

    vis_computed_inv = [r for r in recs if r.get("vis_computed") == "invisible"]
    print(f"\n--- COMPUTED visibility == invisible (incl. ancestor pruning): {len(vis_computed_inv)} ---")
    for r in vis_computed_inv[: args.max_print]:
        print(f"  {r['path']}  ({r['type']})")

    purposes = Counter(r.get("purpose_computed") for r in recs if "purpose_computed" in r)
    print("\n--- computed purpose histogram ---")
    for p, n in purposes.most_common():
        print(f"  {n:5d}  {p}")
    odd_purpose = [
        r for r in recs
        if r.get("purpose_computed") not in (None, "default", "render")
    ]
    print(f"  non-render/default purpose prims: {len(odd_purpose)}")
    for r in odd_purpose[: args.max_print]:
        print(f"    {r['path']}  purpose={r.get('purpose_computed')}  ({r['type']})")

    meshes = [r for r in recs if r["type"] == "Mesh"]
    empty = [r for r in meshes if not r.get("n_points")]
    print(f"\n--- meshes: {len(meshes)}, of which EMPTY (0 points): {len(empty)} ---")
    for r in empty[: args.max_print]:
        print(f"  {r['path']}")

    # Which meshes are visual (i.e. would be drawn) vs collision-tagged.
    unbound = [r for r in meshes if not r.get("bound_material")]
    broken = [r for r in meshes if r.get("bound_material") and not r.get("bound_material_exists")]
    print(f"\n--- meshes with NO bound material: {len(unbound)} ---")
    for r in unbound[: 40]:
        print(f"  {r['path']}")
    print(f"--- meshes whose bound material prim DOES NOT EXIST: {len(broken)} ---")
    for r in broken[: 40]:
        print(f"  {r['path']} -> {r['bound_material']}")

    bindings = Counter(r.get("bound_material") for r in meshes)
    print("\n--- material binding histogram (meshes) ---")
    for m, n in bindings.most_common():
        print(f"  {n:5d}  {m}")

    mats = material_report(stage)
    print(f"\n--- materials: {len(mats)} ---")
    for m in mats:
        print(f"  {m['path']}  surface_output={m.get('has_surface_output')} "
              f"connected={m.get('surface_connected')}")
        for sh in m["shaders"]:
            print(f"      shader {sh['path']}")
            for k, v in sh.items():
                if k == "path" or k == "textures":
                    continue
                print(f"        {k} = {v}")
            for t in sh.get("textures", []):
                ok = "OK " if t["resolved"] else "MISSING"
                print(f"        tex[{t['input']}] {ok} {t['path']}  -> {t['resolved']}")

    # Full subtree listing, shallow-first, so the hierarchy is legible.
    print("\n--- hierarchy (type / vis / purpose / points) ---")
    for r in recs[: args.max_print]:
        depth = str(r["path"]).count("/") - 1
        extra = []
        if r.get("vis_authored"):
            extra.append(f"vis={r['vis']}")
        if r.get("purpose_authored"):
            extra.append(f"purpose={r['purpose']}")
        if "n_points" in r:
            extra.append(f"pts={r['n_points']} faces={r.get('n_faces')}")
        if r.get("bound_material"):
            extra.append(f"mat={r['bound_material'].rsplit('/', 1)[-1]}")
        if not r["active"]:
            extra.append("INACTIVE")
        print(f"  {'  ' * depth}{str(r['path']).rsplit('/', 1)[-1]}  [{r['type'] or '-'}]  "
              + " ".join(extra))
    if len(recs) > args.max_print:
        print(f"  ... {len(recs) - args.max_print} more (raise --max-print)")

    if args.json:
        with open(args.json, "w") as f:
            json.dump({"prims": recs, "materials": mats}, f, indent=1)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
