from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING

import omnigibson as og
import omnigibson.lazy as lazy

if TYPE_CHECKING:
    from realm.environments.env_dynamic import RealmEnvironmentDynamic

#: Intensity draw (lux): spans dim indoor lighting to studio-bright.
LIGHT_INTENSITY_RANGE = (20000, 750000)

#: Color noise: per-channel normal around a warm white (8-bit RGB), then normalised to [0, 1].
LIGHT_COLOR_MEAN = (255, 214, 170)
LIGHT_COLOR_STD = 15


def v_light(env: "RealmEnvironmentDynamic", intensity=None) -> None:
    """Set every scene light to one shared random intensity and a slightly noised warm color.

    Mutates only the scene's light prims (via USD writes); nothing on @env changes. Raises if the
    scene exposes no light prims, rather than silently leaving the scene unperturbed.
    """
    if intensity is None:
        intensity = np.random.uniform(*LIGHT_INTENSITY_RANGE)

    def find_lights_recursive(obj): # TODO: move the search to new scene instantiation, pointless to call it everytime unless we are swapping scene
        lights = []
        if "light" in obj.name:
            lights.append(obj)

        if hasattr(obj, "_links"):
            for link in obj._links.values():
                lights.extend(find_lights_recursive(link))

        return lights

    all_lights = []
    for obj in env.omnigibson_env.scene.objects:
        all_lights.extend(find_lights_recursive(obj))

    color = np.random.normal(loc=np.array(LIGHT_COLOR_MEAN), scale=LIGHT_COLOR_STD, size=(3,))
    color = np.clip(color, 0, 255).astype(float) / 255.0

    # Collect the actual light prims under each candidate link.
    #
    # This used to build the path by hand as "/World/scene_0" + <relative link path> + "/light_0".
    # In OG 3.9.1 the light sits one level deeper -- ".../<link>/lights/light_0" -- so every lookup
    # returned an invalid prim, the `continue` below swallowed it, and V-LIGHT silently did nothing
    # while still passing the integrity test. Searching the link's subtree for prims that actually
    # carry `inputs:intensity` avoids depending on that layout at all, and also drops the hardcoded
    # scene index (which was already flagged as wrong for vectorized envs).
    light_prims = []
    for light_link in all_lights:
        link_prim = lazy.omni.isaac.core.utils.prims.get_prim_at_path(light_link.prim_path)
        if link_prim is None or not link_prim.IsValid():
            # The recursive search also returns links that do not contain a light; skip those.
            continue
        for prim in lazy.pxr.Usd.PrimRange(link_prim):
            if prim.HasAttribute("inputs:intensity"):
                light_prims.append(prim)

    # Fail loudly rather than returning a silently unperturbed scene -- a no-op here would quietly
    # turn every V-LIGHT evaluation into a duplicate of the Default condition.
    if not light_prims:
        raise RuntimeError(
            f"V-LIGHT found {len(all_lights)} light link(s) but no light prims carrying "
            "'inputs:intensity' underneath them -- the scene's light prim layout has changed."
        )

    # OG 3.9.1 (Fabric Scene Delegate) requires USD writes to be inside this context, and the
    # context must not be nested, so every light is written in one block.
    with og.sim.editing_usd():
        for light_prim in light_prims:
            light_prim.GetAttribute("inputs:intensity").Set(intensity)
            if light_prim.HasAttribute("inputs:color"):
                light_prim.GetAttribute("inputs:color").Set(lazy.pxr.Gf.Vec3f(*color))
