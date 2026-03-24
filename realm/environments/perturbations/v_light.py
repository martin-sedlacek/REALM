from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING

import omnigibson.lazy as lazy

if TYPE_CHECKING:
    from realm.environments.env_dynamic import RealmEnvironmentDynamic


def v_light(env: "RealmEnvironmentDynamic", intensity=None) -> None:
    if intensity is None:
        intensity = np.random.uniform(20000, 750000)

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

    col_mean = np.array([255, 214, 170])
    col_std = 15
    color = np.random.normal(loc=col_mean, scale=col_std, size=(3,))
    color = np.clip(color, 0, 255).astype(float) / 255.0

    world_path = "/World/scene_0" # TODO: is this always the case? what about vectorized envs
    for light in all_lights:
        light_prim_path = world_path + light._relative_prim_path + "/light_0" # TODO: ^^^
        light_prim = lazy.omni.isaac.core.utils.prims.get_prim_at_path(light_prim_path)
        if light_prim is None or not light_prim.IsValid(): # the recursive search also takes links that do not contain the light object, these are skipped here
            continue

        light_prim.GetAttribute("inputs:intensity").Set(intensity)
        light_prim.GetAttribute("inputs:color").Set(lazy.pxr.Gf.Vec3f(*color))
