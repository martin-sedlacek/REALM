
import omnigibson as og
import omnigibson.lazy as lazy


def repair_init_queue(envs):

    our_scenes = {id(env.omnigibson_env.scene) for env in envs}

    stale = [
        obj
        for obj in og.sim._objects_to_initialize
        if id(obj.scene) in our_scenes and _is_dead(obj)
    ]
    if stale:
        og.log.warning(
            "Dropping %d removed object(s) that a sibling scene's eviction left on the sim init "
            "queue: %s"
            % (len(stale), ", ".join(f"scene{obj.scene.idx}/{obj.name}" for obj in stale))
        )
        stale_ids = {id(obj) for obj in stale}
        og.sim._objects_to_initialize = [
            obj for obj in og.sim._objects_to_initialize if id(obj) not in stale_ids
        ]

    queued = {id(obj) for obj in og.sim._objects_to_initialize}
    orphans = [
        obj
        for env in envs
        for obj in env.omnigibson_env.scene.objects
        if not obj.initialized and id(obj) not in queued
    ]
    if not orphans:
        return []

    og.log.warning(
        "Re-queueing %d object(s) evicted from the sim init queue by a sibling scene: %s"
        % (len(orphans), ", ".join(f"scene{obj.scene.idx}/{obj.name}" for obj in orphans))
    )
    og.sim._objects_to_initialize.extend(orphans)
    return orphans


def _is_dead(obj):

    registered = obj.scene.object_registry("name", obj.name)
    if registered is obj:
        return False        # the live object, about to be initialized normally
    if registered is not None:
        return True         # a same-named REPLACEMENT holds the name; this one is the corpse
    return not lazy.isaacsim.core.utils.prims.is_prim_path_valid(obj.prim_path)
