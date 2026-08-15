"""Repairing the simulator's global object-init queue after a sibling scene's removal.

Adding an object appends it to the GLOBAL ``og.sim._objects_to_initialize``
(``Simulator._post_import_object``), and ``og.sim.play()`` initializes everything on that queue via
``_non_physics_step()``. Stock OmniGibson 3.9.1's ``Simulator._pre_remove_object()`` prunes that
queue by NAME ALONE, and object names are unique per SCENE rather than per simulator -- so in a
vector env, where every member is built from the same task config and all members' perturbations run
inside ONE stopped window, member 1's ``remove_object("corkscrew")`` pops MEMBER 0's freshly added
corkscrew instead. The last member is always fine, which is why the failure reads as "every scene but
the last one". Single-env never hit it: with one scene there is no sibling to collide with, and each
perturbation's own ``play()`` drains the queue before the next thing runs.

One wrong pop, two symmetric repairs:

    (a) a LIVE object was knocked off the queue and would never be initialized
    (b) the REMOVED object kept its slot and would be initialized after its prim was deleted

FIXED UPSTREAM 2026-08-14: OG-lite's ``_pre_remove_object`` matches on IDENTITY, so against that fork
this module finds nothing and prints nothing. KEPT ANYWAY, as a net rather than a workaround, because
the fix does not travel with the image -- ``scripts/clara/interactive/rr`` defaults to ``MODE=stock``
and ``MODE=stockfix`` binds only ``scenes/scene_base.py``, so both still run the stock
``simulator.py``. The full account, that argument, and the measurement table
(``t9_vbpose_nostopplay.py --num_envs 2 --resets 3``: 4 re-queue warnings per perturbation against
stock, 0 against OG-lite, both PASS either way) are in ``docs/vector_env/PERTURBATIONS.md`` §2-3.

Must run BEFORE the shared ``og.sim.play()``, never after: ``play()`` initializes the queue and THEN
calls ``update()`` on every object's states, and ``update()`` asserts the state is initialized -- so
an evicted object makes ``play()`` itself raise "Cannot update uninitialized state." before any later
repair could run. Re-queueing while still stopped lets ``play()`` do the initialization itself, in
the right order.
"""
import omnigibson as og
import omnigibson.lazy as lazy


def repair_init_queue(envs):
    """Undo the damage a SIBLING member's remove_object() did to the sim's global init queue.

    Returns the objects re-queued, so the caller can assert they came up initialized after its
    play(). Restricted to @envs' own scenes, so anything queued by code outside this vector env is
    left strictly alone.
    """
    our_scenes = {id(env.omnigibson_env.scene) for env in envs}

    # (b) first, so the queue is clean before (a) reads it.
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

    # (a) Anything still queued is not an orphan -- it is about to be initialized normally.
    # (After a play() the queue should already be empty; this only guards against a future caller.)
    queued = {id(obj) for obj in og.sim._objects_to_initialize}
    orphans = [
        obj
        for env in envs
        for obj in env.omnigibson_env.scene.objects
        if not obj.initialized and id(obj) not in queued
    ]
    if not orphans:
        return []

    # Warning rather than info: OmniGibson pins the root logger to WARNING, so og.log.info() would
    # never be seen -- and papering over an upstream bug is genuinely warning-worthy.
    og.log.warning(
        "Re-queueing %d object(s) evicted from the sim init queue by a sibling scene: %s"
        % (len(orphans), ", ".join(f"scene{obj.scene.idx}/{obj.name}" for obj in orphans))
    )
    # Just re-queue. The caller invokes this while the sim is STOPPED, so the following
    # og.sim.play() performs the initialization itself via _non_physics_step(). Calling
    # _non_physics_step() here instead would assert, because the sim is not playing yet; and
    # og.Environment.post_play_load() would be far too much -- it also reloads the
    # observation/action spaces, rebases the scene's initial file and calls reset().
    og.sim._objects_to_initialize.extend(orphans)
    return orphans


def _is_dead(obj):
    """Is this queued object one that has been removed from its scene?

    Deliberately NOT "is there a prim at obj.prim_path": replace_obj re-creates the replacement at
    the SAME relative prim path, so the path is occupied again a moment later and says nothing about
    which instance owns it. (Tried; it silently disabled the whole repair and the crash came straight
    back.) Identity against the registry is what distinguishes them.
    """
    registered = obj.scene.object_registry("name", obj.name)
    if registered is obj:
        return False        # the live object, about to be initialized normally
    if registered is not None:
        return True         # a same-named REPLACEMENT holds the name; this one is the corpse
    # Registered under its name by nothing at all. Either it was removed without a replacement, or
    # it was never registered -- scene.add_object(..., register=False), which OmniGibson uses for
    # particle system templates and which must NOT be dropped. The stage tells them apart: a removed
    # object with no replacement leaves its prim path empty.
    return not lazy.isaacsim.core.utils.prims.is_prim_path_valid(obj.prim_path)
