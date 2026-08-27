"""Robot-link contact queries against the OmniGibson 3.9.1 contact API.

Before 3.9.1, REALM read collisions per link with ``RigidPrim.contact_list()``, which returned one
entry per contact carrying an ``.impulse`` field, and discarded any contact whose impulse norm fell
below 1e-3. That threshold is what kept resting contacts -- the arm sitting on its mount, an object
at rest on the table -- from being counted as collisions.

3.9.1 removed ``contact_list()``. Contacts now come from ``RigidContactAPI``, which aggregates them
per scene into a *boolean* matrix using an ``impulses != 0`` test, i.e. strictly looser than the old
threshold: every resting contact registers. This module restores the old semantics on top of the new
API -- one batched query for all queried links, then a magnitude filter -- so that the collision
counters stay comparable to the pre-3.9.1 numbers.

The impulse magnitudes themselves are not retained by the aggregation (``update_contact_cache``
reduces them to booleans and clears its pending buffers), so they are re-read from the live contact
view at query time.
"""
from realm.config.shared import DEFAULT_IMPULSE_THRESHOLD

from collections import defaultdict

import torch as th

import omnigibson as og
from omnigibson.utils.usd_utils import RigidContactAPI

# Matches the pre-3.9.1 filter in RealmEnvironmentBase.check_collisions: contacts weaker than this
# are resting contacts, not collisions.


def _live_impulse_matrix(scene_idx):
    """Return the (R, C, 3) contact-force matrix for ``scene_idx``, or None if unavailable.

    Reaches into RigidContactAPI's private view because 3.9.1 exposes no public accessor for
    contact *magnitudes* -- the public surface (``get_contact_pairs``, ``is_in_contact``) is boolean
    only. Guarded so that a future rename degrades to unfiltered contacts instead of crashing.
    """
    view = getattr(RigidContactAPI, "_CONTACT_VIEW", {}).get(scene_idx)
    if view is None:
        return None
    try:
        return view.get_contact_force_matrix(dt=og.sim.get_physics_dt())
    except Exception:
        # The physx tensor view is invalid between a scene load and the first physics step.
        return None


def get_impulse_contacts(scene_idx, links, impulse_threshold=DEFAULT_IMPULSE_THRESHOLD):
    """Contacts on ``links`` whose impulse magnitude reaches ``impulse_threshold``.

    Args:
        scene_idx (int): scene index the links belong to (``robot.scene.idx``).
        links (Iterable[RigidPrim]): links to query. Typically every robot link except the root.
        impulse_threshold (float): minimum contact impulse norm (N*s) for a contact to count.
            Contacts below it are resting contacts and are dropped.

    Returns:
        dict[str, set[str]]: maps each queried link's prim path to the prim paths it is in contact
            with. Links with no qualifying contact are absent rather than mapped to an empty set,
            so callers should use ``.get(path, ())``.
    """
    links = list(links)
    if not links:
        return {}

    # One batched query for every link, rather than the per-link call the old contact_list() forced.
    pairs = RigidContactAPI.get_contact_pairs(
        scene_idx=scene_idx,
        query_set=set(links),
        with_set=None,
        current_only=True,
    )
    if not pairs:
        return {}

    contacts = defaultdict(set)

    impulses = _live_impulse_matrix(scene_idx)
    if impulses is None:
        # No magnitudes available: report the boolean contacts unfiltered. Over-reporting a resting
        # contact is the safer failure direction for a collision counter than dropping a real hit.
        for query_path, other_path in pairs:
            contacts[query_path].add(other_path)
        return dict(contacts)

    row_map = getattr(RigidContactAPI, "_PATH_TO_ROW_IDX", {}).get(scene_idx, {})
    col_map = getattr(RigidContactAPI, "_PATH_TO_COL_IDX", {}).get(scene_idx, {})

    for query_path, other_path in pairs:
        row, col = row_map.get(query_path), col_map.get(other_path)
        if row is None or col is None:
            # Path not in the contact matrix (e.g. a kinematic-only body that has no row). Keep the
            # contact: it was reported, we just cannot weigh it.
            contacts[query_path].add(other_path)
            continue
        if th.linalg.norm(impulses[row, col]).item() >= impulse_threshold:
            contacts[query_path].add(other_path)

    return dict(contacts)
