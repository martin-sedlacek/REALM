"""Name -> implementation table for REALM's perturbations.

A run applies the perturbation its command line names: ``realm/eval.py`` and ``realm/vector_eval.py``
each index their own ``SUPPORTED_PERTURBATIONS`` list by id and hand the resulting name to the
environment. The mapping from that name to a callable belongs here rather than inside the
environment, which only has to dispatch.

Every entry takes the environment and mutates it in place. V-AUG maps to the same no-op as Default
because it has no scene-side effect at all: it distorts the images on the way out of ``get_obs``,
which ``RealmEnvironmentDynamic`` does itself.

REALM does not implement V-OBJ, VB-ISC, VS-PROP, SB-ADV or SB-SMO.
"""
from realm.environments.perturbations.b_hobj import b_hobj
from realm.environments.perturbations.default import default
from realm.environments.perturbations.sb_noun import sb_noun
from realm.environments.perturbations.sb_vrb import sb_vrb
from realm.environments.perturbations.semantic import s_aff, s_int, s_lang, s_mo, s_prop
from realm.environments.perturbations.v_light import v_light
from realm.environments.perturbations.v_sc import v_sc
from realm.environments.perturbations.v_view import v_view
from realm.environments.perturbations.vb_mobj import vb_mobj
from realm.environments.perturbations.vb_pose import vb_pose
from realm.environments.perturbations.vsb_nobj import vsb_nobj

PERTURBATION_FNS = {
    'Default':  default,
    "V-AUG":    default,
    "V-VIEW":   v_view,
    "V-SC":     v_sc,
    "V-LIGHT":  v_light,
    "S-PROP":   s_prop,
    "S-LANG":   s_lang,
    "S-MO":     s_mo,
    "S-AFF":    s_aff,
    "S-INT":    s_int,
    "B-HOBJ":   b_hobj,
    "SB-NOUN":  sb_noun,
    "SB-VRB":   sb_vrb,
    "VB-POSE":  vb_pose,
    "VB-MOBJ":  vb_mobj,
    "VSB-NOBJ": vsb_nobj,
}
