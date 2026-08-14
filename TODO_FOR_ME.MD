# TODO for Martin

Things parked here because they need a human decision, GUI access, or a machine that can build
container images — not because they are unfinished engineering. Everything else found during the
vectorization work is either fixed or tracked in `docs/vector_env/PERTURBATIONS.md`.

Each item says what was measured, so none of it has to be rediscovered.

---

## 1. Scene config for tasks 2 and 6 — check in the GUI

Both are `Benevolence_1_int` entries in `realm/config/scenes/scenes.yaml`. **These may simply differ
from the 1.1.1 configuration rather than be a port bug** — worth eyeballing in the GUI before
changing anything.

### Task 6 `stack_cubes` (`Dining_Table`) renders nothing but sky

Both cameras show only the sky dome: no table, no cubes, no room, no floor; the wrist camera sees
the Robotiq fingers against clouds. **Every artifact and metric check passes**, and the joint
metrics are in family with task 0, so this is invisible to anything except looking at the frames.

Measured from `Benevolence_1_int_best.json`:

| | y range |
|---|---|
| spawn box (`Dining_Table`) | **-2.55 .. -2.15** |
| walls | -1.22 .. 9.29 |
| floors | 1.96 .. 8.48 |
| the scene's actual furniture | `breakfast_table` at (1.92, 4.95), countertops at y 6.2-8.8 |

So the rig spawns roughly 1 m beyond the nearest wall with no floor within 4 m. Reproduced at
`--num_envs 1`, so it is **not** a vectorization problem.

Implication worth weighing: any `stack_cubes` result collected on this port was a robot waving at
the sky.

### Task 2 `rotate_marker` (`Kitchen_Counter`) — external camera unusable

Spawn box y ∈ [-0.80, -0.40] is inside the wall envelope but still short of any floor. The scene
renders an interior, but the external camera ends up against the work surface and shows nothing
usable; only the wrist view is informative. Lower confidence, flagged not asserted: in the wrist
view the plate and cup appear to sit on a straw-textured surface *beyond* the table edge.

---

## 2. SB-NOUN degenerate instruction — you said you would fix this later

SB-NOUN pops a distractor into `main_objects[0]` and pushes the old main object into
`env.distractors`, which is never restored — so 25 rollouts is one chain and the original object can
be re-drawn. `PrimitiveObject.category` defaults to `"object"` and the task YAMLs never set one.

| task | degenerate instruction | rate over 25 rollouts |
|---|---|---|
| 0 `put_green_block_into_bowl` | "put **the object** in the bowl" | 16.1% |
| 6 `stack_cubes` (two PrimitiveObjects) | "stack **the object** on the yellow block" | **66.2%** |
| 1, 2, 3 | instruction becomes the task default verbatim → SB-NOUN silently no-ops | 20-25% |

Separately, task 1 has two `toy_dice`, so its instruction names two objects while only one is
scored.

Options, none chosen (each changes what SB-NOUN *means*): do nothing and document the rate; add
`category:` to the PrimitiveObjects in the task YAML (cheapest, but check `category` is inert for a
PrimitiveObject, since it feeds OG's semantic-segmentation registry); stop pushing the old main
object back into `distractors` (also removes the silent no-op); skip candidates whose category is
`"object"`; or add a per-object `instruction_noun:` falling back to category.

---

## 3. V-SC spawn box is over-subscribed — you said keep as is

Recorded so the number is known rather than to reopen it. V-SC re-places and re-models five
distractors into one box; roughly **2 objects per member per reset** fail after 2500 attempts and
are dropped from the air instead of placed collision-free (`table_knife` most often). Measured 14
failures over warmup + 3 resets at Vec=2, stable rather than escalating. The objects do land on the
table, so the checks pass — but the scatter is physics-determined rather than the controlled
placement the perturbation intends.

Same root as SB-VRB's `receiver` on task 4: `pick_spoon`'s 0.4 x 0.5 m box already holds a 0.2 m
plate and a 0.33 x 0.48 m tray.

---

## 4. Rebuild the SIF somewhere that is not Lustre

`apptainer build --fakeroot` cannot complete here: it unpacks all 644k files and then dies with
`set_attributes: failed to change uid and gids on /image/rootfs/var/mail`, a Lustre limitation, not
a recipe problem.

**Everything needed is committed** — **five** patches under `realm/misc/`, applied by both
`.docker/realm_og391.def` and `.docker/realm_og391.Dockerfile`, each with a `grep -q` guard that
fails the build if a patch stops applying:

| patch | fixes |
|---|---|
| `entity_prim_og391` | pre-existing: relaxes kinematic-tree assertions for REALM's custom robot USDs |
| `usd_object_og391` | pre-existing, same reason |
| `scene_base_zoffset_og391` | scene-file objects load ~100 m high in every scene but 0 |
| `simulator_initqueue_og391` | the global object-init queue was pruned by object NAME, so a sibling scene evicted freshly-added objects |
| `material_prim_preset_og391` | `OmniSurfaceMaterialPrim` required `preset_name` although it is written for `None` |
| `xform_prim_rootlink_og391` + `entity_prim_rootlink_og391` | an articulation's pose was READ from the root link and WRITTEN to the entity prim |

(Seven patch files, five of them added during this work; `entity_prim.py` is patched twice by
independent hunks.)

Why it matters: with these the vectorized eval path works against the image's own OmniGibson with
**no OG-lite bind at all** — measured, Default and V-SC pass every check. Without the z-offset one
they fail with the task object at z=0.015 in every scene but 0; without the others the drawer tasks
do not load and a runtime repair in `RealmVectorEnvironment` is load-bearing rather than a safety
net. The remaining OG-lite delta is performance at these scene counts, visible as 268 contact rows
versus 49-51.

Until a rebuilt image exists, `scripts/clara/interactive/make_stock_patch.sh` plus
`MODE=stockfix ./scripts/clara/interactive/rr` gives the same behaviour by binding the patched files
individually, with no image build. **A rebuilt image has never been verified** — only the bind path
has.

---

## 5. Move to `~/projects/REALM` (whenever you do it)

`realm/` and `examples/` are already clean. **15 scripts** hardcode
`/mnt/home_lustre/sedlam56/projects/REALM_og391`, one `REALM_ROOT=` line each. The robust version is
one `scripts/clara/lib/paths.sh` that derives both roots from its own location, sourced by all 15 —
then the move costs zero edits. `rr` and `t9_sweep.sh` already derive theirs.

Related trap, since the profile already points at the destination: `$REALM_ROOT`, `$REALM_SIF` and
`$REALM_LOGS` are exported pointing at the **pre-port 1.1.1** tree and image, so any
`${REALM_SIF:-<og391 default>}` silently selects the wrong container — the failure reads as "the
patch no longer applies" rather than "wrong image". Being fixed where found.
