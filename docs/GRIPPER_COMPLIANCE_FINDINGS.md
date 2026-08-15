# Robotiq 2F-85 compliance in REALM — what was tried, what failed, what had potential

Written 2026-08-15, at the point the work was parked. All code changes were reverted; this document
and the diagnostic probes under `scripts/debug_probes/` are what survive.

**The goal:** make the 2F-85's fingertips visibly bend inward when pressed against a surface, as they
do on the real gripper and reportedly in RoboLab.

**The outcome:** not achieved. A real and serious OmniGibson bug was found and fixed, which improved
the measured curl ~10×, but the result is **~0.35°** — Martin's assessment, and it is correct: *"that
is not even one degree, it's nothing."* The behaviour being chased is probably a different mechanism
than the one that was measured. See [What had potential](#what-had-potential).

---

## 1. The one real finding: OmniGibson inflates gripper-link inertia 77×

**`RigidPrim.update_meshes()` composes each collision geom's centre of mass with
`get_position_orientation(frame="parent")` — the geom's *immediate* parent — while the comment above
it claims the link frame.** `GEOM_TYPES` excludes `Xform`, so when an intermediate Xform carries the
collision APIs and the Mesh sits beneath it, the Xform→link transform is silently dropped.

On this asset that dropped transform is the left/right **mirror** — 90° on one side, 180° on the
other. Consequences, all measured:

| quantity | broken | correct |
|---|--:|--:|
| pad CoM (link frame, mm) | `[-54.20, 116.34]` on **both** pads, identical incl. sign of y | `[-3.59, ∓4.94]`, mirrored |
| pad inertia about its pivot | 1.496e-04 (**77.3×**) | 1.937e-06 |
| `nf_eq` | 113.8 / 116.1 | 810 (fix) … 1018 (fix + authored tensor) |

**The diagnostic signature:** a mirrored left/right pair receiving an *identical* CoM including the
sign of y. Two mirror-image bodies cannot share a CoM y-component. Only objects whose collision
geometry sits under an intermediate Xform are affected; where geoms are direct children of the link
the old and new expressions are algebraically identical.

**Why it mattered here:** a PhysX mimic joint realises a constraint stiffness of roughly ω²·I, so a
77× inertia error made the fingertips ~77× too stiff at the asset's own authored `naturalFrequency`.
The `m·d²` parallel-axis term accounts for ~95% of the inflation.

**The two-way confirmation.** Independently: (a) compute what `nf` REALM would need to reach RoboLab's
ω²·I product → **113.8**; (b) measure which `nf` empirically produces a curl → **100–200**. They agree.
That is what turned a plausible story into a diagnosis, and it retrospectively explained why lowering
`nf` "worked" — it was hand-compensation for links the loader had made too heavy.

### It is worse than one call site

`grep 'frame="parent"'` finds **three** sites composing to the wrong level:

| site | feeds | was patched? |
|---|---|---|
| `prims/rigid_prim.py:324` | centre of mass → inertia | yes |
| `prims/geom_prim.py:250` `points_in_parent_frame` | collision + visual **hull** | **no** |
| `utils/object_utils.py:88` `compute_base_aligned_bboxes` | bounding boxes (offline tooling only) | **no** |

The **hull** site is the one that reaches live REALM code: `get_base_aligned_bbox()` at
`perturbations/_helpers.py:200` and `sb_vrb.py:74` routes through it. Measured on the Robotiq links,
every collision *and visual* hull is **61.09–192.66 mm** off centre with extents wrong by up to
**31.80 mm** — against **0.00 mm on all eight `panda_link*`** under the same loader and same robot.
The defect is asset-structure dependent, which is why it went unnoticed.

**This bug is still present in the reverted tree.** It is genuine and worth fixing on its own merits,
independently of gripper compliance — see [Restoring the fix](#restoring-the-fix).

### A second, separate mass bug (found, never fixed)

`dataset_object.py:289` divides category mass by a `total_volume` that includes **meta links**, whose
`volume` returns their *visual* mesh volume. `bottom_cabinet/glefdh` receives **36.25%** of its
category mass; `microwave/vuezel` **75.24%**. Every BEHAVIOR object with meta links is underweight.
Unrelated to the gripper; recorded because it was found here.

---

## 2. Three fixes, all working, all landing in the same place

| route | how | curl @ authored `nf=1000` |
|---|---|--:|
| broken baseline | — | +0.034° |
| **loader patch** | compose CoM to the link frame in `rigid_prim.py` | +0.330° / +0.391° |
| **Xform flatten** | bake the `Defeatured_*_01` Xform onto its Mesh; stock loader, no patch | +0.335° / +0.383° |
| **author mass + re-anchor** | apply `MassAPI` with correct tensors, plus the re-anchor | +0.359° / +0.403° |

All INWARD, all three hull-free observables agreeing, noise floor 0.0012°. Three structurally
different repairs landing within 0.08° is strong evidence the diagnosis was right — if anything else
were losing a frame, they would diverge.

**The asset routes are better than the loader patch**, on evidence rather than taste: the patch fixes
the CoM but leaves the hull reading backwards, and the hull is what live perturbation code consumes.
Measured hull/hull-free ratio — loader-patched run is as backwards as the unpatched one (−0.49…−1.06);
only the re-anchored asset reads correctly (+0.969…+0.995).

**Authoring mass properties alone does not work**, and the reason is worth knowing: `update_meshes()`
ends with `self.center_of_mass = com`, whose setter writes `physics:centerOfMass` directly into the
**scene stage's edit target**, outranking any referenced layer. **OmniGibson computes-then-overwrites
— an asset cannot defend that field.** `mass`, `diagonalInertia` and `principalAxes` *are* consumed
verbatim (matching RoboLab's runtime tensor to 0.00062%), and PhysX does not parallel-axis-shift an
authored tensor even with the CoM 128 mm away.

---

## 3. What failed, and why — so none of it is retried

Every one of these is a **measured negative**, not an untried idea.

| hypothesis | verdict | evidence |
|---|---|---|
| Drive gains on `finger_joint` | **no effect** | 1e7→1e4, and RoboLab's exact 5729.578/0.011459. Drive saturates at `max_effort` 16.5 regardless |
| `max_effort` on `finger_joint` | **no effect** | flat over 0.5 → 33 N·m (66×) while peak force moved 38× |
| `dampingRatio` on the mimic joints | **inert** | 0.005 → 0.3 (60×) changes deflection 0.4% |
| Restoring the followers' DriveAPI | **inert** | bit-identical to six decimals at every load rung; also predicts the wrong sign |
| Lowering mimic `naturalFrequency` | works, but is **compensation** | nf=100 gives +1.78° on the broken build — it was masking the inertia bug |
| Solver iteration counts | **not a difference** | both stacks solve at 32 position iterations; RoboLab's scene caps its own request of 64 |
| dt / decimation | **identical** | 1/120 and 8 on both |
| GPU vs CPU dynamics | **~6%** | RoboLab run on CPU keeps 94% of its compliance; the 18–26× gap survives |
| Self-collisions | **zero contact** | audit returns `n_self_max=0` everywhere; the 28-pair filter is complete |
| Gravity on the robot | **already matched** | REALM already sets `disableGravity=True` on all nine gripper links |
| Compliant contact material | **not the difference** | Isaac Lab's default ships `compliant_contact_stiffness=0.0` |
| Joint limits | **identical** | apparent mismatch was a radians/degrees double-conversion |
| Link masses | **bit-identical** | the two USDs' colliders agree to 5.4e-11 m, world anchors to 7.6 nm |
| Engine version (5.1 vs 5.0) | **ruled out** | Martin's call |
| Pad-pivot spring (replace mimic with a soft drive) | works, **partial** | +0.96° under press; ~3.4× of the gap, and it saturates |
| Gripper tilt | **second-order** | 15° of tilt (29.7 mm across the jaw) buys only 2.1× |
| Flat-table press | **cannot work at any depth** | a flat pad beds flat, so there is almost no moment about the pivot |
| Leader back-drive | **structurally impossible as tested** | `finger_joint` = 0.0000° across **21 rungs** |

### The load case matters more than any parameter

Curl scales with how **concentrated** the tip load is:

| geometry | default | soft variant |
|---|--:|--:|
| flat table, tips levelled | 0.11° | 0.30° |
| flat table, hand tilted 4.25 mm | 0.31° | 5.37° |
| object pinned into one fingertip | 0.048° | 3.39° |

---

## 4. What had potential

**The untested case, and the most promising remaining lead: closing the gripper THROUGH contact.**

Every press measured held `finger_joint` at a **commanded fixed angle** — either jaws shut (no travel
left) or jaws open and *held* (the drive resists). That is why the leader read exactly 0.0000° across
all 21 rungs: **it was never asked to move against a blocked pad.** What was measured throughout was
how far the mimic constraint yields under load, which is correctly a few tenths of a degree.

The real gripper's fold is **driven**, not passive: command it shut, the pads meet an obstruction, and
the underactuated linkage keeps travelling, rolling the tips inward. No parameter sweep would find
this, because it is a different experiment rather than a different value. This was launched and then
parked; nothing was measured.

**Two things to be careful of if it is picked up:**

- The four-bar is modelled as a **tree** with PhysX mimic joints (`inner_knuckle` is a leaf, not
  joined back to `base_link`). That is RoboLab's own structure and **loop closure is not available in
  PhysX articulations** — do not try to change the topology.
- A mimic joint is a solver equality, so it slaves a follower to the leader but does not obviously
  transmit load back. Whether a *driven* leader produces the fold anyway is exactly the open question.

**Also unresolved:** RoboLab's own measured compliance is small — mimic residual 0.328° at 5 N and
2.250° at 50 N. It is ~20× more compliant than broken-REALM, but it is not 45° either. If 45° has been
seen in RoboLab, it did not come from this mechanism under these loads.

---

## 5. Things that wasted time, recorded so they do not again

- **The load case drifted three times**, each substitution looking like reasonable test design:
  squeeze-an-object replaced press; press-with-jaws-**shut** replaced press-with-jaws-open; and the
  press was run with the hand **4.25 mm out of level**, which produced a spurious 5.37° that was
  reported as a result before being retracted.
- **`collision_boundary_points_world` sits 116 mm off the pad link origins**, so hull-derived tip
  separation reads **backwards**. This made the curl *direction* unreadable for hours and produced two
  wrong conclusions. It is the same CoM bug in a different consumer. Use hull-free observables
  (`d_tipg_sep`); `d_tip_sep`/`d_base_sep` are retired as `HULL_INVALID`.
- **Premature reporting.** Numbers were stated for runs that had not produced output — four times in
  one session, once wrongly enough to carry a bad scoping decision into three pushed records. Read the
  artifact, then write the claim.
- **Unit reconciliation.** PhysX getters return per-radian; USD authors angular drives per-degree. An
  unreconciled "1e5× stiffer" figure was wrong; the correct ratio is ~1745×.
- **`max_effort` reading `100.0` is a display sentinel** (`joint_prim.py:370` returns
  `DEFAULT_MAX_EFFORT` when `|raw| > INF_EFFORT_THRESHOLD`), not a real limit.
- **Isaac exits 139 at teardown regardless of outcome.** Never gate on exit code. And its teardown
  hang makes a time-limit kill routine, so without `python -u` a block-buffered log ending abruptly is
  indistinguishable from a crash.
- **`--cam-pose` values start with `-`**, so argparse reads them as a flag unless the `--flag=value`
  form is used. Cost five runs in one chain.
- **`sed -i` drops the exec bit.**

---

## 6. What was reverted, and how to restore it

Reverted: the `curlgrip` (nf=200), `padspring` and `xflat` robot configs, definitions and variant
USDs; `padspring_gripper_controller.py` and its registry entry; the three `make_*_gripper_usd.py`
generators; and the OmniGibson CoM fix in OG-lite.

**Kept:** everything under `scripts/debug_probes/` (the measurement harness and its validated
observables), this document, `CHANGE_LEDGER.md`, and the `docs/og_deviations/` audit chapters.

### Restoring the fix

The CoM fix is a **genuine bug fix independent of gripper compliance** — it corrects a 77× inertia
error and a hull that is 61–193 mm off on any asset whose collision geometry sits under an
intermediate Xform. If it is wanted back:

```sh
cd ~/projects/OG-lite_og391 && git revert --no-commit <the revert commit> && git commit
```

Note it only reaches a run via `MODE=oglite` or `MODE=stockfix`; `rr` defaults to `MODE=stock`, and it
was never added to the seven `realm/misc/` patches, so a rebuilt image does not contain it.

The asset-side equivalent (`make_xflat_gripper_usd.py`) is preferable if it is ever revisited, because
it repairs the hull as well as the CoM and needs no engine patch — but both are reverted here.
