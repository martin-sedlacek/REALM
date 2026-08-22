# Can the benchmark tell two stacks apart? VB-POSE, OG 3.9.1 vs OG 1.1.1

Measured 21-22 Aug 2026. Every number here came out of `scripts/separability_test.py` or straight
out of the report CSVs; nothing is estimated. **Read the confound in §1 before quoting anything
below** — it limits what these numbers can support far more than the statistics do.

## 1. What this compares, and what it cannot

Two full VB-POSE benchmarks, 30 cells each, 25 rollouts per cell, 1500 rollouts total:

| | A = `vbpose_bench_391_v3` | B = `vbpose_bench_111_v3` |
| --- | --- | --- |
| Simulator | OmniGibson 3.9.1 / Isaac 5.1 | OmniGibson 1.1.1 / Isaac 4.1 |
| Execution | vectorized (`vector_eval`, waves of 4) | sequential (`02_evaluate.py`) — 1.1.1 has no vector path |
| **Robot** | **`DROID_robolab_v2`, mounted** | **stock `DROID`** |
| Perturbation | `VB-POSE` (id 13) — same id on both sides, checked | same |
| Tasks | 0-9, same ten names in the same order, checked | same |
| Protocol | 25 repeats, `max_steps` 800, `horizon` 8, `rt` rendering | same |

**The robot differs, so no delta here is attributable to the simulator.** `DROID_robolab_v2` does
not exist in the 1.1.1 tree — there is no robolab asset and no definition directory for it — so a
robot-matched pair was not possible without re-running the 3.9.1 side with `--robot DROID`. Asset and
simulator move together in every figure below. The defensible form of any claim here is "the
benchmark distinguishes these two **arms**", never "3.9.1 differs from 1.1.1".

A second, smaller mismatch: pi0's checkpoint lives under a different parent on each side, so
`compute_model_name()` derives `checkpoints_pi0_droid_jointpos` (A) and `ckpt_pi0_droid_jointpos`
(B). The separability script pairs checkpoint directories **by name**, so pi0 would have been
silently skipped. It is paired via an additive symlink (§6), not a rename.

## 2. Method, and why it is this method

`scripts/separability_test.py`; its module docstring carries the full rationale and
`~/runbook/references/eval_statistics.md` the derivations. In brief:

* **Paired, not pooled.** REALM reuses one RNG stream per `run_id` within a (task, perturbation), so
  rollout *i* of A and rollout *i* of B face the same condition. Pairing is on
  (task, perturbation, run_id) and roughly doubles power for free.
* **SR and RF** are per-rollout binary → exact McNemar (two-sided binomial on discordant pairs).
* **TP** is continuous on [0,1] and not normal → Wilcoxon signed-rank on paired differences.
* **Pooled SR is task-stratified**, never a pooled binomial across task strata.
* **Every cell scanned counts toward multiplicity** → Holm correction. See the caveat in §7 about
  which family that correction actually spans.

Noise floor: at n=25 the Clopper-Pearson half-width is ~20pp, so one cell resolves only ~30pp
differences. **`separable = no` means "not resolvable at n=25", not "these are the same".**

## 3. Headline: pooled task-stratified

The rows with power. Read these first.

| model | metric | A | B | d | p | verdict |
| --- | --- | --- | --- | --- | --- | --- |
| pi0 | SR | 0.200 | 0.244 | +4.4pp | 0.101 | not separable |
| pi0 | TP | 0.348 | 0.372 | +0.024 | 0.313 | not separable |
| pi0 | RF | 0.472 | 0.504 | +3.2pp | 0.305 | not separable |
| pi0-FAST | **SR** | 0.292 | 0.392 | **+10.0pp** | **0.0018** | **separable** |
| pi0-FAST | TP | 0.563 | 0.594 | +0.031 | 0.220 | not separable |
| pi0-FAST | RF | 0.128 | 0.168 | +4.0pp | 0.138 | not separable |
| pi0.5 | SR | 0.396 | 0.456 | +6.0pp | 0.065 | not at alpha=0.05 |
| pi0.5 | TP | 0.651 | 0.644 | -0.007 | 0.764 | not separable |
| pi0.5 | RF | 0.080 | 0.124 | +4.4pp | 0.046 | significant, **but see below** |

**Only pi0-FAST clears the bar on a metric worth acting on.** pi0.5 misses at p=0.065 on SR, and its
one significant pooled row is RF — a **disproven** selection surrogate (rho = -0.554 against SR).
The script's own header says: report it, never select on it. So pi0.5 is not separable on anything
actionable.

**pi0 is the informative negative.** Nothing separable anywhere, and it is the model with six of ten
tasks at SR 0.000 on *both* sides. At the floor there is nothing for the benchmark to resolve — this
is a statement about pi0's competence, not about the benchmark's sensitivity.

## 4. Every cell, every metric

3 models x 10 tasks x 3 metrics = 90 cell-metrics. `d` is B - A. `p_holm` is Holm-adjusted across
the 10 tasks of that metric. Separable = `p_holm < 0.05`.

### pi0

| t | task | dSR | p_holm | SR? | dTP | p_holm | TP? | dRF | p_holm | RF? |
|---|---|---|---|---|---|---|---|---|---|---|
| t0 | put_green_block_into_bowl | -4.0p | 1.000 | no | -0.064 | 1.000 | no | +4.0p | 1.000 | no |
| t1 | put_banana_into_box | +4.0p | 1.000 | no | -0.072 | 1.000 | no | +4.0p | 1.000 | no |
| t2 | rotate_marker | +0.0p | 1.000 | no | -0.093 | 0.850 | no | +24.0p | 1.000 | no |
| t3 | rotate_mug | +20.0p | 1.000 | no | +0.280 | 0.094 | no | -12.0p | 1.000 | no |
| t4 | pick_spoon | +0.0p | 1.000 | no | +0.027 | 1.000 | no | -8.0p | 1.000 | no |
| t5 | pick_water_bottle | +0.0p | 1.000 | no | -0.013 | 1.000 | no | +4.0p | 1.000 | no |
| t6 | stack_cubes | +16.0p | 1.000 | no | +0.056 | 1.000 | no | +8.0p | 1.000 | no |
| t7 | push_switch | +8.0p | 1.000 | no | +0.040 | 1.000 | no | +4.0p | 1.000 | no |
| t8 | open_drawer | +0.0p | 1.000 | no | +0.000 | 1.000 | no | +0.0p | 1.000 | no |
| t9 | close_drawer | +0.0p | 1.000 | no | +0.080 | 0.850 | no | +4.0p | 1.000 | no |

SR 0/10 · TP 0/10 · RF 0/10

### pi0-FAST

| t | task | dSR | p_holm | SR? | dTP | p_holm | TP? | dRF | p_holm | RF? |
|---|---|---|---|---|---|---|---|---|---|---|
| t0 | put_green_block_into_bowl | -4.0p | 1.000 | no | -0.120 | 0.591 | no | +8.0p | 1.000 | no |
| t1 | put_banana_into_box | **+44.0p** | 0.009 | **YES** | +0.096 | 0.591 | no | +0.0p | 1.000 | no |
| t2 | rotate_marker | -8.0p | 1.000 | no | -0.133 | 0.591 | no | +12.0p | 1.000 | no |
| t3 | rotate_mug | +12.0p | 1.000 | no | +0.133 | 0.591 | no | -4.0p | 1.000 | no |
| t4 | pick_spoon | -4.0p | 1.000 | no | -0.160 | 0.591 | no | **+36.0p** | 0.039 | **YES** |
| t5 | pick_water_bottle | +8.0p | 1.000 | no | +0.213 | 0.479 | no | -32.0p | 0.691 | no |
| t6 | stack_cubes | **+60.0p** | 0.007 | **YES** | +0.256 | 0.117 | no | -4.0p | 1.000 | no |
| t7 | push_switch | -16.0p | 1.000 | no | -0.240 | 0.537 | no | +20.0p | 1.000 | no |
| t8 | open_drawer | +8.0p | 1.000 | no | +0.088 | 0.268 | no | +0.0p | 1.000 | no |
| t9 | close_drawer | +0.0p | 1.000 | no | **+0.176** | 0.044 | **YES** | +4.0p | 1.000 | no |

SR 2/10 · TP 1/10 · RF 1/10

### pi0.5

| t | task | dSR | p_holm | SR? | dTP | p_holm | TP? | dRF | p_holm | RF? |
|---|---|---|---|---|---|---|---|---|---|---|
| t0 | put_green_block_into_bowl | +12.0p | 1.000 | no | +0.000 | 1.000 | no | +0.0p | 1.000 | no |
| t1 | put_banana_into_box | +32.0p | 0.193 | no | +0.016 | 1.000 | no | +0.0p | 1.000 | no |
| t2 | rotate_marker | -32.0p | 0.309 | no | -0.200 | 0.201 | no | -8.0p | 1.000 | no |
| t3 | rotate_mug | +4.0p | 1.000 | no | -0.027 | 1.000 | no | +0.0p | 1.000 | no |
| t4 | pick_spoon | +8.0p | 1.000 | no | -0.120 | 1.000 | no | +16.0p | 1.000 | no |
| t5 | pick_water_bottle | +4.0p | 1.000 | no | +0.240 | 0.296 | no | -32.0p | 0.770 | no |
| t6 | stack_cubes | +44.0p | 0.074 | no | +0.072 | 0.918 | no | +0.0p | 1.000 | no |
| t7 | push_switch | -24.0p | 1.000 | no | -0.333 | 0.055 | no | **+40.0p** | 0.020 | **YES** |
| t8 | open_drawer | +12.0p | 1.000 | no | +0.120 | 0.238 | no | -4.0p | 1.000 | no |
| t9 | close_drawer | +0.0p | 1.000 | no | **+0.160** | 0.029 | **YES** | +32.0p | 0.070 | no |

SR 0/10 · TP 1/10 · RF 1/10

### Totals

| metric | separable |
| --- | --- |
| SR | 2 / 30 |
| TP | 2 / 30 |
| RF | 2 / 30 |
| **all** | **6 / 90** |

What the grid says beyond the counts:

* **All six hits are pi0-FAST or pi0.5.** pi0 is blank across all 30.
* **`stack_cubes` moves the same direction for all three models** (+60.0 / +44.0 / +16.0 pp SR), and
  is the largest single effect in the study.
* **Two of six hits are RF**, the metric you are told never to select on. On SR and TP alone the
  score is 4/60.
* **Signs are not consistent within a row.** pi0-FAST `push_switch` is -16.0pp SR and -0.240 TP but
  +20.0pp RF; `pick_spoon` is -4.0pp SR with +36.0pp RF. That is the rho = -0.554 SR/RF
  anti-correlation appearing per cell — another reason not to select on RF.
* **Three large near-misses that must not be quoted as findings:** pi0.5 `stack_cubes` +44.0pp SR
  (p=0.074), pi0.5 `push_switch` TP -0.333 (0.055), pi0.5 `close_drawer` RF +32.0pp (0.070).

## 5. The drawer tasks, in detail

Asked separately because SR is at the floor there and the means hide the shape.

### Success

| task | model | 3.9.1 | 1.1.1 |
| --- | --- | --- | --- |
| open_drawer | pi0 | 0/25 | 0/25 |
| open_drawer | pi0-FAST | 0/25 | **2/25** (0.080) |
| open_drawer | pi0.5 | 0/25 | **3/25** (0.120) |
| close_drawer | pi0 | 0/25 | 0/25 |
| close_drawer | pi0-FAST | 0/25 | 0/25 |
| close_drawer | pi0.5 | 0/25 | 0/25 |

**5 successes in 150 drawer rollouts, all five on 1.1.1, all five on `open_drawer`.** 3.9.1 is
0-for-75. `close_drawer` is 0-for-50 on both stacks.

### Task progression

| task | model | TP 3.9.1 | TP 1.1.1 | best rollout, 391 -> 111 |
| --- | --- | --- | --- | --- |
| open_drawer | pi0 | 0.000 | 0.000 | 0.0 -> 0.0 |
| open_drawer | pi0-FAST | 0.400 | 0.488 | 0.4 -> 1.0 |
| open_drawer | pi0.5 | 0.384 | 0.504 | 0.4 -> 1.0 |
| close_drawer | pi0 | 0.072 | 0.152 | 0.4 -> 0.6 |
| close_drawer | pi0-FAST | 0.168 | 0.344 | 0.2 -> 0.6 |
| close_drawer | pi0.5 | 0.208 | 0.368 | 0.4 -> 0.6 |

TP is higher on 1.1.1 in **all six** pairs. `close_drawer` roughly doubles for every model, and both
`close_drawer` TP gains for the strong models are the two TP cells that survived correction (§4).

Three distributional facts that matter more than the means:

1. **The 3.9.1 ceiling on `open_drawer` is hard, not merely low.** No 3.9.1 rollout of any model ever
   exceeds TP 0.400. pi0-FAST sits at exactly 0.400 in all 25 rollouts — zero variance, the same
   stage every time. 1.1.1 reaches 1.000. So 3.9.1 does not progress *less on average*; it never
   gets past that stage at all, while 1.1.1 finishes the task outright a few times.
2. **`close_drawer` is capped below success on both stacks.** The best rollout anywhere is TP 0.600.
   Nothing approaches 1.0, consistent with a stage that is never cleared rather than noisy
   near-misses.
3. **pi0 on `open_drawer` is exactly 0.000 in all 50 rollouts, both stacks.** Flat, not low — it
   never leaves the first stage. The one drawer cell where the two stacks agree completely, and a
   property of pi0 rather than of the simulator.

**The confound bites hardest here.** Reaching a drawer handle depends on base placement and reach,
which is exactly what mounted-vs-stock changes. The `open_drawer` success gap is the *last* result in
this document to attribute to the simulator version.

## 6. Reproduction

Login node. Pure Python — no scipy, no container, no GPU.

```sh
cd /mnt/home_lustre/sedlam56/projects/REALM_og391

# One-time: make pi0 pair by name. ADDITIVE symlink beside the real directory; nothing is renamed.
ln -s ckpt_pi0_droid_jointpos \
  /mnt/home_lustre/sedlam56/projects/REALM/logs/vbpose_bench_111_v3/checkpoints_pi0_droid_jointpos

REALM_LOGS=/mnt/home_lustre/sedlam56/projects/REALM/logs \
python3 scripts/separability_test.py \
    vbpose_bench_391_v3 \
    vbpose_bench_111_v3 \
    --reps 25 --alpha 0.05 \
    --csv /mnt/home_lustre/sedlam56/projects/REALM/logs/todo_clara/separability_391_vs_111.csv
```

Argument order sets the sign: `expA expB`, and all deltas are **B - A**.

`--reps 25` is not decoration — a cell short of 25 in *either* arm is excluded and listed rather than
silently truncated. All 30 cells reported `OK` at 25/25; exclusions would mean something is wrong
with the tree, not with the statistics.

The run prints `UNPAIRED checkpoints, skipped (B only): ckpt_pi0_droid_jointpos`. That is the same
data under its original name, already counted through the symlink — not a dropped arm.

**Do not reach for `--exclude-tasks` here.** Its help text says dropping tasks 2 and 6 is "usually
right" because they spawn objects off the work surface, but t6 `stack_cubes` is the largest effect in
the study. Dropping it would remove the most informative cell and make the aggregate
non-comparable with the tables above.

## 7. Caveats

* **The robot confound (§1) dominates everything.** Only a `--robot DROID` re-run of the 3.9.1 side
  separates simulator from arm. 30 jobs.
* **`separable = no` is a power statement.** At n=25 a cell needs roughly 30pp before it can show
  anything alone. Absence of separability is absence of evidence.
* **RF is a disproven surrogate** (rho = -0.554 with SR). Two of the six hits sit on it. Report,
  never select.
* **The Holm family label in the tool output is wrong.** The footer prints
  `(Holm-corrected over 30 tests)`, but `holm()` is called once **per metric** with
  `m = len(cell_rows) = 10` (`scripts/separability_test.py:337-343`), so the per-cell correction
  spans the 10 tasks of one metric, not all 30 tests — roughly 3x more permissive than the label
  claims. The pooled rows in §3 are unaffected. Recorded, not changed: whether the correct family is
  10-per-metric or 30-across-metrics is a judgement call for the benchmark owner, and the fix is
  either the label or the family, not both.
* **Execution model differs** (vectorized A, sequential B) because 1.1.1 has no vector path. Same
  total work, and `rollout.py` semantics are shared in the 3.9.1 tree, but it is not nothing.
* **pi0's model directory name differs between the two trees**, so never join these two experiments
  on model-directory name alone.

## See also

* `scripts/separability_test.py` — the tool, with the statistical rationale in its docstring
* `~/runbook/streams/realm_og391_port.md` — the run log: job IDs, submission scripts, raw tables
* `docs/code_archaeology.md` — the mounted-asset repair that made the 3.9.1 arm usable at all
