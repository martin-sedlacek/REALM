# The two evaluation paths

REALM evaluates a policy through one of two entry points. They write the same artifacts and are
meant to produce comparable numbers.

| | single-env | vectorized |
| --- | --- | --- |
| module | `realm/eval.py` (`evaluate`) | `realm/vector_eval.py` (`evaluate_vectorized`) |
| driver | `examples/02_evaluate.py` | `examples/04_vector_evaluate.py` |
| environment | `RealmEnvironmentDynamic` | `RealmVectorEnvironment`, `num_envs` members |
| rollouts | `repeats`, one after another | `repeats`, in waves of `num_envs` |
| returns | nothing | the result rows |
| extras | `--resume` from an existing report | -- |

Both write, under `log_dir`:

```
reports/{task}_{perturbation}.csv     rewritten in full after every rollout (single) / wave (vector)
qpos/{task}.parquet                   appended per rollout
actions/{task}.parquet                appended per rollout
videos/{task}.parquet                 appended per rollout, unless --no_record
```

`tests/test_vector_integrity.py` checks exactly these four, and the report's row count -- a run that
dies half way leaves a complete-*looking* prefix, so row count is what separates the two.

Everything the two paths must agree on lives in `realm/rollout.py`: what a rollout measures
(`RolloutMetrics`), which control steps render (`RenderSchedule`), how a rollout is driven
(`Rollout`), what a result row looks like (`build_result_entry`) and where artifacts land
(`write_rollout_artifacts`).

## Where they still differ, and why that is deliberate

Both of these predate the shared layer and are preserved as parameters of it, not harmonised.
Changing either would change results or artifact names for runs already in the results tree.

**Artifact naming under a non-default task config.** `resolve_task(..., name_includes_config=)`.
Given `--task_cfg_path REALM_DROID10/pick_spoon/no_distractors.yaml`, the single-env path files the
run under `pick_spoon_no_distractors`; the vector path files it under `pick_spoon`, so two configs
of one task overwrite each other's parquets there. Turning it on for the vector path would rename
every vector artifact written so far.

**Gripper convention.** `Rollout(gripper_inverted=)`. The single-env path looks `model_type` up and
raises `NotImplementedError` for anything unmapped; the vector path always reads `(1, 0)` as
(open, closed). Unreachable in practice: `InferenceClient.__init__` accepts only `debug`, `openpi`
and `dreamzero`, and all three are `(1, 0)` on both paths. The `molmoact` `(0, 1)` mapping in
`rollout.GRIPPER_OPEN_BELOW_HALF` is therefore currently dead -- it is kept because it records the
convention, and `InferenceClient.infer` still has a molmoact branch that a future client
constructor could reach.

### What the 2026-08-16 refactor did unify

Four things, all where the two paths did the same work in a different order or with different
strictness. Each is inert in every configuration anything actually runs, and each was measured
rather than assumed:

1. **An action chunk with `ndim > 2`** now trips the single-env path's assert on both paths, instead
   of being silently queued on the vector path as a 2-D array whose last row the gripper mapping
   would then overwrite. No in-tree client returns one.
2. **The video frame is recorded before inference, not after,** on the single-env path -- the vector
   path always did it first. `VideoRecorder.add_frame` and `InferenceClient.infer` both build new
   arrays and neither mutates the other's inputs.
3. **`recorder.cleanup()` runs before `append_trajectory`, not after,** on the single-env path --
   again the vector path's order. `VideoRecorder.cleanup` is a no-op unless `disk_mode=True`, which
   nothing sets, and in that mode it removes a frame directory that `append_trajectory` does not
   touch.
4. **`results.append(entry)` happens after the artifact writes** rather than between two of them, on
   the single-env path. `results` is in memory and is only serialised afterwards.

Everything else -- which files are written, in which order, with which arguments, and every number
in them -- is unchanged.

## Controller notes that have no other home

From the same pass over `realm/robots/`:

- **`DroidEndEffectorController` accepts `use_gravity_compensation` and never applies it.**
  `droid_joint_controller.py` does apply it (and measured 0.2968 rad of droop on the robolab arm
  without it). Every EE-control config sets it False, so the flag has never had an effect on the EE
  controller -- but the two controllers do not mean the same thing by the same YAML key.
- **The EE controller does not pre-clamp its cartesian command.** A `_scale_cartesian_6d_velocity`
  helper and a commented-out call to it sat in `_update_goal` until 2026-08-16. It clamped to
  0.075 m / 0.15 rad, which is exactly what `RobotIKSolver` applies to the cartesian delta on the
  way through `cartesian_delta_to_velocity` / `cartesian_velocity_to_delta` -- so the disabled call
  was a second, identical clamp. Both are gone; the fact is in the class docstring.
- **The EE controller's IK does not use `panda_arm.urdf`.** A dead `urdf_path` pointing at it was
  removed. `RobotIKSolver` takes no arguments and solves against the dm_control MuJoCo model under
  `realm/robots/robot_ik/franka/`.
- **`_update_goal` writes the height offset back into the command tensor.** In `absolute_pose` mode
  `target_pos = command[:3]` is a *view*, so `target_pos[-1] += self.height_offset` mutates
  `command`. Whether that aliases a caller-owned buffer depends on whether
  `cb.to_torch(command).to(og.sim.device)` copied -- `Tensor.to()` returns `self` when the tensor is
  already on the target device. Untouched by the refactor and not investigated. If the same command
  is ever applied twice, suspect this first.
- **No config in `realm/config/robots/` selects `cartesian_velocity` mode**; only `absolute_pose`
  and `pose_delta_ori` are reachable.

## Verifying a change to either path

Neither path has a unit test. What exists:

- `tests/test_integrity.py` / `tests/test_perturbations_integrity.py` sweep the single-env path over
  tasks and perturbations; `tests/test_vector_integrity.py` does both for the vector path. All three
  check that a run produced its artifacts and did not crash -- **not** that any number is right.
- **A `--model_type debug` run is byte-reproducible on this stack.** Two independent runs of the
  same command produce identical `reports/*.csv` and identical `qpos`/`actions` parquets, on both
  paths -- measured 2026-08-16, task 0 / `Default` / 2 repeats / 30 steps, `MODE=stock` single-env
  and `MODE=oglite` vector at `num_envs=2`. That makes a before/after artifact diff a real
  regression check for a refactor, which it would not be under a real policy: rollouts are
  nondeterministic within a condition in the original port investigation.
- **mp4 bytes are NOT reproducible.** Two identical runs gave video payloads differing by 11% in
  size. Compare **frame counts** instead -- parse the `stsz` box, or count what the recorder was
  given. Frame counts are stable.
- **A debug run never exercises the render-on-demand blind step.** The debug client returns a
  single 1-D action rather than a chunk, so the action buffer empties every control step, inference
  runs every step, and every step renders (29 frames from 30 steps). Exercising the blind-step path
  end to end needs a real policy server.
