# Clara verification

## Mounted RoboLab v2 unification

Before tagging 1.0.0, validate the single remaining DROID asset in the release SIF:

- Run the smoke suite with default `ROBOT=DROID`, including vector scene placement.
- Run a short `DROID_ee_control` rollout and confirm Cartesian error converges with
  `height_offset: 0.863891`.
- Run the same EE probe with `DROID_mounted_ee_control`; `DROID_ee_control` must resolve to
  `droid.usd` with offset `0.0`, while the mounted profile resolves to `droid_mounted.usd` with
  offset `0.863891`.
- Exercise gripper open/close once and confirm `finger_joint` proprio normalizes as 0=open,
  1=closed through the `DROID_mounted` observation profile.
- Run one no-column scene with `DROID` and confirm the spawn-height adjustment puts
  its arm base at the same world height as the mounted model.
