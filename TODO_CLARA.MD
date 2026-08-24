# Clara verification

## Mounted RoboLab v2 unification

Before tagging 1.0.0, validate the single remaining DROID asset in the release SIF:

- Run the smoke suite with default `ROBOT=DROID`, including vector scene placement.
- Run a short `DROID_ee_control` rollout and confirm Cartesian error converges with
  `height_offset: 0.863891`.
- Run the same EE probe with `DROID_robolab_v2_ee_control`; both filenames must resolve to
  `droid_robolab_v2_mounted.usd` and produce equivalent arm motion.
- Exercise gripper open/close once and confirm `finger_joint` proprio normalizes as 0=open,
  1=closed through the `DROID_robolab_v2` observation profile.
- Run one no-column scene with `DROID_robolab_v2_bare` and confirm the spawn-height adjustment puts
  its arm base at the same world height as the mounted model.
