"""Gripper controller for the "pad spring" variant of the robolab Robotiq 2F-85.

The real 2F-85 has a spring at each inner-finger pivot. `droid_robolab_padspring.usd` (built by
scripts/make_padspring_gripper_usd.py) removes the PhysX **mimic** constraint from those two pivots
and gives them a real drive instead, so the pad can rotate under load. That turns them into ordinary
driven DOFs, and two OmniGibson invariants then apply:

  * `Robot.update_controller_mode` asserts that no *un*-controlled DOF carries a DriveAPI, and
  * it force-writes `isaac_kp` / `isaac_kd` onto every controlled DOF on **each** `og.sim.play()`.

Both point the same way: the pad pivots have to be claimed by a controller, and their spring gains
have to come from that controller's config -- which is also the only place they will survive a
stop/play cycle. So the gripper group owns three DOFs here instead of one:

    dof_idx = [finger_joint, left_inner_finger_joint, right_inner_finger_joint]

with, in realm/config/robots/DROID_robolab_padspring.yaml,

    isaac_kp: [1e7, <pad kp>, <pad kp>]     # leader stays at OmniGibson's default stiffness
    isaac_kd: [1e5, <pad kd>, <pad kd>]

`max_effort` for the pads is NOT settable from here -- OmniGibson only forces joint max efforts for
holonomic bases -- so it is authored as `drive:angular:physics:maxForce` in the variant USD.

What this class does
--------------------
Exactly what the removed mimic constraint did, but as a *soft* position target rather than a hard
equality: each pad pivot is commanded to `gearing * q_leader_commanded + offset`, the same affine
relation PhysX was enforcing (`gearing` +1 on the left pivot, -1 on the right, `offset` 0, read off
the original asset's `physxMimicJoint:rotX:*`), anchored to the leader's **measured** angle.

The pad then sits at that target minus `contact_torque / kp`: it stays parallel through the stroke
when unloaded, and rotates visibly when the pad is loaded. That deflection is the whole point --
`kp` sets how much of it there is, and the USD's `maxForce` sets when the pad gives up and folds.

Known limitation: the anchor is updated once per 15 Hz control step, while the leader slews the whole
45 deg stroke in a single step, so the pads lag the leader by one step during a free open/close. A
mimic constraint has no such lag -- it is solved every physics substep. The lag is a transient with no
load on the pads; it does not affect the loaded stall, which is what the compliance numbers measure.

Deliberately NOT subclassing realm.robots.droid_gripper_controller: its `control_dim > 2` branch is
the stock 4-DOF 2F-85's outer-finger-from-inner-finger mapping, which would fire here (control_dim
is 3) and overwrite the pad targets with prismatic-derived nonsense.
"""

from omnigibson.controllers import ControlType
from omnigibson.controllers.multi_finger_gripper_controller import (
    MultiFingerGripperController as OmniGibsonMultiFingerGripperController,
)
from omnigibson.utils.usd_utils import ControllableObjectViewAPI


class PadSpringGripperController(OmniGibsonMultiFingerGripperController):
    """Binary 2F-85 gripper whose two inner-finger pivots are sprung followers, not mimic joints."""

    def __init__(self, *args, pad_gearing=(1.0, -1.0), pad_offset=(0.0, 0.0), **kwargs):
        """
        Args:
            pad_gearing (Array[float]): one entry per pad DOF (i.e. per dof_idx entry after the
                first), the `physxMimicJoint:*:gearing` the original asset authored for that joint.
            pad_offset (Array[float]): matching `physxMimicJoint:*:offset` values, in radians.
        """
        super().__init__(*args, **kwargs)
        self._pad_gearing = [float(g) for g in pad_gearing]
        self._pad_offset = [float(o) for o in pad_offset]
        n_pads = self.control_dim - 1
        assert n_pads >= 1, (
            f"{type(self).__name__} expects the leader plus at least one sprung pad DOF, but "
            f"control_dim is {self.control_dim}. Check finger_joint_names in the robot definition."
        )
        assert len(self._pad_gearing) == n_pads and len(self._pad_offset) == n_pads, (
            f"pad_gearing / pad_offset must have one entry per sprung pad DOF ({n_pads}); got "
            f"{len(self._pad_gearing)} / {len(self._pad_offset)}"
        )

    def compute_control(self, goals):
        """Binary open/close on the leader; every other DOF tracks the mimic relation softly.

        Args:
            goals (Dict[str, Any]): batched goals; must include "target" with shape (N, command_dim)

        Returns:
            Array: (N, control_dim) outputted (non-clipped!) control signal to deploy
        """
        assert self._mode == "binary", (
            f"{type(self).__name__} only implements mode='binary' (got '{self._mode}'); a smooth or "
            f"independent command has no defined meaning for a sprung follower."
        )
        target_batch = goals["target"]  # (N, command_dim)
        rows = self.view_row_indices
        all_joint_pos = ControllableObjectViewAPI.get_all_joint_positions(self.routing_path)[rows, :][
            :, self.dof_idx
        ]  # (N, ctrl_dim)

        # Leader: identical to OmniGibson's binary branch. On this asset finger_joint runs
        # 0 rad = OPEN -> upper limit = CLOSED, which is why CLOSE takes the *upper* limit.
        pos_limits = self._control_limits[ControlType.get_type(self._motor_type)]
        open_limit = pos_limits[1][self.dof_idx] if self._open_qpos is None else self._open_qpos
        closed_limit = pos_limits[0][self.dof_idx] if self._closed_qpos is None else self._closed_qpos
        should_open = target_batch[:, 0] >= 0.0 if not self._inverted else target_batch[:, 0] > 0.0
        u = self._backend_where(should_open, open_limit, closed_limit)  # (N, ctrl_dim)

        # Sprung pads: the affine relation the mimic constraint used to enforce, anchored to the
        # leader's MEASURED angle.
        #
        # It must be the measured angle, not the commanded one. Measured 2026-08-14 (jobs on
        # l40s-01, logs padspring_squeeze_kp{10,40}_*): anchoring to the command instead makes the
        # pads fold to the full 45 deg by themselves the moment the leader stalls on an object --
        # the command stays at the closed limit while the leader sits at 0, so the pad chases 45 deg
        # of *target*, not load. Both rungs ended with the leader at ~0.000 rad and the pads at
        # 0.73 rad, pinching the cube against base_link with the jaw still wide open. The mimic
        # constraint this replaces is likewise a relation on the reference joint's POSITION.
        q_leader = all_joint_pos[:, 0]
        for k, (g, o) in enumerate(zip(self._pad_gearing, self._pad_offset), start=1):
            u[:, k] = g * q_leader + o

        self._update_grasping_state(all_joint_pos, u)
        u[self._unregistered_controllers == 1] = 0.0
        return u  # (N, control_dim)

    @staticmethod
    def _backend_where(mask, a, b):
        from omnigibson.utils.backend_utils import _compute_backend as cb

        return cb.where(mask[:, None], a, b)
