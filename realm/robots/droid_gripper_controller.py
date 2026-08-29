"""REALM's gripper controller for the Robotiq 2F-85 on the DROID/UR platforms.

Before OG 3.9.1 this file was a full copy of OmniGibson's `multi_finger_gripper_controller.py` with
one functional change. 3.9.1 rewrote controllers to be *batched*: one controller instance backs N
group members, state is read from `ControllableObjectViewAPI` rather than a per-step `control_dict`,
and every entry point is indexed by `controller_idx`. Rather than re-copying the (now much larger)
upstream file, this is a subclass that overrides only `compute_control` to re-apply REALM's change:

    when opening, the two outer finger joints are driven from the measured inner finger positions
    instead of being sent to their joint limit

which keeps the 2F-85's four-bar linkage consistent instead of fighting itself.
"""

from omnigibson.controllers import ControlType
from omnigibson.controllers.multi_finger_gripper_controller import (
    MultiFingerGripperController as OmniGibsonMultiFingerGripperController,
)
from omnigibson.utils.backend_utils import _compute_backend as cb
from omnigibson.utils.usd_utils import ControllableObjectViewAPI

# Inner finger travel (m) mapped onto outer finger rotation (rad) for the Robotiq 2F-85 linkage.
INNER_FINGER_OPEN_POS = 0.05
OUTER_FINGER_OPEN_ANGLE = 0.785


class MultiFingerGripperController(OmniGibsonMultiFingerGripperController):
    """OmniGibson's multi-finger gripper controller with REALM's outer-finger handling.

    Registered as `CustomGripperController`; see `controller_registry.py`, and the module docstring
    above for what the one override changes and why this is a subclass rather than a copy.
    """

    def compute_control(self, goals):
        """
        Identical to OmniGibson's implementation except for the outer-finger handling in the
        "binary" open branch. See module docstring.

        Args:
            goals (Dict[str, Any]): batched goals; must include "target" with shape (N, command_dim)

        Returns:
            Array: (N, control_dim) outputted (non-clipped!) control signal to deploy
        """
        target_batch = goals["target"]  # (N, command_dim)

        rows = self.view_row_indices
        all_joint_pos = ControllableObjectViewAPI.get_all_joint_positions(self.routing_path)[rows, :][
            :, self.dof_idx
        ]  # (N, ctrl_dim)

        unregistered_mask = self._unregistered_controllers == 1  # (N,)

        if self._mode == "binary":
            should_open = target_batch[:, 0] >= 0.0 if not self._inverted else target_batch[:, 0] > 0.0  # (N,)
            open_limit = (
                self._control_limits[ControlType.get_type(self._motor_type)][1][self.dof_idx]
                if self._open_qpos is None
                else self._open_qpos
            )  # (ctrl_dim,)
            closed_limit = (
                self._control_limits[ControlType.get_type(self._motor_type)][0][self.dof_idx]
                if self._closed_qpos is None
                else self._closed_qpos
            )  # (ctrl_dim,)
            u = cb.where(should_open[:, None], open_limit, closed_limit)  # (N, ctrl_dim)

            # REALM: drive the outer fingers from the measured inner finger positions rather than
            # sending them to the joint limit. dof_idx order comes from the definition's
            # finger_joint_names: [left_inner_prismatic, right_inner_prismatic, left_inner, right_inner].
            if self.control_dim > 2:
                outer_from_inner = all_joint_pos[:, :2] / INNER_FINGER_OPEN_POS * OUTER_FINGER_OPEN_ANGLE
                u[should_open, 2:] = outer_from_inner[should_open]
        else:
            if target_batch.shape[1] == 1:
                u = target_batch * cb.ones(self.control_dim)
            else:
                u = target_batch  # (N, ctrl_dim)

        # If we're near the joint limits and we're using velocity / effort control, zero out the action
        if self._motor_type in {"velocity", "effort"}:
            pos_hi = self._control_limits[ControlType.POSITION][1][self.dof_idx]  # (ctrl_dim,)
            pos_lo = self._control_limits[ControlType.POSITION][0][self.dof_idx]  # (ctrl_dim,)
            violate_upper_limit = all_joint_pos > pos_hi - self._limit_tolerance  # (N, ctrl_dim)
            violate_lower_limit = all_joint_pos < pos_lo + self._limit_tolerance  # (N, ctrl_dim)
            violation = (violate_upper_limit & (u > 0)) | (violate_lower_limit & (u < 0))
            u = u * ~violation

        self._update_grasping_state(all_joint_pos, u)

        u[unregistered_mask] = 0.0

        return u  # (N, control_dim)
