"""Scoring a rollout against its task's ordered progression rubric.

``config/tasks/task_progressions.yaml`` names, per task type, the ordered stages a rollout has to
pass -- REACH, GRASP, LIFT_SLIGHT, ... . :meth:`TaskProgressionMixin.recompute_task_progression`
walks them in order, stops at the first unmet one, and returns the fraction reached. Stages latch
once reached, so progression is monotone within a rollout and an already-True stage is never
re-checked.

Seven task types can actually be declared by a task config: put, pick, rotate, push, stack,
open_drawer and close_drawer. The rubric file carries two more, ``pour`` and ``turn_faucet``, which
no task uses yet -- they are the TODO markers on POUR and MOVE_JOINT_SMALL below.

Every environment must own a DEEP COPY of its rubric. ``TASK_PROGRESS_RUBRICS`` is built once at
module import and the scan MUTATES the dict it walks. Handing environments the module-level object
gave every one of them in the process the SAME progression state -- harmless with one env per
process, catastrophic in a vector env: one member reaching GRASP latched it for all of them, and
because the scan short-circuits on an already-True stage it was never re-checked per member.
Progression became an OR across members and stuck there, so every member reported the same stage and
the same timestamps, the 15-step terminal countdown started for all of them at once (which read as
the members naturally converging), and SR was 1 whenever ANY member succeeded. Measured 2026-08-13:
it inflated a 25-rollout vectorized pi0.5 eval to SR 0.960 -- an upper bound over waves of 4 -- with
rollouts scored SUCCESS whose block was never grasped.
"""
import copy

import numpy as np

import omnigibson as og

from realm.environments.utils import load_task_progressions
from realm.geometry import compute_rot_diff_magnitude

TASK_PROGRESS_RUBRICS = load_task_progressions()

#: The two task types whose main object is the `impact_drawer` cabinet -- the ones that articulate
#: a joint instead of moving a free body. These are the literals
#: `realm/config/tasks/REALM_DROID10/{open,close}_drawer/default.yaml` declare, and the only two
#: values of `task_type` for which `mo_joint` is ever set.
#:
#: Stated once here because ``"open_close_drawer"`` -- a value NO task config has ever produced, in
#: this checkout or in the pre-port 1.1.1 one -- was compared against at two sites (this module's
#: `check_reach_condition` and `realm/rollout.py`'s second-camera selection) and was constant False
#: at both. Both now use this tuple. The perturbation modules still spell the pair out inline;
#: that is a duplication, not a bug, and is left alone.
DRAWER_TASK_TYPES = ("open_drawer", "close_drawer")


class TaskProgressionMixin:
    """The stage checkers behind a task's rubric.

    Expects the host to provide ``task_type``, the scene handles (``main_objects``,
    ``target_objects``, ``robot``, ``robot_finger_links``), the start-of-rollout scoring reference
    (``mo_pos_orig``, ``mo_rot_orig``), the drawer-joint state ``JointResetMixin`` records
    (``mo_joint``, ``joint_range``, ``init_openness_fraction``), and the contact predicates
    ``is_grasping`` / ``is_touching``.
    """

    def _init_task_progression(self, task_type):
        """Give this environment its own rubric copy and its own stage -> checker map."""
        self.was_lifted = False
        self.task_progression = (
            copy.deepcopy(TASK_PROGRESS_RUBRICS[task_type])
            if task_type in TASK_PROGRESS_RUBRICS
            else None
        )
        self.success_conditions = {
            "REACH": self.check_reach_condition,
            "GRASP": self.check_grasp_condition,
            "TOUCH": self.check_touch_condition,
            "LIFT_SLIGHT": self.check_lift_slight_condition,
            "LIFT_LARGE": self.check_lift_large_condition,
            "ROTATED": self.check_rotated,
            "PUSH": self.check_push,
            "MOVE_CLOSE": self.check_move_close_condition,
            "PLACE_INTO": self.check_place_condition,
            "PLACE_ONTO": self.check_place_onto_condition,
            "TOUCH_AND_MOVE_JOINT": self.check_touching_and_moved_mo_joint,
            "OPEN_JOINT_SMALL": self.check_opened_mo_joint_small,
            "OPEN_JOINT_LARGE": self.check_opened_mo_joint_large,
            "OPEN_JOINT_FULL": self.check_opened_mo_joint_full,
            "CLOSE_JOINT_SMALL": self.check_closed_mo_joint_small,
            "CLOSE_JOINT_LARGE": self.check_closed_mo_joint_large,
            "CLOSE_JOINT_FULL": self.check_closed_mo_joint_full,
            "MOVE_JOINT_SMALL": self.check_moved_mo_joint_small, # TODO: turn faucet
            "MOVE_JOINT_LARGE": self.check_moved_mo_joint_large,
            "MOVE_JOINT_FULL": self.check_moved_mo_joint_full,
            "TOGGLED_ON": self.check_toggled_on_condition,
            # "POUR", not "POURED": the key has to match the stage name the rubric uses, and
            # task_progressions.yaml's `pour` rubric names POUR. Registered as POURED here and in
            # the pre-port 1.1.1 tree, which left recompute_task_progression looking POUR up,
            # getting None, and calling it. See check_pour.
            "POUR": self.check_pour  # TODO: pouring
        }

    def recompute_task_progression(self, obs):
        """Fraction of this task's rubric reached, latching every stage passed."""
        reward = 0.0

        if self.task_progression is not None:
            for stage, is_completed_flag in self.task_progression.items():
                checker_function = self.success_conditions.get(stage)
                if is_completed_flag or checker_function(obs):
                    if not is_completed_flag:
                        self.task_progression[stage] = True
                    reward += 1 / len(self.task_progression.keys())
                else:
                    break
            assert 0.0 <= reward <= 1.0
        return reward

    # ============================== [PROXIMITY AND GRASP STAGES] ==============================
    def check_reach_condition(self, obs):
        """REACH: fingers near the main object, or touching it.

        The drawer tasks take the touch-only branch. Centre-to-centre distance is the wrong
        measure for a cabinet -- ``mo`` is the whole `impact_drawer` asset, whose origin sits
        inside the carcass, well outside anything the robot reaches for -- so the 10 cm test below
        is not a proximity test on a point a rollout can approach.

        THIS BRANCH WAS DEAD UNTIL 2026-08-16, in this checkout and in the pre-port 1.1.1 one
        (`~/projects/REALM/realm/environments/env_base.py:234`). It read
        ``if self.task_progression in ["open_close_drawer"]``, which is wrong twice over:
        ``self.task_progression`` is this environment's rubric -- an ``OrderedDict`` of
        stage -> bool -- not a task type, and ``"open_close_drawer"`` is not a value any task
        config declares (they declare ``open_drawer`` / ``close_drawer``). An ``OrderedDict`` is
        never ``in`` a list of strings, so the condition was constant False.

        Making it live makes REACH STRICTLY HARDER for the two drawer tasks, and for nothing else.
        The general path returns ``d1 < 0.1 or d2 < 0.1 or check_touch_condition(obs)`` and this
        branch returns ``is_touching(obs, mo)``, which is exactly ``check_touch_condition``'s body
        -- so the branch drops the two centre-distance disjuncts and keeps the touch term. Any
        drawer rollout that scored REACH on touch still does; one that scored it purely on being
        within 10 cm of the cabinet's origin no longer does.

        Do NOT reach for the obvious next sentence, that pre-2026-08-16 drawer
        ``task_progression`` numbers are an upper bound on what this code now reports. They are not
        a bound on anything. The `impact_drawer` cabinet carried ``purpose = "guide"`` on all 56 of
        its geoms until `8598e59`, so it contributed 0 px to every camera including the wrist: every
        drawer rollout on record was scored on a scene where the object to be manipulated was absent
        from the policy's inputs. Those cells are INVALID, not failed, and rescoring them under any
        rubric -- this one or the old one -- does not recover a measurement.
        """
        mo = self.main_objects[0]

        if self.task_type in DRAWER_TASK_TYPES:
            return self.is_touching(obs, mo)

        pos1 = mo.get_position_orientation()[0]
        finger1 = list(self.robot_finger_links)[0]
        pos_finger1 = finger1.get_position_orientation()[0]
        finger2 = list(self.robot_finger_links)[1]
        pos_finger2 = finger2.get_position_orientation()[0]

        distance_1 = np.linalg.norm(pos1 - pos_finger1)
        distance_2 = np.linalg.norm(pos1 - pos_finger2)

        dist = 0.1
        return distance_1 < dist or distance_2 < dist or self.check_touch_condition(obs)

        # TODO: make the distance computation bbox dependent rather than centre-to-centre

    def check_grasp_condition(self, obs):
        return self.is_grasping(obs, self.main_objects[0])

    def check_touch_condition(self, obs):
        return self.is_touching(obs, self.main_objects[0])

    # ============================== [MOTION STAGES] ==============================
    # NOTE: switched to checking Z axis rotation only, possible it is still bad but seems to be working well now
    def check_rotated(self, obs, rot_threshold=1.1):
        mo = self.main_objects[0]
        mo_rot_curr = mo.get_position_orientation()[1]

        rot_diff = compute_rot_diff_magnitude(self.mo_rot_orig, mo_rot_curr)

        return abs(rot_diff) > rot_threshold

    def check_lift_and_distance_condition(self, distance_threshold=0.05, lift_threshold=0.01):
        """Has the main object risen and travelled far enough from where the rollout started?

        Both terms are measured against ``mo_pos_orig``, the START-OF-ROLLOUT reference -- see
        ``RealmEnvironmentBase.capture_mo_reference``.
        """
        mo = self.main_objects[0]
        mo_pos_curr = mo.get_position_orientation()[0]

        distance = np.linalg.norm(mo_pos_curr - self.mo_pos_orig)

        return mo_pos_curr[2] - self.mo_pos_orig[2] > lift_threshold and distance > distance_threshold

    def check_lift_slight_condition(self, obs):
        return self.check_lift_and_distance_condition()  # lifted at least 1cm and traveled at least 5cm

    def check_lift_large_condition(self, obs):
        return self.check_lift_and_distance_condition(distance_threshold=0.1, lift_threshold=0.075)

    def check_push(self, obs):
        mo = self.main_objects[0]
        push_cond = self.check_lift_and_distance_condition(distance_threshold=0.1, lift_threshold=-0.05)
        is_lifted = self.check_lift_and_distance_condition(distance_threshold=-0.05, lift_threshold=0.05)
        self.was_lifted = is_lifted or self.was_lifted
        is_robot_touching_obj = self.robot.states[og.object_states.Touching].get_value(mo)
        return push_cond and is_robot_touching_obj and not self.was_lifted

    # ============================== [PLACEMENT STAGES] ==============================
    def check_move_close_condition(self, obs):
        assert len(self.main_objects) == 1
        assert len(self.target_objects) == 1

        mo = self.main_objects[0]
        pos1 = mo.get_position_orientation()[0]

        target = self.target_objects[0]
        pos2 = target.get_position_orientation()[0]

        distance = np.linalg.norm(pos1 - pos2)
        return distance < 0.125 #0.075 #TODO: adjust for size of receiver, this might not always be enough it seems

    def check_place_condition(self, obs):
        mo = self.main_objects[0]
        target = self.target_objects[0]
        inside_or_on_top = mo.states[og.object_states.OnTop].get_value(target) or mo.states[og.object_states.Inside].get_value(target)
        return inside_or_on_top and not self.is_grasping(obs, mo)

    def check_place_onto_condition(self, obs):
        mo = self.main_objects[0]
        target = self.target_objects[0]
        return mo.states[og.object_states.OnTop].get_value(target) and not self.is_grasping(obs, mo)

    def check_toggled_on_condition(self, obs):
        mo = self.main_objects[0]
        return mo.states[og.object_states.ToggledOn].get_value()

    def check_pour(self, obs):
        """POUR: not implemented. Returns False so a `pour` rubric stops here rather than crashing.

        Still a stub -- no task config declares `task_type: pour`, so nothing reaches it -- but it
        is now a stub that can be CALLED. `recompute_task_progression` invokes every checker as
        `checker_function(obs)`, unconditionally, so both of the following were crashes waiting on
        the first `pour` task, in this checkout and in the pre-port 1.1.1 one:

          * the rubric names the stage POUR while the registry keyed it POURED, so
            `self.success_conditions.get("POUR")` returned None and the next line called
            `None(obs)` -> `TypeError: 'NoneType' object is not callable`;
          * and this method took no `obs`, so fixing only the key would have moved the crash one
            line down to `TypeError: check_pour() takes 1 positional argument but 2 were given`.

        Both halves go together; neither alone helps. Latent either way -- with `pour` declared by
        no task config, no live rollout changes behaviour.
        """
        return False

    # ============================== [DRAWER JOINT STAGES] ==============================
    def get_mo_joint_openness_fraction(self):
        assert self.mo_joint is not None
        return (self.mo_joint.get_state()[0][0] - self.mo_joint.lower_limit) / self.joint_range

    def get_mo_joint_delta(self):
        """How far the drawer has moved since the reset, as a fraction of its range."""
        openness_fraction = self.get_mo_joint_openness_fraction()
        delta_openness_fraction = self.init_openness_fraction - openness_fraction
        return delta_openness_fraction

    def check_touching_and_moved_mo_joint(self, obs, threshold=0.025):
        delta_openness_fraction = self.get_mo_joint_delta()
        if self.task_type == "open_drawer":
            return self.check_touch_condition(obs) and delta_openness_fraction < threshold
        elif self.task_type == "close_drawer":
            return self.check_touch_condition(obs) and delta_openness_fraction > threshold
        else:
            raise NotImplementedError()

    def check_opened_mo_joint_small(self, obs):
        return self.get_mo_joint_openness_fraction() > 0.125

    def check_opened_mo_joint_large(self, obs):
        return self.get_mo_joint_openness_fraction() > 0.65

    def check_opened_mo_joint_full(self, obs):
        return self.get_mo_joint_openness_fraction() > 0.95

    def check_closed_mo_joint_small(self, obs):
        return self.get_mo_joint_openness_fraction() < 0.875

    def check_closed_mo_joint_large(self, obs):
        return self.get_mo_joint_openness_fraction() < 0.35

    def check_closed_mo_joint_full(self, obs):
        return self.get_mo_joint_openness_fraction() < 0.05

    def check_moved_mo_joint_small(self, obs):
        return self.check_closed_mo_joint_small(obs) or self.check_opened_mo_joint_small(obs)

    def check_moved_mo_joint_large(self, obs):
        return self.check_closed_mo_joint_large(obs) or self.check_opened_mo_joint_large(obs)

    def check_moved_mo_joint_full(self, obs):
        """MOVE_JOINT_FULL: delegates to the _LARGE checkers, not the _FULL ones.

        DELIBERATELY LEFT AS IS. It reads like a copy-paste slip -- `_small` calls the `_small`
        pair, `_large` calls the `_large` pair, and this one calls the `_large` pair again instead
        of `check_closed_mo_joint_full` / `check_opened_mo_joint_full` -- but it is identical to
        the pre-port 1.1.1 implementation (`~/projects/REALM`,
        `realm/environments/env_base.py:330-331`). So this is not port breakage, and the behaviour
        it produces is the behaviour every REALM number was ever scored against.

        What "fixing" it would do: MOVE_JOINT_FULL would demand openness > 0.95 or < 0.05 instead
        of > 0.65 or < 0.35, making the last stage of a `turn_faucet` rubric strictly harder.
        Nothing reaches it today -- MOVE_JOINT_FULL is named only by the `turn_faucet` rubric and
        no task config declares `task_type: turn_faucet` -- so the change would alter no number
        now while silently redefining the stage for whoever adds that task. That decision belongs
        to them, with the thresholds in front of them, not to a drive-by cleanup.
        """
        return self.check_closed_mo_joint_large(obs) or self.check_opened_mo_joint_large(obs)
