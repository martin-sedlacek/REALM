"""Registers REALM's custom controllers with OmniGibson 3.9.1.

Two things have to be in place before a REALM robot config that names, say, ``CustomJointController``
can be loaded:

1. **The name must resolve to a class.** OmniGibson auto-registers controller subclasses under their
   own *class* name, but REALM's configs refer to them by REALM-facing aliases, so the aliases are
   added to ``REGISTERED_CONTROLLERS`` explicitly. The mapping is carried over verbatim from the
   pre-3.9.1 ``realm/environments/env_base.py``; note in particular that ``CustomJointController`` is
   the Jacobian-weighted controller in ``droid_joint_controller.py``, while the plain
   ``IndividualJointPDController`` name refers to the scalar-gain one in
   ``custom_joint_controller.py``. They are different classes with the same class name.

2. **A default controller config must exist under that name.**
   ``Robot._generate_controller_config`` looks the config's ``name`` up in
   ``self._default_controller_config[group]`` to find the base dict it merges the robot YAML's
   overrides into, and raises ``KeyError`` for anything it does not know. Pre-3.9.1, each REALM robot
   class supplied these via its own ``_default_controller_config`` override; robots are not Python
   classes any more in 3.9.1, so the entries are injected by wrapping ``Robot``'s property instead.

The injected entries are clones of OmniGibson's own defaults for the equivalent stock controller with
``name`` swapped, plus ``link_name`` for the two Jacobian-weighted controllers -- they look up the
task-space Jacobian by eef link name, which OmniGibson's own configs have no reason to carry. Gains
(``Kq``/``Kqd``/``Kx``/``Kxd``, ``kp``/``kd``) are deliberately *not* defaulted here: they are
hyperparameters that belong in ``realm/config/robots/*.yaml``, and a missing one should fail loudly
rather than silently pick up a value invented in this file.

Importing this module also has one side effect inherited from the pre-3.9.1 layout:
``droid_gripper_controller.MultiFingerGripperController`` subclasses OmniGibson's controller of the
same class name, so OmniGibson's auto-registration rebinds the stock ``MultiFingerGripperController``
entry to REALM's subclass. That was already true before the port. It is harmless for REALM, whose
robot configs always name ``CustomGripperController`` explicitly.
"""

from copy import deepcopy

from omnigibson.controllers import REGISTERED_CONTROLLERS
from omnigibson.robots.robot import Robot

from realm.robots.custom_joint_controller import IndividualJointPDController
from realm.robots.droid_ee_controller import DroidEndEffectorController
from realm.robots.droid_gripper_controller import MultiFingerGripperController as DROIDGripperController
from realm.robots.droid_joint_controller import IndividualJointPDController as DROIDJointPDController
from realm.robots.padspring_gripper_controller import PadSpringGripperController

# REALM-facing controller name -> implementing class. Mirrors pre-3.9.1 env_base.py exactly, plus
# PadSpringGripperController -- the compliant-pad 2F-85 variant, which needs its own entry because it
# claims three gripper DOFs instead of one. See its module docstring.
REALM_CONTROLLERS = {
    "IndividualJointPDController": IndividualJointPDController,
    "DroidEndEffectorController": DroidEndEffectorController,
    "CustomJointController": DROIDJointPDController,
    "CustomGripperController": DROIDGripperController,
    "PadSpringGripperController": PadSpringGripperController,
}

# Arm-group controllers, split by which OmniGibson default they clone and whether they need the eef
# link name for their Jacobian lookup.
_ARM_JOINT_CONTROLLERS = ("CustomJointController", "IndividualJointPDController")
_ARM_IK_CONTROLLERS = ("DroidEndEffectorController",)
_NEEDS_EEF_LINK_NAME = ("CustomJointController", "DroidEndEffectorController")

_GRIPPER_CONTROLLERS = ("CustomGripperController", "PadSpringGripperController")

_PATCHED_FLAG = "_realm_default_controller_config_patched"


def _realm_arm_configs(robot, arm):
    """Base config dicts for REALM's arm controllers, keyed by REALM controller name."""
    joint_base = robot._default_arm_joint_controller_configs[arm]
    ik_base = robot._default_arm_ik_controller_configs[arm]

    configs = {}
    for name in _ARM_JOINT_CONTROLLERS:
        configs[name] = deepcopy(joint_base)
    for name in _ARM_IK_CONTROLLERS:
        cfg = deepcopy(ik_base)
        # The IK default carries no motor_type; REALM's ee controller requires one (it forces
        # "effort" internally but still takes the argument).
        cfg.setdefault("motor_type", "position")
        configs[name] = cfg

    eef_link_name = robot.eef_link_names[arm]
    for name, cfg in configs.items():
        cfg["name"] = name
        if name in _NEEDS_EEF_LINK_NAME:
            cfg["link_name"] = eef_link_name
    return configs


def _realm_gripper_configs(robot, arm):
    """Base config dicts for REALM's gripper controllers, keyed by REALM controller name."""
    gripper_base = robot._default_gripper_multi_finger_controller_configs[arm]
    configs = {}
    for name in _GRIPPER_CONTROLLERS:
        cfg = deepcopy(gripper_base)
        cfg["name"] = name
        configs[name] = cfg
    return configs


def _patch_default_controller_config():
    """Wrap ``Robot._default_controller_config`` so it also advertises REALM's controllers."""
    if getattr(Robot, _PATCHED_FLAG, False):
        return

    base_property = Robot._default_controller_config

    @property
    def _default_controller_config(self):
        cfg = base_property.fget(self)
        if not self.is_manipulation:
            return cfg
        for arm in self.arm_names:
            cfg.setdefault(f"arm_{arm}", {}).update(_realm_arm_configs(self, arm))
            cfg.setdefault(f"gripper_{arm}", {}).update(_realm_gripper_configs(self, arm))
        return cfg

    Robot._default_controller_config = _default_controller_config
    setattr(Robot, _PATCHED_FLAG, True)


def register_realm_controllers():
    """Make REALM's controllers loadable by name. Idempotent; safe to call on every import."""
    REGISTERED_CONTROLLERS.update(REALM_CONTROLLERS)
    _patch_default_controller_config()
