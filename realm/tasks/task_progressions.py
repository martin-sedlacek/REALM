from collections import OrderedDict

TASK_PROGRESSIONS = {
    "put": OrderedDict(
        {
            "REACH": False,
            "GRASP": False,
            "LIFT_SLIGHT": False,
            "MOVE_CLOSE": False,
            "PLACE_INTO": False
        }
    ),
    "pick": OrderedDict(
        {
            "REACH": False,
            "GRASP": False,
            "LIFT_LARGE": False
        }
    ),
    "rotate": OrderedDict(
        {
            "REACH": False,
            "GRASP": False,
            "ROTATED": False
        }
    ),
    "push": OrderedDict(
        {
            "REACH": False,
            "TOUCH": False,
            "TOGGLED_ON": False
        }
    ),
    "stack": OrderedDict(
        {
            "REACH": False,
            "GRASP": False,
            "LIFT_SLIGHT": False,
            "MOVE_CLOSE": False,
            "PLACE_ONTO": False
        }
    ),
    "open_drawer": OrderedDict(
        {
            "REACH": False,
            "TOUCH_AND_MOVE_JOINT": False,
            "OPEN_JOINT_SMALL": False,
            "OPEN_JOINT_LARGE": False,
            "OPEN_JOINT_FULL": False
        }
    ),
    "close_drawer": OrderedDict(
        {
            "REACH": False,
            "TOUCH_AND_MOVE_JOINT": False,
            "CLOSE_JOINT_SMALL": False,
            "CLOSE_JOINT_LARGE": False,
            "CLOSE_JOINT_FULL": False
        }
    ),
    # TODO:
    "pour": OrderedDict(
        {
            "REACH": False,
            "GRASP": False,
            "LIFT_SLIGHT": False,
            "MOVE_CLOSE": False,
            "POUR": False
        }
    ),
    "turn_faucet": OrderedDict(
        {
            "REACH": False,
            "TOUCH_AND_MOVE_JOINT": False,
            "MOVE_JOINT_SMALL": False,
            "MOVE_JOINT_LARGE": False,
            "MOVE_JOINT_FULL": False
        }
    )
}
