"""Host-safe tilt math checks without importing the simulator."""
import ast
from pathlib import Path

import numpy as np
import pytest


SOURCE = Path(__file__).resolve().parents[1] / 'realm/environments/perturbations/v_view.py'
node = next(node for node in ast.parse(SOURCE.read_text()).body
            if isinstance(node, ast.FunctionDef) and node.name == '_camera_tilt_degrees')
namespace = {'np': np}
exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), 'exec'), namespace)
tilt = namespace['_camera_tilt_degrees']


@pytest.mark.parametrize('angle', [-30, -21, 0, 21, 30])
def test_camera_right_axis_tilt(angle):
    half = np.radians(angle) / 2
    quaternion = [0, -np.sin(half), 0, np.cos(half)]
    assert tilt(quaternion) == pytest.approx(angle)
    assert tilt(np.array(quaternion) * 2) == pytest.approx(angle)


def test_downward_view_does_not_tilt_image_right():
    assert tilt([np.sin(np.pi / 8), 0, 0, np.cos(np.pi / 8)]) == pytest.approx(0)


def test_zero_quaternion_is_rejected():
    with pytest.raises(ValueError):
        tilt([0, 0, 0, 0])
