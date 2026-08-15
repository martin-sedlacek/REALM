"""Gain preparation for REALM's Jacobian-weighted controllers.

`droid_joint_controller.py` and `droid_ee_controller.py` both build their control law out of four
gains -- joint-space Kq/Kqd and task-space Kx/Kxd -- given in `realm/config/robots/*.yaml` as
scalars, per-DOF lists or full matrices.
"""
import omnigibson as og
import torch as th


def prepare_gain(gain):
    """A gain given as a scalar, list or tensor, as a square matrix on the simulator's device.

    A 1-D gain is the diagonal of the matrix the control law wants; a 2-D one is used as given.

    Returned detached and as a plain tensor, never a `th.nn.Parameter`: nothing here is ever
    optimized, and under OG 3.9.1 the compute backend converts controls with `Tensor.numpy()`,
    which raises on grad-tracking tensors. Values are identical -- only `requires_grad` differs.
    """
    tensor = gain.to(th.Tensor()) if th.is_tensor(gain) else th.tensor(gain).to(th.Tensor())
    if tensor.dim() == 1:
        tensor = th.diag(tensor)
    elif tensor.dim() != 2:
        raise ValueError(f"Gain tensor must be 1D or 2D, but got {tensor.dim()}D.")
    return tensor.detach().to(og.sim.device)
