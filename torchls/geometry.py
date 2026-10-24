"""Small tensor-only geometry primitives, compatible with torch.func."""

import math

import torch
from torch import Tensor

from .variables import Var


def skew(vector: Tensor) -> Tensor:
    x, y, z = vector.unbind(-1)
    zero = torch.zeros_like(x)
    return torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), dim=-1).reshape(
        *vector.shape[:-1], 3, 3
    )


def _so3_coefficients(rotation: Tensor):
    # Retain a length-one axis: scalar sinc JVPs in PyTorch 2.11 can
    # promote float32 tangents to float64 and break later matrix products.
    theta2 = rotation.square().sum(dim=-1, keepdim=True)
    safe_theta2 = theta2.clamp_min(torch.finfo(rotation.dtype).eps)
    theta = safe_theta2.sqrt()
    small = theta2 < 1e-4
    a = torch.where(small, 1 - theta2 / 6 + theta2.square() / 120, torch.sinc(theta / math.pi))
    b = torch.where(
        small,
        0.5 - theta2 / 24 + theta2.square() / 720,
        0.5 * torch.sinc(theta / (2 * math.pi)).square(),
    )
    c = torch.where(
        small,
        1 / 6 - theta2 / 120 + theta2.square() / 5040,
        (1 - torch.sinc(theta / math.pi)) / safe_theta2,
    )
    return a[..., None], b[..., None], c[..., None]


def so3_exp(rotation: Tensor) -> Tensor:
    """Axis-angle vectors (..., 3) to rotation matrices (..., 3, 3)."""
    omega = skew(rotation)
    a, b, _ = _so3_coefficients(rotation)
    eye = torch.eye(3, dtype=rotation.dtype, device=rotation.device)
    return eye + a * omega + b * (omega @ omega)


def se3_exp(tangent: Tensor) -> Tensor:
    """SE(3) exponential; tangent order is [translation(3), rotation(3)]."""
    translation, rotation = tangent[..., :3], tangent[..., 3:]
    omega = skew(rotation)
    _, b, c = _so3_coefficients(rotation)
    eye = torch.eye(3, dtype=tangent.dtype, device=tangent.device)
    v = eye + b * omega + c * (omega @ omega)
    top = torch.cat((so3_exp(rotation), (v @ translation[..., None])), dim=-1)
    bottom = tangent.new_tensor([0, 0, 0, 1]).expand(*tangent.shape[:-1], 1, 4)
    return torch.cat((top, bottom), dim=-2)


def _se2_retract(pose: Tensor, delta: Tensor) -> Tensor:
    angle = pose[2]
    dtheta = delta[2]
    a = torch.sinc(delta[2:3] / math.pi)[0]
    b = (0.5 * delta[2:3] * torch.sinc(delta[2:3] / (2 * math.pi)).square())[0]
    local_x = a * delta[0] - b * delta[1]
    local_y = b * delta[0] + a * delta[1]
    return torch.stack(
        (
            pose[0] + angle.cos() * local_x - angle.sin() * local_y,
            pose[1] + angle.sin() * local_x + angle.cos() * local_y,
            angle + dtheta,
        )
    )


class SE2Var(Var, default_factory=lambda: torch.zeros(3), tangent_dim=3, retract_fn=_se2_retract):
    """SE(2) poses stored as [x, y, angle], updated on the right."""


class SE3Var(
    Var,
    default_factory=lambda: torch.eye(4),
    tangent_dim=6,
    retract_fn=lambda pose, delta: pose @ se3_exp(delta),
):
    """SE(3) poses stored as 4x4 matrices, updated on the right."""
