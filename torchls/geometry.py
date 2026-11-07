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


def sim3_exp(tangent: Tensor) -> Tensor:
    """Sim(3) exponential to (..., 4, 4), with positive scale in the 3x3 block.

    Tangent order: [translation(3), rotation(3), log_scale(1)]. Translation is
    coupled to both rotation and scale through the matrix exponential, not a
    direct copy of the tangent. Unlike SO(3)/SE(3), the finite Sim(3) update uses
    torch.matrix_exp for stable behavior at zero rotation/scale and their limits.
    Sim3Var linearization uses sim3_retract_jvp and does not differentiate Exp.
    """
    eye = torch.eye(3, dtype=tangent.dtype, device=tangent.device)
    algebra = skew(tangent[..., 3:6]) + tangent[..., 6:7, None] * eye
    top = torch.cat((algebra, tangent[..., :3, None]), dim=-1)
    bottom = tangent.new_zeros((*tangent.shape[:-1], 1, 4))
    return torch.matrix_exp(torch.cat((top, bottom), dim=-2))


def so3_retract_jvp(rotation: Tensor, direction: Tensor) -> Tensor:
    """D_delta (R @ Exp(delta)) at delta=0, applied to direction (..., 3)."""
    return rotation @ skew(direction)


def se3_retract_jvp(pose: Tensor, direction: Tensor) -> Tensor:
    """Right SE(3) retraction derivative at zero; order [translation, rotation].

    Returns the storage-space differential (..., 4, 4), not an updated pose.
    """
    rotation = pose[..., :3, :3]
    top = torch.cat(
        (so3_retract_jvp(rotation, direction[..., 3:6]), rotation @ direction[..., :3, None]),
        dim=-1,
    )
    bottom = torch.zeros_like(top[..., :1, :])
    return torch.cat((top, bottom), dim=-2)


def sim3_retract_jvp(pose: Tensor, direction: Tensor) -> Tensor:
    """Right Sim(3) retraction derivative at zero; order [translation, rotation, scale].

    The last direction component is log-scale. The pose's 3x3 block stores sR.
    Returns a (..., 4, 4) differential; supports broadcasting batch dimensions.
    """
    scaled_rotation = pose[..., :3, :3]
    rotation_part = (
        so3_retract_jvp(scaled_rotation, direction[..., 3:6])
        + scaled_rotation * direction[..., 6:7, None]
    )
    top = torch.cat((rotation_part, scaled_rotation @ direction[..., :3, None]), dim=-1)
    return torch.cat((top, torch.zeros_like(top[..., :1, :])), dim=-2)


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
    retract_jvp_fn=se3_retract_jvp,
):
    """SE(3) 4x4 poses; right update with [translation(3), rotation(3)] tangent."""


class SO3Var(
    Var,
    default_factory=lambda: torch.eye(3),
    tangent_dim=3,
    retract_fn=lambda rotation, delta: rotation @ so3_exp(delta),
    retract_jvp_fn=so3_retract_jvp,
):
    """SO(3) 3x3 rotations; right update with a 3D axis-angle tangent."""


class Sim3Var(
    Var,
    default_factory=lambda: torch.eye(4),
    tangent_dim=7,
    retract_fn=lambda pose, delta: pose @ sim3_exp(delta),
    retract_jvp_fn=sim3_retract_jvp,
):
    """Sim(3) 4x4 poses storing [sR,t;0,1], s>0, updated on the right.

    Tangent order: [translation(3), rotation(3), log_scale(1)].
    """
