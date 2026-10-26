"""Block robust losses, evaluated on each observation's squared residual norm."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from torch import Tensor


class Loss(ABC):
    """Define rho(s), rho'(s), rho''(s) for F = 0.5 * sum(rho(||r_i||²)).

    Implement ``evaluate`` using tensor operations, preserving shape and device.
    Losses must have nonnegative values and first derivatives. Linearization
    uses Ceres's clamped curvature correction: nonpositive rho'' is ignored;
    positive rho'' receives the full rank-one correction.
    """

    @abstractmethod
    def evaluate(self, squared_norm: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Return the loss value and first two derivatives with respect to s."""

    def value(self, squared_norm: Tensor) -> Tensor:
        """Evaluate only rho(s); override to avoid unused derivative work."""
        return self.evaluate(squared_norm)[0]

    def _linearize(self, residual, blocks, squared_norm, evaluation):
        _, first, second = evaluation
        tiny = torch.finfo(squared_norm.dtype).tiny
        positive = (squared_norm > 0) & (second > 0)
        ratio = torch.where(positive, 2 * squared_norm * second / first.clamp_min(tiny), 0)
        # Equivalent to 1 - sqrt(1 + ratio), without cancellation near zero.
        alpha = -ratio / (1 + (1 + ratio).sqrt())
        correction = (alpha / squared_norm.clamp_min(tiny)).to(residual.dtype)
        weight = first.sqrt().to(residual.dtype)
        corrected = tuple(
            weight[:, None, None]
            * (
                block
                - correction[:, None, None]
                * residual[:, :, None]
                * torch.einsum("nr,nrd->nd", residual, block)[:, None, :]
            )
            for block in blocks
        )
        model_residual = residual * (weight / (1 - alpha).to(residual.dtype))[:, None]
        return model_residual, corrected

    def _reduction(self, before, after, squared_difference):
        return self.value(before) - self.value(after)


@dataclass(frozen=True)
class HuberLoss(Loss):
    """Huber loss with a threshold on the norm of a complete residual block.

    rho(s) = s for ||r|| <= delta, otherwise 2 * delta * ||r|| - delta².
    The robust GN model scales raw residuals/Jacobians by sqrt(rho'(s));
    its squared norm is not the objective used for trial acceptance.
    """

    delta: float = 1.0

    def __post_init__(self):
        if not math.isfinite(self.delta) or self.delta <= 0:
            raise ValueError("Huber delta must be finite and positive")

    def evaluate(self, squared_norm: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        safe_squared = squared_norm.clamp_min(torch.finfo(squared_norm.dtype).tiny)
        norm = safe_squared.sqrt()
        outlier = squared_norm > self.delta * self.delta
        first = torch.where(outlier, self.delta / norm, torch.ones_like(norm))
        value = torch.where(outlier, self.delta * (2 * norm - self.delta), squared_norm)
        second = torch.where(outlier, -0.5 * first / safe_squared, torch.zeros_like(norm))
        return value, first, second

    def value(self, squared_norm: Tensor) -> Tensor:
        norm = squared_norm.clamp_min(torch.finfo(squared_norm.dtype).tiny).sqrt()
        return torch.where(
            squared_norm > self.delta * self.delta,
            self.delta * (2 * norm - self.delta),
            squared_norm,
        )

    def _linearize(self, residual, blocks, squared_norm, evaluation):
        weight = evaluation[1].sqrt().to(residual.dtype)
        return residual * weight[:, None], tuple(block * weight[:, None, None] for block in blocks)

    def _reduction(self, before, after, squared_difference):
        # A divided difference of rho preserves small improvements even when
        # subtracting two rounded squared norms/loss values would give zero.
        a, b = before.sqrt(), after.sqrt()
        low, high = torch.minimum(a, b), torch.maximum(a, b)
        tiny = torch.finfo(before.dtype).tiny
        shortfall = (self.delta - low).clamp_min(0)
        crossing = (2 * self.delta - shortfall * (shortfall / (high - low).clamp_min(tiny))) / (
            a + b
        ).clamp_min(tiny)
        coefficient = torch.where(
            high <= self.delta,
            torch.ones_like(before),
            torch.where(low >= self.delta, 2 * self.delta / (a + b).clamp_min(tiny), crossing),
        )
        return squared_difference * coefficient
