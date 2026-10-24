from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import wraps
from typing import Literal

from torch import Tensor

JacMode = Literal["auto", "reverse", "forward"]


@dataclass(frozen=True)
class Cost:
    """One batch of residuals. Functions receive local values, not Var handles.

    Variable IDs select observations. Tensor arguments default to a leading
    observation axis (a length-one axis broadcasts). Use in_dims=None for an
    argument shared by every observation, e.g. camera intrinsics.
    """

    residual_fn: Callable[..., Tensor]
    args: tuple
    jac_mode: JacMode = "auto"
    batch_size: int | None = None
    in_dims: tuple[int | None, ...] | None = None
    jacobian_fn: Callable[..., tuple[Tensor, ...]] | None = None

    def __post_init__(self):
        if self.jac_mode not in ("auto", "reverse", "forward"):
            raise ValueError("jac_mode must be auto, reverse or forward")
        if self.batch_size is not None and self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.in_dims is not None and (
            len(self.in_dims) != len(self.args) or any(x not in (0, None) for x in self.in_dims)
        ):
            raise ValueError("in_dims must contain one 0 or None per argument")

    @staticmethod
    def factory(fn=None, **options):
        """Decorate a single-observation residual to create batched Costs.

        A custom jacobian_fn returns one (residual_dim, tangent_dim) matrix
        per Var argument, in argument order, using the same retractions.
        """

        def decorate(residual_fn):
            @wraps(residual_fn)
            def make(*args):
                return Cost(residual_fn, args, **options)

            return make

        return decorate(fn) if fn is not None else decorate
