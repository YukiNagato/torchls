"""Memory-based vmap chunk selection shared by both modeling interfaces."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class AutoBatchSize:
    """Estimate a chunk size for temporary residual/AD working tensors.

    This is a heuristic, not a hard memory limit: arbitrary residual functions
    may create intermediates whose sizes cannot be inferred from their inputs.
    Full gathered inputs and the returned Jacobian are outside this budget.
    Decrease the budget or specify an integer batch_size for such functions.
    None continues to mean an unchunked vmap, independently of this policy.
    """

    memory_budget_bytes: int = 256 * 1024**2
    safety_factor: float = 8.0

    def __post_init__(self):
        if (
            isinstance(self.memory_budget_bytes, bool)
            or not isinstance(self.memory_budget_bytes, int)
            or self.memory_budget_bytes < 1
        ):
            raise ValueError("memory_budget_bytes must be a positive integer")
        if not math.isfinite(self.safety_factor) or self.safety_factor < 1:
            raise ValueError("safety_factor must be finite and at least one")

    def resolve(self, *, count, input_bytes, residual_dim, tangent_dim, mode, itemsize, device):
        derivative_width = (
            residual_dim
            if mode == "reverse" or (mode == "auto" and residual_dim <= tangent_dim)
            else tangent_dim
        )
        # Include the local values, output, full Jacobian, tangent and AD basis.
        row_bytes = input_bytes + itemsize * (
            residual_dim * (tangent_dim + 1) + tangent_dim + derivative_width**2
        )
        budget = self.memory_budget_bytes
        if device.type == "cuda":
            free, _ = torch.cuda.mem_get_info(device)
            budget = min(budget, max(1, free // 4))
        chunk = max(1, int(budget / max(1, row_bytes) / self.safety_factor))
        return None if chunk >= count else chunk


BatchSize = int | AutoBatchSize | None
