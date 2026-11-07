"""Fixed sparse linear least squares with reusable dense, multi-RHS factors.

This is intended for small/medium column counts. COO rows may have arbitrary
coefficients; neither rotation graphs nor a particular RHS width are assumed.
The normal-product plan costs sum(nnz_per_row**2), the factor costs O(n**2).
"""

from dataclasses import dataclass
import math

import torch
from torch import Tensor


class FixedLinearSystem:
    def __init__(self, matrix: Tensor, *, max_dense_columns: int = 2048):
        if matrix.ndim != 2 or min(matrix.shape) < 1:
            raise ValueError("matrix must be a nonempty two-dimensional tensor")
        if matrix.dtype not in (torch.float32, torch.float64):
            raise ValueError("matrix must use float32 or float64")
        if max_dense_columns < 1 or matrix.shape[1] > max_dense_columns:
            raise ValueError("matrix exceeds max_dense_columns; use an iterative solver")
        matrix = matrix.detach().to_sparse_coo().coalesce().clone()
        self.shape = matrix.shape
        self.rows, self.cols = matrix.indices().unbind(0)
        self.values = matrix.values()
        if not bool(torch.isfinite(self.values).all()):
            raise ValueError("matrix coefficients must be finite")
        counts = torch.bincount(self.rows, minlength=self.shape[0])
        per_entry = counts[self.rows]
        entries = torch.arange(len(self.values), device=matrix.device)
        left = entries.repeat_interleave(per_entry)
        offsets = counts.cumsum(0) - counts
        starts = per_entry.cumsum(0) - per_entry
        right = offsets[self.rows[left]] + torch.arange(len(left), device=matrix.device) - starts[left]
        self._normal_index = self.cols[left] * self.shape[1] + self.cols[right]
        self._normal_row = self.rows[left]
        self._normal_value = self.values[left] * self.values[right]

    def _input(self, value, rows):
        if value.ndim not in (1, 2) or value.shape[0] != rows:
            raise ValueError("incompatible vector or matrix shape")
        if value.device != self.values.device or value.dtype != self.values.dtype:
            raise ValueError("input and system must have matching device and dtype")
        return value[:, None] if value.ndim == 1 else value

    def matvec(self, value: Tensor) -> Tensor:
        v = self._input(value, self.shape[1])
        out = v.new_zeros(self.shape[0], v.shape[1]).index_add_(
            0, self.rows, self.values[:, None] * v[self.cols])
        return out[:, 0] if value.ndim == 1 else out

    def rmatvec(self, value: Tensor) -> Tensor:
        v = self._input(value, self.shape[0])
        out = v.new_zeros(self.shape[1], v.shape[1]).index_add_(
            0, self.cols, self.values[:, None] * v[self.rows])
        return out[:, 0] if value.ndim == 1 else out

    def factor(self, weights: Tensor | None = None, *, ridge: float = 0.):
        if not math.isfinite(ridge) or ridge < 0:
            raise ValueError("ridge must be finite and nonnegative")
        if weights is not None:
            if weights.ndim != 1:
                raise ValueError("weights must have one entry per matrix row")
            self._input(weights, self.shape[0])
            if not bool(torch.isfinite(weights).all() & (weights >= 0).all()):
                raise ValueError("weights must be finite and nonnegative")
            weights = weights.detach().clone()
        products = self._normal_value
        if weights is not None:
            products = products * weights[self._normal_row]
        n = self.shape[1]
        normal = self.values.new_zeros(n*n).index_add_(0, self._normal_index, products).reshape(n, n)
        normal.diagonal().add_(ridge)
        return LinearFactor(self, torch.linalg.cholesky(normal), weights)


@dataclass
class LinearFactor:
    system: FixedLinearSystem
    cholesky: Tensor
    weights: Tensor | None

    def solve(self, rhs: Tensor) -> Tensor:
        """Solve the factored normal system for one or several right-hand sides."""
        value = self.system._input(rhs, self.system.shape[1])
        result = torch.cholesky_solve(value, self.cholesky)
        return result[:, 0] if rhs.ndim == 1 else result

    def least_squares(self, target: Tensor) -> Tensor:
        """Solve min ||sqrt(weights) (A x - target)||² + ridge ||x||²."""
        self.system._input(target, self.system.shape[0])
        weighted = target
        if self.weights is not None:
            weighted = target * (self.weights if target.ndim == 1 else self.weights[:, None])
        return self.solve(self.system.rmatvec(weighted))


@dataclass
class LADResult:
    solution: Tensor
    iterations: int
    converged: bool


def least_absolute_deviations(system: FixedLinearSystem, target: Tensor, *,
                             factor: LinearFactor | None = None, max_iterations: int = 100,
                             rho: float = 1., absolute_tolerance: float = 1e-4,
                             relative_tolerance: float = 1e-2) -> LADResult:
    """ADMM for entrywise ||A X - target||_1, sharing a factor across RHS columns.

    Pass a precomputed unweighted factor to reuse its numerical factorization
    across calls. Ridge stabilization, if requested when factoring, is retained.
    Stopping norms are over all RHS entries, as for the equivalent stacked system.
    """
    if (max_iterations < 1 or not math.isfinite(rho) or rho <= 0
            or not all(math.isfinite(v) and v >= 0 for v in (absolute_tolerance, relative_tolerance))):
        raise ValueError("invalid ADMM iteration limit, rho or tolerance")
    system._input(target, system.shape[0])
    factor = system.factor() if factor is None else factor
    if factor.system is not system or factor.weights is not None:
        raise ValueError("ADMM requires an unweighted factor of this system")
    z, u = torch.zeros_like(target), torch.zeros_like(target)
    columns = 1 if target.ndim == 1 else target.shape[1]
    target_norm = target.norm()
    eps_primal = math.sqrt(target.numel()) * absolute_tolerance
    eps_dual = math.sqrt(system.shape[1] * columns) * absolute_tolerance
    for iteration in range(1, max_iterations + 1):
        x = factor.least_squares(target + z - u)
        ax = system.matvec(x)
        v = ax - target + u
        old_z, z = z, v.sign() * (v.abs() - 1/rho).clamp_min(0)
        primal = ax - z - target
        u = u + primal
        primal_limit = eps_primal + relative_tolerance * torch.maximum(
            target_norm, torch.maximum(ax.norm(), z.norm()))
        dual_limit = eps_dual + relative_tolerance * rho * system.rmatvec(u).norm()
        dual = rho * system.rmatvec(z - old_z).norm()
        if bool((primal.norm() < primal_limit) & (dual < dual_limit)):
            return LADResult(x, iteration, True)
    return LADResult(x, max_iterations, False)
