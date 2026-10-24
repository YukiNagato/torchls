from __future__ import annotations

import math
import time
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass, field
from functools import partial

import torch
from torch import Tensor

from .schur import SchurLinearization, SchurPlan
from .sparse import BlockJacobian


@dataclass(frozen=True)
class ConjugateGradientConfig:
    rtol: float = 1e-6
    atol: float = 1e-10
    max_iterations: int = 200
    preconditioner: str | None = "block_jacobi"

    def __post_init__(self):
        if self.rtol < 0 or self.atol < 0 or self.max_iterations < 1:
            raise ValueError("Invalid CG tolerance or iteration limit")
        if self.preconditioner not in ("block_jacobi", None):
            raise ValueError("preconditioner must be block_jacobi or None")


@dataclass(frozen=True)
class TrustRegionConfig:
    lambda_initial: float = 1e-3
    lambda_min: float = 1e-12
    lambda_max: float = 1e12
    lambda_factor: float = 4.0
    step_quality_min: float = 1e-4
    max_trials: int = 20
    strategy: str = "fixed_factor"
    damping_mode: str = "identity"
    diagonal_min: float = 1e-6
    diagonal_max: float = 1e32

    def __post_init__(self):
        if not (0 < self.lambda_min <= self.lambda_initial <= self.lambda_max):
            raise ValueError("Require 0 < lambda_min <= lambda_initial <= lambda_max")
        if self.lambda_factor <= 1 or self.max_trials < 1 or not (0 <= self.step_quality_min < 1):
            raise ValueError("Invalid trust region settings")
        if self.strategy not in ("fixed_factor", "adaptive"):
            raise ValueError("Trust region strategy must be fixed_factor or adaptive")
        if self.damping_mode not in ("identity", "diagonal"):
            raise ValueError("damping_mode must be identity or diagonal")
        if not (0 < self.diagonal_min <= self.diagonal_max < math.inf):
            raise ValueError("Require finite positive diagonal damping bounds")


@dataclass(frozen=True)
class TerminationConfig:
    max_iterations: int = 50
    cost_tolerance: float = 1e-10
    gradient_tolerance: float = 1e-8
    parameter_tolerance: float = 1e-10
    target_cost: float | None = None
    max_time_seconds: float | None = None

    def __post_init__(self):
        if (
            self.max_iterations < 1
            or min(self.cost_tolerance, self.gradient_tolerance, self.parameter_tolerance) < 0
        ):
            raise ValueError("Invalid termination settings")
        if self.target_cost is not None and (
            not math.isfinite(self.target_cost) or self.target_cost < 0
        ):
            raise ValueError("target_cost must be finite and nonnegative")
        if self.max_time_seconds is not None and (
            not math.isfinite(self.max_time_seconds) or self.max_time_seconds <= 0
        ):
            raise ValueError("max_time_seconds must be finite and positive")


_DEFAULT_CG = ConjugateGradientConfig()
_DEFAULT_TRUST_REGION = TrustRegionConfig()
_DEFAULT_TERMINATION = TerminationConfig()


@dataclass(frozen=True)
class PCGResult:
    solution: Tensor
    iterations: int
    residual_norm: float
    converged: bool
    reason: str


@dataclass
class SolveSummary:
    """Histories include the initial point and each accepted update."""

    initial_cost: float
    final_cost: float
    iterations: int = 0  # outer iterations; trial count is separate
    accepted_steps: int = 0
    rejected_steps: int = 0
    linear_iterations: int = 0
    termination_reason: str = "max_iterations"
    gradient_norm: float = math.inf
    cost_history: list[float] = field(default_factory=list)
    lambda_history: list[float] = field(default_factory=list)
    linear_iterations_history: list[int] = field(default_factory=list)
    linear_converged_history: list[bool] = field(default_factory=list)
    linear_solver: str = "conjugate_gradient"
    jacobi_scaling: bool = False
    reduced_dim: int | None = None


def pcg(
    matvec: Callable[[Tensor], Tensor],
    rhs: Tensor,
    *,
    preconditioner: Callable[[Tensor], Tensor] | None = None,
    config: ConjugateGradientConfig = _DEFAULT_CG,
) -> PCGResult:
    """Preconditioned CG with true-residual checks before declaring convergence."""
    if rhs.ndim != 1 or rhs.numel() == 0:
        raise ValueError("PCG expects a nonempty vector")
    apply = (lambda x: x) if preconditioner is None else preconditioner
    x = torch.zeros_like(rhs)
    residual = rhs.clone()
    rhs_norm = float(torch.linalg.vector_norm(rhs))
    threshold = max(config.atol, config.rtol * rhs_norm)
    if not math.isfinite(rhs_norm):
        return PCGResult(x, 0, rhs_norm, False, "nonfinite")
    if rhs_norm <= threshold:
        return PCGResult(x, 0, rhs_norm, True, "residual_tolerance")
    z = apply(residual)
    rz = torch.dot(residual, z)
    direction = z.clone()
    iterations = 0
    reason = "max_iterations"
    for iterations in range(1, config.max_iterations + 1):
        product = matvec(direction)
        curvature = torch.dot(direction, product)
        if not bool(torch.isfinite(curvature) & torch.isfinite(rz)):
            reason = "nonfinite"
            break
        if float(curvature) <= 0 or float(rz) <= 0:
            reason = "nonpositive_curvature"
            break
        alpha = rz / curvature
        x = x + alpha * direction
        residual = residual - alpha * product
        recurrence_norm = float(torch.linalg.vector_norm(residual))
        if recurrence_norm <= threshold or iterations % 50 == 0:
            residual = rhs - matvec(x)
            true_norm = float(torch.linalg.vector_norm(residual))
            if true_norm <= threshold:
                return PCGResult(x, iterations, true_norm, True, "residual_tolerance")
            # Restart after replacing the recurrence residual.
            z = apply(residual)
            rz = torch.dot(residual, z)
            direction = z.clone()
            continue
        z = apply(residual)
        rz_new = torch.dot(residual, z)
        direction = z + (rz_new / rz) * direction
        rz = rz_new
    true_norm = float(torch.linalg.vector_norm(rhs - matvec(x)))
    return PCGResult(x, iterations, true_norm, true_norm <= threshold, reason)


def block_jacobi(jacobian: BlockJacobian, damping: float, *, diagonal=None):
    """Factor damped variable diagonal blocks without building J.T J."""
    diagonal = jacobian.diagonal_blocks() if diagonal is None else diagonal
    factors = tuple(
        torch.linalg.cholesky(
            block + damping * torch.eye(block.shape[-1], device=block.device, dtype=block.dtype)
        )
        for block in diagonal
    )

    def apply(vector):
        pieces = []
        for entry, factor in zip(jacobian.layout, factors, strict=True):
            part = vector[entry.offset : entry.offset + entry.count * entry.dim]
            pieces.append(
                torch.cholesky_solve(part.reshape(entry.count, entry.dim, 1), factor).reshape(-1)
            )
        return torch.cat(pieces)

    return apply


@torch.no_grad()
def solve(
    problem,
    initial_vals,
    *,
    linear_solver: str = "conjugate_gradient",
    conjugate_gradient: ConjugateGradientConfig = _DEFAULT_CG,
    trust_region: TrustRegionConfig | None = _DEFAULT_TRUST_REGION,
    termination: TerminationConfig = _DEFAULT_TERMINATION,
    return_summary: bool = False,
    jacobi_scaling: bool = False,
    eliminate=None,
    profile: bool = False,
):
    """Optimize with LM (or GN when trust_region=None).

    The first implementation is an eager Python solver. It returns detached
    estimates; differentiating through solve() is not part of this interface.
    The default path stores local Jacobians but never materializes J or J.T J.
    """
    if linear_solver not in ("conjugate_gradient", "dense_qr", "schur_dense", "schur_cg"):
        raise ValueError("Unknown linear solver")
    is_schur = linear_solver.startswith("schur_")
    if eliminate is not None and not is_schur:
        raise ValueError("eliminate requires a Schur linear solver")
    solve_start = time.perf_counter()

    def region(name):
        return torch.profiler.record_function("torchls/" + name) if profile else nullcontext()

    def timed_out():
        return (
            termination.max_time_seconds is not None
            and time.perf_counter() - solve_start >= termination.max_time_seconds
        )

    vals = initial_vals.detach()
    initial_cost = float(problem.cost(vals))
    if not math.isfinite(initial_cost):
        raise ValueError("Initial residuals are nonfinite")
    summary = SolveSummary(
        initial_cost,
        initial_cost,
        cost_history=[initial_cost],
        linear_solver=linear_solver,
        jacobi_scaling=jacobi_scaling,
    )
    damping = trust_region.lambda_initial if trust_region else 0.0
    rejection_factor = 2.0
    with region("analyze_schur"):
        schur_plan = SchurPlan(problem, eliminate, vals.device) if is_schur else None
    if linear_solver == "schur_dense" and schur_plan.dim > 8192:
        raise ValueError("Dense Schur is limited to 8192 retained dimensions; use schur_cg")
    summary.reduced_dim = schur_plan.dim if is_schur else None
    column_scale = None
    with region("linearize"):
        linearization = problem.linearize(vals)
    for iteration in range(termination.max_iterations):
        if timed_out():
            summary.termination_reason = "time_limit"
            break
        summary.iterations = iteration + 1
        residual, jacobian = linearization.residual, linearization.jacobian
        if not bool(torch.isfinite(residual).all()) or any(
            not bool(torch.isfinite(block).all())
            for group in jacobian.groups
            for block in group.blocks
        ):
            summary.termination_reason = "nonfinite_linearization"
            break
        gradient = linearization.gradient
        summary.gradient_norm = float(gradient.abs().max())
        if (
            termination.target_cost is not None
            and float(linearization.cost) <= termination.target_cost
        ):
            summary.termination_reason = "target_cost"
            break
        if summary.gradient_norm <= termination.gradient_tolerance:
            summary.termination_reason = "gradient_tolerance"
            break
        original_jacobian = jacobian
        with region("prepare_linear_system"):
            diagonal_damping = trust_region is not None and trust_region.damping_mode == "diagonal"
            needs_diagonal = is_schur or (
                linear_solver == "conjugate_gradient"
                and conjugate_gradient.preconditioner is not None
            )
            diagonal = (
                jacobian.diagonal_blocks()
                if needs_diagonal or diagonal_damping or (jacobi_scaling and column_scale is None)
                else None
            )
            if jacobi_scaling:
                if column_scale is None:
                    # Same fixed-at-initialization convention as Ceres. Adding
                    # one keeps unobservable columns well-defined.
                    column_scale = torch.cat(
                        [
                            (
                                1 / (1 + block.diagonal(dim1=-2, dim2=-1).clamp_min(0).sqrt())
                            ).reshape(-1)
                            for block in diagonal
                        ]
                    ).reshape(-1)
            solve_scale = column_scale
            if diagonal_damping:
                normal_diagonal = torch.cat(
                    [block.diagonal(dim1=-2, dim2=-1).reshape(-1) for block in diagonal]
                )
                if column_scale is not None:
                    normal_diagonal = normal_diagonal * column_scale.square()
                inverse_diagonal = normal_diagonal.clamp(
                    trust_region.diagonal_min, trust_region.diagonal_max
                ).rsqrt()
                solve_scale = (
                    inverse_diagonal if column_scale is None else column_scale * inverse_diagonal
                )
            if solve_scale is not None:
                # Solve in coordinates that turn either damping metric into
                # lambda I, then map the step back to the original tangent.
                jacobian = jacobian.scale_columns(solve_scale)
                linear_gradient = gradient * solve_scale
                if diagonal is not None:
                    diagonal = tuple(
                        block
                        * solve_scale[
                            entry.offset : entry.offset + entry.count * entry.dim
                        ].reshape(entry.count, entry.dim, 1)
                        * solve_scale[
                            entry.offset : entry.offset + entry.count * entry.dim
                        ].reshape(entry.count, 1, entry.dim)
                        for entry, block in zip(problem.layout, diagonal, strict=True)
                    )
            else:
                linear_gradient = gradient
            schur_linear = (
                SchurLinearization(
                    jacobian, diagonal, schur_plan, dense=linear_solver == "schur_dense"
                )
                if is_schur
                else None
            )
        old_cost = float(linearization.cost)
        accepted = False
        trial_limit = trust_region.max_trials if trust_region else 1
        for _ in range(trial_limit):
            if timed_out():
                summary.termination_reason = "time_limit"
                break
            summary.lambda_history.append(damping)
            try:
                if linear_solver == "dense_qr":
                    dense = jacobian.to_dense()
                    eye = torch.eye(problem.tangent_dim, dtype=vals.dtype, device=vals.device)
                    augmented = torch.cat((dense, math.sqrt(damping) * eye))
                    rhs = torch.cat((-residual, gradient.new_zeros(problem.tangent_dim)))
                    delta = torch.linalg.lstsq(augmented, rhs).solution
                    usable = bool(torch.isfinite(delta).all())
                elif is_schur:
                    with region("prepare_schur"):
                        system = schur_linear.damped(damping)
                        rhs = system.reduced_rhs(-linear_gradient)
                    with region("linear_solve"):
                        if linear_solver == "schur_dense":
                            factor = torch.linalg.cholesky(system.dense_matrix())
                            retained = torch.cholesky_solve(rhs[:, None], factor).reshape(-1)
                            iterations, converged = 1, True
                            usable = bool(torch.isfinite(retained).all())
                        else:
                            result = pcg(
                                system.matvec,
                                rhs,
                                preconditioner=system.preconditioner()
                                if conjugate_gradient.preconditioner
                                else None,
                                config=conjugate_gradient,
                            )
                            retained, iterations, converged = (
                                result.solution,
                                result.iterations,
                                result.converged,
                            )
                            usable = result.reason in (
                                "residual_tolerance",
                                "max_iterations",
                            ) and bool(torch.isfinite(retained).all())
                        delta = system.backsubstitute(retained)
                    summary.linear_iterations += iterations
                    summary.linear_iterations_history.append(iterations)
                    summary.linear_converged_history.append(converged)
                else:
                    precondition = (
                        block_jacobi(jacobian, damping, diagonal=diagonal)
                        if conjugate_gradient.preconditioner is not None
                        else None
                    )
                    with region("linear_solve"):
                        result = pcg(
                            partial(jacobian.normal_matvec, damping=damping),
                            -linear_gradient,
                            preconditioner=precondition,
                            config=conjugate_gradient,
                        )
                    summary.linear_iterations += result.iterations
                    summary.linear_iterations_history.append(result.iterations)
                    summary.linear_converged_history.append(result.converged)
                    delta = result.solution
                    # An inexact but finite iterate may still produce a valid LM step.
                    usable = result.reason in (
                        "residual_tolerance",
                        "max_iterations",
                    ) and bool(torch.isfinite(delta).all())
            except torch.linalg.LinAlgError:
                usable = False
            if usable:
                if solve_scale is not None:
                    delta = solve_scale * delta
                proposed = vals.retract(delta, problem.layout)
                with region("trial_residual"):
                    proposed_residual = problem.residuals(proposed)
                proposed_cost = float(0.5 * proposed_residual.double().square().sum())
                predicted = float(
                    -torch.dot(gradient.double(), delta.double())
                    - 0.5 * original_jacobian.matvec(delta).double().square().sum()
                )
                # Expanded difference avoids subtracting two nearly equal
                # objectives. Accumulate reductions in float64 even when the
                # Jacobians and linear solve use float32.
                old_r, new_r = residual.double(), proposed_residual.double()
                actual = float(0.5 * torch.dot(old_r - new_r, old_r + new_r))
                quality = actual / predicted if predicted > 0 else -math.inf
                accepted = math.isfinite(proposed_cost) and (
                    trust_region is None
                    or (actual > 0 and quality >= trust_region.step_quality_min)
                )
                if accepted:
                    vals = proposed
                    summary.accepted_steps += 1
                    summary.final_cost = proposed_cost
                    summary.cost_history.append(proposed_cost)
                    if trust_region:
                        if trust_region.strategy == "adaptive":
                            multiplier = max(1 / 3, 1 - (2 * min(quality, 1.0) - 1) ** 3)
                            damping = max(
                                trust_region.lambda_min,
                                min(trust_region.lambda_max, damping * multiplier),
                            )
                            rejection_factor = 2.0
                        else:
                            damping = (
                                max(
                                    trust_region.lambda_min,
                                    damping / trust_region.lambda_factor,
                                )
                                if quality > 0.75
                                else damping
                            )
                    # Keep the summary gradient at the returned solution, even
                    # when cost/step tolerances stop immediately after a step.
                    with region("linearize"):
                        linearization = problem.linearize(vals)
                    summary.gradient_norm = float(linearization.gradient.abs().max())
                    if (
                        termination.target_cost is not None
                        and proposed_cost <= termination.target_cost
                    ):
                        summary.termination_reason = "target_cost"
                    elif summary.gradient_norm <= termination.gradient_tolerance:
                        summary.termination_reason = "gradient_tolerance"
                    elif abs(actual) <= termination.cost_tolerance * max(
                        old_cost, torch.finfo(vals.dtype).tiny
                    ):
                        summary.termination_reason = "cost_tolerance"
                    elif float(torch.linalg.vector_norm(delta)) <= termination.parameter_tolerance:
                        # Absolute tangent threshold: storage coordinates for
                        # manifold variables need not have meaningful norms.
                        summary.termination_reason = "parameter_tolerance"
                    break
            summary.rejected_steps += 1
            if trust_region is None or damping >= trust_region.lambda_max:
                break
            if trust_region.strategy == "adaptive":
                damping = min(trust_region.lambda_max, damping * rejection_factor)
                rejection_factor = min(
                    rejection_factor * 2, trust_region.lambda_max / trust_region.lambda_min
                )
            else:
                damping = min(trust_region.lambda_max, damping * trust_region.lambda_factor)
        if not accepted:
            if summary.termination_reason != "time_limit":
                summary.termination_reason = "no_acceptable_step"
            break
        if summary.termination_reason != "max_iterations":
            break
    return (vals, summary) if return_summary else vals
