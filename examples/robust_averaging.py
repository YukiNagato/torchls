"""Robust rotation averaging and translation-direction averaging.

Run from the repository root:
    PYTHONPATH=. python examples/robust_averaging.py --compare-l2
    PYTHONPATH=. python examples/robust_averaging.py --device cuda

Conventions:
    R_i maps world coordinates into camera i, and C_i is its world center.
    R_ji = R_j @ R_i.T; t_ji points from camera j to camera i, expressed in j:
    t_ji = normalize(R_j @ (C_i - C_j)).

Inputs are noisy pairwise poses and approximate initial values. Synthetic
ground truth generates measurements/initialization and evaluates the result;
it is not added as a set of pose constraints. This is a modeling example,
not a full GLOMAP initializer or a performance benchmark.
"""

import argparse
import json
import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

import torchls as tls


def so3_log(rotation):
    """SO(3) log through a quaternion, including small angles and near pi.

    Tensor-only branch selection works inside vmap + jacrev/jacfwd.
    """
    r00, r01, r02 = rotation[..., 0, :].unbind(-1)
    r10, r11, r12 = rotation[..., 1, :].unbind(-1)
    r20, r21, r22 = rotation[..., 2, :].unbind(-1)
    magnitudes = (
        torch.stack(
            (1 + r00 + r11 + r22, 1 + r00 - r11 - r22, 1 - r00 + r11 - r22, 1 - r00 - r11 + r22),
            dim=-1,
        )
        .clamp_min(1e-12)
        .sqrt()
    )
    candidates = torch.stack(
        (
            torch.stack((magnitudes[..., 0].square(), r21 - r12, r02 - r20, r10 - r01), -1),
            torch.stack((r21 - r12, magnitudes[..., 1].square(), r10 + r01, r02 + r20), -1),
            torch.stack((r02 - r20, r10 + r01, magnitudes[..., 2].square(), r12 + r21), -1),
            torch.stack((r10 - r01, r02 + r20, r12 + r21, magnitudes[..., 3].square()), -1),
        ),
        dim=-2,
    ) / (2 * magnitudes[..., :, None])
    index = magnitudes.argmax(-1)[..., None, None].expand(*magnitudes.shape[:-1], 1, 4)
    quaternion = candidates.gather(-2, index).squeeze(-2)
    quaternion = torch.where(quaternion[..., :1] < 0, -quaternion, quaternion)
    length = quaternion[..., 1:].square().sum(-1, keepdim=True).clamp_min(1e-24).sqrt()
    angle = 2 * torch.atan2(length, quaternion[..., :1])
    return quaternion[..., 1:] * (angle / length)


class SO3Var(
    tls.Var,
    default_factory=lambda: torch.eye(3),
    tangent_dim=3,
    retract_fn=lambda rotation, delta: rotation @ tls.so3_exp(delta),
):
    """Store a rotation matrix and optimize its three-dimensional tangent."""


class PositiveScaleVar(
    tls.Var,
    default_factory=lambda: torch.ones(1),
    tangent_dim=1,
    retract_fn=lambda scale, delta: scale * delta.exp(),
):
    """Positive inverse-baseline scale, with a multiplicative/log update."""


def rotation_error(rotation_i, rotation_j, measured_relative):
    # Raw three-vector in radians. Huber is declared on the cost, not here.
    return so3_log(rotation_j @ rotation_i.T @ measured_relative.T)


def rotation_anchor(rotation):
    return so3_log(rotation)  # Choose R_0 = I as the world orientation.


def direction_error(center_i, center_j, scale, measured_world_direction):
    # scale is the inverse baseline length, not a robust weight.
    return measured_world_direction - scale[0] * (center_j - center_i)


def center_anchor(center):
    return center  # Choose C_0 = 0 as the world origin.


def baseline_anchor(center_i, center_j):
    # Direction-only measurements leave one global scale undetermined.
    return ((center_j - center_i).norm() - 1).reshape(1)


@dataclass
class Input:
    camera_i: torch.Tensor  # [edges], integer indices
    camera_j: torch.Tensor  # [edges], integer indices
    relative_rotations: torch.Tensor  # [edges, 3, 3], measured R_ji
    relative_directions: torch.Tensor  # [edges, 3], measured unit t_ji
    initial_rotations: torch.Tensor  # [cameras, 3, 3]
    initial_centers: torch.Tensor  # [cameras, 3]
    truth_rotations: torch.Tensor
    truth_centers: torch.Tensor
    outliers: torch.Tensor


def make_input(device="cpu", dtype=torch.float64):
    generator = torch.Generator().manual_seed(23)
    cameras = 12
    camera_i, camera_j = torch.triu_indices(cameras, cameras, offset=1)
    edges = len(camera_i)

    def random(*shape):
        # Generate one scene, then cast: --dtype changes solver precision only.
        return torch.randn(*shape, generator=generator, dtype=torch.float64)

    truth_rotations = tls.so3_exp(0.3 * random(cameras, 3))
    truth_rotations = truth_rotations @ truth_rotations[0].T
    angles = torch.arange(cameras, dtype=torch.float64) * (2 * math.pi / cameras)
    truth_centers = torch.stack((angles.cos(), angles.sin(), 0.3 * (2 * angles).sin()), -1)
    truth_centers = truth_centers - truth_centers[0]
    truth_centers = truth_centers / truth_centers[1].norm()

    outliers = torch.zeros(edges, dtype=torch.bool)
    outliers[torch.randperm(edges - 1, generator=generator)[:8] + 1] = True
    rotation_noise = math.radians(0.3) * random(edges, 3)
    rotation_noise[outliers] = math.radians(35) * F.normalize(random(8, 3), dim=-1)
    relative_rotations = (
        tls.so3_exp(rotation_noise)
        @ truth_rotations[camera_j]
        @ truth_rotations[camera_i].transpose(-1, -2)
    )
    displacement = truth_centers[camera_i] - truth_centers[camera_j]
    camera_direction = F.normalize(
        (truth_rotations[camera_j] @ displacement[..., None]).squeeze(-1), dim=-1
    )
    direction_noise = math.radians(0.3) * random(edges, 3)
    direction_noise[outliers] = math.radians(50) * F.normalize(random(8, 3), dim=-1)
    relative_directions = (tls.so3_exp(direction_noise) @ camera_direction[..., None]).squeeze(-1)

    initial_rotations = torch.eye(3, dtype=torch.float64).repeat(cameras, 1, 1)
    # Approximate centers are inputs to this example, as in a refinement stage.
    initial_centers = truth_centers + 0.2 * random(cameras, 3)
    initial_centers = initial_centers - initial_centers[0]
    initial_centers = initial_centers / initial_centers[1].norm()
    return Input(
        *[
            tensor.to(device=device, dtype=dtype if tensor.is_floating_point() else tensor.dtype)
            for tensor in (
                camera_i,
                camera_j,
                relative_rotations,
                relative_directions,
                initial_rotations,
                initial_centers,
                truth_rotations,
                truth_centers,
                outliers,
            )
        ]
    )


def solve_options(dtype):
    return dict(
        jacobi_scaling=True,
        trust_region=tls.TrustRegionConfig(
            lambda_initial=1e-3, strategy="adaptive", damping_mode="diagonal"
        ),
        conjugate_gradient=tls.ConjugateGradientConfig(rtol=1e-5, max_iterations=200),
        termination=tls.TerminationConfig(
            max_iterations=100,
            cost_tolerance=1e-10 if dtype == torch.float64 else 1e-6,
            gradient_tolerance=1e-8 if dtype == torch.float64 else 1e-5,
            parameter_tolerance=1e-9 if dtype == torch.float64 else 1e-6,
        ),
        return_summary=True,
    )


def average_rotations(data, *, robust=True):
    problem = tls.Problem({"rotations": data.initial_rotations}, manifolds={"rotations": SO3Var})
    problem.add_residual(
        rotation_error,
        ("rotations", data.camera_i),
        ("rotations", data.camera_j),
        data.relative_rotations,
        loss=tls.HuberLoss(delta=math.radians(2)) if robust else None,
    )
    # Gauge priors use ordinary least squares, without robust downweighting.
    problem.add_residual(rotation_anchor, ("rotations", 0))
    solution, summary = problem.solve(
        linear_solver="dense_qr", **solve_options(data.initial_rotations.dtype)
    )
    return solution["rotations"], summary


def average_translations(data, rotations, *, robust=True, solver="schur_dense"):
    # t_ji is expressed in camera j; the world direction C_j - C_i is -R_j.T t_ji.
    directions = -(
        rotations[data.camera_j].transpose(-1, -2) @ data.relative_directions[..., None]
    ).squeeze(-1)
    baselines = (data.initial_centers[data.camera_j] - data.initial_centers[data.camera_i]).norm(
        dim=-1
    )
    problem = tls.Problem(
        {"centers": data.initial_centers, "scales": baselines.reciprocal()[:, None]},
        manifolds={"scales": PositiveScaleVar},
    )
    problem.add_residual(
        direction_error,
        ("centers", data.camera_i),
        ("centers", data.camera_j),
        ("scales", torch.arange(len(directions), device=directions.device)),
        directions,
        loss=tls.HuberLoss(delta=0.05) if robust else None,
    )
    problem.add_residual(center_anchor, ("centers", 0))
    problem.add_residual(baseline_anchor, ("centers", 0), ("centers", 1))
    solution, summary = problem.solve(
        linear_solver=solver, eliminate="scales", **solve_options(data.initial_centers.dtype)
    )
    return solution["centers"], summary


def report_stage(summary):
    return {
        "initial_cost": summary.initial_cost,
        "final_cost": summary.final_cost,
        "accepted_steps": summary.accepted_steps,
        "rejected_steps": summary.rejected_steps,
        "termination": summary.termination_reason,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    parser.add_argument(
        "--translation-solver", choices=["schur_dense", "schur_cg"], default="schur_dense"
    )
    parser.add_argument("--compare-l2", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    data = make_input(args.device, getattr(torch, args.dtype))
    results = {}
    for robust in [True, False] if args.compare_l2 else [True]:
        rotations, rotation_summary = average_rotations(data, robust=robust)
        centers, translation_summary = average_translations(
            data, rotations, robust=robust, solver=args.translation_solver
        )
        rotation_degrees = so3_log(rotations @ data.truth_rotations.transpose(-1, -2)).norm(
            dim=-1
        ) * (180 / math.pi)
        results["huber" if robust else "l2"] = {
            "rotation": {
                **report_stage(rotation_summary),
                "mean_error_degrees": float(rotation_degrees.mean()),
            },
            "translation": {
                **report_stage(translation_summary),
                "center_rmse": float((centers - data.truth_centers).square().sum(-1).mean().sqrt()),
            },
        }
    print(
        json.dumps(
            {
                "device": args.device,
                "dtype": args.dtype,
                "cameras": len(data.initial_centers),
                "edges": len(data.camera_i),
                "outlier_edges": int(data.outliers.sum()),
                "results": results,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
