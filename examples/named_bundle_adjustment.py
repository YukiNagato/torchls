"""BA using a parameter dictionary and (name, index) residual bindings.

PYTHONPATH=. python examples/named_bundle_adjustment.py --device cuda
All poses are world-to-camera SE(3) transforms; pixels are normalized.
"""

import argparse
import json

import torch

import torchls as tls


def reprojection(camera, point, observation):
    local = camera[:3, :3] @ point + camera[:3, 3]
    return local[:2] / local[2] - observation


def camera_prior(camera, reference):
    return (camera[:3, :] - reference[:3, :]).reshape(-1)


def point_prior(point, reference):
    return point - reference


def make_problem(device="cpu", dtype=torch.float64):
    generator = torch.Generator().manual_seed(12)
    n_cameras, n_points = 4, 20
    cameras = torch.eye(4, dtype=dtype).repeat(n_cameras, 1, 1)
    cameras[:, 0, 3] = torch.linspace(-1, 1, n_cameras, dtype=dtype)
    points = torch.randn(n_points, 3, generator=generator, dtype=dtype) * 0.5
    points[:, 2] += 5
    cam_ids = torch.arange(n_cameras).repeat_interleave(n_points)
    point_ids = torch.arange(n_points).repeat(n_cameras)
    local = points[point_ids] + cameras[cam_ids, :3, 3]
    observations = (local[:, :2] / local[:, 2:]).to(device)
    camera0 = cameras @ tls.se3_exp(
        torch.randn(n_cameras, 6, generator=generator, dtype=dtype) * 0.03
    )
    point0 = points + torch.randn(n_points, 3, generator=generator, dtype=dtype) * 0.1

    problem = tls.Problem(
        {"cams": camera0.to(device), "points": point0.to(device)},
        manifolds={"cams": tls.SE3Var},
    )
    problem.add_residual(reprojection, ("cams", cam_ids), ("points", point_ids), observations)
    problem.add_residual(camera_prior, ("cams", 0), cameras[0].to(device))
    problem.add_residual(point_prior, ("points", 0), points[0].to(device))
    return problem


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    parser.add_argument(
        "--solver", choices=["conjugate_gradient", "schur_cg", "schur_dense"], default="schur_cg"
    )
    options = parser.parse_args()
    problem = make_problem(options.device, getattr(torch, options.dtype))
    solution, summary = problem.solve(
        linear_solver=options.solver,
        eliminate="points" if options.solver.startswith("schur") else None,
        jacobi_scaling=True,
        trust_region=tls.TrustRegionConfig(
            lambda_initial=1e-4, strategy="adaptive", damping_mode="diagonal"
        ),
        return_summary=True,
    )
    print(
        json.dumps(
            {
                "initial_cost": summary.initial_cost,
                "final_cost": summary.final_cost,
                "termination": summary.termination_reason,
                "parameter_shapes": {name: list(value.shape) for name, value in solution.items()},
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
