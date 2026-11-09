# torchls

torchls is a block-sparse nonlinear least-squares framework modeled after
[jaxls](https://github.com/brentyi/jaxls), built with **PyTorch** and accelerated
with **NVIDIA Warp**. Define residuals as ordinary eager PyTorch functions;
torchls batches their evaluation and differentiation. No `torch.compile` is required.

```bash
pip install -e .          # PyTorch backend
pip install -e '.[warp]'  # optional Warp CUDA acceleration
```

## Minimal example

Recover a 3D point from two fixed cameras using normalized image coordinates:

```python
import torch
import torchls as tls


def reprojection(point, camera, observed_uv):
    xyz = camera[:3, :3] @ point + camera[:3, 3]
    valid = xyz[2] > 1e-6
    z = torch.where(valid, xyz[2], torch.ones_like(xyz[2]))
    error = xyz[:2] / z - observed_uv
    return torch.where(valid, error, torch.zeros_like(error))


cameras = torch.eye(4, dtype=torch.float64).repeat(2, 1, 1)  # world-to-camera
cameras[1, 0, 3] = -1.0
observations = torch.tensor([[0.2, 0.1], [0.0, 0.1]], dtype=torch.float64)

problem = tls.Problem({"points": torch.tensor([[0.8, 0.4, 4.0]], dtype=torch.float64)})
problem.add_residual(
    reprojection,
    ("points", torch.tensor([0, 0])),  # both observations reference point 0
    cameras,
    observations,
)
solution = problem.solve()
print(solution["points"])  # approximately [[1.0, 0.5, 5.0]]
```

`("points", indices)` selects optimizable parameter rows; plain tensors supply
fixed data. Residuals operate on one observation and must support `torch.func`
transforms. The depth guard returns zero for `z <= 1e-6` and uses a safe denominator
before division, keeping those residuals and their derivatives zero.

## Features

- **Two modeling APIs:** named parameter dictionaries with `(name, index)` bindings,
  or jaxls-style typed variables with `Cost.factory` and `VarValues`.
- **Batched residuals:** multiple residual functions, shared parameters and constants;
  `vmap` with `jacrev` / `jacfwd`, optional analytic Jacobians, configurable chunking.
- **Manifolds:** Euclidean variables, SO(3), SE(2), SE(3), Sim(3), and custom retractions.
- **Optimization:** Levenberg–Marquardt and Gauss–Newton; full-system PCG, dense QR,
  Schur-PCG and dense Schur solvers.
- **Robustness and constraints:** Huber and custom losses, Euclidean bounds, and fixed
  coordinates through equal lower/upper bounds.
- **Preconditioning:** block Jacobi, joint variable blocks, column scaling, diagonal
  damping, and residual checks with adaptive or periodic PCG restarts.
- **Warp acceleration:** fused normal matvec and diagonal-block accumulation;
  fused PCG updates with CUDA Graph replay. Select
  `ConjugateGradientConfig(backend="warp", normal_matvec="warp")` for CUDA problems.
- **Execution and diagnostics:** CPU/CUDA, float32/float64, convergence summaries and
  optional profiling. Fixed linear least squares and L1 regression helpers are also available.
