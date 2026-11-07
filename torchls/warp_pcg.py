"""Optional CUDA PCG with fused updates and batched CUDA graph replay.

The operator and preconditioner remain ordinary, graph-capturable Torch callables.
Warp is imported only when this backend is explicitly requested. Requires Warp
1.17+ and a CUDA-enabled Torch build. No BA-specific dimensions or
residual functions are assumed.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import warp as wp


@wp.kernel
def _step(
    x: wp.array(dtype=Any),
    r: wp.array(dtype=Any),
    p: wp.array(dtype=Any),
    ap: wp.array(dtype=Any),
    curvature: wp.array(dtype=Any),
    state: wp.array(dtype=Any),
    control: wp.array(dtype=int),
):
    i = wp.tid()
    if control[1] != 0 or control[0] >= control[2]:
        return
    c, rz = curvature[0], state[0]
    if wp.isfinite(c) and wp.isfinite(rz) and c > 0.0 and rz > 0.0:
        alpha = rz / c
        x[i] = x[i] + alpha * p[i]
        r[i] = r[i] - alpha * ap[i]


@wp.kernel
def _direction(
    p: wp.array(dtype=Any),
    z: wp.array(dtype=Any),
    curvature: wp.array(dtype=Any),
    new_rz: wp.array(dtype=Any),
    state: wp.array(dtype=Any),
    control: wp.array(dtype=int),
):
    i = wp.tid()
    if control[1] != 0 or control[0] >= control[2]:
        return
    c, rz, next_rz = curvature[0], state[0], new_rz[0]
    if wp.isfinite(c) and wp.isfinite(rz) and wp.isfinite(next_rz) and c > 0.0 and rz > 0.0:
        p[i] = z[i] + (next_rz / rz) * p[i]


@wp.kernel
def _finish(
    curvature: wp.array(dtype=Any),
    new_rz: wp.array(dtype=Any),
    norm2: wp.array(dtype=Any),
    state: wp.array(dtype=Any),
    control: wp.array(dtype=int),
):
    # control = [iterations, status, stop_at]; state = [r.z, threshold**2]
    # status = active / recurrence tolerance / nonfinite / nonpositive curvature
    if control[1] != 0 or control[0] >= control[2]:
        return
    c, rz = curvature[0], state[0]
    control[0] += 1
    if not wp.isfinite(c) or not wp.isfinite(rz) or not wp.isfinite(norm2[0]):
        control[1] = 2
    elif c <= 0.0 or rz <= 0.0:
        control[1] = 3
    elif norm2[0] <= state[1]:
        control[1] = 1
    elif not wp.isfinite(new_rz[0]):
        control[1] = 2
    state[0] = new_rz[0]


@torch.no_grad()
def warp_pcg(matvec, rhs, *, preconditioner, config):
    if rhs.device.type != "cuda" or rhs.dtype not in (torch.float32, torch.float64):
        raise ValueError("Warp PCG requires CUDA float32 or float64 tensors")
    wp.init()
    caller = torch.cuda.current_stream(rhs.device)
    stream = torch.cuda.Stream(device=rhs.device)
    stream.wait_stream(caller)
    with torch.cuda.stream(stream):
        result = _run(matvec, rhs, preconditioner=preconditioner, config=config)
    caller.wait_stream(stream)
    result.solution.record_stream(caller)
    return result


def _run(matvec, rhs, *, preconditioner, config):
    from .solvers import PCGResult

    rhs = rhs.contiguous()
    apply = (lambda v: v) if preconditioner is None else preconditioner
    x = torch.zeros_like(rhs)
    r = rhs.clone()
    rhs_norm = float(rhs.norm())
    threshold = max(config.atol, config.rtol * rhs_norm)
    if not math.isfinite(rhs_norm) or rhs_norm <= threshold:
        finite = math.isfinite(rhs_norm)
        return PCGResult(
            x,
            0,
            rhs_norm,
            finite,
            "residual_tolerance" if finite else "nonfinite",
            rhs_norm=rhs_norm,
        )

    p = apply(r).clone()
    state = torch.stack((torch.dot(r, p), rhs.new_tensor(threshold * threshold)))
    control = torch.tensor([0, 0, config.max_iterations], dtype=torch.int32, device=rhs.device)
    stream = torch.cuda.current_stream(rhs.device)
    wp_stream = wp.stream_from_torch(stream)
    wx, wr, wp_, ws, wc = map(wp.from_torch, (x, r, p, state, control))

    def iteration():
        ap = matvec(p)
        curvature = torch.dot(p, ap).reshape(1)
        wp.launch(
            _step,
            dim=rhs.numel(),
            inputs=[wx, wr, wp_, wp.from_torch(ap), wp.from_torch(curvature), ws, wc],
            stream=wp_stream,
        )
        z = apply(r)
        new_rz = torch.dot(r, z).reshape(1)
        norm2 = torch.dot(r, r).reshape(1)
        wp.launch(
            _direction,
            dim=rhs.numel(),
            inputs=[wp_, wp.from_torch(z), wp.from_torch(curvature), wp.from_torch(new_rz), ws, wc],
            stream=wp_stream,
        )
        wp.launch(
            _finish,
            dim=1,
            inputs=[wp.from_torch(curvature), wp.from_torch(new_rz), wp.from_torch(norm2), ws, wc],
            stream=wp_stream,
        )

    # Warm up libraries and specialized Warp kernels before graph capture.
    iteration()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        iteration()

    x.zero_()
    r.copy_(rhs)
    p.copy_(apply(r))
    state[0].copy_(torch.dot(r, p))
    control.zero_()
    iterations = checks = restarts = 0
    reason = "max_iterations"
    while iterations < config.max_iterations:
        interval = config.residual_check_interval
        stop_at = min(
            config.max_iterations,
            ((iterations // interval + 1) * interval) if interval else config.max_iterations,
        )
        control[1] = 0
        control[2] = stop_at
        for _ in range(min(config.graph_batch_iterations, stop_at - iterations)):
            graph.replay()
        iterations, status, _ = control.tolist()
        if status == 0 and iterations < stop_at:
            continue
        true_r = rhs - matvec(x)
        true_norm = float(true_r.norm())
        checks += 1
        if not math.isfinite(true_norm):
            reason = "nonfinite"
            break
        if true_norm <= threshold:
            return PCGResult(
                x, iterations, true_norm, True, "residual_tolerance", restarts, checks, rhs_norm
            )
        if status in (2, 3):
            reason = "nonfinite" if status == 2 else "nonpositive_curvature"
            break
        if iterations >= config.max_iterations:
            break
        drift = float((true_r - r).norm())
        if (
            status == 1
            or config.restart_strategy == "periodic"
            or drift > config.residual_drift_tolerance * max(true_norm, threshold)
        ):
            r.copy_(true_r)
            p.copy_(apply(r))
            state[0].copy_(torch.dot(r, p))
            restarts += 1
    return PCGResult(x, iterations, true_norm, False, reason, restarts, checks, rhs_norm)
