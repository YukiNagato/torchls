"""Fused factor-wise J.T @ (J @ v), without Schur or a global normal matrix.

Local Jacobians are stored in structure-of-arrays order. Frequently referenced
columns use sharded accumulation to avoid serializing shared-variable atomics.
Only the symbolic column map is reused between nonlinear linearizations.
"""

from __future__ import annotations

from functools import cache
from typing import Any

import torch
import warp as wp


@cache
def _factor_kernel(rows, dim, scalar):
    local_matrix = wp.types.matrix(shape=(rows, dim), dtype=scalar)
    local_vector = wp.types.vector(length=rows, dtype=scalar)

    @wp.kernel
    def factor(
        jac: wp.array3d(dtype=scalar),
        columns: wp.array2d(dtype=wp.int32),
        hot: wp.array(dtype=wp.int32),
        vector: wp.array(dtype=scalar),
        accumulated: wp.array(dtype=scalar),
        size: int,
        hot_count: int,
        shards: int,
    ):
        observation = wp.tid()
        j = local_matrix()
        q = local_vector()
        for c in range(wp.static(dim)):
            value = vector[columns[c, observation]]
            for r in range(wp.static(rows)):
                j[r, c] = jac[r, c, observation]
                q[r] = q[r] + j[r, c] * value
        for c in range(wp.static(dim)):
            contribution = scalar(0.0)
            for r in range(wp.static(rows)):
                contribution = contribution + j[r, c] * q[r]
            column = columns[c, observation]
            target = column
            if hot[column] >= 0:
                target = size + (observation % shards) * hot_count + hot[column]
            wp.atomic_add(accumulated, target, contribution)

    return factor


@wp.kernel
def _finish(
    accumulated: wp.array(dtype=Any),
    hot: wp.array(dtype=wp.int32),
    vector: wp.array(dtype=Any),
    out: wp.array(dtype=Any),
    damping: Any,
    size: int,
    hot_count: int,
    shards: int,
):
    c = wp.tid()
    value = accumulated[c]
    if hot[c] >= 0:
        for shard in range(shards):
            value = value + accumulated[size + shard * hot_count + hot[c]]
    out[c] = value + damping * vector[c]


class WarpNormalPlan:
    """Fixed incidence structure; arbitrary residual sizes, slots and aliases."""

    def __init__(self, jacobian, *, shards=128, hot_threshold=512):
        prototype = jacobian.groups[0].blocks[0]
        if prototype.device.type != "cuda" or prototype.dtype not in (torch.float32, torch.float64):
            raise ValueError("Warp normal matvec requires CUDA float32 or float64")
        if shards < 1 or hot_threshold < 1:
            raise ValueError("shards and hot_threshold must be positive")
        if jacobian.shape[1] > 2**31 - 1:
            raise ValueError("Warp normal matvec requires 32-bit column indices")
        wp.init()
        self.size = jacobian.shape[1]
        self.shards = shards
        self.dtype, self.device = prototype.dtype, prototype.device
        self.structure = tuple(
            (tuple(block.shape for block in group.blocks), group.columns)
            for group in jacobian.groups
        )
        self.columns = tuple(
            torch.cat(group.columns, dim=1).T.to(torch.int32).contiguous()
            for group in jacobian.groups
        )
        degree = torch.zeros(self.size, dtype=torch.int64, device=self.device)
        for columns in self.columns:
            degree += torch.bincount(columns.flatten(), minlength=self.size)
        selected = (degree > hot_threshold).nonzero().flatten()
        self.hot_count = selected.numel()
        if self.size + self.hot_count * shards > 2**31 - 1:
            raise ValueError("Sharded accumulation exceeds the 32-bit index range")
        self.hot = torch.full((self.size,), -1, dtype=torch.int32, device=self.device)
        self.hot[selected] = torch.arange(self.hot_count, dtype=torch.int32, device=self.device)
        self.scalar = wp.float64 if self.dtype == torch.float64 else wp.float32
        self.kernels = tuple(
            _factor_kernel(group.blocks[0].shape[1], cols.shape[0], self.scalar)
            for group, cols in zip(jacobian.groups, self.columns, strict=True)
        )
        self.wp_columns = tuple(wp.from_torch(c) for c in self.columns)
        self.wp_hot = wp.from_torch(self.hot)

    def bind(self, jacobian):
        if jacobian.shape[1] != self.size or len(jacobian.groups) != len(self.structure):
            raise ValueError("Warp normal plan requires unchanged incidence structure")
        for group, (shapes, columns) in zip(jacobian.groups, self.structure, strict=True):
            if tuple(block.shape for block in group.blocks) != shapes or any(
                block.dtype != self.dtype or block.device != self.device for block in group.blocks
            ):
                raise ValueError(
                    "Warp normal plan requires unchanged block shapes, dtype and device"
                )
            if len(group.columns) != len(columns) or any(
                new is not old and not torch.equal(new, old)
                for new, old in zip(group.columns, columns, strict=True)
            ):
                raise ValueError("Warp normal plan requires unchanged incidence structure")
        return WarpNormalLinearization(self, jacobian)


class WarpNormalLinearization:
    def __init__(self, plan, jacobian):
        self.plan = plan
        self.values = tuple(
            torch.cat(group.blocks, dim=2).permute(1, 2, 0).contiguous()
            for group in jacobian.groups
        )
        self.wp_values = tuple(wp.from_torch(value) for value in self.values)

    def normal_matvec(self, vector, damping=0.0):
        plan = self.plan
        if (
            vector.shape != (plan.size,)
            or vector.device != plan.device
            or vector.dtype != plan.dtype
        ):
            raise ValueError("Vector must match Warp normal operator dimensions, device and dtype")
        accumulated = vector.new_zeros(plan.size + plan.hot_count * plan.shards)
        out = torch.empty_like(vector)
        stream = wp.stream_from_torch(torch.cuda.current_stream(vector.device))
        wvector, waccumulated = wp.from_torch(vector), wp.from_torch(accumulated)
        for kernel, values, columns in zip(
            plan.kernels, self.wp_values, plan.wp_columns, strict=True
        ):
            if values.shape[2]:
                wp.launch(
                    kernel,
                    dim=values.shape[2],
                    inputs=[
                        values,
                        columns,
                        plan.wp_hot,
                        wvector,
                        waccumulated,
                        plan.size,
                        plan.hot_count,
                        plan.shards,
                    ],
                    stream=stream,
                )
        wp.launch(
            _finish,
            dim=plan.size,
            inputs=[
                waccumulated,
                plan.wp_hot,
                wvector,
                wp.from_torch(out),
                plan.scalar(damping),
                plan.size,
                plan.hot_count,
                plan.shards,
            ],
            stream=stream,
        )
        return out
