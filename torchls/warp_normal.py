"""Fused factor-wise J.T @ (J @ v), without Schur or a global normal matrix.

Small Jacobians are packed in structure-of-arrays order. Larger operators read
the existing blocks, bounding the extra storage. Frequently referenced columns
use sharded accumulation to avoid serializing shared-variable atomics. Only the
symbolic column map is reused between nonlinear linearizations.
"""

from functools import cache
from typing import Any

import torch
import warp as wp


@wp.struct
class _Block32:
    values: wp.array3d(dtype=wp.float32)


@wp.struct
class _Block64:
    values: wp.array3d(dtype=wp.float64)


@cache
def _block_factor_kernel(rows, dims, scalar):
    dim = sum(dims)
    slots = tuple(s for s, width in enumerate(dims) for _ in range(width))
    local_columns = tuple(c for width in dims for c in range(width))
    block_type = _Block64 if scalar == wp.float64 else _Block32
    local_matrix = wp.types.matrix(shape=(rows, dim), dtype=scalar)
    local_vector = wp.types.vector(length=rows, dtype=scalar)

    @wp.kernel
    def factor(
        blocks: wp.array(dtype=block_type),
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
                j[r, c] = blocks[wp.static(slots[c])].values[
                    observation, r, wp.static(local_columns[c])
                ]
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
    """Fixed incidence structure; arbitrary residual sizes, slots and aliases.

    max_packed_bytes caps the additional SoA copy of local Jacobians; zero
    always reads existing blocks. It does not bound the whole solver's memory.
    """

    def __init__(self, jacobian, *, shards=128, hot_threshold=512,
                 max_packed_bytes=256 * 1024**2):
        prototype = jacobian.groups[0].blocks[0]
        if prototype.device.type != "cuda" or prototype.dtype not in (torch.float32, torch.float64):
            raise ValueError("Warp normal matvec requires CUDA float32 or float64")
        if shards < 1 or hot_threshold < 1:
            raise ValueError("shards and hot_threshold must be positive")
        if (isinstance(max_packed_bytes, bool) or not isinstance(max_packed_bytes, int)
                or max_packed_bytes < 0):
            raise ValueError("max_packed_bytes must be a nonnegative integer")
        if jacobian.shape[1] > 2**31 - 1:
            raise ValueError("Warp normal matvec requires 32-bit column indices")
        wp.init()
        self.size = jacobian.shape[1]
        self.shards = shards
        self.dtype, self.device = prototype.dtype, prototype.device
        self.packed = sum(b.numel() * b.element_size()
                          for g in jacobian.groups for b in g.blocks) <= max_packed_bytes
        self.counts = tuple(g.blocks[0].shape[0] for g in jacobian.groups)
        self.structure = tuple(
            (tuple(block.shape for block in group.blocks), group.columns)
            for group in jacobian.groups
        )
        columns = []
        for group in jacobian.groups:
            target = torch.empty(
                (sum(c.shape[1] for c in group.columns), group.blocks[0].shape[0]),
                dtype=torch.int32, device=self.device,
            )
            offset = 0
            for source in group.columns:
                width = source.shape[1]
                target[offset : offset + width].copy_(source.T)
                offset += width
            columns.append(target)
        self.columns = tuple(columns)
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
            (_factor_kernel(group.blocks[0].shape[1], cols.shape[0], self.scalar)
             if self.packed else _block_factor_kernel(
                 group.blocks[0].shape[1], tuple(b.shape[2] for b in group.blocks), self.scalar))
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
        if not plan.packed:
            # Keep tensor/Warp views alive for the descriptors' device pointers.
            self.values = tuple(group.blocks for group in jacobian.groups)
            self.block_views = tuple(tuple(wp.from_torch(b) for b in group)
                                     for group in self.values)
            block_type = _Block64 if plan.scalar == wp.float64 else _Block32
            descriptors = []
            storage = []
            with wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream(plan.device))):
                for group in self.block_views:
                    records = []
                    for view in group:
                        record = block_type()
                        record.values = view
                        records.append(record)
                    # Keep even these small allocations in Torch's pool. A
                    # native cudaMalloc cannot reclaim Torch's cached memory.
                    host = wp.array(records, dtype=block_type, device='cpu')
                    data = torch.from_numpy(host.numpy().view('uint8')).to(plan.device)
                    storage.append(data)
                    descriptors.append(wp.array(ptr=data.data_ptr(), dtype=block_type,
                        shape=(len(records),), device=str(plan.device), copy=False))
            self.descriptor_storage = tuple(storage)
            self.wp_values = tuple(descriptors)
            return
        values = []
        for group in jacobian.groups:
            count, rows, _ = group.blocks[0].shape
            target = group.blocks[0].new_empty(
                rows, sum(b.shape[2] for b in group.blocks), count
            )
            offset = 0
            for source in group.blocks:
                width = source.shape[2]
                target[:, offset : offset + width].copy_(source.permute(1, 2, 0))
                offset += width
            values.append(target)
        self.values = tuple(values)
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
        for kernel, values, columns, count in zip(
            plan.kernels, self.wp_values, plan.wp_columns, plan.counts, strict=True
        ):
            if count:
                wp.launch(
                    kernel,
                    dim=count,
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
