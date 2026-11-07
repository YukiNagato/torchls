"""Fused variable-block Gram accumulation, with sharding for shared variables.

Only incidence metadata is cached. Values are rebuilt on every linearization;
duplicate slots include their cross terms, exactly as BlockJacobian does.
"""

from functools import cache
from typing import Any

import torch
import warp as wp


@cache
def _gram_kernel(rows, dim, cross, scalar):
    @wp.kernel
    def gram(
        a: wp.array3d(dtype=scalar),
        b: wp.array3d(dtype=scalar),
        ia: wp.array(dtype=wp.int64),
        ib: wp.array(dtype=wp.int64),
        hot: wp.array(dtype=wp.int32),
        out: wp.array3d(dtype=scalar),
        count: int,
        hot_count: int,
        shards: int,
    ):
        n, element = wp.tid()
        i = element // dim
        j = element % dim
        if i <= j:
            valid = True
            if wp.static(cross):
                valid = ia[n] == ib[n]
            if valid:
                value = scalar(0.0)
                for r in range(wp.static(rows)):
                    value += a[n, r, i] * b[n, r, j]
                    if wp.static(cross):
                        value += b[n, r, i] * a[n, r, j]
                index = int(ia[n])
                target = index
                if hot[index] >= 0:
                    target = count + (n % shards) * hot_count + hot[index]
                wp.atomic_add(out, target, i, j, value)

    return gram


@wp.kernel
def _finish(
    acc: wp.array3d(dtype=Any),
    hot: wp.array(dtype=wp.int32),
    out: wp.array3d(dtype=Any),
    hot_count: int,
    shards: int,
):
    n, i, j = wp.tid()
    if i <= j:
        value = acc[n, i, j]
        if hot[n] >= 0:
            for s in range(shards):
                value += acc[out.shape[0] + s * hot_count + hot[n], i, j]
        out[n, i, j] = value
        if i != j:
            out[n, j, i] = value


class WarpDiagonalPlan:
    def __init__(self, jacobian, *, shards=128, hot_threshold=512):
        prototype = jacobian.groups[0].blocks[0]
        if prototype.device.type != "cuda" or prototype.dtype not in (torch.float32, torch.float64):
            raise ValueError("Warp diagonal blocks require CUDA float32 or float64")
        if shards < 1 or hot_threshold < 1:
            raise ValueError("shards and hot_threshold must be positive")
        wp.init()
        self.device, self.dtype = prototype.device, prototype.dtype
        self.scalar = wp.float64 if self.dtype == torch.float64 else wp.float32
        self.shards = shards
        self.layout = tuple((e.count, e.dim) for e in jacobian.layout)
        self.structure = tuple(
            (g.type_indices, g.variable_indices, tuple(b.shape for b in g.blocks))
            for g in jacobian.groups
        )
        degree = [torch.zeros(n, device=self.device, dtype=torch.int64) for n, _ in self.layout]
        self.indices = []
        self.calls = []
        for gi, group in enumerate(jacobian.groups):
            ids = tuple(idx.to(torch.int64).contiguous() for idx in group.variable_indices)
            self.indices.append(tuple(wp.from_torch(idx) for idx in ids))
            for slot, (block, ti, idx) in enumerate(
                zip(group.blocks, group.type_indices, ids, strict=True)
            ):
                degree[ti] += torch.bincount(idx, minlength=self.layout[ti][0])
                self.calls.append(
                    (
                        gi,
                        slot,
                        slot,
                        ti,
                        _gram_kernel(block.shape[1], block.shape[2], False, self.scalar),
                    )
                )
                for other in range(slot):
                    if group.type_indices[other] == ti:
                        self.calls.append(
                            (
                                gi,
                                slot,
                                other,
                                ti,
                                _gram_kernel(block.shape[1], block.shape[2], True, self.scalar),
                            )
                        )
        self.hot, self.hot_count = [], []
        for d in degree:
            selected = (d > hot_threshold).nonzero().flatten()
            hot = torch.full(d.shape, -1, device=self.device, dtype=torch.int32)
            hot[selected] = torch.arange(selected.numel(), device=self.device, dtype=torch.int32)
            self.hot.append(wp.from_torch(hot))
            self.hot_count.append(selected.numel())

    def diagonal_blocks(self, jacobian):
        if tuple((e.count, e.dim) for e in jacobian.layout) != self.layout or len(
            jacobian.groups
        ) != len(self.structure):
            raise ValueError("Warp diagonal plan requires unchanged incidence structure")
        for group, (types, indices, shapes) in zip(jacobian.groups, self.structure, strict=True):
            if (
                group.type_indices != types
                or tuple(b.shape for b in group.blocks) != shapes
                or any(b.device != self.device or b.dtype != self.dtype for b in group.blocks)
                or len(group.variable_indices) != len(indices)
                or any(
                    a is not b and not torch.equal(a, b)
                    for a, b in zip(group.variable_indices, indices, strict=True)
                )
            ):
                raise ValueError(
                    "Warp diagonal plan requires unchanged incidence structure, shapes and dtype"
                )
        stream = wp.stream_from_torch(torch.cuda.current_stream(self.device))
        accum = [
            torch.zeros((n + h * self.shards, d, d), device=self.device, dtype=self.dtype)
            for (n, d), h in zip(self.layout, self.hot_count, strict=True)
        ]
        outputs = tuple(
            torch.empty((n, d, d), device=self.device, dtype=self.dtype) for n, d in self.layout
        )
        wa = [wp.from_torch(a) for a in accum]
        blocks = [tuple(wp.from_torch(b) for b in g.blocks) for g in jacobian.groups]
        for gi, slot, other, ti, kernel in self.calls:
            a = blocks[gi][slot]
            if a.shape[0]:
                wp.launch(
                    kernel,
                    dim=(a.shape[0], a.shape[2] ** 2),
                    inputs=[
                        a,
                        blocks[gi][other],
                        self.indices[gi][slot],
                        self.indices[gi][other],
                        self.hot[ti],
                        wa[ti],
                        self.layout[ti][0],
                        self.hot_count[ti],
                        self.shards,
                    ],
                    stream=stream,
                )
        for i, out in enumerate(outputs):
            if out.numel():
                wp.launch(
                    _finish,
                    dim=out.shape,
                    inputs=[wa[i], self.hot[i], wp.from_torch(out), self.hot_count[i], self.shards],
                    stream=stream,
                )
        return outputs
