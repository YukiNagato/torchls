from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .variables import VariableLayout


@dataclass(frozen=True)
class JacobianGroup:
    """Local Jacobian blocks and global tangent-column indices, per slot."""

    blocks: tuple[Tensor, ...]  # each [observations, residual_dim, slot_dim]
    columns: tuple[Tensor, ...]  # each [observations, slot_dim]
    type_indices: tuple[int, ...]
    variable_indices: tuple[Tensor, ...]

    @property
    def num_rows(self):
        return self.blocks[0].shape[0] * self.blocks[0].shape[1]


@dataclass(frozen=True)
class BlockJacobian:
    groups: tuple[JacobianGroup, ...]
    layout: tuple[VariableLayout, ...]
    shape: tuple[int, int]

    def matvec(self, vector: Tensor) -> Tensor:
        if vector.shape != (self.shape[1],):
            raise ValueError("matvec input has incorrect shape")
        rows = []
        for group in self.groups:
            parts = [
                torch.einsum("nrd,nd->nr", block, vector[columns])
                for block, columns in zip(group.blocks, group.columns, strict=True)
            ]
            rows.append(sum(parts).reshape(-1))
        return torch.cat(rows)

    def rmatvec(self, vector: Tensor) -> Tensor:
        if vector.shape != (self.shape[0],):
            raise ValueError("rmatvec input has incorrect shape")
        out = vector.new_zeros(self.shape[1])
        offset = 0
        for group in self.groups:
            count, rows, _ = group.blocks[0].shape
            local = vector[offset : offset + count * rows].reshape(count, rows)
            for block, columns in zip(group.blocks, group.columns, strict=True):
                contribution = torch.einsum("nrd,nr->nd", block, local)
                out = out.index_add(0, columns.reshape(-1), contribution.reshape(-1))
            offset += count * rows
        return out

    def normal_matvec(self, vector: Tensor, damping: float = 0.0) -> Tensor:
        return self.rmatvec(self.matvec(vector)) + damping * vector

    def scale_columns(self, scale: Tensor) -> BlockJacobian:
        """Return J diag(scale), retaining the same symbolic block structure."""
        if scale.shape != (self.shape[1],):
            raise ValueError("Column scale has incorrect shape")
        groups = tuple(
            JacobianGroup(
                tuple(
                    block * scale[columns][:, None, :]
                    for block, columns in zip(group.blocks, group.columns, strict=True)
                ),
                group.columns,
                group.type_indices,
                group.variable_indices,
            )
            for group in self.groups
        )
        return BlockJacobian(groups, self.layout, self.shape)

    def diagonal_blocks(self) -> tuple[Tensor, ...]:
        """Exact variable diagonal blocks of J.T J, including aliased slots."""
        prototype = self.groups[0].blocks[0]
        diagonal = [prototype.new_zeros(entry.count, entry.dim, entry.dim) for entry in self.layout]
        for group in self.groups:
            for slot, (block, type_index, indices) in enumerate(
                zip(group.blocks, group.type_indices, group.variable_indices, strict=True)
            ):
                gram = block.transpose(-1, -2) @ block
                # Two arguments can refer to the same variable in one residual.
                # Its effective Jacobian is the sum, so include cross terms.
                for other in range(slot):
                    if group.type_indices[other] == type_index:
                        cross = block.transpose(-1, -2) @ group.blocks[other]
                        mask = (indices == group.variable_indices[other])[:, None, None]
                        gram = gram + (cross + cross.transpose(-1, -2)) * mask
                diagonal[type_index] = diagonal[type_index].index_add(0, indices, gram)
        return tuple(diagonal)

    def to_dense(self) -> Tensor:
        """Diagnostic/reference conversion; the PCG path never calls this."""
        prototype = self.groups[0].blocks[0]
        dense = prototype.new_zeros(self.shape)
        offset = 0
        for group in self.groups:
            count, rows, _ = group.blocks[0].shape
            row_ids = (
                torch.arange(count * rows, device=dense.device).reshape(count, rows, 1) + offset
            )
            for block, columns in zip(group.blocks, group.columns, strict=True):
                dense = dense.index_put(
                    (
                        row_ids.expand_as(block).reshape(-1),
                        columns[:, None, :].expand_as(block).reshape(-1),
                    ),
                    block.reshape(-1),
                    accumulate=True,
                )
            offset += count * rows
        return dense
