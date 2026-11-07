"""Variable-group block Jacobi without assembling the global normal matrix."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .variables import Var


def cholesky_apply(rhs, factor, *, graph_safe=False):
    """Apply a Cholesky factor without MAGMA's capture-unsafe allocation path."""
    if graph_safe:
        intermediate = torch.linalg.solve_triangular(factor, rhs, upper=False)
        return torch.linalg.solve_triangular(factor.transpose(-1, -2), intermediate, upper=True)
    return torch.cholesky_solve(rhs, factor)


def normalize_groups(groups):
    if not isinstance(groups, (tuple, list)):
        raise ValueError("preconditioner_groups must be a sequence of groups")
    normalized, seen = [], set()
    for group in groups:
        if not isinstance(group, (tuple, list)) or len(group) < 2:
            raise ValueError("Each joint preconditioner group needs at least two variable keys")
        for key in group:
            if not (
                (isinstance(key, str) and key) or (isinstance(key, type) and issubclass(key, Var))
            ):
                raise ValueError("Preconditioner keys must be parameter names or Var types")
            if key in seen:
                raise ValueError("A variable may occur in only one preconditioner group")
            seen.add(key)
        normalized.append(tuple(group))
    return tuple(normalized)


@dataclass(frozen=True)
class JacobiBlock:
    members: tuple[int, ...]
    count: int
    dim: int
    columns: torch.Tensor


class BlockJacobiPlan:
    """Join selected variable groups row-wise, leaving other groups separate.

    Members must have identical IDs in identical order. Grouping affects only
    the preconditioner, not parameter storage, retractions or the linear system.
    """

    def __init__(self, layout, groups, device):
        self.layout = layout
        lookup = {entry.key: i for i, entry in enumerate(layout)}
        grouped, selected = [], set()
        for group in normalize_groups(groups):
            if any(key not in lookup for key in group):
                raise ValueError(
                    "Unknown preconditioner key (Schur groups must be retained variables)"
                )
            members = tuple(lookup[key] for key in group)
            first = layout[members[0]]
            if any(not torch.equal(first.ids.cpu(), layout[i].ids.cpu()) for i in members[1:]):
                raise ValueError(
                    "Joint preconditioner groups require identical IDs in identical order"
                )
            grouped.append(members)
            selected.update(members)
        grouped.extend((i,) for i in range(len(layout)) if i not in selected)
        self.blocks, self.locations = [], {}
        for block_index, members in enumerate(grouped):
            offset, columns = 0, []
            for i in members:
                entry = layout[i]
                self.locations[i] = (block_index, offset)
                columns.append(
                    entry.offset
                    + torch.arange(entry.count, device=device)[:, None] * entry.dim
                    + torch.arange(entry.dim, device=device)[None, :]
                )
                offset += entry.dim
            self.blocks.append(
                JacobiBlock(members, layout[members[0]].count, offset, torch.cat(columns, dim=1))
            )
        self._cross_records = None

    def normal_blocks(self, jacobian, diagonal=None):
        diagonal = jacobian.diagonal_blocks() if diagonal is None else diagonal
        blocks = [diagonal[0].new_zeros(b.count, b.dim, b.dim) for b in self.blocks]
        for i, entry in enumerate(self.layout):
            block, offset = self.locations[i]
            blocks[block][:, offset : offset + entry.dim, offset : offset + entry.dim] = diagonal[i]
        if self._cross_records is None:
            self._cross_records = []
            for g, group in enumerate(jacobian.groups if jacobian is not None else ()):
                for p, i in enumerate(group.type_indices):
                    block, a = self.locations[i]
                    for q in range(p):
                        j = group.type_indices[q]
                        other, b = self.locations[j]
                        # Same-variable aliases are already included in diagonal_blocks().
                        if block == other and i != j:
                            mask = group.variable_indices[p] == group.variable_indices[q]
                            self._cross_records.append((g, p, q, block, a, b, mask))
        for g, p, q, block, a, b, mask in self._cross_records:
            group = jacobian.groups[g]
            left, right = group.blocks[p], group.blocks[q]
            cross = (left.transpose(-1, -2) @ right) * mask[:, None, None]
            da, db = cross.shape[-2:]
            rows = group.variable_indices[p]
            blocks[block][:, a : a + da, b : b + db].index_add_(0, rows, cross)
            blocks[block][:, b : b + db, a : a + da].index_add_(0, rows, cross.transpose(-1, -2))
        return blocks

    def factor(self, blocks, damping, *, graph_safe=False):
        factors = [
            torch.linalg.cholesky(
                value + damping * torch.eye(value.shape[-1], device=value.device, dtype=value.dtype)
            )
            for value in blocks
        ]

        def apply(vector):
            result = torch.empty_like(vector)
            for block, factor in zip(self.blocks, factors, strict=True):
                solved = cholesky_apply(
                    vector[block.columns][..., None], factor, graph_safe=graph_safe
                )
                result.index_copy_(0, block.columns.reshape(-1), solved.reshape(-1))
            return result

        return apply
