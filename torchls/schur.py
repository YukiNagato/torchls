"""Block Schur elimination with a single independent eliminated parameter group.

Only the reduced system may become dense. Cross blocks use cached CSR
structure, and point/landmark blocks are factored independently. All code is
Python/PyTorch; both public modeling interfaces use this implementation.
"""

from __future__ import annotations

import torch

from .preconditioning import BlockJacobiPlan, cholesky_apply
from .sparse import BlockJacobian, JacobianGroup
from .variables import VariableLayout


class SchurPlan:
    def __init__(self, problem, eliminate, device, *, preconditioner_groups=()):
        matches = [i for i, entry in enumerate(problem.layout) if entry.key == eliminate]
        if len(matches) != 1:
            raise ValueError("eliminate must identify one parameter group by name or Var type")
        self.eliminated = matches[0]
        self.e = problem.layout[self.eliminated]
        self.retained = tuple(i for i in range(len(problem.layout)) if i != self.eliminated)
        if not self.retained:
            raise ValueError("Schur elimination needs a retained parameter group")
        self.original_layout = problem.layout
        layout, offset, global_indices = [], 0, []
        for i in self.retained:
            entry = problem.layout[i]
            layout.append(VariableLayout(entry.var_type, entry.ids, offset, entry.name))
            global_indices.append(
                torch.arange(entry.offset, entry.offset + entry.count * entry.dim)
            )
            offset += entry.count * entry.dim
        self.layout, self.dim = tuple(layout), offset
        self.global_indices = torch.cat(global_indices).to(device)
        local_types = {original: local for local, original in enumerate(self.retained)}
        records = {i: [] for i in self.retained}
        self.retained_groups = []
        self.off_diagonal = []
        for group_index, plan in enumerate(problem._plans):
            eslots = [slot for slot, t in enumerate(plan.type_indices) if t == self.eliminated]
            if eslots and any(
                not torch.equal(plan.indices[slot], plan.indices[eslots[0]]) for slot in eslots[1:]
            ):
                raise ValueError(
                    "Eliminated variables must be independent: a residual connects distinct eliminated indices"
                )
            rslots = [slot for slot, t in enumerate(plan.type_indices) if t != self.eliminated]
            if rslots:
                cols = tuple(
                    (
                        plan.indices[slot][:, None]
                        * self.layout[local_types[plan.type_indices[slot]]].dim
                        + self.layout[local_types[plan.type_indices[slot]]].offset
                        + torch.arange(self.layout[local_types[plan.type_indices[slot]]].dim)[
                            None, :
                        ]
                    ).to(device)
                    for slot in rslots
                )
                types = tuple(local_types[plan.type_indices[slot]] for slot in rslots)
                indices = tuple(plan.indices[slot].to(device) for slot in rslots)
                self.retained_groups.append((group_index, tuple(rslots), cols, types, indices))
                for p, slot in enumerate(rslots):
                    for q in range(p):
                        other = rslots[q]
                        mask = (
                            (plan.indices[slot] != plan.indices[other])
                            if types[p] == types[q]
                            else torch.ones(plan.count, dtype=torch.bool)
                        )
                        if bool(mask.any()):
                            self.off_diagonal.append(
                                (group_index, slot, other, cols[p], cols[q], mask.to(device))
                            )
            if eslots:
                for slot in rslots:
                    key = plan.indices[slot] * self.e.count + plan.indices[eslots[0]]
                    records[plan.type_indices[slot]].append((group_index, slot, tuple(eslots), key))

        self.edges, self.cross_records = [], []
        row_counts, csr_columns, value_orders = [], [], []
        nnz_offset = 0
        for original, entry in zip(self.retained, self.layout, strict=True):
            contributions = records[original]
            keys = (
                torch.cat([record[3] for record in contributions])
                if contributions
                else torch.empty(0, dtype=torch.long)
            )
            unique, inverse = torch.unique(keys, sorted=True, return_inverse=True)
            ri, ei = unique // self.e.count, unique % self.e.count
            degree = torch.bincount(ri, minlength=entry.count)
            starts = torch.cat((torch.zeros(1, dtype=torch.long), degree.cumsum(0)))
            d, e = entry.dim, self.e.dim
            local_edge = torch.arange(len(unique)) - starts[ri]
            positions = (
                starts[ri, None, None] * d * e
                + torch.arange(d)[None, :, None] * degree[ri, None, None] * e
                + local_edge[:, None, None] * e
                + torch.arange(e)[None, None, :]
            )
            order = torch.empty(len(unique) * d * e, dtype=torch.long)
            order[positions.reshape(-1)] = torch.arange(order.numel())
            columns = (
                (ei[:, None, None] * e + torch.arange(e)[None, None, :])
                .expand(-1, d, -1)
                .reshape(-1)
            )
            csr_columns.append(columns[order])
            value_orders.append(order + nnz_offset)
            row_counts.append((degree * e).repeat_interleave(d))
            nnz_offset += order.numel()
            self.edges.append((ri.to(device), ei.to(device)))
            cursor = 0
            group_records = []
            for group_index, slot, eslots, key in contributions:
                group_records.append(
                    (group_index, slot, eslots, inverse[cursor : cursor + len(key)].to(device))
                )
                cursor += len(key)
            self.cross_records.append(tuple(group_records))
        counts = torch.cat(row_counts)
        row_ptr_cpu = torch.cat((torch.zeros(1, dtype=torch.long), counts.cumsum(0)))
        if nnz_offset > 2**31 - 1 or max(self.dim, self.e.count * self.e.dim) > 2**31 - 1:
            raise ValueError("Schur CSR dimensions exceed the supported 32-bit sparse index range")
        self.row_ptr = row_ptr_cpu.to(device=device, dtype=torch.int32)
        # Bound cuSPARSE's symbolic SpGEMM workspace. A full W W.T can
        # allocate many gigabytes even when the reduced dense matrix is tiny.
        pointers = row_ptr_cpu.tolist()
        self.dense_chunks = tuple(
            (start, min(start + 64, self.dim), pointers[start], pointers[min(start + 64, self.dim)])
            for start in range(0, self.dim, 64)
        )
        self.columns = torch.cat(csr_columns).to(device=device, dtype=torch.int32)
        self.value_order = torch.cat(value_orders).to(device)
        self.block_columns = tuple(
            (
                torch.arange(entry.count)[:, None] * entry.dim
                + entry.offset
                + torch.arange(entry.dim)[None, :]
            ).to(device)
            for entry in self.layout
        )
        self.joint_plan = (
            BlockJacobiPlan(self.layout, preconditioner_groups, device)
            if preconditioner_groups
            else None
        )
        self.preconditioner_edges = []
        if self.joint_plan is not None:
            for block in self.joint_plan.blocks:
                if len(block.members) == 1:
                    self.preconditioner_edges.append(None)
                    continue
                keys = [self.edges[i][0] * self.e.count + self.edges[i][1] for i in block.members]
                unique, inverse = torch.unique(torch.cat(keys), sorted=True, return_inverse=True)
                maps = inverse.split([len(key) for key in keys])
                self.preconditioner_edges.append((unique // self.e.count, maps))

    def retained_normal_pattern(self):
        """Cache block-pair aggregation, without expanding observation-wise scalar COO."""
        if hasattr(self, "normal_indices"):
            return self.normal_indices, self.normal_records
        indices, records = [], []
        for columns in self.block_columns:
            rows, cols = torch.broadcast_tensors(columns[:, :, None], columns[:, None, :])
            indices.append(torch.stack((rows.reshape(-1), cols.reshape(-1))))
        for g, slot, other, columns, other_columns, mask in self.off_diagonal:
            selected = mask.nonzero().flatten()
            keys = columns[selected, 0] * self.dim + other_columns[selected, 0]
            unique, inverse = torch.unique(keys, sorted=True, return_inverse=True)
            d, e = columns.shape[1], other_columns.shape[1]
            rows = (
                unique[:, None, None] // self.dim
                + torch.arange(d, device=keys.device)[None, :, None]
            )
            cols = (
                unique[:, None, None] % self.dim
                + torch.arange(e, device=keys.device)[None, None, :]
            )
            rows, cols = torch.broadcast_tensors(rows, cols)
            pair = torch.stack((rows.reshape(-1), cols.reshape(-1)))
            indices.extend((pair, pair.flip(0)))
            records.append((g, slot, other, selected, inverse, len(unique)))
        self.normal_indices, self.normal_records = torch.cat(indices, dim=1), tuple(records)
        return self.normal_indices, self.normal_records


class SchurLinearization:
    def __init__(self, jacobian, diagonal, plan, *, dense, cache_retained=False):
        self.plan, self.diagonal = plan, diagonal
        groups = tuple(
            JacobianGroup(
                tuple(jacobian.groups[g].blocks[s] for s in slots), columns, types, indices
            )
            for g, slots, columns, types, indices in plan.retained_groups
        )
        self.retained_jacobian = (
            BlockJacobian(groups, plan.layout, (sum(g.num_rows for g in groups), plan.dim))
            if groups
            else None
        )
        self.preconditioner_blocks = (
            plan.joint_plan.normal_blocks(
                self.retained_jacobian, tuple(diagonal[i] for i in plan.retained)
            )
            if plan.joint_plan is not None
            else None
        )
        self.cross = []
        for entry, (ri, _), records in zip(
            plan.layout, plan.edges, plan.cross_records, strict=True
        ):
            values = jacobian.groups[0].blocks[0].new_zeros(len(ri), entry.dim, plan.e.dim)
            for group_index, slot, eslots, edge_ids in records:
                blocks = jacobian.groups[group_index].blocks
                eliminated_block = sum(blocks[s] for s in eslots)
                values = values.index_add(
                    0, edge_ids, blocks[slot].transpose(-1, -2) @ eliminated_block
                )
            self.cross.append(values)
        self.retained_normal = None
        if cache_retained and plan.off_diagonal:
            indices, records = plan.retained_normal_pattern()
            values = [diagonal[i].reshape(-1) for i in plan.retained]
            for g, slot, other, selected, inverse, count in records:
                blocks = jacobian.groups[g].blocks
                cross = blocks[slot][selected].transpose(-1, -2) @ blocks[other][selected]
                summed = cross.new_zeros(count, *cross.shape[1:]).index_add_(0, inverse, cross)
                # Transposing the COO coordinates above already transposes this contribution.
                values.extend((summed.reshape(-1), summed.reshape(-1)))
            self.retained_normal = (
                torch.sparse_coo_tensor(
                    indices, torch.cat(values), size=(plan.dim, plan.dim), check_invariants=False
                )
                .coalesce()
                .to_sparse_csr()
            )
        self.base_matrix = None
        if dense:
            self.base_matrix = jacobian.groups[0].blocks[0].new_zeros(plan.dim, plan.dim)
            for i, columns in zip(plan.retained, plan.block_columns, strict=True):
                self.base_matrix[columns[:, :, None], columns[:, None, :]] = diagonal[i]
            for g, slot, other, columns, other_columns, mask in plan.off_diagonal:
                blocks = jacobian.groups[g].blocks
                cross = (blocks[slot].transpose(-1, -2) @ blocks[other]) * mask[:, None, None]
                self.base_matrix.index_put_(
                    (columns[:, :, None], other_columns[:, None, :]), cross, accumulate=True
                )
                self.base_matrix.index_put_(
                    (other_columns[:, :, None], columns[:, None, :]),
                    cross.transpose(-1, -2),
                    accumulate=True,
                )

    def damped(self, damping):
        return DampedSchurSystem(self, damping)


class DampedSchurSystem:
    def __init__(self, linear, damping):
        self.linear, self.plan, self.damping = linear, linear.plan, damping
        c = linear.diagonal[self.plan.eliminated]
        eye = torch.eye(c.shape[-1], device=c.device, dtype=c.dtype)
        factor = torch.linalg.cholesky(c + damping * eye)
        self.inverse_l = torch.linalg.solve_triangular(factor, eye.expand_as(factor), upper=False)
        self.whitened = tuple(
            block @ self.inverse_l[ei].transpose(-1, -2)
            for block, (_, ei) in zip(linear.cross, self.plan.edges, strict=True)
        )
        values = torch.cat([block.reshape(-1) for block in self.whitened])[self.plan.value_order]
        self.w = torch.sparse_csr_tensor(
            self.plan.row_ptr,
            self.plan.columns,
            values,
            size=(self.plan.dim, self.plan.e.count * self.plan.e.dim),
            device=c.device,
            check_invariants=False,
        )
        self.wt = self.w.transpose(0, 1).to_sparse_csr()

    def reduced_rhs(self, rhs):
        e = self.plan.e
        self.eliminated_rhs = (
            self.inverse_l @ rhs[e.offset : e.offset + e.count * e.dim].reshape(e.count, e.dim, 1)
        ).reshape(-1)
        return rhs[self.plan.global_indices] - self.w @ self.eliminated_rhs

    def backsubstitute(self, retained):
        e = self.plan.e
        eliminated = self.inverse_l.transpose(-1, -2) @ (
            self.eliminated_rhs - self.wt @ retained
        ).reshape(e.count, e.dim, 1)
        total = sum(entry.count * entry.dim for entry in self.plan.original_layout)
        result = retained.new_zeros(total)
        result[self.plan.global_indices] = retained
        result[e.offset : e.offset + e.count * e.dim] = eliminated.reshape(-1)
        return result

    def matvec(self, vector):
        jac = self.linear.retained_jacobian
        if self.linear.retained_normal is not None:
            base = self.linear.retained_normal @ vector + self.damping * vector
        elif not self.plan.off_diagonal:
            base = (
                torch.cat(
                    [
                        (
                            self.linear.diagonal[i]
                            @ vector[entry.offset : entry.offset + entry.count * entry.dim].reshape(
                                entry.count, entry.dim, 1
                            )
                        ).reshape(-1)
                        for i, entry in zip(self.plan.retained, self.plan.layout, strict=True)
                    ]
                )
                + self.damping * vector
            )
        else:
            base = (
                jac.normal_matvec(vector, self.damping)
                if jac is not None
                else self.damping * vector
            )
        return base - self.w @ (self.wt @ vector)

    def preconditioner(self, *, graph_safe=False):
        if self.plan.joint_plan is not None:
            diagonals = []
            for block, base, edge_plan in zip(
                self.plan.joint_plan.blocks,
                self.linear.preconditioner_blocks,
                self.plan.preconditioner_edges,
                strict=True,
            ):
                if edge_plan is None:
                    i = block.members[0]
                    rows, _ = self.plan.edges[i]
                    whitened = self.whitened[i]
                else:
                    rows, maps = edge_plan
                    whitened = base.new_zeros(len(rows), block.dim, self.plan.e.dim)
                    start = 0
                    for i, mapping in zip(block.members, maps, strict=True):
                        dim = self.plan.layout[i].dim
                        whitened[:, start : start + dim].index_copy_(0, mapping, self.whitened[i])
                        start += dim
                diagonals.append(base.index_add(0, rows, -(whitened @ whitened.transpose(-1, -2))))
            return self.plan.joint_plan.factor(diagonals, self.damping, graph_safe=graph_safe)
        factors = []
        for i, entry, block, (ri, _) in zip(
            self.plan.retained, self.plan.layout, self.whitened, self.plan.edges, strict=True
        ):
            diagonal = self.linear.diagonal[i].clone()
            diagonal = diagonal.index_add(0, ri, -(block @ block.transpose(-1, -2)))
            diagonal = diagonal + self.damping * torch.eye(
                entry.dim, device=block.device, dtype=block.dtype
            )
            factors.append(torch.linalg.cholesky(diagonal))

        def apply(vector):
            return torch.cat(
                [
                    cholesky_apply(
                        vector[entry.offset : entry.offset + entry.count * entry.dim].reshape(
                            entry.count, entry.dim, 1
                        ),
                        factor,
                        graph_safe=graph_safe,
                    ).reshape(-1)
                    for entry, factor in zip(self.plan.layout, factors, strict=True)
                ]
            )

        return apply

    def dense_matrix(self):
        matrix = self.linear.base_matrix
        if matrix is None:
            raise ValueError("Dense Schur matrix was not requested")
        matrix = matrix + self.damping * torch.eye(
            self.plan.dim, device=matrix.device, dtype=matrix.dtype
        )
        for start, end, nnz_start, nnz_end in self.plan.dense_chunks:
            rows = torch.sparse_csr_tensor(
                self.plan.row_ptr[start : end + 1] - nnz_start,
                self.plan.columns[nnz_start:nnz_end],
                self.w.values()[nnz_start:nnz_end],
                size=(end - start, self.w.shape[1]),
                device=matrix.device,
                check_invariants=False,
            )
            matrix[start:end] -= torch.sparse.mm(rows, self.wt).to_dense()
        return 0.5 * (matrix + matrix.T)
