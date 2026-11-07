from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch
from torch import Tensor

from .batching import AutoBatchSize
from .cost import Cost
from .linearization import make_linearizer
from .sparse import BlockJacobian, JacobianGroup
from .variables import Var, VariableLayout, VarValues, _ids, _NamedValues, _ParameterRef


def _is_reference(arg):
    return isinstance(arg, (Var, _ParameterRef))


@dataclass(frozen=True)
class Linearization:
    """Residual and Jacobian of the local solve model, plus the true objective.

    With a loss, ``raw_residual`` holds the original errors and ``cost`` is
    the robust objective, not half the squared norm of the model residual.
    """

    residual: Tensor
    jacobian: BlockJacobian
    raw_residual: Tensor | None = None
    objective: Tensor | None = None

    @property
    def cost(self) -> Tensor:
        """True objective, including each residual block's declared loss."""
        if self.objective is not None:
            return self.objective
        return 0.5 * self.residual.to(torch.float64).square().sum()

    @property
    def gradient(self) -> Tensor:
        return self.jacobian.rmatvec(self.residual)


class _CostPlan:
    def __init__(self, cost: Cost, layout: tuple[VariableLayout, ...]):
        self.cost = cost
        self.slots = tuple(i for i, arg in enumerate(cost.args) if _is_reference(arg))
        if not self.slots:
            raise ValueError("Every cost must reference at least one variable or named parameter")
        types = {entry.key: i for i, entry in enumerate(layout)}
        sizes = [
            _ids(cost.args[i].id).numel() for i in self.slots if _ids(cost.args[i].id).ndim == 1
        ]
        self.batched = bool(sizes)
        self.count = max(sizes, default=1)
        if self.count == 0 or any(size not in (1, self.count) for size in sizes):
            raise ValueError("Variable batch lengths must match (or be one)")
        self.type_indices, self.indices = [], []
        for slot in self.slots:
            var = cost.args[slot]
            key = var.name if isinstance(var, _ParameterRef) else type(var)
            if key not in types:
                raise ValueError(f"Undeclared parameter group {key!r}")
            type_index = types[key]
            available = layout[type_index].ids
            ids = _ids(var.id).reshape(-1).cpu().expand(self.count)
            index = torch.searchsorted(available, ids.contiguous())
            if bool(torch.any(index >= len(available))) or not torch.equal(available[index], ids):
                raise ValueError(
                    f"Cost references undeclared IDs or out-of-range indices in {key!r}"
                )
            self.type_indices.append(type_index)
            self.indices.append(index)
        self.type_indices = tuple(self.type_indices)
        self.indices = tuple(self.indices)
        self._device_cache = {}
        self._batch_cache = {}
        self._linearizer_cache = {}
        self._mode_cache = {}
        # Validate measurement batch shapes during analysis, not inside AD.
        for i, arg in enumerate(cost.args):
            if _is_reference(arg):
                expected_axis = 0 if _ids(arg.id).ndim == 1 else None
                if cost.in_dims is not None and cost.in_dims[i] != expected_axis:
                    raise ValueError("Parameter reference in_dims must match its index shape")
                continue
            tensor = torch.as_tensor(arg)
            axis = self._data_axis(i, tensor)
            if axis == 0 and (tensor.ndim == 0 or tensor.shape[0] not in (1, self.count)):
                raise ValueError(
                    "Data batch length must match observations; use in_dims=None for shared data"
                )

    def _data_axis(self, i, tensor):
        if self.cost.in_dims is not None:
            return self.cost.in_dims[i]
        return 0 if self.batched and tensor.ndim > 0 else None

    def batch_size(self, args, vals, layout):
        policy = self.cost.batch_size
        if not isinstance(policy, AutoBatchSize):
            return policy
        key = (policy, vals.device, vals.dtype)
        if key not in self._batch_cache:
            sample = self.cost.residual_fn(*(arg[0] for arg in args))
            self._batch_cache[key] = policy.resolve(
                count=self.count,
                input_bytes=sum(arg[0].numel() * arg.element_size() for arg in args),
                residual_dim=sample.numel(),
                tangent_dim=sum(layout[t].dim for t in self.type_indices),
                mode=self.cost.jac_mode,
                itemsize=sample.element_size(),
                device=vals.device,
            )
        return self._batch_cache[key]

    def device_indices(self, device, layout):
        if device not in self._device_cache:
            indices = tuple(index.to(device) for index in self.indices)
            columns = tuple(
                index[:, None] * layout[t].dim
                + layout[t].offset
                + torch.arange(layout[t].dim, device=device)[None, :]
                for index, t in zip(indices, self.type_indices, strict=True)
            )
            self._device_cache[device] = indices, columns
        return self._device_cache[device]

    def linearize(self, args, vals, layout, has_loss):
        types = tuple(layout[t].var_type for t in self.type_indices)
        batch = self.batch_size(args, vals, layout)
        signature = (vals.device, vals.dtype, tuple((a.shape[1:], a.dtype) for a in args))
        mode = self.cost.jac_mode
        if mode == "auto" and self.cost.jacobian_fn is None:
            if signature not in self._mode_cache:
                sample = self.cost.residual_fn(*(a[0] for a in args))
                self._mode_cache[signature] = (
                    "reverse" if sample.numel() <= sum(t.tangent_dim for t in types) else "forward"
                )
            mode = self._mode_cache[signature]
        key = (mode, has_loss, batch)
        if key not in self._linearizer_cache:
            self._linearizer_cache[key] = make_linearizer(
                self.cost, self.slots, types, mode, batch, has_loss
            )
        function = self._linearizer_cache[key]
        zeros = tuple(args[0].new_zeros(t.tangent_dim, dtype=vals.dtype) for t in types)
        blocks, residual, raw, objective = function(zeros, *args)
        return blocks, residual, raw, objective.sum() if objective is not None else None

    def gather(self, vals, layout):
        indices, columns = self.device_indices(vals.device, layout)
        args = []
        slot_index = 0
        for i, arg in enumerate(self.cost.args):
            if _is_reference(arg):
                key = layout[self.type_indices[slot_index]].key
                args.append(vals._values[key][indices[slot_index]])
                slot_index += 1
            else:
                tensor = torch.as_tensor(arg, device=vals.device)
                if tensor.is_floating_point():
                    # Convert Python numbers directly to the requested dtype;
                    # first rounding through the default float32 loses bits.
                    tensor = torch.as_tensor(arg, device=vals.device, dtype=vals.dtype)
                if self._data_axis(i, tensor) is None:
                    tensor = tensor.unsqueeze(0)
                args.append(tensor.expand(self.count, *tensor.shape[1:]))
        return args, indices, columns


class LeastSquaresProblem:
    def __init__(self, costs: Iterable[Cost], variables: Iterable[Var]):
        self.costs = tuple(costs)
        self.variables = tuple(variables)

    def analyze(self) -> AnalyzedLeastSquaresProblem:
        if not self.costs or not self.variables:
            raise ValueError("A problem needs costs and variables")
        by_type = {}
        for var in self.variables:
            by_type.setdefault(type(var), []).append(_ids(var.id).reshape(-1).cpu())
        layout = []
        offset = 0
        for key, parts in by_type.items():
            ids = torch.cat(parts).sort().values
            if len(ids) == 0 or bool(torch.any(ids[1:] == ids[:-1])):
                raise ValueError("Declared variable IDs must be nonempty and unique per type")
            layout.append(VariableLayout(key, ids, offset))
            offset += len(ids) * key.tangent_dim
        return AnalyzedLeastSquaresProblem(tuple(layout), self.costs, offset)


class AnalyzedLeastSquaresProblem:
    def __init__(self, layout, costs, tangent_dim):
        self.layout = layout
        self.tangent_dim = tangent_dim
        self._plans = tuple(_CostPlan(cost, layout) for cost in costs)
        self._has_loss = any(plan.cost.loss is not None for plan in self._plans)

    def _validate_values(self, vals):
        if set(vals._values) != {entry.key for entry in self.layout}:
            raise ValueError("Initial value types do not match the analyzed problem")
        for entry in self.layout:
            if not torch.equal(vals._ids[entry.key], entry.ids):
                raise ValueError("Initial value IDs do not match the analyzed problem")

    def default_values(self):
        if self.layout[0].name is not None:
            return _NamedValues(
                {
                    entry.key: entry.var_type.default_factory().expand(
                        entry.count, *entry.var_type.default_factory().shape
                    )
                    for entry in self.layout
                },
                self.layout,
            )
        return VarValues.make(entry.var_type(entry.ids) for entry in self.layout)

    def residuals(self, vals: VarValues) -> Tensor:
        """Return raw residuals, without robust weights or loss transforms."""
        self._validate_values(vals)
        residuals = []
        for plan in self._plans:
            args, _, _ = plan.gather(vals, self.layout)
            residual = torch.func.vmap(
                plan.cost.residual_fn, chunk_size=plan.batch_size(args, vals, self.layout)
            )(*args)
            residuals.append(residual.reshape(-1))
        return torch.cat(residuals)

    def cost(self, vals: VarValues) -> Tensor:
        if not self._has_loss:
            return 0.5 * self.residuals(vals).to(torch.float64).square().sum()
        self._validate_values(vals)
        values = []
        for plan in self._plans:
            args, _, _ = plan.gather(vals, self.layout)
            residual = torch.func.vmap(
                plan.cost.residual_fn, chunk_size=plan.batch_size(args, vals, self.layout)
            )(*args).reshape(plan.count, -1)
            squared = residual.to(torch.float64).square().sum(dim=-1)
            loss = plan.cost.loss
            values.append((squared if loss is None else loss.value(squared)).sum())
        return 0.5 * torch.stack(values).sum()

    def _evaluate_trial(self, vals, linearization):
        proposed = self.residuals(vals).to(torch.float64)
        if not self._has_loss:
            before = linearization.residual.to(torch.float64)
            return (
                0.5 * proposed.square().sum(),
                0.5 * torch.dot(before - proposed, before + proposed),
            )
        before = linearization.raw_residual.to(torch.float64)
        objectives, reductions = [], []
        offset = 0
        for plan, group in zip(self._plans, linearization.jacobian.groups, strict=True):
            count, rows, _ = group.blocks[0].shape
            old = before[offset : offset + count * rows].reshape(count, rows)
            new = proposed[offset : offset + count * rows].reshape(count, rows)
            offset += count * rows
            old_squared, new_squared = old.square().sum(-1), new.square().sum(-1)
            difference = ((old - new) * (old + new)).sum(-1)
            loss = plan.cost.loss
            objectives.append((new_squared if loss is None else loss.value(new_squared)).sum())
            reductions.append(
                (
                    difference
                    if loss is None
                    else loss._reduction(old_squared, new_squared, difference)
                ).sum()
            )
        return 0.5 * torch.stack(objectives).sum(), 0.5 * torch.stack(reductions).sum()

    def linearize(self, vals: VarValues) -> Linearization:
        self._validate_values(vals)
        groups, residuals, raw_residuals, objectives = [], [], [], []
        for plan in self._plans:
            args, indices, columns = plan.gather(vals, self.layout)
            blocks, residual, raw, objective = plan.linearize(
                args, vals, self.layout, self._has_loss
            )
            if self._has_loss:
                raw_residuals.append(raw.reshape(-1))
                objectives.append(objective)
            residuals.append(residual.reshape(-1))
            groups.append(JacobianGroup(tuple(blocks), columns, plan.type_indices, indices))
        residual = torch.cat(residuals)
        jacobian = BlockJacobian(tuple(groups), self.layout, (residual.numel(), self.tangent_dim))
        return Linearization(
            residual,
            jacobian,
            torch.cat(raw_residuals) if self._has_loss else None,
            0.5 * torch.stack(objectives).sum() if self._has_loss else None,
        )

    def solve(self, initial_vals=None, **kwargs):
        from .solvers import solve

        return solve(
            self,
            self.default_values() if initial_vals is None else initial_vals,
            **kwargs,
        )
