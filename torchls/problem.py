from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch
from torch import Tensor

from .batching import AutoBatchSize
from .cost import Cost
from .linearization import make_linearizer, stream_linearization
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
        self._weights = None
        self._weight_cache = {}
        if cost.weights is not None:
            weights = torch.as_tensor(cost.weights)
            if weights.ndim > 1 or weights.numel() not in (1, self.count) or weights.is_complex():
                raise ValueError("Weights must be a real scalar or one value per observation")
            weights = torch.as_tensor(cost.weights, dtype=torch.float64).detach()
            if not (torch.isfinite(weights) & (weights >= 0)).all():
                raise ValueError("Weights must be finite and nonnegative")
            self._weights = weights.reshape(-1).clone()
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

    def resolved_batch_size(self, vals, layout):
        policy = self.cost.batch_size
        if not isinstance(policy, AutoBatchSize):
            return policy
        key = (policy, vals.device, vals.dtype)
        if key not in self._batch_cache:
            sample, _, _ = self.gather(vals, layout, 0, 1)
            return self.batch_size(sample, vals, layout)
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

    def observation_weights(self, device):
        if self._weights is None:
            return None
        if device not in self._weight_cache:
            self._weight_cache[device] = self._weights.to(device).expand(self.count)
        return self._weight_cache[device]

    def linearize(self, args, vals, layout, has_loss, *, weights=None, reduce_objective=True):
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
        if weights is not None:
            scale = weights.sqrt().to(residual.dtype)
            residual = residual * scale[:, None]
            blocks = tuple(block * scale[:, None, None] for block in blocks)
            objective = objective * weights
        return blocks, residual, raw, (objective.sum()
            if objective is not None and reduce_objective else objective)

    def linearize_values(self, vals, layout, has_loss):
        batch = self.resolved_batch_size(vals, layout)
        weights = self.observation_weights(vals.device)
        if batch is None or self.count <= batch:
            args, _, _ = self.gather(vals, layout)
            return self.linearize(args, vals, layout, has_loss, weights=weights)

        def evaluate(start, end):
            args, _, _ = self.gather(vals, layout, start, end)
            return self.linearize(args, vals, layout, has_loss,
                                  weights=None if weights is None else weights[start:end],
                                  reduce_objective=False)

        blocks, residual, raw, objective = stream_linearization(evaluate, self.count, batch)
        return blocks, residual, raw, objective.sum() if objective is not None else None

    def gather(self, vals, layout, start=0, end=None):
        indices, columns = self.device_indices(vals.device, layout)
        args = []
        slot_index = 0
        for i, arg in enumerate(self.cost.args):
            if _is_reference(arg):
                key = layout[self.type_indices[slot_index]].key
                args.append(vals._values[key][indices[slot_index][start:end]])
                slot_index += 1
            else:
                tensor = torch.as_tensor(arg)
                if tensor.is_floating_point() and not isinstance(arg, torch.Tensor):
                    # Convert Python numbers directly to the requested dtype;
                    # first rounding through the default float32 loses bits.
                    tensor = torch.as_tensor(arg, dtype=vals.dtype)
                if self._data_axis(i, tensor) is None:
                    tensor = tensor.unsqueeze(0)
                tensor = tensor.expand(self.count, *tensor.shape[1:])[start:end]
                args.append(tensor.to(device=vals.device,
                    dtype=vals.dtype if tensor.is_floating_point() else tensor.dtype))
        return args, indices, columns

    def residuals(self, vals, layout):
        # Determine the batch size from one observation before gathering all
        # poses/points. In BA these expanded inputs can exceed the Jacobian.
        batch = self.resolved_batch_size(vals, layout)
        evaluate = torch.func.vmap(self.cost.residual_fn)
        if batch is None or self.count <= batch:
            args, _, _ = self.gather(vals, layout)
            return evaluate(*args)
        for start in range(0, self.count, batch):
            end = min(start + batch, self.count)
            args, _, _ = self.gather(vals, layout, start, end)
            values = evaluate(*args)
            if start == 0:
                output = values.new_empty((self.count, *values.shape[1:]))
            output[start:end].copy_(values)
            del args, values
        return output


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
        self._has_loss = any(plan.cost.loss is not None or plan.cost.weights is not None
                             for plan in self._plans)

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
            residual = plan.residuals(vals, self.layout)
            residuals.append(residual.reshape(-1))
        return torch.cat(residuals)

    def cost(self, vals: VarValues) -> Tensor:
        if not self._has_loss:
            return 0.5 * self.residuals(vals).to(torch.float64).square().sum()
        self._validate_values(vals)
        values = []
        for plan in self._plans:
            residual = plan.residuals(vals, self.layout).reshape(plan.count, -1)
            squared = residual.to(torch.float64).square().sum(dim=-1)
            loss = plan.cost.loss
            value = squared if loss is None else loss.value(squared)
            weights = plan.observation_weights(vals.device)
            values.append(value.sum() if weights is None else (weights * value).sum())
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
            loss = plan.cost.loss
            batch = plan.resolved_batch_size(vals, self.layout) or count
            values = old.new_empty(count)
            changes = old.new_empty(count)
            for start in range(0, count, batch):
                end = min(start + batch, count)
                a, b = old[start:end], new[start:end]
                old_squared, new_squared = a.square().sum(-1), b.square().sum(-1)
                difference = ((a - b) * (a + b)).sum(-1)
                values[start:end] = new_squared if loss is None else loss.value(new_squared)
                changes[start:end] = (difference if loss is None
                    else loss._reduction(old_squared, new_squared, difference))
                del a, b, old_squared, new_squared, difference
            weights = plan.observation_weights(vals.device)
            objectives.append(values.sum() if weights is None else (weights * values).sum())
            reductions.append(changes.sum() if weights is None else (weights * changes).sum())
            del values, changes
        return 0.5 * torch.stack(objectives).sum(), 0.5 * torch.stack(reductions).sum()

    def linearize(self, vals: VarValues) -> Linearization:
        self._validate_values(vals)
        groups, residuals, raw_residuals, objectives = [], [], [], []
        for plan in self._plans:
            indices, columns = plan.device_indices(vals.device, self.layout)
            blocks, residual, raw, objective = plan.linearize_values(vals, self.layout, self._has_loss)
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
