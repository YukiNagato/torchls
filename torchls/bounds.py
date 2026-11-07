"""Active bounds and projected LM steps for Euclidean parameter groups."""

from collections.abc import Mapping

import torch

from .variables import VarValues, _NamedValues


class BoundsPlan:
    def __init__(self, bounds, layout, initial):
        if not isinstance(bounds, Mapping):
            raise TypeError("bounds must map parameter names or Var types to (lower, upper)")
        entries = {entry.key: entry for entry in layout}
        self.groups = []
        self.layout = layout
        for key, pair in bounds.items():
            if key not in entries:
                raise ValueError(f"Bounds refer to undeclared parameter group {key!r}")
            entry = entries[key]
            if not getattr(entry.var_type, "is_euclidean", False):
                raise ValueError("Bounds require Euclidean groups without a custom retraction")
            if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                raise ValueError("Each bounds entry must be a (lower, upper) pair")
            value = initial._values[key]
            limits = []
            for limit, default in zip(pair, (-torch.inf, torch.inf), strict=True):
                tensor = torch.as_tensor(
                    default if limit is None else limit, dtype=value.dtype, device=value.device
                )
                try:
                    limits.append(torch.broadcast_to(tensor, value.shape).clone())
                except RuntimeError as error:
                    raise ValueError(
                        f"Bounds for {key!r} do not broadcast to {value.shape}"
                    ) from error
            lower, upper = limits
            if bool(torch.isnan(lower).any() | torch.isnan(upper).any() | (lower > upper).any()):
                raise ValueError("Bounds must be ordered and cannot contain NaN")
            if bool(((value < lower) | (value > upper)).any()):
                raise ValueError(f"Initial values of {key!r} are outside their bounds")
            self.groups.append((entry, lower, upper))

    def free_mask(self, values, gradient):
        mask = torch.ones_like(gradient)
        for entry, lower, upper in self.groups:
            start, end = entry.offset, entry.offset + entry.count * entry.dim
            value = values._values[entry.key]
            local_gradient = gradient[start:end].reshape_as(value)
            active = (
                (lower == upper)
                | ((value <= lower) & (local_gradient > 0))
                | ((value >= upper) & (local_gradient < 0))
            )
            mask[start:end] = (~active).reshape(-1)
        return mask

    def projected_gradient(self, values, gradient):
        return gradient * self.free_mask(values, gradient)

    def retract(self, values, tangent):
        proposed = values.retract(tangent, self.layout)
        projected = dict(proposed._values)
        effective = tangent.clone()
        for entry, lower, upper in self.groups:
            value = torch.clamp(projected[entry.key], min=lower, max=upper)
            projected[entry.key] = value
            effective[entry.offset : entry.offset + entry.count * entry.dim] = (
                value - values._values[entry.key]
            ).reshape(-1)
        if isinstance(values, VarValues):
            proposed = VarValues(projected, values._ids)
        else:
            proposed = _NamedValues(projected, self.layout)
        return proposed, effective
