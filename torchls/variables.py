from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass

import torch
from torch import Tensor


def _ids(ids: int | Tensor) -> Tensor:
    out = torch.as_tensor(ids)
    if out.dtype not in (torch.int32, torch.int64) or out.ndim > 1:
        raise ValueError("Variable IDs must be scalar or 1D integer tensors")
    return out.to(dtype=torch.long)


@dataclass(frozen=True)
class Var:
    """Symbolic variable(s), identified by their subclass and integer ID."""

    id: int | Tensor

    def __init_subclass__(
        cls,
        *,
        default_factory: Callable[[], Tensor],
        tangent_dim: int | None = None,
        retract_fn: Callable[[Tensor, Tensor], Tensor] | None = None,
        **kwargs,
    ):
        super().__init_subclass__(**kwargs)
        default = default_factory()
        if not isinstance(default, Tensor) or not default.is_floating_point():
            raise TypeError("default_factory must return a floating point Tensor")
        if (tangent_dim is None) != (retract_fn is None):
            raise ValueError("Specify tangent_dim and retract_fn together")
        cls.default_factory = staticmethod(default_factory)
        cls.tangent_dim = default.numel() if tangent_dim is None else tangent_dim
        if cls.tangent_dim <= 0:
            raise ValueError("tangent_dim must be positive")
        cls.retract_fn = staticmethod(
            (lambda x, delta: x + delta.reshape(x.shape)) if retract_fn is None else retract_fn
        )

    def __post_init__(self):
        _ids(self.id)

    def __getitem__(self, index) -> Var:
        return type(self)(_ids(self.id)[index])

    def with_value(self, value: Tensor) -> VarWithValue:
        return VarWithValue(self, value)


@dataclass(frozen=True)
class VarWithValue:
    variable: Var
    value: Tensor


@dataclass(frozen=True)
class _ParameterRef:
    name: str
    id: int | Tensor

    def __post_init__(self):
        _ids(self.id)


@dataclass(frozen=True)
class _EuclideanParameterization:
    shape: tuple[int, ...]

    @property
    def tangent_dim(self):
        return math.prod(self.shape)

    def default_factory(self):
        return torch.zeros(self.shape)

    @staticmethod
    def retract_fn(value, delta):
        return value + delta.reshape(value.shape)


@dataclass(frozen=True)
class VariableLayout:
    var_type: type[Var] | _EuclideanParameterization
    ids: Tensor
    offset: int
    name: str | None = None

    @property
    def key(self):
        return self.var_type if self.name is None else self.name

    @property
    def count(self) -> int:
        return self.ids.numel()

    @property
    def dim(self) -> int:
        return self.var_type.tangent_dim


class _NamedValues:
    """Solver state keyed directly by parameter-group name."""

    def __init__(self, values: Mapping[str, Tensor], layout: tuple[VariableLayout, ...]):
        if not layout or set(values) != {entry.key for entry in layout}:
            raise ValueError("Parameter names do not match the analyzed problem")
        first = values[layout[0].key]
        for entry in layout:
            value = values[entry.key]
            expected = (entry.count, *entry.var_type.default_factory().shape)
            if not isinstance(value, Tensor) or value.shape != expected:
                raise ValueError(f"{entry.key}: expected parameter tensor of shape {expected}")
            if value.dtype not in (torch.float32, torch.float64):
                raise ValueError("Use float32 or float64 for least squares")
            if value.dtype != first.dtype or value.device != first.device:
                raise ValueError("All parameters must have the same dtype and device")
        self._values = dict(values)
        self._ids = {entry.key: entry.ids for entry in layout}
        self._layout = layout

    @property
    def dtype(self):
        return next(iter(self._values.values())).dtype

    @property
    def device(self):
        return next(iter(self._values.values())).device

    def detach(self):
        return _NamedValues(
            {name: value.detach() for name, value in self._values.items()}, self._layout
        )

    def retract(self, tangent, layout):
        values = {}
        for entry in layout:
            delta = tangent[entry.offset : entry.offset + entry.count * entry.dim]
            values[entry.key] = torch.func.vmap(entry.var_type.retract_fn)(
                self._values[entry.key], delta.reshape(entry.count, entry.dim)
            )
        return _NamedValues(values, layout)


class VarValues:
    """Values stacked by variable type and sorted by ID."""

    def __init__(self, values: dict[type[Var], Tensor], ids: dict[type[Var], Tensor]):
        if not values or values.keys() != ids.keys():
            raise ValueError("Values and IDs must contain the same nonempty types")
        self._values = dict(values)
        self._ids = {key: _ids(value).reshape(-1).cpu() for key, value in ids.items()}
        first = next(iter(values.values()))
        for key, value in values.items():
            expected = (len(self._ids[key]), *key.default_factory().shape)
            if value.shape != expected:
                raise ValueError(f"{key.__name__}: expected values of shape {expected}")
            if (
                not value.is_floating_point()
                or value.dtype != first.dtype
                or value.device != first.device
            ):
                raise ValueError("All variable values must have the same floating dtype and device")
            if value.dtype not in (torch.float32, torch.float64):
                raise ValueError("Use float32 or float64 for least squares")
            if len(self._ids[key]) == 0 or not bool(
                torch.all(self._ids[key][1:] > self._ids[key][:-1])
            ):
                raise ValueError("Variable IDs must be nonempty, sorted and unique")

    @classmethod
    def make(cls, variables: Iterable[Var | VarWithValue]) -> VarValues:
        value_parts: dict[type[Var], list[Tensor]] = {}
        id_parts: dict[type[Var], list[Tensor]] = {}
        for item in variables:
            var = item.variable if isinstance(item, VarWithValue) else item
            key = type(var)
            ids = _ids(var.id)
            shape = key.default_factory().shape
            if isinstance(item, VarWithValue):
                value = item.value
                expected = (*ids.shape, *shape)
                if value.shape != expected:
                    raise ValueError(
                        f"{key.__name__}: expected shape {expected}, got {value.shape}"
                    )
            else:
                value = key.default_factory().to(device=ids.device).expand((*ids.shape, *shape))
            value_parts.setdefault(key, []).append(value.reshape(len(ids.reshape(-1)), *shape))
            id_parts.setdefault(key, []).append(ids.reshape(-1).cpu())
        values, all_ids = {}, {}
        for key, parts in value_parts.items():
            ids = torch.cat(id_parts[key])
            order = ids.argsort()
            all_ids[key] = ids[order]
            stacked = torch.cat(parts)
            values[key] = stacked[order.to(stacked.device)]
        return cls(values, all_ids)

    @property
    def dtype(self):
        return next(iter(self._values.values())).dtype

    @property
    def device(self):
        return next(iter(self._values.values())).device

    def __getitem__(self, var: Var | type[Var]) -> Tensor:
        if isinstance(var, type):
            return self._values[var]
        ids = _ids(var.id).cpu()
        available = self._ids[type(var)]
        index = torch.searchsorted(available, ids)
        if bool(torch.any(index >= len(available))) or not torch.equal(available[index], ids):
            raise KeyError(f"Unknown {type(var).__name__} ID")
        return self._values[type(var)][index.to(self.device)]

    def to(self, *args, **kwargs) -> VarValues:
        return VarValues({key: x.to(*args, **kwargs) for key, x in self._values.items()}, self._ids)

    def detach(self) -> VarValues:
        return VarValues({key: x.detach() for key, x in self._values.items()}, self._ids)

    def retract(self, tangent: Tensor, layout: tuple[VariableLayout, ...]) -> VarValues:
        values = {}
        for entry in layout:
            delta = tangent[entry.offset : entry.offset + entry.count * entry.dim]
            values[entry.var_type] = torch.func.vmap(entry.var_type.retract_fn)(
                self._values[entry.var_type], delta.reshape(entry.count, entry.dim)
            )
        return VarValues(values, self._ids)
