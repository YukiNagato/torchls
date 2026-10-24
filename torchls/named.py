"""Name-and-index modeling API, sharing the block-sparse solver implementation."""

from __future__ import annotations

from collections.abc import Callable, Mapping

import torch
from torch import Tensor

from .cost import Cost, JacMode
from .problem import AnalyzedLeastSquaresProblem
from .variables import (
    Var,
    VariableLayout,
    _EuclideanParameterization,
    _ids,
    _NamedValues,
    _ParameterRef,
)


class Problem:
    """Own parameter tensors by name and bind residual inputs with (name, index).

    Each parameter tensor has shape [count, *storage_shape]. Unspecified
    groups use Euclidean addition. ``manifolds`` optionally assigns existing
    Var parameterizations, e.g. {"cams": SE3Var, "objects": SE3Var}; names,
    rather than those types, identify independent groups.

    ``solve`` returns a dict and writes the solved tensors into ``parameters``.
    Inputs are not modified in place. The numerical backend matches the
    typed-variable API.
    """

    def __init__(
        self,
        parameters: Mapping[str, Tensor],
        *,
        manifolds: Mapping[str, type[Var]] | None = None,
    ):
        self.parameters = dict(parameters)
        self.manifolds = dict(manifolds or {})
        self._costs: list[Cost] = []
        self._analyzed = None
        self._signature = None
        self.last_summary = None

    def add_residual(
        self,
        residual_fn: Callable[..., Tensor],
        *args,
        jac_mode: JacMode = "auto",
        batch_size: int | None = None,
        in_dims: tuple[int | None, ...] | None = None,
        jacobian_fn: Callable[..., tuple[Tensor, ...]] | None = None,
    ) -> Problem:
        """Add a batch using an ordinary, undecorated residual function.

        ("name", indices) selects rows of an optimizable parameter group.
        Other arguments are constant measurements/data. Index tensors define
        the batch; scalar indices broadcast. ``in_dims`` controls shared data
        exactly as in Cost. Indices are copied when the batch is added.
        """
        if not callable(residual_fn):
            raise TypeError("The first argument must be a residual function")
        bound = tuple(
            _ParameterRef(arg[0], _ids(arg[1]).detach().cpu().clone())
            if isinstance(arg, tuple) and len(arg) == 2 and isinstance(arg[0], str)
            else arg
            for arg in args
        )
        self._costs.append(Cost(residual_fn, bound, jac_mode, batch_size, in_dims, jacobian_fn))
        self._analyzed = None
        return self

    def _ensure_analyzed(self):
        if not self.parameters or not self._costs:
            raise ValueError("A problem needs parameters and residuals")
        if set(self.manifolds) - set(self.parameters):
            raise ValueError("Manifold names must refer to declared parameter groups")
        signature, specs = [], []
        for name, value in self.parameters.items():
            if not isinstance(name, str) or not name:
                raise ValueError("Parameter group names must be nonempty strings")
            if not isinstance(value, Tensor) or value.ndim < 1 or value.numel() == 0:
                raise ValueError(
                    f"{name}: expected a nonempty tensor with a leading parameter axis"
                )
            spec = self.manifolds.get(name)
            if spec is None:
                spec = _EuclideanParameterization(tuple(value.shape[1:]))
            elif not isinstance(spec, type) or not issubclass(spec, Var):
                raise TypeError(f"{name}: manifold must be a Var parameterization such as SE3Var")
            specs.append(spec)
            signature.append((name, tuple(value.shape), spec))
        signature = tuple(signature)
        if self._analyzed is None or signature != self._signature:
            layout, offset = [], 0
            for (name, value), spec in zip(self.parameters.items(), specs, strict=True):
                entry = VariableLayout(
                    spec, torch.arange(value.shape[0], device="cpu"), offset, name=name
                )
                layout.append(entry)
                offset += entry.count * entry.dim
            layout = tuple(layout)
            _NamedValues(self.parameters, layout)
            self._analyzed = AnalyzedLeastSquaresProblem(layout, tuple(self._costs), offset)
            self._signature = signature
        return self._analyzed

    def analyze(self) -> Problem:
        """Precompute layouts and local-to-global indices; also done lazily."""
        self._state(None)
        return self

    @property
    def tangent_dim(self):
        return self._ensure_analyzed().tangent_dim

    def _state(self, parameters):
        analyzed = self._ensure_analyzed()
        return analyzed, _NamedValues(
            self.parameters if parameters is None else parameters, analyzed.layout
        )

    def residuals(self, parameters: Mapping[str, Tensor] | None = None):
        analyzed, state = self._state(parameters)
        return analyzed.residuals(state)

    def cost(self, parameters: Mapping[str, Tensor] | None = None):
        analyzed, state = self._state(parameters)
        return analyzed.cost(state)

    def linearize(self, parameters: Mapping[str, Tensor] | None = None):
        analyzed, state = self._state(parameters)
        return analyzed.linearize(state)

    def solve(self, *, return_summary: bool = False, **options):
        analyzed, initial = self._state(None)
        solution, summary = analyzed.solve(initial, return_summary=True, **options)
        result = dict(solution._values)
        self.parameters.update(result)
        self.last_summary = summary
        return (result, summary) if return_summary else result
