"""Reusable batched local autodiff functions."""

from functools import lru_cache

import torch


def make_linearizer(cost, slots, types, mode, chunk_size, has_loss):
    """Build once per cost plan; numeric parameter/measurement tensors are inputs."""
    key = (
        cost.residual_fn,
        cost.jacobian_fn,
        len(cost.args),
        cost.loss,
        slots,
        types,
        mode,
        chunk_size,
        has_loss,
    )
    try:
        hash(key)
    except TypeError:
        # User-defined callable/loss objects need not be hashable.
        return _make_linearizer(*key)
    return _cached_linearizer(*key)


@lru_cache(maxsize=64)
def _cached_linearizer(*key):
    return _make_linearizer(*key)


def _make_linearizer(
    residual_fn, jacobian_fn, arg_count, loss, slots, types, mode, chunk_size, has_loss
):
    retract_jvps = tuple(getattr(spec, "retract_jvp_fn", None) for spec in types)

    def local_residual(deltas, *args):
        updated = list(args)
        for pos, spec, jvp, delta in zip(slots, types, retract_jvps, deltas, strict=True):
            # AD is evaluated ONLY at delta=0. This affine local chart has the
            # exact value and first derivative of the true retraction there.
            # Actual solver updates always use spec.retract_fn, including Exp.
            updated[pos] = (
                args[pos] + jvp(args[pos], delta)
                if jvp is not None
                else spec.retract_fn(args[pos], delta)
            )
        residual = residual_fn(*updated).reshape(-1)
        return residual, residual

    if jacobian_fn is None:
        differentiate = torch.func.jacrev if mode == "reverse" else torch.func.jacfwd
        batched_jac = torch.func.vmap(
            differentiate(local_residual, argnums=0, has_aux=True),
            in_dims=(None, *([0] * arg_count)),
            chunk_size=chunk_size,
        )
    else:
        batched_jac = torch.func.vmap(jacobian_fn, chunk_size=chunk_size)
        batched_residual = torch.func.vmap(residual_fn, chunk_size=chunk_size)

    def evaluate(zeros, *args):
        if jacobian_fn is None:
            blocks, residual = batched_jac(zeros, *args)
        else:
            blocks = batched_jac(*args)
            residual = batched_residual(*args).reshape(args[0].shape[0], -1)
            if not isinstance(blocks, tuple) or len(blocks) != len(types):
                raise ValueError("Custom Jacobian must return one block per Var argument")
        for block, spec in zip(blocks, types, strict=True):
            if block.shape != (args[0].shape[0], residual.shape[1], spec.tangent_dim):
                raise ValueError("Incorrect local Jacobian shape")
        raw, objective = None, None
        if has_loss:
            raw = residual
            squared = residual.to(torch.float64).square().sum(dim=-1)
            if loss is None:
                objective = squared
            else:
                evaluation = loss.evaluate(squared)
                objective = evaluation[0]
                residual, blocks = loss._linearize(residual, blocks, squared, evaluation)
        return tuple(blocks), residual, raw, objective

    return evaluate
