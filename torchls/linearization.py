"""Reusable batched local autodiff functions."""

from functools import lru_cache

import torch


def stream_linearization(evaluate_chunk, count, chunk_size):
    """Evaluate and write chunks without retaining their tensors for a final cat."""
    def allocate(value):
        return None if value is None else value.new_empty((count, *value.shape[1:]))

    def write(destination, value, start, end):
        if destination is not None:
            destination[start:end].copy_(value)

    for start in range(0, count, chunk_size):
        end = min(start + chunk_size, count)
        blocks, residual, raw, objective = evaluate_chunk(start, end)
        if start == 0:
            output_blocks = tuple(allocate(block) for block in blocks)
            output_residual = allocate(residual)
            output_raw = output_residual if raw is residual else allocate(raw)
            output_objective = allocate(objective)
        for index in range(len(blocks)):
            write(output_blocks[index], blocks[index], start, end)
        write(output_residual, residual, start, end)
        if output_raw is not output_residual:
            write(output_raw, raw, start, end)
        write(output_objective, objective, start, end)
        # copy_ retains autograd connectivity if callers enable gradients.
        del blocks, residual, raw, objective
    return output_blocks, output_residual, output_raw, output_objective


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
        )
    else:
        batched_jac = torch.func.vmap(jacobian_fn)
        batched_residual = torch.func.vmap(residual_fn)

    def evaluate_chunk(zeros, *args):
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

    def evaluate(zeros, *args):
        count = args[0].shape[0]
        if chunk_size is None or count <= chunk_size:
            return evaluate_chunk(zeros, *args)

        # vmap(chunk_size=...) retains every chunk before concatenating, so its
        # peak includes two copies of the complete Jacobian. Write directly to
        # final buffers instead. Robust-loss transforms also run per chunk to
        # avoid a second full set of unweighted Jacobian blocks.
        return stream_linearization(
            lambda start, end: evaluate_chunk(zeros, *(arg[start:end] for arg in args)),
            count, chunk_size,
        )

    return evaluate
