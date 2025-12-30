import time
import math
from functools import partial
from typing import Any, Callable, Dict, Sequence, Tuple

import chex
import jax
import jax.numpy as jnp

import flax
import optax
from flax.core.scope import FrozenVariableDict

ParamsType = FrozenVariableDict | Dict[str, Any]

import numpy as np
try:
    import matplotlib.pyplot as plt
    HAS_PLT = True
except:
    HAS_PLT = False

from evogo_jax.utils import print_with_prefix


@partial(jax.jit, static_argnums=[0], static_argnames=["shuffle"])
def prepare_epoch(batch_size: int,
                  key: jax.Array,
                  *datasets: jax.Array,
                  shuffle: bool = True) -> Tuple[jax.Array, ...]:
    """
    Prepare a (shuffled) epoch with given batch size over given datasets

    ### Args
    `batch_size`: the batch size
    `key`: the random key
    `datasets`: the datasets, each one has dimension [num_parallels, dataset_size, *dim]
    `shuffle`: whether shuffle or not

    ### Return
    The datasets with each one splitted to [num_parallels, num_batches, batch_size, *dim]
    """
    num_parallels = datasets[0].shape[0]
    dataset_size = datasets[0].shape[1]
    num_batches = dataset_size // batch_size
    if shuffle:
        keys = jax.random.split(key, num=num_parallels)
        perms = [jax.random.permutation(k, dataset_size) for k in keys]
        perms = [p[:num_batches * batch_size].reshape(num_batches, batch_size)
                 for p in perms]  # skip incomplete batch
        datasets = tuple(
            jnp.stack([d[p, ...] for d, p in zip(dataset, perms)]).swapaxes(0, 1)
            for dataset in datasets)
    else:
        datasets = tuple(jnp.stack(jnp.split(dataset, num_batches, axis=1)) for dataset in datasets)
    return datasets


def breakpoint_if_nonfinite(x):
    is_finite = jnp.isfinite(x).all()

    def true_fn(x):
        pass

    def false_fn(x):
        jax.debug.breakpoint()

    jax.lax.cond(is_finite, true_fn, false_fn, x)


GRAD_BREAKPOINT = False


@partial(jax.jit, static_argnums=[0, 1, 2])
def train_epoch(
    loss_grad_fn: Callable[
        [chex.ArrayTree, jax.Array],
        Tuple[Tuple[float, jax.Array], chex.ArrayTree],
    ],
    loss_shape: int | Sequence[int],
    tx: optax.GradientTransformation,
    params_in: chex.ArrayTree,
    opt_state_in: chex.ArrayTree,
    key: jax.Array,
    *batches_all: jax.Array,
) -> Tuple[jax.Array, chex.ArrayTree, chex.ArrayTree]:
    """
    Train a epoch with prepared epoch batches

    ### Args
    `loss_grad_fn`: the loss value and its gradient function with inputs (`params_in`, *batched_inputs)
    `loss_shape`: the shape of loss values
    `tx`: the optimizer
    `params_in`: the input network parameters
    `opt_state_in`: the input optimizer state
    `key`: the random key
    `batches_all`: the prepared batches, each one has dimension [num_parallels, num_batches, batch_size, *dim] of each one

    ### Return
    `loss_val`: the averaged loss value of this epoch
    `params_out`: the output network parameters
    `opt_state_out`: the output optimizer state
    """
    num_batches = batches_all[0].shape[0]
    if num_batches == 0:
        return jnp.zeros(loss_shape, dtype=batches_all[0].dtype), params_in, opt_state_in

    _keys = jax.random.split(key, num_batches)

    def _loop_fn(b: int, inputs: Tuple[
        chex.ArrayTree,
        chex.ArrayTree,
        jax.Array,
    ]):
        params_in, opt_state_in, loss_val_all = inputs
        (_, loss_vals), grads = loss_grad_fn(params_in,
                                             *(batches[b] for batches in batches_all),
                                             rngs={'dropout': _keys[b]})  # type: ignore
        if GRAD_BREAKPOINT:
            jax.tree_map(breakpoint_if_nonfinite, grads)
        updates, opt_state_out = tx.update(grads, opt_state_in)
        params_out = optax.apply_updates(params_in, updates)
        loss_val_all += loss_vals
        return params_out, opt_state_out, loss_val_all

    params_out, opt_state_out, loss_val_avg = jax.lax.fori_loop(
        0,
        num_batches,
        _loop_fn,
        (
            params_in,
            opt_state_in,
            jnp.zeros(loss_shape, dtype=batches_all[0].dtype),
        ),
    )
    return loss_val_avg / num_batches, params_out, opt_state_out


@partial(jax.jit, static_argnums=[0, 1, 2, 3], static_argnames=["shuffle"])
def full_epoch(
    batch_size: int,
    loss_fn: Callable[[chex.ArrayTree, jax.Array], jax.Array | float],
    loss_shape: int | Sequence[int],
    tx: optax.GradientTransformation,
    params_in: chex.ArrayTree,
    opt_state_in: chex.ArrayTree,
    key: jax.Array,
    *datasets: jax.Array,
    shuffle: bool = True,
) -> Tuple[jax.Array, chex.ArrayTree, chex.ArrayTree]:
    """
    Train a full epoch with given batch size over given datasets

    ### Args
    `batch_size`: the batch size
    `loss_grad_fn`: the loss value(s) function with inputs (`params_in`, *batched_inputs)
    `loss_shape`: the shape of loss values
    `tx`: the optimizer
    `params_in`: the input network parameters
    `opt_state_in`: the input optimizer state
    `key`: the random key
    `datasets`: the datasets, each one has dimension [num_parallels, dataset_size, *dim]
    `shuffle`: whether shuffle or not

    ### Return
    `loss_val`: the averaged loss value(s) of this epoch
    `params_out`: the output network parameters
    `opt_state_out`: the output optimizer state
    """

    def _aux_sum_loss(params, *inputs, **kwargs):
        losses = loss_fn(params, *inputs, **kwargs)
        return jnp.sum(losses), losses

    loss_grad_fn = jax.value_and_grad(_aux_sum_loss, has_aux=True)
    key1, key2 = jax.random.split(key)
    batches_all = prepare_epoch(batch_size, key1, *datasets, shuffle=shuffle)
    return train_epoch(loss_grad_fn, loss_shape, tx, params_in, opt_state_in, key2, *batches_all)


def train_valid_split(key: jax.Array, valid_portion: float = 0.1, **org_datasets: jax.Array):
    """
    Split train and validation sets from given datasets

    ### Args
    `key`: the random key
    `org_datasets`: the datasets, each one has dimension [num_parallels, dataset_size, *dim]
    `valid_portion`: the validation set size portion

    ### Return
    `datasets`: the output training sets
    `validsets`: the output validation sets
    """
    print(
        f"[DEBUG] Splitting datasets {tuple(org_datasets.keys())} of size {tuple(a.shape for a in org_datasets.values())} with validation portion = {valid_portion}"
    )
    datasets: Sequence[jax.Array] = tuple(org_datasets.values())
    num_parallels = datasets[0].shape[0]
    dataset_size = datasets[0].shape[1]
    num_valid = math.floor(dataset_size * valid_portion)
    keys = jax.random.split(key, num=num_parallels)
    perms = [jax.random.permutation(k, dataset_size) for k in keys]
    valid_perms = [p[:num_valid] for p in perms]
    data_perms = [p[num_valid:] for p in perms]
    validsets = tuple(
        jnp.stack([d[p, ...] for d, p in zip(dataset, valid_perms)]) for dataset in datasets)
    datasets = tuple(
        jnp.stack([d[p, ...] for d, p in zip(dataset, data_perms)]) for dataset in datasets)
    return datasets, validsets


def _get_parallel_best_params(loss_vals: jax.Array, params: ParamsType,
                              *prev) -> Tuple[jax.Array, ParamsType]:
    prev_loss: jax.Array = prev[0]
    prev_params: ParamsType = prev[1]
    replace_mask = loss_vals <= prev_loss

    def _replace_value(new_param: Dict[str, Any] | jax.Array,
                       old_param: Dict[str, Any] | jax.Array) -> Dict[str, Any] | jax.Array:
        if isinstance(old_param, jax.Array):
            assert (isinstance(new_param, jax.Array))
            return jnp.where(jnp.expand_dims(replace_mask, range(1, new_param.ndim)), new_param,
                             old_param)
        # assert isinstance(new_param, dict)
        params = {}
        for k, nv in new_param.items():
            pv = old_param[k]
            params[k] = _replace_value(nv, pv)
        return params

    return jnp.where(replace_mask, loss_vals, prev_loss), _replace_value(params, prev_params)  # type: ignore


_COLORS = ['b', 'g', 'r', 'c', 'm', 'y', 'k']


def train_model(key: jax.Array,
                batch_size: int,
                epochs: int,
                net: flax.linen.Module,
                tx: optax.GradientTransformation,
                params: ParamsType,
                opt_state: chex.ArrayTree,
                loss_names: Dict[str, int],
                shuffle: bool = True,
                DEBUG: bool = False,
                valid_portion: float = 0.0,
                net_eval_kwargs: Dict[str, bool] = {"training": False},
                **org_datasets: jax.Array):
    """
    Train the given `net` with initial `params` for `epochs` with given `datasets`

    ### Return
    `best_params`
    """
    datasets = tuple(org_datasets.values())
    t0 = t_prev = time.time()
    if len(loss_names) > 0:
        prefix_cnt = max(max(len(n) for n, i in loss_names.items() if i >= 0), len("history")) + 15
    else:
        prefix_cnt = 0
    num_parallels = datasets[0].shape[0]
    num_losses = len(loss_names)
    loss_shape = (num_losses, num_parallels) if num_losses > 1 else (num_parallels, )
    best_params = (jnp.full(shape=num_parallels, fill_value=1e10), params)
    loss_vals_all = jnp.zeros(shape=(epochs, num_parallels))
    if valid_portion > 0:  # with validation set
        key, key1 = jax.random.split(key)
        datasets, validsets = train_valid_split(key1, valid_portion=valid_portion, **org_datasets)
        batch_size = min(batch_size, datasets[0].shape[1])
    i_prev = 0
    for i in range(1, epochs + 1):
        # run epoch
        key = jax.random.split(key, num=())
        loss_vals: jax.Array
        loss_vals, params, opt_state = full_epoch(batch_size,
                                                  net.apply,
                                                  loss_shape,
                                                  tx,
                                                  params,
                                                  opt_state,
                                                  key,
                                                  *datasets,
                                                  shuffle=shuffle)  # type: ignore
        # deal with loss
        if valid_portion > 0:  # with validation set
            key = jax.random.split(key, num=())
            loss_vals = net.apply(params, *validsets, **net_eval_kwargs, rngs={"dropout": key})  # type: ignore
        total_loss = jnp.sum(loss_vals, axis=0) if loss_vals.ndim > 1 else loss_vals
        loss_vals_all = loss_vals_all.at[i - 1].set(total_loss)
        # preserve best parameters
        best_params = _get_parallel_best_params(total_loss, params, *best_params)
        # prints
        if i % (epochs // 4) == 0 or \
                time.time() - t_prev >= 600 or \
                (DEBUG and i % (epochs // 20) == 0):
            print(
                f"[INFO]  Loss epoch {i}/{epochs}, total_loss = {print_with_prefix(total_loss)}, elapsed = {time.time() - t0}:"
            )
            if len(loss_names) <= 0:
                t_prev = time.time()
                i_prev = i
                continue
            print(
                " " * (prefix_cnt - len("history") - 3) +
                f"history = {print_with_prefix(loss_vals_all[i_prev:i - 1], prefix_cnt=prefix_cnt)}"
            )
            for name, idx in loss_names.items():
                if idx < 0:
                    continue
                print(" " * (prefix_cnt - len(name) - 3) +
                      f"{name} = {print_with_prefix(loss_vals[idx], prefix_cnt=prefix_cnt)}")
            t_prev = time.time()
            i_prev = i
    # output
    print(
        f"[INFO]  Best total_loss = {print_with_prefix(best_params[0])}, elapsed = {time.time() - t0}"
    )
    if DEBUG and HAS_PLT:
        loss_vals_all = np.transpose(jax.device_get(loss_vals_all))
        for i in range(num_parallels):
            plt.plot(loss_vals_all[i], _COLORS[i % len(_COLORS)])
        plt.show()
    return best_params[1], best_params[0] != 1e10
