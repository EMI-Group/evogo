from typing import Any, Callable, Sequence, Tuple
from math import prod
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp


def init_wrapper(func):
    """
    Flax network init helper
    """
    return lambda *args: func(*(args[-3:])) if len(args) > 3 else func(*args)


def vmap_last_dim(func: Callable,
                  *inputs: jax.Array,
                  last_ndim: int = 1,
                  last_dims_out_shape: None | Sequence[int] = None) -> jax.Array:
    """
    `jax.vmap` the last (few) dimension(s) to the input array(s)
    """
    assert last_ndim >= 1
    for input in inputs:
        assert input.ndim >= last_ndim
        assert input.shape[:-last_ndim] == inputs[0].shape[:-last_ndim]
    if last_dims_out_shape is None:
        last_dims_out_shape = inputs[0].shape[-last_ndim:]
    reshaped = [
        input.reshape(prod(input.shape[:-last_ndim]), *input.shape[-last_ndim:]) for input in inputs
    ]
    output = jax.vmap(func)(*reshaped)
    return output.reshape(inputs[0].shape[:-last_ndim] + tuple(last_dims_out_shape))


@partial(jax.jit, static_argnums=[1, 2])
def latin_hyper_cube(key: jax.Array, num: int, dim: int) -> jax.Array:
    key1, key2 = jax.random.split(key)
    perms = jax.random.permutation(key1,
                                   jnp.tile(jnp.arange(num), (dim, 1)),
                                   axis=1,
                                   independent=True)
    reals = jax.random.uniform(key2, shape=(dim, num))
    samples = (reals + perms) / num
    return samples.T


@partial(jax.jit, static_argnames=["num"])
def sort_select(xs: jax.Array,
                ys: jax.Array,
                num: int | None = None) -> Tuple[jax.Array, jax.Array]:
    if num is None:
        num = len(xs) // 2
    perm = jnp.argsort(ys)
    return xs[perm[:num]], ys[perm[:num]]


def print_with_prefix(array: Any, prefix_cnt: int | None = None) -> str:
    if isinstance(array, jax.Array):
        array = jax.device_get(array)
    else:
        array = np.asarray(array)
    if prefix_cnt is not None:
        return np.array2string(array, prefix=" " * prefix_cnt, max_line_width=1000)
    else:
        return np.array2string(array, max_line_width=1000)
