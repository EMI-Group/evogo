from typing import Callable, List

import jax
import jax.numpy as jnp
import numpy as np
import math

from test_functions.simple_functions_jax import HarderNumerical  # , Ackley, Rosenbrock, Rastrigin, Levy
# from test_functions.numerical.cec2022 import CEC2022


def get_functions(FUNC_ID: int = 0, NUM_PARALLEL: int = 1):
    """
    Returns a list of functions and their corresponding dimensions.
    :param NUM_PARALLEL: Number of parallel evaluations.
    :return: Tuple of functions and their dimensions.
    """
    key = jax.random.PRNGKey(FUNC_ID * 1000)
    keys = jax.random.split(key, 4)

    dimensions: List[int] = []
    functions: List[Callable[[jax.Array], jax.Array]] = []

    _ndim = 5
    functions += [
        HarderNumerical(_ndim, keys[0], HarderNumerical.Ackley, NUM_PARALLEL),
        HarderNumerical(_ndim, keys[1], HarderNumerical.Rosenbrock, NUM_PARALLEL),
        HarderNumerical(_ndim, keys[2], HarderNumerical.Rastrigin, NUM_PARALLEL),
        HarderNumerical(_ndim, keys[3], HarderNumerical.Levy, NUM_PARALLEL),
    ]
    dimensions += [_ndim] * 4
    _ndim = 10
    functions += [
        HarderNumerical(_ndim, keys[0], HarderNumerical.Ackley, NUM_PARALLEL),
        HarderNumerical(_ndim, keys[1], HarderNumerical.Rosenbrock, NUM_PARALLEL),
        HarderNumerical(_ndim, keys[2], HarderNumerical.Rastrigin, NUM_PARALLEL),
        HarderNumerical(_ndim, keys[3], HarderNumerical.Levy, NUM_PARALLEL),
    ]
    dimensions += [_ndim] * 4
    _ndim = 20
    functions += [
        HarderNumerical(_ndim, keys[0], HarderNumerical.Ackley, NUM_PARALLEL),
        HarderNumerical(_ndim, keys[1], HarderNumerical.Rosenbrock, NUM_PARALLEL),
        HarderNumerical(_ndim, keys[2], HarderNumerical.Rastrigin, NUM_PARALLEL),
        HarderNumerical(_ndim, keys[3], HarderNumerical.Levy, NUM_PARALLEL),
    ]
    dimensions += [_ndim] * 4

    from test_functions.push_function import get_push_func
    from test_functions.rover_function import get_rover_func
    from test_functions.mujoco_functions import get_heuristic_landing_func, get_walker_func, get_ant_func
    for _fn in (get_push_func, get_heuristic_landing_func):
        _ff, _ndim = _fn()
        functions.append(convert_host_reward_fn(_ff))
        dimensions.append(_ndim)
    _ndim = 200
    functions += [
        HarderNumerical(_ndim, keys[0], HarderNumerical.Ackley, NUM_PARALLEL, affine=False),  # 14
        HarderNumerical(_ndim, keys[1], HarderNumerical.Rosenbrock, NUM_PARALLEL, affine=False),
        HarderNumerical(_ndim, keys[2], HarderNumerical.Rastrigin, NUM_PARALLEL, affine=False),
        HarderNumerical(_ndim, keys[3], HarderNumerical.Levy, NUM_PARALLEL, affine=False),
    ]
    dimensions += [_ndim] * 4
    
    _ndim = 1000
    functions += [
        HarderNumerical(_ndim, keys[0], HarderNumerical.Ackley, NUM_PARALLEL, affine=False),  # 14
        HarderNumerical(_ndim, keys[1], HarderNumerical.Rosenbrock, NUM_PARALLEL, affine=False),
        HarderNumerical(_ndim, keys[2], HarderNumerical.Rastrigin, NUM_PARALLEL, affine=False),
        HarderNumerical(_ndim, keys[3], HarderNumerical.Levy, NUM_PARALLEL, affine=False),
    ]
    dimensions += [_ndim] * 4

    from test_functions.push_function_high_dim import get_push_func_high_dim
    from test_functions.rover_function_high_dim import get_rover_func_high_dim
    from test_functions.mujoco_functions_high_dim import get_large_heuristic_landing_func, get_large_walker_func, get_large_ant_func
    for _fn in (get_large_heuristic_landing_func, get_push_func_high_dim, get_rover_func_high_dim, get_large_walker_func, get_large_ant_func):
        _ff, _ndim = _fn()
        functions.append(convert_host_reward_fn(_ff))
        dimensions.append(_ndim)

    return functions, dimensions

def convert_host_reward_fn(f: Callable[[np.ndarray], float]) -> Callable[[jax.Array], jax.Array]:
    
    def _converted_fn(xs: jax.Array) -> jax.Array:
        xs_host = xs.reshape(math.prod(xs.shape[:2]), xs.shape[2])
        xs_host = jax.device_get(xs_host)
        fs = list(map(f, xs_host))
        fs = jnp.array(fs)
        return -fs.reshape(xs.shape[:2])
    
    return _converted_fn