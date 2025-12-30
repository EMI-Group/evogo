from typing import Callable, List
import math

import numpy as np
import torch


def convert_host_reward_fn(f: Callable[[np.ndarray], float]) -> Callable[[torch.Tensor], torch.Tensor]:
    def _converted_fn(xs: torch.Tensor) -> torch.Tensor:
        xs_host = xs.reshape(math.prod(xs.shape[:2]), xs.shape[2])
        xs_host = xs_host.cpu().numpy()
        fs = list(map(f, xs_host))
        fs = torch.tensor(fs, dtype=xs.dtype, device=xs.device)
        return -fs.reshape(xs.shape[:2])

    return _converted_fn


def get_functions(seed: int, device: torch.device, instances: int):
    torch.manual_seed(seed)

    import sys
    import os

    if os.path.abspath(os.path.join(__file__, "../../")) not in sys.path:
        sys.path.append(os.path.abspath(os.path.join(__file__, "../../")))

    from test_functions.simple_functions import HarderNumerical

    
    functions: List[Callable[[torch.Tensor], torch.Tensor]] = []
    dimensions: List[int] = []

    _ndim = 5
    functions += [
        HarderNumerical(_ndim, device, HarderNumerical.Ackley, instances), # 12
        HarderNumerical(_ndim, device, HarderNumerical.Rosenbrock, instances),
        HarderNumerical(_ndim, device, HarderNumerical.Rastrigin, instances),
        HarderNumerical(_ndim, device, HarderNumerical.Levy, instances),
    ]
    dimensions += [_ndim] * 4
    _ndim = 10
    functions += [
        HarderNumerical(_ndim, device, HarderNumerical.Ackley, instances), # 16
        HarderNumerical(_ndim, device, HarderNumerical.Rosenbrock, instances),
        HarderNumerical(_ndim, device, HarderNumerical.Rastrigin, instances),
        HarderNumerical(_ndim, device, HarderNumerical.Levy, instances),
    ]
    dimensions += [_ndim] * 4
    _ndim = 20
    functions += [
        HarderNumerical(_ndim, device, HarderNumerical.Ackley, instances), # 20
        HarderNumerical(_ndim, device, HarderNumerical.Rosenbrock, instances),
        HarderNumerical(_ndim, device, HarderNumerical.Rastrigin, instances),
        HarderNumerical(_ndim, device, HarderNumerical.Levy, instances),
    ]
    dimensions += [_ndim] * 4


    from test_functions.push_function import get_push_func
    from test_functions.rover_function import get_rover_func
    from test_functions.mujoco_functions import get_heuristic_landing_func, get_walker_func, get_ant_func

    for _fn in (get_push_func, get_heuristic_landing_func): # 24-25
        _ff, _ndim = _fn()
        functions.append(convert_host_reward_fn(_ff))
        dimensions.append(_ndim)
    
    _ndim = 200
    functions += [
        HarderNumerical(_ndim, device, HarderNumerical.Ackley, instances, affine=False),  # 26
        HarderNumerical(_ndim, device, HarderNumerical.Rosenbrock, instances, affine=False),
        HarderNumerical(_ndim, device, HarderNumerical.Rastrigin, instances, affine=False),
        HarderNumerical(_ndim, device, HarderNumerical.Levy, instances, affine=False),
    ]
    dimensions += [_ndim] * 4
    
    for _fn in (get_rover_func, get_walker_func, get_ant_func): # 30-32
        _ff, _ndim = _fn()
        functions.append(convert_host_reward_fn(_ff))
        dimensions.append(_ndim)
    
    # from test_functions.mujoco_functions import get_ant_func

    # for _fn in (get_ant_func,):
    #     _ff, _ndim = _fn()
    #     functions.append(convert_host_reward_fn(_ff))
    #     dimensions.append(_ndim)
    # For Ablation Study only
    # _ndim = 200
    # functions += [
    #     HarderNumerical(_ndim, device, HarderNumerical.Ackley, instances, affine=False),
    #     HarderNumerical(_ndim, device, HarderNumerical.Rosenbrock, instances, affine=False),
    #     HarderNumerical(_ndim, device, HarderNumerical.Rastrigin, instances, affine=False),
    #     HarderNumerical(_ndim, device, HarderNumerical.Levy, instances, affine=False),
    # ]
    # dimensions += [_ndim] * 4
    
    # from test_functions.rover_function import get_rover_func
    # from test_functions.mujoco_functions import get_heuristic_landing_func, get_walker_func, get_ant_func

    # for _fn in (get_heuristic_landing_func, get_rover_func, get_walker_func, get_ant_func):
    #     _ff, _ndim = _fn()
    #     functions.append(convert_host_reward_fn(_ff))
    #     dimensions.append(_ndim)

    # End Ablation Study
    
    # For 1000D Functions
    # _ndim = 1000
    # functions += [
    #     HarderNumerical(_ndim, device, HarderNumerical.Ackley, instances, affine=False),
    #     HarderNumerical(_ndim, device, HarderNumerical.Rosenbrock, instances, affine=False),
    #     HarderNumerical(_ndim, device, HarderNumerical.Rastrigin, instances, affine=False),
    #     HarderNumerical(_ndim, device, HarderNumerical.Levy, instances, affine=False),
    # ]
    # dimensions += [_ndim] * 4
    
    # from test_functions.push_function_high_dim import get_push_func_high_dim
    # from test_functions.rover_function_high_dim import get_rover_func_high_dim
    # from test_functions.mujoco_functions_high_dim import get_large_heuristic_landing_func, get_large_walker_func, get_large_ant_func
    # for _fn in (get_large_heuristic_landing_func, get_push_func_high_dim, get_rover_func_high_dim, get_large_walker_func, get_large_ant_func):
    #     _ff, _ndim = _fn()
    #     functions.append(convert_host_reward_fn(_ff))
    #     dimensions.append(_ndim)
    
    
    # from test_functions.rover_function import get_rover_func
    # from test_functions.mujoco_functions import get_heuristic_landing_func, get_walker_func, get_ant_func

    # for _fn in (get_heuristic_landing_func, get_rover_func, get_walker_func, get_ant_func):
    #     _ff, _ndim = _fn()
    #     functions.append(convert_host_reward_fn(_ff))
    #     dimensions.append(_ndim)

    
    # from test_functions.brax_functions import get_brax_pendulum_func, get_brax_hopper_func, get_brax_pusher_func

    # for _fn in (get_brax_pendulum_func, get_brax_hopper_func, get_brax_pusher_func):
    #     _ff, _ndim = _fn()
    #     functions.append(_ff)
    #     dimensions.append(_ndim)
    

    return functions, dimensions
