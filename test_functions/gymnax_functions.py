import jax
from flax import linen as nn
import brax.envs
from evox.utils.common import TreeAndVector
from evox.problems.neuroevolution.reinforcement_learning.brax import Brax
import gymnax

from typing import Any, Dict, Optional, Union

import chex
from flax import struct
import jax
from gymnax.environments import environment
from brax import envs


@struct.dataclass
class State:  # Lookalike for brax.envs.env.State.
    qp: environment.EnvState  # Brax QP is roughly equivalent to our EnvState
    obs: Any  # depends on environment
    reward: float
    done: bool
    metrics: Dict[str, Union[chex.Array, chex.Scalar]] = struct.field(default_factory=dict)
    info: Dict[str, Any] = struct.field(default_factory=dict)


class GymnaxToBraxWrapper(envs.Env):
    """Wrap Gymnax environment as Brax environment.

    Primarily useful for including obs, reward, and done as part of state.
    Compatible with all brax wrappers, but AutoResetWrapper is redundant
    since Gymnax environments already reset state.
    """
    
    def __init__(self, env: environment.Environment):
        super().__init__()
        self.env = env
    
    def reset(self, rng: chex.PRNGKey, params: Optional[environment.EnvParams] = None):  # -> State:
        """Reset, return brax State. Save rng and params in info field for step."""
        if params is None:
            params = self.env.default_params
        obs, env_state = self.env.reset(rng, params)
        return State(
            env_state,
            obs,
            0.0,
            False,
            {},
            {
                "_rng": jax.random.split(rng)[0],
                "_env_params": params
            },
        )
    
    def step(
            self,
            state,  #: State,
            action,  #: Union[chex.Scalar, chex.Array],
            params=None,  #: Optional[environment.EnvParams] = None,
    ):  # -> State:
        """Step brax State. Update stored rng and params in info field."""
        rng, step_rng = jax.random.split(state.info["_rng"])
        if params is None:
            params = self.env.default_params
        state.info.update(_rng=rng, _env_params=params)
        o, env_state, r, d, _ = self.env.step(step_rng, state.qp, action, params)
        r = r if r.ndim < 1 else r.flatten()[0]
        return state.replace(qp=env_state, obs=o, reward=r.astype(jax.numpy.float32), done=d)
    
    def action_size(self) -> int:
        """DEFAULT size of action vector expected by step."""
        a_space = self.env.action_space(self.env.default_params)
        example_a = a_space.sample(jax.random.PRNGKey(0))
        return len(jax.tree_util.tree_flatten(example_a)[0])
    
    def observation_size(self) -> int:
        """DEFAULT size of observation vector expected by step."""
        o_space = self.env.observation_space(self.env.default_params)
        example_o = o_space.sample(jax.random.PRNGKey(0))
        return len(jax.tree_util.tree_flatten(example_o)[0])
    
    def backend(self) -> str:
        """Return backend of the environment."""
        return "jax"


def get_gymnax_moutaincar_func(key: jax.Array = jax.random.PRNGKey(0), cap_episode: int = 1000, repeats: int = 100):
    
    class Controller(nn.Module):
        
        @nn.compact
        def __call__(self, obs: jax.Array) -> jax.Array:
            hidden = nn.Dense(2)(obs)
            hidden = nn.tanh(hidden)
            output = nn.Dense(3)(hidden)
            output = jax.numpy.argmax(output)
            return output
    
    gymnax_env = gymnax.make("MountainCar-v0")[0]
    brax_env = lambda **_: GymnaxToBraxWrapper(gymnax_env)
    brax.envs.register_environment("mountain_car", brax_env)
    
    N_OBS = 2
    controller = Controller()
    params = controller.init(key, jax.numpy.zeros((N_OBS, )))
    tree_vec = TreeAndVector(params)
    env = Brax(controller.apply, "mountain_car", cap_episode, repeats)
    state = env.setup(key)
    MAX_RANGE = 5
    
    def f(_x):
        ws = jax.numpy.array(_x)
        ws = ws * MAX_RANGE * 2 - MAX_RANGE
        if ws.ndim > 2:
            weights = ws.reshape(-1, ws.shape[-1])
        else:
            weights = ws
        weights = tree_vec.batched_to_tree(weights)
        reward, _ = env.evaluate(state, weights)
        # state = state.update(key=jax.random.split(key, num=1))
        reward = -jax.numpy.nan_to_num(reward.reshape(ws.shape[:-1]), nan=jax.numpy.nanmin(reward))
        return reward
    
    return f, tree_vec.to_vector(params).shape[0]


def get_gymnax_pointrobot_func(key: jax.Array = jax.random.PRNGKey(0), cap_episode: int = 1000, repeats: int = 100):
    
    class Controller(nn.Module):
        
        @nn.compact
        def __call__(self, obs: jax.Array) -> jax.Array:
            hidden = nn.Dense(2)(obs)
            hidden = nn.tanh(hidden)
            output = nn.Dense(2)(hidden)
            output = 0.1 * nn.tanh(output)
            return output
    
    gymnax_env = gymnax.make("PointRobot-misc")[0]
    brax_env = lambda **_: GymnaxToBraxWrapper(gymnax_env)
    brax.envs.register_environment("point_robot", brax_env)
    
    N_OBS = 6
    controller = Controller()
    params = controller.init(key, jax.numpy.zeros((N_OBS, )))
    tree_vec = TreeAndVector(params)
    env = Brax(controller.apply, "point_robot", cap_episode, repeats)
    state = env.setup(key)
    MAX_RANGE = 5
    
    def f(_x):
        ws = jax.numpy.array(_x)
        ws = ws * MAX_RANGE * 2 - MAX_RANGE
        if ws.ndim > 2:
            weights = ws.reshape(-1, ws.shape[-1])
        else:
            weights = ws
        weights = tree_vec.batched_to_tree(weights)
        reward, _ = env.evaluate(state, weights)
        # state = state.update(key=jax.random.split(key, num=1))
        reward = -jax.numpy.nan_to_num(reward.reshape(ws.shape[:-1]), nan=jax.numpy.nanmin(reward))
        return reward
    
    return f, tree_vec.to_vector(params).shape[0]


def get_gymnax_acrobot_func(key: jax.Array = jax.random.PRNGKey(0), cap_episode: int = 1000, repeats: int = 100):
    
    class Controller(nn.Module):
        
        @nn.compact
        def __call__(self, obs: jax.Array) -> jax.Array:
            hidden = nn.Dense(2)(obs)
            hidden = nn.tanh(hidden)
            output = nn.Dense(3)(hidden)
            output = jax.numpy.argmax(output)
            return output
    
    gymnax_env = gymnax.make("Acrobot-v1")[0]
    brax_env = lambda **_: GymnaxToBraxWrapper(gymnax_env)
    brax.envs.register_environment("acrobot", brax_env)
    
    N_OBS = 6
    controller = Controller()
    params = controller.init(key, jax.numpy.zeros((N_OBS, )))
    tree_vec = TreeAndVector(params)
    env = Brax(controller.apply, "acrobot", cap_episode, repeats)
    state = env.setup(key)
    MAX_RANGE = 5
    
    def f(_x):
        ws = jax.numpy.array(_x)
        ws = ws * MAX_RANGE * 2 - MAX_RANGE
        if ws.ndim > 2:
            weights = ws.reshape(-1, ws.shape[-1])
        else:
            weights = ws
        weights = tree_vec.batched_to_tree(weights)
        reward, _ = env.evaluate(state, weights)
        # state = state.update(key=jax.random.split(key, num=1))
        reward = -jax.numpy.nan_to_num(reward.reshape(ws.shape[:-1]), nan=jax.numpy.nanmin(reward))
        return reward
    
    return f, tree_vec.to_vector(params).shape[0]


def get_gymnax_bandit_func(key: jax.Array = jax.random.PRNGKey(0), cap_episode: int = 100, repeats: int = 100):
    
    class Controller(nn.Module):
        
        @nn.compact
        def __call__(self, obs: jax.Array) -> jax.Array:
            hidden = nn.Dense(2)(obs)
            hidden = nn.tanh(hidden)
            output = nn.Dense(2)(hidden)
            output = jax.numpy.argmax(output)
            return output
    
    gymnax_env = gymnax.make("GaussianBandit-misc")[0]
    brax_env = lambda **_: GymnaxToBraxWrapper(gymnax_env)
    brax.envs.register_environment("bandit", brax_env)
    
    N_OBS = 4
    controller = Controller()
    params = controller.init(key, jax.numpy.zeros((N_OBS, )))
    tree_vec = TreeAndVector(params)
    env = Brax(controller.apply, "bandit", cap_episode, repeats)
    state = env.setup(key)
    MAX_RANGE = 5
    
    def f(_x):
        ws = jax.numpy.array(_x)
        ws = ws * MAX_RANGE * 2 - MAX_RANGE
        if ws.ndim > 2:
            weights = ws.reshape(-1, ws.shape[-1])
        else:
            weights = ws
        weights = tree_vec.batched_to_tree(weights)
        reward, _ = env.evaluate(state, weights)
        # state = state.update(key=jax.random.split(key, num=1))
        reward = -jax.numpy.nan_to_num(reward.reshape(ws.shape[:-1]), nan=jax.numpy.nanmin(reward))
        return reward
    
    return f, tree_vec.to_vector(params).shape[0]


def get_gymnax_bandit2_func(key: jax.Array = jax.random.PRNGKey(0), cap_episode: int = 100, repeats: int = 100):
    
    class Controller(nn.Module):
        
        @nn.compact
        def __call__(self, obs: jax.Array) -> jax.Array:
            hidden = nn.Dense(2)(obs)
            hidden = nn.tanh(hidden)
            output = nn.Dense(2)(hidden)
            output = jax.numpy.argmax(output)
            return output
    
    gymnax_env = gymnax.make("BernoulliBandit-misc")[0]
    brax_env = lambda **_: GymnaxToBraxWrapper(gymnax_env)
    brax.envs.register_environment("bandit2", brax_env)
    
    N_OBS = 4
    controller = Controller()
    params = controller.init(key, jax.numpy.zeros((N_OBS, )))
    tree_vec = TreeAndVector(params)
    env = Brax(controller.apply, "bandit2", cap_episode, repeats)
    state = env.setup(key)
    MAX_RANGE = 5
    
    def f(_x):
        ws = jax.numpy.array(_x)
        ws = ws * MAX_RANGE * 2 - MAX_RANGE
        if ws.ndim > 2:
            weights = ws.reshape(-1, ws.shape[-1])
        else:
            weights = ws
        weights = tree_vec.batched_to_tree(weights)
        reward, _ = env.evaluate(state, weights)
        # state = state.update(key=jax.random.split(key, num=1))
        reward = -jax.numpy.nan_to_num(reward.reshape(ws.shape[:-1]), nan=jax.numpy.nanmin(reward))
        return reward
    
    return f, tree_vec.to_vector(params).shape[0]
