import jax
from flax import linen as nn
from evox.utils.common import TreeAndVector
from evox.problems.neuroevolution.reinforcement_learning.brax import Brax


def get_brax_pendulum_func(key: jax.Array = jax.random.PRNGKey(0), cap_episode: int = 1000, repeats: int = 10):
    class Controller(nn.Module):
        @nn.compact
        def __call__(self, obs: jax.Array) -> jax.Array:
            hidden = nn.Dense(2)(obs)
            hidden = nn.tanh(hidden)
            output = nn.Dense(1)(hidden)
            output = nn.tanh(output)
            return output
    
    N_OBS = 4
    controller = Controller()
    params = controller.init(key, jax.numpy.zeros((N_OBS,)))
    tree_vec = TreeAndVector(params)
    env = Brax(controller.apply, "inverted_pendulum", cap_episode, repeats)
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


def get_brax_hopper_func(key: jax.Array = jax.random.PRNGKey(0), cap_episode: int = 10000, repeats: int = 10):
    class Controller(nn.Module):
        @nn.compact
        def __call__(self, obs: jax.Array) -> jax.Array:
            hidden = nn.Dense(13)(obs)
            hidden = nn.tanh(hidden)
            output = nn.Dense(3)(hidden)
            output = nn.tanh(output)
            return output
    
    N_OBS = 11
    controller = Controller()
    params = controller.init(key, jax.numpy.zeros((N_OBS,)))
    tree_vec = TreeAndVector(params)
    env = Brax(controller.apply, "hopper", cap_episode, repeats)
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


def get_brax_pusher_func(key: jax.Array = jax.random.PRNGKey(0), cap_episode: int = 10000, repeats: int = 10):
    class Controller(nn.Module):
        @nn.compact
        def __call__(self, obs: jax.Array) -> jax.Array:
            hidden = nn.Dense(11)(obs)
            hidden = nn.tanh(hidden)
            output = nn.Dense(7)(hidden)
            output = 2 * nn.tanh(output) # action range = [-2, 2]
            return output
    
    N_OBS = 23
    controller = Controller()
    params = controller.init(key, jax.numpy.zeros((N_OBS,)))
    tree_vec = TreeAndVector(params)
    env = Brax(controller.apply, "pusher", cap_episode, repeats)
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

