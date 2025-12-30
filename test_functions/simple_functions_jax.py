from functools import partial

import jax
import jax.numpy as jnp
import jax.numpy.linalg
import numpy as np


class HarderNumerical:
    
    def __init__(self,
                 dim: int,
                 key: jax.Array,
                 eval_fn,
                 instances: int = 1,
                 lb: float = -10.0,
                 ub: float = 10.0,
                 scaling: float = 2.5,
                 hyper_cube_x: bool = True,
                 affine: bool = True):
        self.dim = dim
        self.scaling = scaling
        self.lb = lb
        self.ub = ub
        self.range = (ub - lb) / scaling
        self.eval_fns = [eval_fn] if callable(eval_fn) else eval_fn
        self.hyper_cube_x = hyper_cube_x
        self.affine = affine
        self.instances = instances
        self.reinit(key)
        print(f"[DEBUG] Min f(x) = {self(self.__optimal)}")
    
    def reinit(self, key: jax.Array):
        self.__offsets = jax.random.uniform(key=key, shape=(self.instances, self.dim), minval=self.lb / self.scaling, maxval=self.ub / self.scaling)
        self.__offsets = self.__offsets  # [instances, dim]
        rand_mat = jax.random.normal(key, shape=(self.instances, self.dim, self.dim))
        self.__isometries = jnp.linalg.eigh(rand_mat)[1]  # [instances, dim, dim] # type: ignore
        self.range = (self.ub - self.lb) / self.scaling
        if not self.affine:
            # self.__offsets = jnp.zeros((instances, dim))
            self.__isometries = jnp.array([jnp.eye(self.dim)] * self.instances)
        self.__optimal = jax.vmap(lambda A, b: A.T.dot(b))(self.__isometries, self.__offsets) / self.range
        if self.hyper_cube_x:
            self.__optimal = (self.__optimal + 2.5) / 5
    
    def get_optimal(self):
        return self.__optimal
    
    def get_transform(self):
        for o, i in zip(self.__offsets, self.__isometries):
            yield np.asarray(jax.device_get(o), dtype=np.float32), np.asarray(jax.device_get(i), dtype=np.float32)
    
    @partial(jax.jit, static_argnums=[0, 4])
    def __call__(self, x: jax.Array, offsets=jnp.zeros(0), isometries=jnp.zeros(0), fn_idx=0) -> jax.Array:
        if x.ndim == 3:
            return self.get_batch_eval()(x)
        if offsets.ndim == 1:
            offsets = self.__offsets
        if isometries.ndim == 1:
            isometries = self.__isometries
        assert x.ndim == 2  # [instances, dim] ~ N(0, 1) or U(0, 1)
        assert x.shape[0] == offsets.shape[0] and x.shape[-1] == offsets.shape[1]
        if self.hyper_cube_x:
            x = x * 5 - 2.5
        # self.__isometries @ x ~ N(0, 1)
        w = self.range * jax.vmap(jnp.dot)(isometries, x) - offsets
        return jax.vmap(self.eval_fns[fn_idx])(w)
    
    def get_batch_eval(self, axis=1):
        
        def _eval_fn(x: jax.Array) -> jax.Array:
            bs = self.__offsets.shape[0] // len(self.eval_fns)
            xs = jnp.split(x, len(self.eval_fns), axis=0)
            for i, xi in enumerate(xs):
                xs[i] = jax.vmap(partial(self.__call__, fn_idx=i), in_axes=(axis, None, None), out_axes=axis) \
                                (xi, self.__offsets[bs * i:bs * (i + 1)], self.__isometries[bs * i:bs * (i + 1)])
            return jnp.concatenate(xs, axis=0)
        
        return _eval_fn
    
    @staticmethod
    def Ackley(w: jax.Array):
        w *= 2
        val = -20 * jnp.exp(-0.2 * jnp.sqrt(jnp.mean(w**2, axis=-1))) - \
            jnp.exp(jnp.mean(jnp.cos(2 * jnp.pi * w), axis=-1)) + 20 + jnp.e
        return val
    
    @staticmethod
    def Rosenbrock(w: jax.Array):
        w *= 1
        val = jnp.sum(100 * (w[1:] + 1 - (w[:-1] + 1)**2)**2 + w[:-1]**2, axis=-1)
        return val
    
    @staticmethod
    def Rastrigin(w: jax.Array):
        w *= 3
        val = 10 * w.shape[0] + jnp.sum(w**2 - 10 * jnp.cos(2 * jnp.pi * w), axis=-1)
        return val
    
    @staticmethod
    def Levy(w: jax.Array):
        w *= 1
        val = jnp.sin(jnp.pi * (w[0] + 1))**2 + \
                w[-1]**2 * (1 + jnp.sin(2 * jnp.pi * (w[-1] + 1))**2) + \
                jnp.sum(w[:-1]**2 * (1 + 10 * jnp.sin(jnp.pi * (w[:-1] + 1) + 1)**2), axis=-1)
        return val
    
    @staticmethod
    def Griewank(w: jax.Array):
        w *= 50
        val = 1 / 4000 * jnp.sum(w**2) - jnp.prod(jnp.cos(w / jnp.sqrt(jnp.arange(1, w.shape[0] + 1)))) + 1
        return val
    
    @staticmethod
    def Schwefel(w: jax.Array):
        w = w * 50 + 420.9687  # [-500, 500]
        val = 418.9829 * w.shape[0] - jnp.sum(w * jnp.sin(jnp.sqrt(jnp.abs(w))))
        return val
    
    @staticmethod
    def HolderTable(w: jax.Array):
        w = w * 1
        val = -jnp.abs(jnp.sin(w[0]) * jnp.cos(w[1]) * jnp.exp(jnp.abs(1 - jax.numpy.linalg.norm(w) / jnp.pi)))
        return val


class Ackley:
    
    def __init__(self, dim: int, key: jax.Array, parallels: int = 1, offseted: bool = True):
        self.dim = dim
        # self.lb = -20 * np.ones(dim)
        # self.ub = 20 * np.ones(dim)
        self.__offsets = jax.random.uniform(key=key, shape=(parallels, dim), minval=-5, maxval=5) if offseted else jnp.zeros(
            (parallels, dim))
        self.__offsets = self.__offsets[:, jnp.newaxis, :]
        print(f"[DEBUG] Ackley min = {jnp.ravel(self.__call__((self.__offsets + 20) / 40))}")
    
    @partial(jax.jit, static_argnums=[0])
    def __call__(self, x: jax.Array):
        assert x.ndim == 3
        assert x.shape[0] == self.__offsets.shape[0] and x.shape[2] == self.__offsets.shape[2]
        # assert np.all(x <= self.ub) and np.all(x >= self.lb)
        w = 40 * x - 20 - self.__offsets
        val = -20 * jnp.exp(-0.2 * jnp.sqrt(jnp.mean(w**2, axis=-1))) - jnp.exp(jnp.mean(jnp.cos(2 * jnp.pi * w),
                                                                                         axis=-1)) + 20 + jnp.e
        return val


class Rosenbrock:
    
    def __init__(self, dim: int, key: jax.Array, parallels: int = 1, offseted: bool = True):
        self.dim = dim
        # self.lb = -20 * np.ones(dim)
        # self.ub = 20 * np.ones(dim)
        self.__offsets = jax.random.uniform(key=key, shape=(parallels,
                                                            dim), minval=-2.5, maxval=2.5) if offseted else jnp.zeros(
                                                                (parallels, dim))
        self.__offsets = self.__offsets[:, jnp.newaxis, :]
        print(f"[DEBUG] Rosenbrock min = {self.__call__((self.__offsets + 10) / 20)}")
    
    @partial(jax.jit, static_argnums=[0])
    def __call__(self, x):
        assert x.ndim == 3
        assert x.shape[0] == self.__offsets.shape[0] and x.shape[2] == self.__offsets.shape[2]
        # assert np.all(x <= self.ub) and np.all(x >= self.lb)
        w = 20 * x - 10 - self.__offsets + 1
        val = jnp.sum(100 * (w[:, :, 1:] - w[:, :, :-1]**2)**2 + (w[:, :, :-1] - 1)**2, axis=-1)
        return val


class Rastrigin:
    
    def __init__(self, dim: int, key: jax.Array, parallels: int = 1, offseted: bool = True):
        self.dim = dim
        # self.lb = -20 * np.ones(dim)
        # self.ub = 20 * np.ones(dim)
        self.__offsets = jax.random.uniform(key=key, shape=(parallels, dim), minval=-8, maxval=8) if offseted else jnp.zeros(
            (parallels, dim))
        self.__offsets = self.__offsets[:, jnp.newaxis, :]
        print(f"[DEBUG] Rastrigin min = {self.__call__((self.__offsets + 32) / 64)}")
    
    @partial(jax.jit, static_argnums=[0])
    def __call__(self, x):
        assert x.ndim == 3
        assert x.shape[0] == self.__offsets.shape[0] and x.shape[2] == self.__offsets.shape[2]
        # assert np.all(x <= self.ub) and np.all(x >= self.lb)
        w = 64 * x - 32 - self.__offsets
        val = 10 * self.dim + jnp.sum(w**2 - 10 * jnp.cos(2 * jnp.pi * w), axis=-1)
        return val


class Levy:
    
    def __init__(self, dim: int, key: jax.Array, parallels: int = 1, offseted: bool = True):
        self.dim = dim
        # self.lb = -20 * np.ones(dim)
        # self.ub = 20 * np.ones(dim)
        self.__offsets = jax.random.uniform(key=key, shape=(parallels,
                                                            dim), minval=-2.5, maxval=2.5) if offseted else jnp.zeros(
                                                                (parallels, dim))
        self.__offsets = self.__offsets[:, jnp.newaxis, :]
        print(f"[DEBUG] Levy min = {self.__call__((self.__offsets + 10) / 20)}")
    
    @partial(jax.jit, static_argnums=[0])
    def __call__(self, x):
        assert x.ndim == 3
        assert x.shape[0] == self.__offsets.shape[0] and x.shape[2] == self.__offsets.shape[2]
        w = 1 + (20 * x - 10 - self.__offsets) / 4
        val = jnp.sin(jnp.pi * w[:, :, 0])**2 + \
                (w[:, :, -1] - 1)**2 * (1 + jnp.sin(2 * jnp.pi * w[:, :, -1])**2) + \
                jnp.sum((w[:, :, :-1] - 1)**2 * (1 + 10 * jnp.sin(jnp.pi * w[:, :, :-1] + 1)**2), axis=-1)
        return val
