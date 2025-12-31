import numpy as np


class HarderNumerical:
    
    def reinit(self):
        self.__offsets = np.random.uniform(size=(self.instances, self.dim),
                                           low=self.lb / self.scaling,
                                           high=self.ub / self.scaling)
        rand_mat = np.random.normal(size=(self.instances, self.dim, self.dim))
        # Use eigh for each instance to get isometries
        self.__isometries = np.stack([np.linalg.eigh(0.5 * (m + m.T))[1] for m in rand_mat])
        self.range = (self.ub - self.lb) / self.scaling
        if not self.affine:
            self.__isometries = np.array([np.eye(self.dim)] * self.instances)
        
        # Calculate optimal: A.T @ b for each instance
        self.__optimal = np.array([A.T @ b for A, b in zip(self.__isometries, self.__offsets)]) / self.range
        if self.hyper_cube_x:
            self.__optimal = (self.__optimal + 2.5) / 5

    def __init__(self,
                 dim: int,
                 eval_fn,
                 instances: int = 1,
                 lb: float = -10.0,
                 ub: float = 10.0,
                 scaling: float = 2.5,
                 hyper_cube_x: bool = True,
                 affine: bool = True):
        self.dim = dim
        self.lb = lb
        self.ub = ub
        self.scaling = scaling
        self.instances = instances
        self.eval_fns = [eval_fn] if callable(eval_fn) else eval_fn
        self.hyper_cube_x = hyper_cube_x
        self.affine = affine
        self.reinit()
        print(f"[DEBUG] Min f(x) = {self(self.__optimal)}")
    
    def get_optimal(self):
        return self.__optimal
    
    def get_transform(self):
        for o, i in zip(self.__offsets, self.__isometries):
            yield o, i
    
    def __call__(self, x: np.ndarray, offsets=None, isometries=None, fn_idx=0) -> np.ndarray:
        x = np.asarray(x)
        if x.ndim == 3:
            return self.get_batch_eval()(x)
        
        if offsets is None:
            offsets = self.__offsets
        if isometries is None:
            isometries = self.__isometries
            
        assert x.ndim == 2  # [instances, dim]
        assert x.shape[0] == offsets.shape[0] and x.shape[-1] == offsets.shape[1]
        
        if self.hyper_cube_x:
            x = x * 5 - 2.5
            
        # w = range * (isometries @ x) - offsets
        # x is [instances, dim], isometries is [instances, dim, dim]
        # We need to compute dot product for each instance
        w = self.range * np.einsum('ijk,ik->ij', isometries, x) - offsets
        return np.array([self.eval_fns[fn_idx](wi) for wi in w])
    
    def get_batch_eval(self, axis=1):
        def _eval_fn(x: np.ndarray) -> np.ndarray:
            num_fns = len(self.eval_fns)
            bs = self.__offsets.shape[0] // num_fns
            xs = np.split(x, num_fns, axis=0)
            results = []
            for i, xi in enumerate(xs):
                # xi is [bs, pop, dim]
                # We need to apply __call__ for each pop member across instances
                # This is a bit complex in NumPy without vmap, but we can use a loop or einsum
                off = self.__offsets[bs * i:bs * (i + 1)]
                iso = self.__isometries[bs * i:bs * (i + 1)]
                
                # Process each population member
                pop_res = []
                for j in range(xi.shape[1]):
                    pop_res.append(self(xi[:, j, :], offsets=off, isometries=iso, fn_idx=i))
                results.append(np.stack(pop_res, axis=1))
            return np.concatenate(results, axis=0)
        return _eval_fn
        
    @staticmethod
    def Ackley(w: np.ndarray):
        w = w * 2
        val = -20 * np.exp(-0.2 * np.sqrt(np.mean(w**2, axis=-1))) - \
            np.exp(np.mean(np.cos(2 * np.pi * w), axis=-1)) + 20 + np.e
        return val
    
    @staticmethod
    def Rosenbrock(w: np.ndarray):
        w = w * 1
        val = np.sum(100 * (w[..., 1:] + 1 - (w[..., :-1] + 1)**2)**2 + w[..., :-1]**2, axis=-1)
        return val
    
    @staticmethod
    def Rastrigin(w: np.ndarray):
        w = w * 3
        val = 10 * w.shape[-1] + np.sum(w**2 - 10 * np.cos(2 * np.pi * w), axis=-1)
        return val
    
    @staticmethod
    def Levy(w: np.ndarray):
        w = w * 1
        val = np.sin(np.pi * (w[..., 0] + 1))**2 + \
                w[..., -1]**2 * (1 + np.sin(2 * np.pi * (w[..., -1] + 1))**2) + \
                np.sum(w[..., :-1]**2 * (1 + 10 * np.sin(np.pi * (w[..., :-1] + 1) + 1)**2), axis=-1)
        return val

    @staticmethod
    def Griewank(w: np.ndarray):
        w = w * 50
        val = 1 / 4000 * np.sum(w**2, axis=-1) - np.prod(np.cos(w / np.sqrt(np.arange(1, w.shape[-1] + 1))), axis=-1) + 1
        return val
    
    @staticmethod
    def Schwefel(w: np.ndarray):
        w = w * 50 + 420.9687  # [-500, 500]
        val = 418.9829 * w.shape[-1] - np.sum(w * np.sin(np.sqrt(np.abs(w))), axis=-1)
        return val
    
    @staticmethod
    def HolderTable(w: np.ndarray):
        w = w * 1
        val = -np.abs(np.sin(w[..., 0]) * np.cos(w[..., 1]) * np.exp(np.abs(1 - np.linalg.norm(w, axis=-1) / np.pi)))
        return val


class Ackley:
    
    def __init__(self, dim: int, parallels: int = 1, offseted: bool = True):
        self.dim = dim
        self.__offsets = np.random.uniform(size=(parallels, dim), low=-5, high=5) if offseted else np.zeros((parallels, dim))
        self.__offsets = self.__offsets[:, np.newaxis, :]
        print(f"[DEBUG] Ackley min = {self.__call__((self.__offsets + 20) / 40)}")
    
    def reinit(self, offseted: bool = True):
        parallels = self.__offsets.shape[0]
        self.__offsets = np.random.uniform(size=(parallels, self.dim), low=-5, high=5) if offseted else np.zeros((parallels, self.dim))
        self.__offsets = self.__offsets[:, np.newaxis, :]
    
    def __call__(self, x: np.ndarray):
        x = np.asarray(x)
        assert x.ndim == 3
        w = 40 * x - 20 - self.__offsets
        val = -20 * np.exp(-0.2 * np.sqrt(np.mean(w**2, axis=-1))) - np.exp(np.mean(np.cos(2 * np.pi * w),
                                                                                    axis=-1)) + 20 + np.e
        return val


class Rosenbrock:
    
    def __init__(self, dim: int, parallels: int = 1, offseted: bool = True):
        self.dim = dim
        self.__offsets = np.random.uniform(size=(parallels, dim), low=-2.5, high=2.5) if offseted else np.zeros((parallels, dim))
        self.__offsets = self.__offsets[:, np.newaxis, :]
        print(f"[DEBUG] Rosenbrock min = {self.__call__((self.__offsets + 10) / 20)}")
    
    def reinit(self, offseted: bool = True):
        parallels = self.__offsets.shape[0]
        self.__offsets = np.random.uniform(size=(parallels, self.dim), low=-2.5, high=2.5) if offseted else np.zeros((parallels, self.dim))
        self.__offsets = self.__offsets[:, np.newaxis, :]
    
    def __call__(self, x: np.ndarray):
        x = np.asarray(x)
        assert x.ndim == 3
        w = 20 * x - 10 - self.__offsets + 1
        val = np.sum(100 * (w[:, :, 1:] - w[:, :, :-1]**2)**2 + (w[:, :, :-1] - 1)**2, axis=-1)
        return val


class Rastrigin:
    
    def __init__(self, dim: int, parallels: int = 1, offseted: bool = True):
        self.dim = dim
        self.__offsets = np.random.uniform(size=(parallels, dim), low=-8, high=8) if offseted else np.zeros((parallels, dim))
        self.__offsets = self.__offsets[:, np.newaxis, :]
        print(f"[DEBUG] Rastrigin min = {self.__call__((self.__offsets + 32) / 64)}")
    
    def reinit(self, offseted: bool = True):
        parallels = self.__offsets.shape[0]
        self.__offsets = np.random.uniform(size=(parallels, self.dim), low=-8, high=8) if offseted else np.zeros((parallels, self.dim))
        self.__offsets = self.__offsets[:, np.newaxis, :]
    
    def __call__(self, x):
        x = np.asarray(x)
        assert x.ndim == 3
        w = 64 * x - 32 - self.__offsets
        val = 10 * self.dim + np.sum(w**2 - 10 * np.cos(2 * np.pi * w), axis=-1)
        return val


class Levy:
    
    def __init__(self, dim: int, parallels: int = 1, offseted: bool = True):
        self.dim = dim
        self.__offsets = np.random.uniform(size=(parallels, dim), low=-2.5, high=2.5) if offseted else np.zeros((parallels, dim))
        self.__offsets = self.__offsets[:, np.newaxis, :]
        print(f"[DEBUG] Levy min = {self.__call__((self.__offsets + 10) / 20)}")
    
    def reinit(self, offseted: bool = True):
        parallels = self.__offsets.shape[0]
        self.__offsets = np.random.uniform(size=(parallels, self.dim), low=-2.5, high=2.5) if offseted else np.zeros((parallels, self.dim))
        self.__offsets = self.__offsets[:, np.newaxis, :]
    
    def __call__(self, x):
        x = np.asarray(x)
        assert x.ndim == 3
        w = 1 + (20 * x - 10 - self.__offsets) / 4
        val = np.sin(np.pi * w[:, :, 0])**2 + \
                (w[:, :, -1] - 1)**2 * (1 + np.sin(2 * np.pi * w[:, :, -1])**2) + \
                np.sum((w[:, :, :-1] - 1)**2 * (1 + 10 * np.sin(np.pi * w[:, :, :-1] + 1)**2), axis=-1)
        return val
