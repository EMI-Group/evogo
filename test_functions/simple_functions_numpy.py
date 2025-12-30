import numpy as np


class HarderNumerical:
    
    def reinit(self):
        self.__offsets = np.random.uniform(size=(self.dim,),
                                           low=self.lb / self.scaling,
                                           high=self.ub / self.scaling)
        self.__offsets = self.__offsets  # [dim]
        rand_mat = np.random.normal(size=(self.dim, self.dim))
        self.__isometries = np.linalg.eigh(0.5 * (rand_mat + rand_mat.T))[1]  # [dim, dim]

    def __init__(self, dim: int, eval_fn, lb=-10.0, ub=10.0, scaling=2.5, hyper_cube_x=True):
        self.dim = dim
        self.lb = lb
        self.ub = ub
        self.scaling = scaling
        self.reinit()
        self.eval_fn = eval_fn
        self.hyper_cube_x = hyper_cube_x
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x)
        assert x.ndim == 1  # [dim] ~ N(0, 1) or U(0, 1)
        assert x.shape[0] == self.__offsets.shape[0]
        if self.hyper_cube_x:
            x = x * 5 - 2.5
        # self.__isometries @ x ~ N(0, 1)
        w = (self.ub - self.lb) / self.scaling * (self.__isometries @ x) - self.__offsets
        return self.eval_fn(w)
        
    @staticmethod
    def Ackley(w: np.ndarray):
        w *= 2
        val = -20 * np.exp(-0.2 * np.sqrt(np.mean(w**2, axis=-1))) - \
            np.exp(np.mean(np.cos(2 * np.pi * w), axis=-1)) + 20 + np.e
        return val
    
    @staticmethod
    def Rosenbrock(w: np.ndarray):
        w *= 1
        val = np.sum(100 * (w[1:] + 1 - (w[:-1] + 1)**2)**2 + w[:-1]**2, axis=-1)
        return val
    
    @staticmethod
    def Rastrigin(w: np.ndarray):
        w *= 3
        val = 10 * w.shape[0] + np.sum(w**2 - 10 * np.cos(2 * np.pi * w), axis=-1)
        return val
    
    @staticmethod
    def Levy(w: np.ndarray):
        w *= 1
        val = np.sin(np.pi * (w[0] + 1))**2 + \
                w[-1]**2 * (1 + np.sin(2 * np.pi * (w[-1] + 1))**2) + \
                np.sum(w[:-1]**2 * (1 + 10 * np.sin(np.pi * (w[:-1] + 1) + 1)**2), axis=-1)
        return val
    

class Ackley:
    
    def __init__(self, dim: int, offseted: bool = True):
        self.dim = dim
        # self.lb = -20 * np.ones(dim)
        # self.ub = 20 * np.ones(dim)
        self.__offsets = np.random.uniform(size=(dim, ), low=-5, high=5) if offseted else np.zeros((dim, ))
        print(f"[DEBUG] Ackley min = {self.__call__((self.__offsets + 20) / 40)}")
    
    def reinit(self, offseted: bool = True):
        self.__offsets = np.random.uniform(size=(self.dim, ), low=-5, high=5) if offseted else np.zeros((self.dim, ))
    
    def __call__(self, x: np.ndarray):
        x = np.asarray(x)
        # assert np.all(x <= self.ub) and np.all(x >= self.lb)
        w = 40 * x - 20 - self.__offsets
        val = -20 * np.exp(-0.2 * np.sqrt(np.mean(w**2, axis=-1))) - np.exp(np.mean(np.cos(2 * np.pi * w),
                                                                                    axis=-1)) + 20 + np.e
        return val


class Rosenbrock:
    
    def __init__(self, dim: int, offseted: bool = True):
        self.dim = dim
        # self.lb = -20 * np.ones(dim)
        # self.ub = 20 * np.ones(dim)
        self.__offsets = np.random.uniform(size=(dim, ), low=-2.5, high=2.5) if offseted else np.zeros((dim, ))
        print(f"[DEBUG] Rosenbrock min = {self.__call__((self.__offsets + 10) / 20)}")
    
    def reinit(self, offseted: bool = True):
        self.__offsets = np.random.uniform(size=(self.dim, ), low=-2.5, high=2.5) if offseted else np.zeros((self.dim, ))
    
    def __call__(self, x: np.ndarray):
        x = np.asarray(x)
        # assert np.all(x <= self.ub) and np.all(x >= self.lb)
        w = 20 * x - 10 - self.__offsets + 1
        val = np.sum(100 * (w[1:] - w[:-1]**2)**2 + (w[:-1] - 1)**2, axis=-1)
        return val


class Rastrigin:
    
    def __init__(self, dim: int, offseted: bool = True):
        self.dim = dim
        # self.lb = -20 * np.ones(dim)
        # self.ub = 20 * np.ones(dim)
        self.__offsets = np.random.uniform(size=(dim, ), low=-8, high=8) if offseted else np.zeros((dim, ))
        print(f"[DEBUG] Rastrigin min = {self.__call__((self.__offsets + 32) / 64)}")
    
    def reinit(self, offseted: bool = True):
        self.__offsets = np.random.uniform(size=(self.dim, ), low=-8, high=8) if offseted else np.zeros((self.dim, ))
    
    def __call__(self, x):
        x = np.asarray(x)
        # assert np.all(x <= self.ub) and np.all(x >= self.lb)
        w = 64 * x - 32 - self.__offsets
        val = 10 * self.dim + np.sum(w**2 - 10 * np.cos(2 * np.pi * w), axis=-1)
        return val


class Levy:
    
    def __init__(self, dim: int, offseted: bool = True):
        self.dim = dim
        # self.lb = -20 * np.ones(dim)
        # self.ub = 20 * np.ones(dim)
        self.__offsets = np.random.uniform(size=(dim, ), low=-2.5, high=2.5) if offseted else np.zeros((dim, ))
        print(f"[DEBUG] Levy min = {self.__call__((self.__offsets + 10) / 20)}")
    
    def reinit(self, offseted: bool = True):
        self.__offsets = np.random.uniform(size=(self.dim, ), low=-2.5, high=2.5) if offseted else np.zeros((self.dim, ))
    
    def __call__(self, x):
        x = np.asarray(x)
        w = 1 + (20 * x - 10 - self.__offsets) / 4
        val = np.sin(np.pi * w[0])**2 + \
                (w[-1] - 1)**2 * (1 + np.sin(2 * np.pi * w[-1])**2) + \
                np.sum((w[:-1] - 1)**2 * (1 + 10 * np.sin(np.pi * w[:-1] + 1)**2), axis=-1)
        return val
