from typing import Callable

import torch


class HarderNumerical(torch.nn.Module):
    def __init__(
        self,
        dim: int,
        device: torch.device,
        eval_fn: Callable,
        instances: int = 1,
        lb: float = -10.0,
        ub: float = 10.0,
        scaling: float = 2.5,
        hyper_cube_x: bool = True,
        affine: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.scaling = scaling
        self.lb = lb
        self.ub = ub
        self.range = (ub - lb) / scaling
        self.eval_fn = eval_fn
        self.hyper_cube_x = hyper_cube_x
        self.affine = affine
        self.instances = instances
        self.re_init(device)
        # print(f"[DEBUG] {self.eval_fn.__name__} with min f({self.__optimal}) = {self.forward(self.__optimal)}")

    def re_init(self, device: torch.device):
        self.range = (self.ub - self.lb) / self.scaling
        self.__offsets = (
            torch.rand(self.instances, self.dim, device=device) * self.range + self.lb / self.scaling
        ) # [instances, dim]
        rand_mat = torch.randn(self.instances, self.dim, self.dim, device=device)
        self.__isometries = torch.linalg.eigh(rand_mat)[1]  # [instances, dim, dim] # type: ignore
        if not self.affine:
            # self.__offsets = torch.zeros((instances, dim))
            self.__isometries = torch.stack([torch.eye(self.dim, device=device)] * self.instances)
        self.__optimal = torch.vmap(lambda A, b: A.T.matmul(b))(self.__isometries, self.__offsets) / self.range
        if self.hyper_cube_x:
            self.__optimal = (self.__optimal + 2.5) / 5

    def get_optimal(self):
        return self.__optimal

    def get_transform(self):
        for o, i in zip(self.__offsets, self.__isometries):
            yield o.cpu().numpy(), i.cpu().numpy()

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        offsets = self.__offsets
        isometries = self.__isometries
        if x.ndim == 3:  # [instances, pop, dim]
            offsets = offsets.unsqueeze(1)
            isometries = isometries.unsqueeze(1)
        # assert x.ndim == 2  # [instances, dim] ~ N(0, 1) or U(0, 1)
        assert x.size(0) == offsets.size(0) and x.size(-1) == offsets.size(-1)
        if self.hyper_cube_x:
            x = x * 5 - 2.5
        # self.__isometries @ x ~ N(0, 1)
        w = self.range * torch.matmul(isometries, x.unsqueeze(-1)).squeeze(-1) - offsets
        f = self.eval_fn(w)
        return f

    @staticmethod
    def Ackley(w: torch.Tensor):
        w *= 2
        val = (
            -20 * torch.exp(-0.2 * torch.mean(w**2, dim=-1).sqrt())
            - torch.cos(2 * torch.pi * w).mean(dim=-1).exp()
            + 20
            + torch.e
        )
        return val

    @staticmethod
    def Rosenbrock(w: torch.Tensor):
        w *= 1
        w_1_n = w.narrow(-1, 1, w.size(-1) - 1)
        w_0_n = w.narrow(-1, 0, w.size(-1) - 1)
        val = torch.sum(100 * (w_1_n + 1 - (w_0_n + 1) ** 2) ** 2 + w_0_n**2, dim=-1)
        return val

    @staticmethod
    def Rastrigin(w: torch.Tensor):
        w *= 3
        val = 10 * w.size(-1) + torch.sum(w**2 - 10 * torch.cos(2 * torch.pi * w), dim=-1)
        return val

    @staticmethod
    def Levy(w: torch.Tensor):
        w *= 1
        w_0 = w.select(-1, 0)
        w_1 = w.select(-1, -1)
        w_0_n = w.narrow(-1, 0, w.size(-1) - 1)
        val = (
            torch.sin(torch.pi * (w_0 + 1)) ** 2
            + w_1**2 * (1 + torch.sin(2 * torch.pi * (w_1 + 1)) ** 2)
            + torch.sum(w_0_n**2 * (1 + 10 * torch.sin(torch.pi * (w_0_n + 1) + 1) ** 2), dim=-1)
        )
        return val

    @staticmethod
    def Griewank(w: torch.Tensor):
        w *= 50
        val = (
            1 / 4000 * torch.sum(w**2, dim=-1)
            - torch.prod(
                torch.cos(w / torch.arange(1, w.shape[0] + 1, device=w.device).sqrt()),
                dim=-1,
            )
            + 1
        )
        return val

    @staticmethod
    def Schwefel(w: torch.Tensor):
        w = w * 50 + 420.9687  # [-500, 500]
        val = 418.9829 * w.size(-1) - torch.sum(w * w.abs().sqrt().sin(), dim=-1)
        return val

    @staticmethod
    def HolderTable(w: torch.Tensor):
        w = w * 1
        val = -torch.abs(
            torch.sin(w.select(-1, 0))
            * torch.cos(w.select(-1, 1))
            * torch.exp(torch.abs(1 - torch.linalg.vector_norm(w, dim=-1) / torch.pi))
        )
        return val


class Ackley(torch.nn.Module):
    def __init__(self, dim: int, device: torch.device, parallels: int = 1, offseted: bool = True):
        super().__init__()
        self.dim = dim
        self.__offsets = (torch.rand(parallels, dim, device=device) * 10 - 5) if offseted else torch.zeros(
            (parallels, dim), device=device)
        self.__offsets = self.__offsets.unsqueeze(1)
        print(f"[DEBUG] Ackley min = {self.forward((self.__offsets + 20) / 40).ravel()}")

    def forward(self, x: torch.Tensor):
        assert x.ndim == 3
        assert x.size(0) == self.__offsets.size(0) and x.size(2) == self.__offsets.size(2)
        w = 40 * x - 20 - self.__offsets
        val = -20 * torch.exp(-0.2 * torch.sqrt(torch.mean(w**2, dim=-1))) - torch.exp(torch.mean(torch.cos(2 * torch.pi * w),
                                                                                         dim=-1)) + 20 + torch.e
        return val


class Rosenbrock(torch.nn.Module):
    def __init__(self, dim: int, device: torch.device, parallels: int = 1, offseted: bool = True):
        super().__init__()
        self.dim = dim
        self.__offsets = (torch.rand(parallels, dim, device=device) * 5 - 2.5) if offseted else torch.zeros(
            (parallels, dim), device=device)
        self.__offsets = self.__offsets.unsqueeze(1)
        print(f"[DEBUG] Rosenbrock min = {self.forward((self.__offsets + 10) / 20)}")

    def forward(self, x: torch.Tensor):
        assert x.ndim == 3
        assert x.size(0) == self.__offsets.size(0) and x.size(2) == self.__offsets.size(2)
        w = 20 * x - 10 - self.__offsets + 1
        val = torch.sum(100 * (w[:, :, 1:] - w[:, :, :-1]**2)**2 + (w[:, :, :-1] - 1)**2, dim=-1)
        return val


class Rastrigin(torch.nn.Module):
    def __init__(self, dim: int, device: torch.device, parallels: int = 1, offseted: bool = True):
        super().__init__()
        self.dim = dim
        self.__offsets = (torch.rand(parallels, dim, device=device) * 16 - 8) if offseted else torch.zeros(
            (parallels, dim), device=device)
        self.__offsets = self.__offsets.unsqueeze(1)
        print(f"[DEBUG] Rastrigin min = {self.forward((self.__offsets + 32) / 64)}")

    def forward(self, x: torch.Tensor):
        assert x.ndim == 3
        assert x.size(0) == self.__offsets.size(0) and x.size(2) == self.__offsets.size(2)
        w = 64 * x - 32 - self.__offsets
        val = 10 * self.dim + torch.sum(w**2 - 10 * torch.cos(2 * torch.pi * w), dim=-1)
        return val


class Levy(torch.nn.Module):
    def __init__(self, dim: int, device: torch.device, parallels: int = 1, offseted: bool = True):
        super().__init__()
        self.dim = dim
        self.__offsets = (torch.rand(parallels, dim, device=device) * 5 - 2.5) if offseted else torch.zeros(
            (parallels, dim), device=device)
        self.__offsets = self.__offsets.unsqueeze(1)
        print(f"[DEBUG] Levy min = {self.forward((self.__offsets + 10) / 20)}")

    def forward(self, x: torch.Tensor):
        assert x.ndim == 3
        assert x.size(0) == self.__offsets.size(0) and x.size(2) == self.__offsets.size(2)
        w = 1 + (20 * x - 10 - self.__offsets) / 4
        val = torch.sin(torch.pi * w[:, :, 0])**2 + \
                (w[:, :, -1] - 1)**2 * (1 + torch.sin(2 * torch.pi * w[:, :, -1])**2) + \
                torch.sum((w[:, :, :-1] - 1)**2 * (1 + 10 * torch.sin(torch.pi * w[:, :, :-1] + 1)**2), dim=-1)
        return val
