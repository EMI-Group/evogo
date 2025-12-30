import math
import copy
from typing import Callable, List, Tuple, Sequence

import torch
from torch import nn
from torch.nn import init

from evogo_torch.utils import sort_select, print_with_prefix


class MultiLaneLinear(nn.Module):
    """
    Batched (over the first dimension) linear transformations applied to the last dimension of the inputs.
    """

    __constants__ = ["head_count", "in_features", "out_features"]
    lane_count: int
    in_features: int
    out_features: int
    weight: torch.Tensor

    def __init__(
        self,
        lane_count: int,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.lane_count = lane_count
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty((lane_count, in_features, out_features), **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty(lane_count, out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight[0])
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            stddev = bound / 0.87962566103423978
            init.trunc_normal_(self.weight, 0, stddev, -2 * stddev, 2 * stddev)
            # for w in self.weight:
            # init.kaiming_uniform_(w, a=math.sqrt(5))
            if self.bias is not None:
                # init.uniform_(self.bias, -bound, bound)
                self.bias.zero_()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # input [lane_count, batch, in_features]
        return torch.bmm(input, self.weight) + self.bias.unsqueeze(1)

    def extra_repr(self) -> str:
        return f"lane_count={self.lane_count}, in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"


def _triangular_solve(
    matrices: torch.Tensor,
    rhs: torch.Tensor,
    lower: bool = True,
    left: bool = True,
    trans: bool = False,
) -> torch.Tensor:
    """Applies a batched triangular solve over the input matrices and right hand sides

    ### Args
    `matrices`: the nd-array as the triangular matrices.
    `rhs`: the nd-array as the right hand sides
    `lower`: input lower triangular matrix or not
    `left`: triangular matrix at left ($A^{trans}$ x = b) or not (x $A^{trans}$ = b)
    `trans`: transpose the triangular matrix or not

    ### Returns
    The transformed input.
    """
    assert matrices.ndim >= 2
    assert matrices.size(-2) == matrices.size(-1)
    if left:
        assert matrices.size(-1) == rhs.size(-2)
    else:
        assert matrices.size(-1) == rhs.size(-1)
    outputs = torch.linalg.solve_triangular(matrices, rhs, upper=not lower, left=left)
    return outputs


def standardize(data: torch.Tensor, full_out: bool = False):
    """Applies the batched dimension-wise standardization over the inputs

    ### Args
    `inputs`: the nd-array to be transformed with dimension [num_parallels, batch_size, *dim]

    ### Returns
    The transformed input.
    """
    mean = torch.mean(data, dim=1)
    std = torch.std(data, dim=1)
    outputs = (data - mean.unsqueeze(1)) / std.unsqueeze(1)
    if full_out:
        return outputs, (mean, std)
    else:
        return outputs


def normalize(data: torch.Tensor, full_out: bool = False):
    """Applies the batched dimension-wise normalization to [0, 1] over the inputs

    ### Args
    `inputs`: the nd-array to be transformed with dimension [num_parallels, batch_size, *dim]

    ### Returns
    The transformed input.
    """
    maxs = torch.max(data, dim=1).values
    mins = torch.min(data, dim=1).values
    outputs = (data - mins.unsqueeze(1)) / (maxs - mins).unsqueeze(1)
    if full_out:
        return outputs, (maxs, mins)
    else:
        return outputs


def normalize_with(data: torch.Tensor, maxs: torch.Tensor, mins: torch.Tensor):
    """Applies the batched dimension-wise normalization to [0, 1] over the inputs

    ### Args
    `inputs`: the nd-array to be transformed with dimension [num_parallels, batch_size, *dim]

    ### Returns
    The transformed input.
    """
    outputs = (data - mins.unsqueeze(1)) / (maxs - mins).unsqueeze(1)
    return outputs


def standardize_with(data: torch.Tensor, mean: torch.Tensor, std: torch.Tensor):
    """Applies the batched dimension-wise standardization over the inputs

    ### Args
    `inputs`: the nd-array to be transformed with dimension [num_parallels, batch_size, *dim]

    ### Returns
    The transformed input.
    """
    outputs = (data - mean.unsqueeze(1)) / std.unsqueeze(1)
    return outputs


def destandardize(data: torch.Tensor, mean: torch.Tensor, std: torch.Tensor):
    """Applies the batched dimension-wise de-standardization over the inputs

    ### Args
    `inputs`: the nd-array to be transformed with dimension [num_parallels, batch_size, *dim]

    ### Returns
    The transformed input.
    """
    outputs = data * std.unsqueeze(1) + mean.unsqueeze(1)
    return outputs


def denormalize(data: torch.Tensor, maxs: torch.Tensor, mins: torch.Tensor):
    """Applies the batched dimension-wise de-normalization over the inputs

    ### Args
    `inputs`: the nd-array to be transformed with dimension [num_parallels, batch_size, *dim]

    ### Returns
    The transformed input.
    """
    outputs = data * (maxs - mins).unsqueeze(1) + mins.unsqueeze(1)
    return outputs


class ConstrainedValue(nn.Module):
    """
    The trainable constrained value layer with logistic sigmoid regularization
    """

    def __init__(
        self,
        init_value: float,
        lb: float,
        ub: float,
        features: int | Sequence[int],
        dtype: torch.dtype | None = None,
    ):
        """
        Initialize the ConstrainedValue

        ### Args
        `init_value`: the initial value of the trainable constrained value
        `lb`: the lower bound of the value
        `ub`: the upper bound of the value
        `features`: the number of the independent values
        `dtype`: the PyTorch data type of the values
        """
        super().__init__()
        values = torch.ones(features, dtype=dtype)
        values *= math.log((init_value - lb) / (ub - init_value))
        self.values = nn.Parameter(values)
        self.lb = lb
        self.ub = ub

    def _sigmoid(self):
        return self.values.sigmoid() * (self.ub - self.lb) + self.lb

    def get_detached(self) -> torch.Tensor:
        return self._sigmoid().detach()

    def forward(self) -> torch.Tensor:
        """Gives the trainable constrained value with logistic sigmoid regularization

        ### Returns
        The trainable constrained value with logistic sigmoid regularization.
        """
        return self._sigmoid()


def marginal_log_likelihood(means: torch.Tensor, covars: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Computes the marginal log likelihood losses of the given inputs

    ### Args
    `means`: the mean values with dimension [num_parallels, dataset_size]
    or the independent mean values with dimension [num_parallels, batch_size]
    `covars`: the covariance matrices with dimension [num_parallels, dataset_size, dataset_size]
    or the independent covariances with dimension [num_parallels, batch_size]
    `targets`: the target values with dimension [num_parallels, dataset_size]
    or the independent target values with dimension [num_parallels, batch_size]

    ### Returns
    The marginal log likelihood losses of the given inputs with dimension [num_parallels].
    """
    assert means.ndim == 2 and 2 <= covars.ndim <= 3 and targets.ndim == 2
    num_parallels = means.size(0)
    dataset_size = covars.size(1)
    assert covars.size(0) == num_parallels and targets.size(0) == num_parallels
    assert targets.size(1) == dataset_size and covars.size(1) == dataset_size
    if covars.ndim == 3:
        assert covars.size(2) == dataset_size
        cholesky: torch.Tensor = torch.linalg.cholesky(covars)
        diff = targets - means
        diff = diff.reshape(num_parallels, dataset_size, 1)
        inv_quad = _triangular_solve(cholesky, diff)
        inv_quad = inv_quad.reshape(num_parallels, dataset_size)
        inv_quad = torch.sum(inv_quad**2, dim=-1)
        log_det = cholesky.diagonal(0, -2, -1)
        log_det = torch.sum(torch.log(log_det**2), dim=-1)
    else:
        dataset_size = 1
        log_det = torch.log(covars)
        inv_quad = covars * (means - targets) ** 2

    LOG_2PI = math.log(2 * math.pi)
    loss = (inv_quad + log_det + dataset_size * LOG_2PI) / 2 / dataset_size
    if covars.ndim == 2:
        loss = torch.mean(loss, dim=1)
    return loss


def _matern_kernel(nu: float, dist: torch.Tensor) -> torch.Tensor:
    matern = torch.exp(-math.sqrt(2 * nu) * dist) * (math.sqrt(2 * nu) * dist + 1 + 2 * nu / 3 * dist**2)
    return matern


def _matern_kernel_fast_dist(nu: float, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    dist = torch.cdist(x, y)
    matern = _matern_kernel(nu, dist)
    return matern


class MaternGaussianProcess(nn.Module):
    """
    The traditional Gaussian Process (GP) with Matern kernel

    ### Attributes
    `nu`: the smoothing coefficient
    """

    nu: float = 2.5

    def __init__(self, dim: int, num_parallels: int, nu: float = 2.5):
        super().__init__()
        self.nu = nu
        self.means = nn.Parameter(torch.zeros(num_parallels))
        self.length_scales = ConstrainedValue(
            init_value=0.5,
            lb=0.005,
            ub=5,
            features=(num_parallels, dim),
            # datatype=inputs.dtype,
            # name="length_scales",
        )
        self.out_scales = ConstrainedValue(
            init_value=1,
            lb=0.05,
            ub=20,
            features=num_parallels,
            # datatype=inputs.dtype,
            # name="out_scales",
        )
        self.noises = ConstrainedValue(
            init_value=0.1,
            lb=0.0005,
            ub=0.5,
            features=num_parallels,
            # datatype=targets.dtype,
            # name="noises",
        )

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes the GP predicted means and covariance matrices of the given inputs

        ### Args
        `inputs`: the input datasets with dimension [num_parallels, dataset, dim(x)]

        ### Returns
        The GP predicted means and covariance matrices of the given inputs.
        """
        means, covars, *_ = self._call_full_out(inputs)
        return means, covars

    def _gp_func(self, input: torch.Tensor, length_scale: torch.Tensor, out_scale: torch.Tensor) -> torch.Tensor:
        input_mean = torch.mean(input, dim=1)
        normalized_input = (input - input_mean.unsqueeze(1)) / length_scale.unsqueeze(1)
        matern = _matern_kernel_fast_dist(self.nu, normalized_input, normalized_input)
        return matern * out_scale.unsqueeze(1).unsqueeze(1)

    def _call_full_out(
        self, inputs: torch.Tensor
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Computes the GP predicted means and covariance matrices of the given inputs

        ### Args
        `inputs`: the input datasets with dimension [num_parallels, dataset, dim(x)]

        ### Returns
        The GP predicted means and covariance matrices of the given inputs.
        Also gives other intermediate values (expanded means, noised covariances, raw covariances, raw means, length scales, out scales, noises)
        """
        # dims
        assert inputs.ndim == 2 or inputs.ndim == 3
        num_parallels = inputs.size(0)
        dim = inputs.size(-1)
        no_batch_dim = inputs.ndim == 2
        if no_batch_dim:
            dataset_size = 1
            inputs = inputs.unsqueeze(1)
        else:
            dataset_size = inputs.size(1)
        # parameters
        length_scales = self.length_scales.forward()
        out_scales = self.out_scales.forward()
        noises = self.noises.forward()
        means = self.means
        # input reshape
        inputs = inputs.reshape(num_parallels, dataset_size, dim)
        means_expanded = means.unsqueeze(1)
        covars = self._gp_func(inputs, length_scales, out_scales)
        noises_eye = torch.stack([n * torch.eye(dataset_size, device=inputs.device) for n in noises])
        covars_noised = covars + noises_eye
        if no_batch_dim:
            means_expanded = means_expanded.squeeze(1)
            covars_noised = covars_noised.squeeze(1)
            covars = covars.squeeze(1)
        return (
            means_expanded,
            covars_noised,
            covars,
            means,
            length_scales,
            out_scales,
            noises,
        )


class MarginalLogLikelihood(nn.Module):
    """
    The traditional Gaussian Process (GP) with Matern kernel and marginal log likelihood loss
    """

    def __init__(self, gp: MaternGaussianProcess):
        super().__init__()
        self.gp = gp

    def forward(self, xs: torch.Tensor, targets: torch.Tensor):
        means, covars = self.gp(xs)
        loss = marginal_log_likelihood(means, covars, targets)
        return loss


USE_INV = True


def _prepare_gp_eval(
    gp_net: MaternGaussianProcess,
    datasets_x: torch.Tensor,
    datasets_y: torch.Tensor,
):
    """Prepare the constants and functions used in GP evaluation forward pass

    ### Args
    `gp_net`: the trained GP model
    `trained_params`: the trained GP model parameters
    `datasets_x`: the original input dataset with dimension [num_parallels, dataset_size, dim]
    `datasets_y`: the original output dataset with dimension [num_parallels, dataset_size]

    ### Return
    Constants `(means, out_scales, noises, solved_means, cholesky, original_xs)`
    and `(mean_x, length_scales)` used to normalized inputs (with `lambda x: (x - mean_x) / length_scales`)
    """
    # get values
    covars_noised: torch.Tensor
    # covars: torch.Tensor
    means: torch.Tensor
    length_scales: torch.Tensor
    out_scales: torch.Tensor
    noises: torch.Tensor
    _, covars_noised, covars, means, length_scales, out_scales, noises = gp_net._call_full_out(datasets_x)
    # set lengths
    assert datasets_x.ndim == 3 and datasets_y.ndim == 2
    assert datasets_x.size()[:-1] == datasets_y.size()

    def _covar_cholesky(covar: torch.Tensor):
        default_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float64)
            covar = covar.to(dtype=torch.float64)
            covar = 0.5 * (covar + covar.T)
            eigval: torch.Tensor
            eigvec: torch.Tensor
            eigval, eigvec = torch.linalg.eigh(covar)
            rcond: float = 2.2204460492503131e-15 * covar.size(0) * eigval.max()
            eigval = torch.maximum(eigval, rcond)
            covar = eigvec @ torch.diag(eigval) @ eigvec.T
            cholesky: torch.Tensor = torch.linalg.cholesky(covar)
            if USE_INV:
                rcond = 2.2204460492503131e-15 * covar.size(0)
                cholesky = torch.triu(torch.linalg.pinv(cholesky, rcond=rcond).T)
        finally:
            torch.set_default_dtype(default_dtype)
        cholesky = cholesky.to(dtype=default_dtype)
        return cholesky

    # make symmetric
    # covars = 0.5 * (covars + covars.swapaxes(-1, -2))
    covars_noised = 0.5 * (covars_noised + covars_noised.swapaxes(-1, -2))
    # [num_parallels, dataset_size]
    solved_means: torch.Tensor = torch.linalg.solve(covars_noised, datasets_y - means.unsqueeze(1))
    # [num_parallels, dataset_size, dataset_size]
    cholesky: torch.Tensor = torch.stack(list(map(_covar_cholesky, covars)))
    # [num_parallels, dim]
    mean_x = torch.mean(datasets_x, dim=1).unsqueeze(1)
    length_scales = length_scales.unsqueeze(1)
    # [num_parallels, dataset_size, dim]
    original_xs = (datasets_x - mean_x) / length_scales
    if (
        torch.any(torch.isnan(means))
        or torch.any(torch.isnan(out_scales))
        or torch.any(torch.isnan(noises))
        or torch.any(torch.isnan(solved_means))
        or torch.any(torch.isnan(cholesky))
        or torch.any(torch.isnan(original_xs))
        or torch.any(torch.isnan(mean_x))
        or torch.any(torch.isnan(length_scales))
    ):
        print("[ERROR] NaN occurred!")
    return (
        means,
        out_scales,
        noises,
        solved_means,
        cholesky,
        original_xs,
    ), (mean_x, length_scales)


def _gp_eval_full_out(
    inputs: torch.Tensor,
    nu: float,
    args: List[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Computes the GP predicted means and covariances of the given inputs

    ### Args
    `inputs`: the inputs with dimension [num_parallels, batch?, dim(x)]

    Other inputs are the outputs of `_prepare_gp_eval`

    ### Returns
    The GP predicted means and covariances of the given inputs.
    Also gives intermediate values (raw_covariances = cholesky(covar_mat)^-1 @ matern).
    """
    means, out_scales, noises, solved_means, cholesky, original_xs = args
    num_parallels = means.size(0)
    dim = original_xs.size(2)
    assert inputs.ndim in [2, 3]
    assert inputs.size(0) == num_parallels and inputs.size(-1) == dim
    no_batch_dim = inputs.ndim == 2
    if no_batch_dim:
        inputs = inputs.unsqueeze(1)
    # Matern
    matern = _matern_kernel_fast_dist(nu, inputs, original_xs)
    matern = matern * out_scales.view(num_parallels, 1, 1)
    # covar
    # [num_parallels, batch_size, dataset_size]
    if USE_INV:
        # print("[DEBUG] Using pseudo inverse rather than triangular solve")
        raw_covars = torch.bmm(matern, cholesky)
    else:
        raw_covars = _triangular_solve(cholesky, matern, left=False, trans=True)
    # raw_covars = _least_square_solve(cholesky, matern, left=False, trans=True)  ## LS
    # [num_parallels, batch_size]
    covars: torch.Tensor = torch.linalg.vecdot(raw_covars, raw_covars)
    # [num_parallels, batch_size]
    covars = out_scales.unsqueeze(1) + noises.unsqueeze(1) - covars
    # mean
    # [num_parallels, batch_size]
    means = torch.bmm(matern, solved_means.unsqueeze(-1)).squeeze(-1) + means.unsqueeze(1)
    # return
    if no_batch_dim:
        means = means.squeeze(1)
        covars = covars.squeeze(1)
        raw_covars = raw_covars.squeeze(1)
    return means, covars, raw_covars


class GPEvaluator(torch.nn.Module):
    args: List[torch.Tensor]
    normalize_vals: Tuple[torch.Tensor, torch.Tensor]

    def __init__(
        self,
        gp_net: MaternGaussianProcess,
        datasets_x: torch.Tensor,
        datasets_y: torch.Tensor,
    ):
        """Prepare the function that computes the GP predicted means and covariances of the given inputs

        Args:
            gp_net (MaternGaussianProcess): the trained GP model
            datasets_x (torch.Tensor): the original input dataset with dimension [num_parallels, dataset_size, dim]
            datasets_y (torch.Tensor): the original output dataset with dimension [num_parallels, dataset_size]

        Returns:
            The function that computes the GP predicted means and covariances of the given inputs
        """
        super().__init__()
        self.init(gp_net, datasets_x, datasets_y)

    def init(self, gp_net: MaternGaussianProcess, datasets_x: torch.Tensor, datasets_y: torch.Tensor):
        self.nu = gp_net.nu
        self.args, self.normalize_vals = _prepare_gp_eval(gp_net, datasets_x, datasets_y)

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        inputs = (inputs - self.normalize_vals[0]) / self.normalize_vals[1]
        return _gp_eval_full_out(inputs, self.nu, self.args)[:2]


def _gp_eval_pair_diff(
    inputs1: torch.Tensor,
    inputs2: torch.Tensor,
    nu: float,
    args: List[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Computes the GP predicted means and covariances of the given input pairs' differences

    ### Args
    `inputs1`: the first inputs with dimension [num_parallels, batch?, dim(x)] to subtract from
    `inputs2`: the seconds inputs with dimension [num_parallels, batch?, dim(x)] to be subtracted

    Other inputs are the outputs of `_prepare_gp_eval`

    ### Returns
    The GP predicted means and covariances of the given input pairs' differences (inputs1 - inputs2).
    """
    means, out_scales, noises, _, _, original_xs = args
    num_parallels = means.size(0)
    dim = original_xs.size(2)
    assert inputs1.ndim in [2, 3] and inputs2.ndim in [2, 3]
    assert inputs1.size(0) == num_parallels and inputs1.size(-1) == dim
    assert inputs1.size() == inputs2.size()
    no_batch_dim = inputs1.ndim == 2
    if no_batch_dim:
        inputs1 = inputs1.unsqueeze(1)
        inputs2 = inputs2.unsqueeze(1)
    # get rho
    # [num_parallels, batch_size]
    distance = torch.linalg.norm(inputs1 - inputs2, dim=-1)
    # [num_parallels, batch_size]
    rho = _matern_kernel(nu, distance) * out_scales.unsqueeze(1)
    # get eval outputs
    m1, c1, raw_covars1 = _gp_eval_full_out(inputs1, nu, args)
    m2, c2, raw_covars2 = _gp_eval_full_out(inputs2, nu, args)
    # [num_parallels, batch_size]
    covars_dot = torch.linalg.vecdot(raw_covars1, raw_covars2)
    # output
    means_diff = m1 - m2
    mask = torch.abs(covars_dot - rho) >= c1 * c2
    rho = torch.where(mask, c1 * c2, covars_dot - rho)
    covars_diff = c1 + c2 + 2 * rho
    covars_diff = torch.maximum(covars_diff, (noises * noises).unsqueeze(1))
    if no_batch_dim:
        means_diff = means_diff.squeeze(1)
        covars_diff = covars_diff.squeeze(1)
    return means_diff, covars_diff


class GPPairDiffEvaluator(torch.nn.Module):
    args: List[torch.Tensor]
    normalize_vals: Tuple[torch.Tensor, torch.Tensor]

    def __init__(
        self,
        gp_net: MaternGaussianProcess,
        datasets_x: torch.Tensor,
        datasets_y: torch.Tensor,
    ):
        """Prepare the function that computes the GP predicted means and covariances of the given input pairs' differences

        Args:
            gp_net (MaternGaussianProcess): the trained GP model
            datasets_x (torch.Tensor): the original input dataset with dimension [num_parallels, dataset_size, dim]
            datasets_y (torch.Tensor): the original output dataset with dimension [num_parallels, dataset_size]

        Returns:
            The function that computes the GP predicted means and covariances of the given input pairs' differences
        """
        super().__init__()
        self.init(gp_net, datasets_x, datasets_y)

    def init(self, gp_net: MaternGaussianProcess, datasets_x: torch.Tensor, datasets_y: torch.Tensor):
        self.nu = gp_net.nu
        self.args, self.normalize_vals = _prepare_gp_eval(gp_net, datasets_x, datasets_y)

    def forward(self, inputs1: torch.Tensor, inputs2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        inputs1 = (inputs1 - self.normalize_vals[0]) / self.normalize_vals[1]
        inputs2 = (inputs2 - self.normalize_vals[0]) / self.normalize_vals[1]
        return _gp_eval_pair_diff(inputs1, inputs2, self.nu, self.args)


class BroadcastDropout(nn.Module):
    def __init__(self, p: float = 0.1, broadcast_dims: Sequence[int] = ()):
        super().__init__()
        self.p = p
        self.broadcast_dims = broadcast_dims

    def forward(self, x: torch.Tensor):
        if not self.training or self.p == 0:
            return x

        mask_shape = list(x.shape)
        for dim in self.broadcast_dims:
            mask_shape[dim] = 1

        mask = (torch.rand(mask_shape, device=x.device) > self.p).float()
        mask = mask.expand_as(x)
        return x * mask / (1 - self.p)


class MLPBase(nn.Module):
    """
    The MLP-based model
    """

    def __init__(
        self,
        dim: int,
        num_parallels: int,
        drop_rate: float,
        activation_fn: Callable[[torch.Tensor], torch.Tensor],
        dim_multipliers: Tuple[float, ...],
        hidden_mins: Tuple[int, ...],
        output_dim: int,
        final_fn: Callable[[torch.Tensor], torch.Tensor] = nn.Identity(),
    ):
        """Initializes the generative model

        Args:
            dim (int): the input dimension
            num_parallels (int): the number of parallel models
            drop_rate (float): the dropout rate.
            activation_fn (Callable[[torch.Tensor], torch.Tensor]): the activation function.
            dim_multipliers (Tuple[int, ...]): the tuple containing each hidden layer's size divided by the dimension.
            hidden_mins (Tuple[int, ...]): the tuple containing each hidden layer's minimum size.
            output_dim (int): the output dimension.
        """
        super().__init__()
        assert dim > 0 and num_parallels > 0
        assert len(dim_multipliers) > 0 and len(dim_multipliers) == len(hidden_mins)
        assert 0 <= drop_rate < 1
        assert all(m > 0 for m in dim_multipliers)
        assert all(m > 0 for m in hidden_mins)
        self.activation_fn = activation_fn

        dropouts: List[BroadcastDropout] = []
        layers: List[MultiLaneLinear] = []
        for i in range(len(dim_multipliers)):
            prev_dim = dim if i == 0 else max(int(dim_multipliers[i - 1] * dim), hidden_mins[i - 1])
            now_dim = max(int(dim_multipliers[i] * dim), hidden_mins[i])
            layer = MultiLaneLinear(num_parallels, prev_dim, now_dim)
            layers.append(layer)
            dropouts.append(BroadcastDropout(p=drop_rate, broadcast_dims=(0,)))
        self.layers = nn.ModuleList(layers)
        self.dropouts = nn.ModuleList(dropouts)
        self.final_layer = MultiLaneLinear(num_parallels, layers[-1].out_features, output_dim)
        self.final_fn = final_fn

    def forward(self, xs: torch.Tensor) -> torch.Tensor:
        """Gives the inputs transformed by the generative model

        ### Args
        `xs`: the inputs with dimension [num_parallels, batch, dim(x)]

        ### Returns
        The transformed inputs.
        """
        assert xs.ndim == 3
        for layer, dropout in zip(self.layers, self.dropouts):
            xs = layer(xs)
            xs = self.activation_fn(xs)
            xs = dropout(xs)
        xs = self.final_layer(xs)
        if xs.size(-1) == 1:
            xs = xs.squeeze(-1)
        xs = self.final_fn(xs)
        return xs


class GenerativeModel(MLPBase):
    """
    The generative model for optimization
    """

    def __init__(
        self,
        dim: int,
        num_parallels: int,
        drop_rate: float = 1 / 128,
        activation_fn: Callable[[torch.Tensor], torch.Tensor] = nn.Tanh(),
        dim_multipliers: Tuple[float, ...] = (2, 4, 4, 4, 2),
        hidden_mins: Tuple[int, ...] = (128, 256, 256, 256, 128),
        # wide: bool = False,
        # deep: bool = False,
    ):
        """Initializes the generative model

        Args:
            dim (int): the input dimension
            num_parallels (int): the number of parallel models
            drop_rate (float, optional): the dropout rate. Defaults to 1/128.
            activation_fn (Callable[[torch.Tensor], torch.Tensor], optional): the activation function. Defaults to torch.tanh.
            dim_multipliers (Tuple[int, ...], optional): the tuple containing each hidden layer's size divided by the dimension. Defaults to (2, 4, 4, 4, 2).
            hidden_mins (Tuple[int, ...], optional): the tuple containing each hidden layer's minimum size. Defaults to (128, 256, 256, 256, 128).
        """
        # if not wide and not deep:
        #     dim_multipliers = (2, 4, 4, 4, 2)
        #     hidden_mins = (128, 256, 256, 256, 128)
        # elif wide:
        #     dim_multipliers = (4, 6, 4)
        #     hidden_mins = (256, 384, 256)
        # else:
        #     dim_multipliers = (2, 3, 3, 3, 3, 3, 2)
        #     hidden_mins = (128, 192, 192, 192, 192, 192, 128)
        super().__init__(
            dim,
            num_parallels,
            drop_rate,
            activation_fn,
            dim_multipliers,
            hidden_mins,
            dim,
            _Arcsinh(),
        )


class MLPPredictor(MLPBase):
    """
    The MLP-based prediction model for optimization
    """

    def __init__(
        self,
        dim: int,
        num_parallels: int,
        drop_rate: float = 1 / 128,
        activation_fn: Callable[[torch.Tensor], torch.Tensor] = nn.Tanh(),
        final_fn: Callable[[torch.Tensor], torch.Tensor] = nn.Identity(),
        dim_multipliers: Tuple[float, ...] = (2, 1, 0.5, 0.25, 0.125),
        hidden_mins: Tuple[int, ...] = (128, 64, 32, 16, 8),
    ):
        """Initializes the prediction model

        Args:
            dim (int): the input dimension
            num_parallels (int): the number of parallel models
            drop_rate (float, optional): the dropout rate. Defaults to 1/128.
            activation_fn (Callable[[torch.Tensor], torch.Tensor], optional): the activation function. Defaults to torch.tanh.
            final_fn (Callable[[torch.Tensor], torch.Tensor], optional): the final activation function. Defaults to torch.identity.
            dim_multipliers (Tuple[float, ...], optional): the tuple containing each hidden layer's size divided by the dimension. Defaults to (2, 1, 0.5, 0.25, 0.125).
            hidden_mins (Tuple[int, ...], optional): the tuple containing each hidden layer's minimum size. Defaults to (128, 64, 32, 16, 8).
        """
        super().__init__(
            dim,
            num_parallels,
            drop_rate,
            activation_fn,
            dim_multipliers,
            hidden_mins,
            1,
            final_fn,
        )
        self.final_fn = final_fn


class _Arcsinh(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (torch.arcsinh(x) + 1.5) / 3


class MultilaneMSELoss(nn.Module):
    def __init__(self, last_ndim: int = 1, model: MLPBase | None = None):
        super().__init__()
        self.last_ndim = last_ndim
        self.model = model

    def forward(self, xs: torch.Tensor, ys: torch.Tensor) -> torch.Tensor:
        if self.model is not None:
            xs = self.model.forward(xs)
        diff = xs - ys
        # diff = diff.flatten(start_dim=-self.last_ndim)
        # inner: torch.Tensor = (diff * diff).sum(dim=-1)
        # return torch.sum(inner, dim=tuple(range(1, inner.ndim))) / (xs.numel() // xs.size(0))
        return (diff * diff).mean(tuple(range(1, xs.ndim)))


class VAELoss(nn.Module):
    def __init__(
        self,
        dim: int,
        num_parallels: int,
        drop_rate: float = 1 / 128,
        activation_fn: Callable[[torch.Tensor], torch.Tensor] = nn.Tanh(),
        dim_multipliers: Tuple[float, ...] = (2, 1, 0.5, 0.25, 0.125),
        hidden_mins: Tuple[int, ...] = (128, 64, 32, 16, 8),
    ):
        super().__init__()
        encode_size = max(int(dim_multipliers[-1] * dim), hidden_mins[-1]) // 2
        self.encode_size = encode_size
        self.encoder = MLPBase(
            dim=dim,
            num_parallels=num_parallels,
            drop_rate=drop_rate,
            activation_fn=activation_fn,
            dim_multipliers=dim_multipliers[:-1],
            hidden_mins=hidden_mins[:-1],
            output_dim=encode_size * 2,
        )
        self.decoder = MLPBase(
            dim=encode_size,
            num_parallels=num_parallels,
            drop_rate=drop_rate,
            activation_fn=activation_fn,
            dim_multipliers=list(map(lambda x: x * dim / encode_size, dim_multipliers[-2::-1])),
            hidden_mins=hidden_mins[-2::-1],
            output_dim=dim,
            final_fn=_Arcsinh(),
        )
        self.mse = MultilaneMSELoss()

    def forward(self, xs: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder.forward(xs)
        encode_size = encoded.size(-1) // 2
        assert encode_size * 2 == encoded.size(-1)
        mean = encoded[..., :encode_size]
        log_var = encoded[..., encode_size:]
        reparameterized = torch.randn_like(mean) * torch.exp(0.5 * log_var) + mean
        reconstruct_xs = self.decoder.forward(reparameterized)
        loss_reconstruct = self.mse(reconstruct_xs, xs)
        loss_KL = -0.5 * torch.mean(torch.sum(1 + log_var - mean**2 - torch.exp(log_var), dim=-1), dim=-1)
        return torch.stack([loss_reconstruct, loss_KL])


class PairedGenerativeLoss(nn.Module):
    """
    The module for giving paired generative model loss
    """

    def __init__(
        self,
        eval_pair_diff: GPPairDiffEvaluator,
        generative_model: GenerativeModel,
        cycle_scale: float = 100,
        out_scale: float = 1,
        mll_scale: float = 1,
        mip_scale: float = 0.1,
        mip_std_scale: float = 1,
        gan: bool = False,
        single_gen: bool = False,
        lcb: bool = False,
        discriminator: MLPPredictor | None = None,
    ):
        """The generative model loss

        Args:
            eval_pair_diff (nn.Module): the pair difference GP evaluation function
            generative_model (nn.Module): the generative model
            cycle_scale (float, optional): the generative models' cyclic consistency loss scale. Defaults to 100.
            out_scale (float, optional): the generative models' output similarity loss scale. Defaults to 1.
            mll_scale (float, optional): the MLL loss scale for GP predictive outputs. Defaults to 1.
            mip_scale (float, optional): the MIP loss scale for GP predicted generative improves. Defaults to 0.1.
            mip_std_scale (float, optional): the standard deviation scale in the MIP loss. Defaults to 1.
            gan (bool, optional): whether to use GAN loss. Defaults to False.
            single_gen (bool, optional): whether to use a single generative model for both win2lose and lose2win. Defaults to False.
            lcb (bool, optional): whether to use the LCB loss or KG loss. Defaults to False.
            discriminator (nn.Module, optional): the discriminator model for GAN loss. Must be present if `gan=True`. Defaults to None.
        """
        super().__init__()
        self.cycle_scale = cycle_scale
        self.out_scale = out_scale
        self.mll_scale = mll_scale
        self.mip_scale = mip_scale
        self.mip_std_scale = mip_std_scale
        self.eval_pair_diff = eval_pair_diff
        assert generative_model is not None
        assert not (gan and single_gen)
        win2lose = None if single_gen else generative_model
        lose2win = generative_model if single_gen else copy.deepcopy(generative_model)
        if gan:
            assert discriminator is not None
            self.gan_loss = nn.CrossEntropyLoss()
            self.discriminator_win = discriminator
            self.discriminator_lose = copy.deepcopy(discriminator)
        else:
            self.discriminator_win = None
            self.discriminator_lose = None
        if lcb:
            self.mll_scale = 0
        self.win2lose = win2lose
        self.lose2win = lose2win
        self.mse = MultilaneMSELoss()

    def mip(self, mean: torch.Tensor, cov: torch.Tensor) -> torch.Tensor:
        return self.mip_scale * torch.mean(self.mip_std_scale * torch.sqrt(cov) + mean, dim=1)

    def mll(self, means: torch.Tensor, covars: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.mll_scale * marginal_log_likelihood(means, covars, targets)

    def forward(
        self,
        wins_x: torch.Tensor,
        loses_x: torch.Tensor,
        wins_y: torch.Tensor,
        loses_y: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes the paired generative losses

        Parameters
        ----------
        wins_x : torch.Tensor
            the winner inputs with dimension [num_parallels, batch, dim(x)]
        loses_x : torch.Tensor
            the winner inputs with dimension [num_parallels, batch, dim(x)]
        wins_y : torch.Tensor
            the winner outputs with dimension [num_parallels, batch]
        loses_y : torch.Tensor
            the winner outputs with dimension [num_parallels, batch]

        Returns
        -------
        The paired generative losses: [lose cycle, win cycle, lose out, win out,
        win - lose MLL, lose - win MLL, G(lose) - win MIP, lose - G(win) MIP].
        """
        assert wins_x.ndim == 3 and wins_y.ndim == 2
        assert wins_x.size() == loses_x.size() and wins_y.size() == loses_y.size()
        num_parallels = wins_x.size(0)
        assert wins_x.size(1) == wins_y.size(1)
        if self.win2lose is not None:
            # generative model outputs
            fake_win = self.lose2win.forward(loses_x)
            fake_lose = self.win2lose.forward(wins_x)
            # fake_win_x = loses_y + self.eval_pair_diff(fake_win, loses_x)[0]
            cycle_lose = self.win2lose.forward(fake_win)
            # fake_lose_x = wins_y + self.eval_pair_diff(fake_lose, wins_x)[0]
            cycle_win = self.lose2win.forward(fake_lose)
            # losses
            win_cycle = self.cycle_scale * self.mse.forward(cycle_win, wins_x)
            lose_cycle = self.cycle_scale * self.mse.forward(cycle_lose, loses_x)
            win_out = self.out_scale * self.mse.forward(fake_win, wins_x)
            lose_out = self.out_scale * self.mse.forward(fake_lose, loses_x)
            if self.eval_pair_diff is not None:
                pair_diff_win = self.eval_pair_diff.forward(wins_x, cycle_lose)
                pair_diff_lose = self.eval_pair_diff.forward(loses_x, cycle_win)
                win_sub_lose_mll_loss = self.mll(pair_diff_win[0], pair_diff_win[1], wins_y - loses_y)
                lose_sub_win_mll_loss = self.mll(pair_diff_lose[0], pair_diff_lose[1], loses_y - wins_y)
                pair_diff_fake_win = self.eval_pair_diff.forward(fake_win, wins_x)
                genWin_sub_win_mip_loss = self.mip(pair_diff_fake_win[0], pair_diff_fake_win[1])
                ##lose_sub_genLose_mip_loss = 0 * self.mip(*self.eval_gp(loses_x, fake_lose))
                lose_sub_genLose_mip_loss = torch.zeros(num_parallels, device=wins_x.device)
            elif self.discriminator_win is not None:
                dis_true_win = self.discriminator_win.forward(wins_x)
                dis_true_lose = self.discriminator_lose.forward(loses_x)
                dis_fake_win = self.discriminator_win.forward(fake_win)
                dis_fake_lose = self.discriminator_lose.forward(fake_lose)
                dis_fake_win_ng = self.discriminator_win.forward(fake_win.detach())
                dis_fake_lose_ng = self.discriminator_lose.forward(fake_lose.detach())
                genWin_sub_win_mip_loss = self.gan_loss.forward(dis_true_win, torch.ones_like(dis_true_win))
                genWin_sub_win_mip_loss += self.gan_loss.forward(dis_true_lose, torch.ones_like(dis_true_lose))
                genWin_sub_win_mip_loss = self.mll_scale * torch.mean(genWin_sub_win_mip_loss, dim=1)
                win_sub_lose_mll_loss = self.gan_loss.forward(dis_fake_win, torch.zeros_like(dis_fake_win))
                win_sub_lose_mll_loss += self.gan_loss.forward(dis_fake_lose, torch.zeros_like(dis_fake_lose))
                win_sub_lose_mll_loss = -self.mll_scale * torch.mean(win_sub_lose_mll_loss, dim=1)
                lose_sub_win_mll_loss = self.gan_loss.forward(dis_fake_win_ng, torch.zeros_like(dis_fake_win_ng))
                lose_sub_win_mll_loss += self.gan_loss.forward(
                    dis_fake_lose_ng, torch.zeros_like(dis_fake_lose_ng)
                )
                lose_sub_win_mll_loss = 2 * self.mll_scale * torch.mean(lose_sub_win_mll_loss, dim=1)
                lose_sub_genLose_mip_loss = torch.zeros(num_parallels, device=wins_x.device)
            else:
                win_sub_lose_mll_loss = lose_sub_win_mll_loss = genWin_sub_win_mip_loss = (
                    lose_sub_genLose_mip_loss
                ) = torch.zeros(num_parallels, device=wins_x.device)
        else:
            fake_win = self.lose2win.forward(loses_x)
            win_out = self.out_scale * self.mse.forward(fake_win, wins_x)
            if self.eval_pair_diff is not None:
                pair_diff_fake_win = self.eval_pair_diff.forward(fake_win, wins_x)
                genWin_sub_win_mip_loss = self.mip(pair_diff_fake_win[0], pair_diff_fake_win[1])
            else:
                genWin_sub_win_mip_loss = torch.zeros(num_parallels, device=wins_x.device)
            lose_sub_genLose_mip_loss = win_sub_lose_mll_loss = lose_sub_win_mll_loss = lose_cycle = win_cycle = (
                lose_out
            ) = torch.zeros(num_parallels, device=wins_x.device)
        # return
        return torch.stack(
            [
                lose_cycle,
                win_cycle,
                lose_out,
                win_out,
                win_sub_lose_mll_loss,
                lose_sub_win_mll_loss,
                genWin_sub_win_mip_loss,
                lose_sub_genLose_mip_loss,
            ]
        )


def data_split(
    datasets_x: torch.Tensor,
    datasets_y: torch.Tensor,
    histories: Tuple[torch.Tensor, torch.Tensor] | None = None,
    portion: float = 0.1,
    sliding_window: float = 0.3,
    same_norm: bool = False,
):
    """Split and normalize the given datasets

    Args:
        datasets_x: the datasets of inputs with dimension [num_parallels, dataset_size, dim(x)]
        datasets_y: the datasets of outputs with dimension [num_parallels, dataset_size]
        histories: the datasets of inputs and outputs of last iteration, can be None
        portion: the portion of data that shall be splitted into the winner datasets
        sliding_window: when using `histories`, the history data with fitness values below
            `max(datasets_y) + sliding_window * std(datasets_y)` will be included to train GP

    Returns:
        (GP input sets, GP output sets),
        (Winner input sets, Loser input sets, Winner output sets, Loser output sets, conditions),
        (de-normalization arguments for x, de-standardization arguments for y)

    ## Note
        `Winner input set` is of size [num_parallels, portion * (1 - portion) * dataset_size**2, dim(x)], others alike
    """
    assert datasets_x.ndim == 3 and datasets_y.ndim == 2
    assert datasets_x.size()[:-1] == datasets_y.size()
    num_parallels = datasets_x.size(0)
    data_size = datasets_x.size(1)
    dim = datasets_x.size(2)
    win_len = round(data_size * portion)
    assert win_len > 0
    _min_len = data_size

    # add history
    if histories is not None:
        assert histories[0].ndim == 3 and histories[1].ndim == 2
        # assert histories[0].size(:2) == datasets_y.size() and \
        #        histories[1].size() == datasets_y.size()
        histories_x = histories[0]
        histories_y = histories[1]
        print(
            f"[DEBUG] Previous fitness: {print_with_prefix(torch.mean(histories_y, dim=1))} ± {print_with_prefix(torch.std(histories_y, dim=1))}"
            + f" ({print_with_prefix(torch.min(histories_y, dim=1).values)} → {print_with_prefix(torch.max(histories_y, dim=1).values)})"
        )
        print(
            f"[DEBUG] Current fitness: {print_with_prefix(torch.mean(datasets_y, dim=1))} ± {print_with_prefix(torch.std(datasets_y, dim=1))}"
            + f" ({print_with_prefix(torch.min(datasets_y, dim=1).values)} → {print_with_prefix(torch.max(datasets_y, dim=1).values)})"
        )
        # remove duplicate from history
        new_histories_x = []
        new_histories_y = []
        for i in range(num_parallels):
            all_x = torch.cat([datasets_x[i], histories_x[i]], dim=0)
            unique_x, inverse_indices = torch.unique(all_x, dim=0, return_inverse=True, sorted=False)
            count = unique_x.size(0)
            unique_y = torch.zeros(count, device=all_x.device)
            unique_y.index_reduce_(
                0, inverse_indices, torch.cat([datasets_y[i], histories_y[i]]), reduce="mean", include_self=False
            )
            range_all = torch.arange(count, device=all_x.device)
            unique_ind = range_all[(range_all[:, None] != inverse_indices[:data_size]).all(dim=1)]
            new_histories_x.append(unique_x[unique_ind])
            new_histories_y.append(unique_y[unique_ind])
        histories_x = new_histories_x
        histories_y = new_histories_y
        # find actual history size
        std_prev = torch.std(datasets_y, dim=1)
        max_prev = torch.max(datasets_y, dim=1).values
        limit = sliding_window * std_prev + max_prev
        _nnz_all = []
        _min_len = min([y.size(0) for y in histories_y])
        for y, lim in zip(histories_y, limit):
            _nnz = torch.count_nonzero(y <= lim).item()
            _nnz_all.append(_nnz)
            _min_len = min(_min_len, _nnz)
        print(f"[DEBUG] All history sizes: {_nnz_all}")
        for i in range(num_parallels):
            histories_x[i], histories_y[i] = sort_select(histories_x[i], histories_y[i], num=_min_len)
        # form full datasets
        all_datasets_x = torch.concatenate([datasets_x, torch.stack(histories_x)], dim=1)
        all_datasets_y = torch.concatenate([datasets_y, torch.stack(histories_y)], dim=1)
    else:
        all_datasets_x = datasets_x
        all_datasets_y = datasets_y
    # standardize
    if same_norm:
        _, (denorm_x_max, denorm_x_min) = normalize(all_datasets_x, full_out=True)
        denorm_x_max = torch.max(denorm_x_max, dim=0).values
        denorm_x_min = torch.min(denorm_x_min, dim=0).values
        print(f"[INFO]  Force same normalization of x: {denorm_x_min} to {denorm_x_max}")
        denorm_x_max = torch.stack([denorm_x_max] * num_parallels)
        denorm_x_min = torch.stack([denorm_x_min] * num_parallels)
        denorm_x = (denorm_x_max, denorm_x_min)
        all_datasets_y, de_std_y = standardize(all_datasets_y, full_out=True)
    else:
        all_datasets_x, denorm_x = normalize(all_datasets_x, full_out=True)
        all_datasets_y, de_std_y = standardize(all_datasets_y, full_out=True)
    datasets_x = all_datasets_x[:, :data_size]
    datasets_y = all_datasets_y[:, :data_size]
    # get wins and loses
    datasets_x, datasets_y = sort_select(datasets_x, datasets_y, num=data_size)
    wins_x = datasets_x[:, :win_len]
    wins_y = datasets_y[:, :win_len]
    loses_x = datasets_x[:, win_len:]
    loses_y = datasets_y[:, win_len:]

    # get tuple dataset
    lose_len = loses_x.size(1)
    wins_x = wins_x.unsqueeze(1).expand(-1, lose_len, -1, -1)
    wins_x = wins_x.reshape(num_parallels, -1, dim)
    loses_x = loses_x.unsqueeze(2).expand(-1, -1, win_len, -1)
    loses_x = loses_x.reshape(num_parallels, -1, dim)
    wins_y = wins_y.unsqueeze(1).expand(-1, lose_len, -1)
    wins_y = wins_y.reshape(num_parallels, -1)
    loses_y = loses_y.unsqueeze(2).expand(-1, -1, win_len)
    loses_y = loses_y.reshape(num_parallels, -1)
    # return
    return (
        (all_datasets_x, all_datasets_y),
        (wins_x, loses_x, wins_y, loses_y),
        (denorm_x, de_std_y),
    )
