import math
from functools import partial
from typing import Any, Mapping, Tuple, Sequence

import jax
import jax.numpy as jnp
from jax.experimental import enable_x64

import flax.linen as nn
from flax.linen.dtypes import promote_dtype
from optax import sigmoid_binary_cross_entropy

from evogo_jax.utils import *


class BatchDenseLayer(nn.Module):
    """
    Batched (over the middle dimensions) linear transformations applied to the last dimension of the inputs.

    ### Attributes
    `features`: the number of output features.
    `use_bias`: whether to add a bias to the output (default: True).
    `dtype`: the dtype of the computation (default: infer from input and params).
    `param_dtype`: the dtype passed to parameter initializers (default: float32).
    `precision`: numerical precision of the computation see `jax.lax.Precision` for details.
    `kernel_init`: initializer function for the weight matrix.
    `bias_init`: initializer function for the bias.
    """
    
    features: int
    use_bias: bool = True
    dtype = None
    param_dtype = jnp.float32
    precision = None
    kernel_init = init_wrapper(nn.initializers.lecun_normal(batch_axis=(0, )))
    bias_init = init_wrapper(nn.initializers.zeros_init())
    
    @nn.compact
    def __call__(self, inputs: jax.Array) -> jax.Array:
        """Applies a batched linear transformation to the inputs along the last dimension.

        ### Args
        `inputs`: the nd-array to be transformed with dimension [num_parallels, *batch, input_features]

        ### Returns
        The transformed input with dimension [num_parallels, *batch, input_features].
        """
        assert inputs.ndim >= 3
        num_parallels = inputs.shape[0]
        input_features = inputs.shape[-1]
        batch_shape = inputs.shape[1:-1]
        batch_size = math.prod(batch_shape)
        inputs = inputs.reshape(num_parallels, batch_size, input_features)
        
        kernels = self.param(
            "kernels",
            self.kernel_init,
            (num_parallels, input_features, self.features),
            self.param_dtype,
        )
        if self.use_bias:
            biases = self.param(
                "biases",
                self.bias_init,
                (num_parallels, self.features),
                self.param_dtype,
            )
        else:
            biases = None
            inputs, kernels, biases = promote_dtype(inputs, kernels, biases, dtype=self.dtype)
        # [num_parallels, batch_size, features]
        y = jax.vmap(partial(jnp.matmul, precision=self.precision))(inputs, kernels)
        if biases is not None:
            y += biases.reshape(num_parallels, self.features)[:, jnp.newaxis, :]
        return y.reshape(num_parallels, *batch_shape, self.features)


def _triangular_solve(matrices: jax.Array,
                      rhs: jax.Array,
                      lower: bool = True,
                      left: bool = True,
                      trans: bool = False) -> jax.Array:
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
    assert matrices.shape[-2] == matrices.shape[-1]
    if left:
        assert matrices.shape[-1] == rhs.shape[-2]
    else:
        assert matrices.shape[-1] == rhs.shape[-1]
    outputs = vmap_last_dim(
        partial(
            jax.lax.linalg.triangular_solve,  # type: ignore
            lower=lower,
            left_side=left,
            transpose_a=trans,
        ),
        matrices,
        rhs,
        last_ndim=2,
        last_dims_out_shape=rhs.shape[-2:],
    )
    return outputs


def _least_square_solve(matrices: jax.Array,
                        rhs: jax.Array,
                        lower: bool = True,
                        left: bool = True,
                        trans: bool = False) -> jax.Array:
    """Applies a batched least square solve over the input matrices and right hand sides

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
    assert matrices.shape[-2] == matrices.shape[-1]
    if left:
        assert matrices.shape[-1] == rhs.shape[-2]
    else:
        assert matrices.shape[-1] == rhs.shape[-1]
    
    def _solve_fn(A: jax.Array, b: jax.Array):
        s: jax.Array
        s, _, _, _ = jax.numpy.linalg.lstsq(A if left != trans else A.T, b if left else b.T, rcond=None)  # type: ignore
        return s if left else s.T
    
    outputs = vmap_last_dim(
        _solve_fn,
        matrices,
        rhs,
        last_ndim=2,
        last_dims_out_shape=rhs.shape[-2:],
    )
    return outputs


def _take_diag(inputs: jax.Array) -> jax.Array:
    """Applies the batched taking diagonal action

    ### Args
    `inputs`: the nd-array to be transformed, the first few dimensions are the batch ones.

    ### Returns
    The transformed input.
    """
    assert inputs.ndim >= 2 and inputs.shape[-1] == inputs.shape[-2]
    outputs = vmap_last_dim(jnp.diag, inputs, last_ndim=2, last_dims_out_shape=(inputs.shape[-1], ))
    return outputs


def standardize(data: jax.Array, full_out: bool = False):
    """Applies the batched dimension-wise standardization over the inputs

    ### Args
    `inputs`: the nd-array to be transformed with dimension [num_parallels, batch_size, *dim]

    ### Returns
    The transformed input.
    """
    mean = jnp.mean(data, axis=1)
    std = jnp.std(data, axis=1)
    outputs = (data - mean[:, jnp.newaxis, ...]) / std[:, jnp.newaxis, ...]
    if full_out:
        return outputs, (mean, std)
    else:
        return outputs


def normalize(data: jax.Array, full_out: bool = False):
    """Applies the batched dimension-wise normalization to [0, 1] over the inputs

    ### Args
    `inputs`: the nd-array to be transformed with dimension [num_parallels, batch_size, *dim]

    ### Returns
    The transformed input.
    """
    maxs = jnp.max(data, axis=1)
    mins = jnp.min(data, axis=1)
    outputs = (data - mins[:, jnp.newaxis, ...]) / (maxs - mins)[:, jnp.newaxis, ...]
    if full_out:
        return outputs, (maxs, mins)
    else:
        return outputs


def normalize_with(data: jax.Array, maxs: jax.Array, mins: jax.Array):
    """Applies the batched dimension-wise normalization to [0, 1] over the inputs

    ### Args
    `inputs`: the nd-array to be transformed with dimension [num_parallels, batch_size, *dim]

    ### Returns
    The transformed input.
    """
    outputs = (data - mins[:, jnp.newaxis, ...]) / (maxs - mins)[:, jnp.newaxis, ...]
    return outputs


def standardize_with(data: jax.Array, mean: jax.Array, std: jax.Array):
    """Applies the batched dimension-wise standardization over the inputs

    ### Args
    `inputs`: the nd-array to be transformed with dimension [num_parallels, batch_size, *dim]

    ### Returns
    The transformed input.
    """
    outputs = (data - mean[:, jnp.newaxis, ...]) / std[:, jnp.newaxis, ...]
    return outputs


def destandardize(data: jax.Array, mean: jax.Array, std: jax.Array):
    """Applies the batched dimension-wise de-standardization over the inputs

    ### Args
    `inputs`: the nd-array to be transformed with dimension [num_parallels, batch_size, *dim]

    ### Returns
    The transformed input.
    """
    outputs = data * std[:, jnp.newaxis, ...] + mean[:, jnp.newaxis, ...]
    return outputs


def denormalize(data: jax.Array, maxs: jax.Array, mins: jax.Array):
    """Applies the batched dimension-wise de-normalization over the inputs

    ### Args
    `inputs`: the nd-array to be transformed with dimension [num_parallels, batch_size, *dim]

    ### Returns
    The transformed input.
    """
    outputs = data * (maxs - mins)[:, jnp.newaxis, ...] + mins[:, jnp.newaxis, ...]
    return outputs


class ConstrainedValueLayer(nn.Module):
    """
    The trainable constrained value layer with logistic sigmoid regularization

    ### Attributes
    `init_value`: the initial value of the trainable constrained value
    `lb`: the lower bound of the value
    `ub`: the upper bound of the value
    `features`: the number of the independent values
    """
    
    init_value: float
    lb: float
    ub: float
    features: int | Sequence[int]
    datatype: None | jnp.dtype = None
    name: str = "values"
    
    def setup(self):
        self.values = self.param(
            self.name,
            lambda *_: jnp.ones(self.features, dtype=self.datatype) * math.log((self.init_value - self.lb) /
                                                                               (self.ub - self.init_value)),
        )
    
    def _sigmoid(self):
        return jax.nn.sigmoid(self.values) * (self.ub - self.lb) + self.lb
    
    def get_detached(self) -> jax.Array:
        return jax.lax.stop_gradient(self._sigmoid())
    
    def __call__(self) -> jax.Array:
        """Gives the trainable constrained value with logistic sigmoid regularization

        ### Returns
        The trainable constrained value with logistic sigmoid regularization.
        """
        return self._sigmoid()


def marginal_log_likelihood(means: jax.Array, covars: jax.Array, targets: jax.Array) -> jax.Array:
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
    num_parallels = means.shape[0]
    dataset_size = covars.shape[1]
    assert covars.shape[0] == num_parallels and targets.shape[0] == num_parallels
    assert targets.shape[1] == dataset_size and covars.shape[1] == dataset_size
    if covars.ndim == 3:
        assert covars.shape[2] == dataset_size
        cholesky = jax.lax.linalg.cholesky(covars)  # type: ignore
        diff = targets - means
        diff = diff.reshape(num_parallels, dataset_size, 1)
        inv_quad = _triangular_solve(cholesky, diff)
        inv_quad = inv_quad.reshape(num_parallels, dataset_size)
        inv_quad = jnp.sum(inv_quad**2, axis=-1)
        log_det = _take_diag(cholesky)
        log_det = jnp.sum(jnp.log(log_det**2), axis=-1)
    else:
        dataset_size = 1
        log_det = jnp.log(covars)
        inv_quad = covars * (means - targets)**2
    
    LOG_2PI = math.log(2 * math.pi)
    loss = (inv_quad + log_det + dataset_size * LOG_2PI) / 2 / dataset_size
    if covars.ndim == 2:
        loss = jnp.mean(loss, axis=1)
    return loss


@partial(jax.jit, static_argnums=[0])
def _matern_kernel(nu: float, dist: jax.Array) -> jax.Array:
    matern = jnp.exp(-math.sqrt(2 * nu) * dist) * (math.sqrt(2 * nu) * dist + 1 + 2 * nu / 3 * dist**2)
    return matern


@partial(jax.jit, static_argnums=[0])
def _matern_kernel_fast_dist(nu: float, x2: jax.Array, y2: jax.Array, xy: jax.Array) -> jax.Array:
    dist = jnp.sqrt(jax.nn.relu(x2 + y2 - 2 * xy))
    matern = _matern_kernel(nu, dist)
    return matern


@partial(jax.jit, static_argnums=[0])
def _matern_kernel_slow_dist(nu: float, xs1: jax.Array, xs2: jax.Array) -> jax.Array:
    # dist_fn = lambda x1, x2: jnp.sqrt(jnp.sum((x1[:, jnp.newaxis, :] - x2[jnp.newaxis, :, :])**2, axis=-1))
    # if xs1.ndim > 2:
    #     dist = jax.vmap(dist_fn)(xs1, xs2)
    # else:
    #     dist = dist_fn(xs1, xs2)
    from cuda.cdist_kernel_bind import cdist
    dist = cdist(xs1, xs2)
    matern = _matern_kernel(nu, dist)
    return matern


def _last_dim_inner(xs: jax.Array) -> jax.Array:
    return vmap_last_dim(lambda x: jnp.inner(x, x), xs, last_dims_out_shape=())


def _last_dim_inner2(xs1: jax.Array, xs2: jax.Array) -> jax.Array:
    return vmap_last_dim(lambda x, y: jnp.inner(x, y), xs1, xs2, last_dims_out_shape=())


USE_FAST_DIST = True


class MaternGaussianProcess(nn.Module):
    """
    The traditional Gaussian Process (GP) with Matern kernel

    ### Attributes
    `nu`: the smoothing coefficient
    """
    
    nu: float = 2.5
    
    def _gp_func(self, input: jax.Array, length_scale: jax.Array, out_scale: jax.Array) -> jax.Array:
        input_mean = jnp.mean(input, axis=0)
        normalized_input = (input - input_mean[jnp.newaxis, :]) / length_scale[jnp.newaxis, :]
        if USE_FAST_DIST:
            x2_ = _last_dim_inner(normalized_input)
            xy_ = jnp.matmul(normalized_input, normalized_input.T)
            y2_ = x2_[:, jnp.newaxis]
            x2_ = x2_[jnp.newaxis, :]
            matern = _matern_kernel_fast_dist(self.nu, x2_, y2_, xy_) * out_scale
        else:
            matern = _matern_kernel_slow_dist(self.nu, normalized_input, normalized_input) * out_scale
        return matern
    
    def __call__(self, inputs: jax.Array) -> Tuple[jax.Array, jax.Array]:
        """Computes the GP predicted means and covariance matrices of the given inputs

        ### Args
        `inputs`: the input datasets with dimension [num_parallels, dataset, dim(x)]

        ### Returns
        The GP predicted means and covariance matrices of the given inputs.
        """
        means, covars, *_ = self._call_full_out(inputs)
        return means, covars
    
    @nn.compact
    def _call_full_out(
            self, inputs: jax.Array) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        """Computes the GP predicted means and covariance matrices of the given inputs

        ### Args
        `inputs`: the input datasets with dimension [num_parallels, dataset, dim(x)]

        ### Returns
        The GP predicted means and covariance matrices of the given inputs.
        Also gives other intermediate values (raw means, raw covariances, length scales, out scales, noises)
        """
        # dims
        assert inputs.ndim == 2 or inputs.ndim == 3
        num_parallels = 1 if inputs.ndim == 2 else inputs.shape[0]
        dataset_size = inputs.shape[-2]
        dim = inputs.shape[-1]
        # trainable parameters
        means = self.param(
            "means",
            lambda *_: jnp.zeros(num_parallels, dtype=inputs.dtype),
        )
        length_scales = ConstrainedValueLayer(
            init_value=0.5,
            lb=0.005,
            ub=5,
            features=(num_parallels, dim),
            # datatype=inputs.dtype,
            name="length_scales",
        )()
        out_scales = ConstrainedValueLayer(
            init_value=1,
            lb=0.05,
            ub=20,
            features=num_parallels,
            # datatype=inputs.dtype,
            name="out_scales",
        )()
        noises = ConstrainedValueLayer(
            init_value=0.1,
            lb=0.0005,
            ub=0.5,
            features=num_parallels,
            # datatype=targets.dtype,
            name="noises",
        )()
        noises_eye = jax.vmap(lambda n: n * jnp.eye(dataset_size))(noises)
        # input reshape
        inputs = inputs.reshape(num_parallels, dataset_size, dim)
        means_expanded = means[:, jnp.newaxis]
        covars = jax.vmap(self._gp_func)(inputs, length_scales, out_scales)
        covars_noised = covars + noises_eye
        return means_expanded, covars_noised, covars, means, length_scales, out_scales, noises


class GpMllNetwork(nn.Module):
    """
    The traditional Gaussian Process (GP) with Matern kernel and marginal log likelihood loss

    ### Attributes
    `nu`: the smoothing coefficient
    """
    
    nu: float = 2.5
    
    def setup(self) -> None:
        self.gp = MaternGaussianProcess(self.nu)
    
    def __call__(self, xs, targets):
        means, covars = self.gp(xs)
        loss = marginal_log_likelihood(means, covars, targets)
        return loss


USE_INV = True


def _prepare_gp_eval(gp_net: MaternGaussianProcess, trained_params: Mapping[str, Mapping[str, Any]], datasets_x: jax.Array,
                     datasets_y: jax.Array):
    """Prepare the constants and functions used in GP evaluation forward pass

    ### Args
    `gp_net`: the trained GP model
    `trained_params`: the trained GP model parameters
    `datasets_x`: the original input dataset with dimension [num_parallels, dataset_size, dim]
    `datasets_y`: the original output dataset with dimension [num_parallels, dataset_size]

    ### Return
    Constants `(means, out_scales, noises, solved_means, cholesky, original_xs, original_x2s)`
    and function `normalize_fn`
    """
    # get values
    covars_noised: jax.Array
    # covars: jax.Array
    means: jax.Array
    length_scales: jax.Array
    out_scales: jax.Array
    noises: jax.Array
    _, covars_noised, covars, means, length_scales, out_scales, noises = gp_net.apply(  # type: ignore
        trained_params, datasets_x, method=gp_net._call_full_out)  # type: ignore
    # set lengths
    assert datasets_x.ndim == 3 and datasets_y.ndim == 2
    assert datasets_x.shape[:2] == datasets_y.shape
    
    def _covar_cholesky(covar: jax.Array):
        org_covar = covar
        with enable_x64():
            covar = covar.astype(jnp.float64)
            eigval: jax.Array
            eigvec: jax.Array
            eigval, eigvec = jnp.linalg.eigh(covar, symmetrize_input=True)
            rcond: float = 10 * jnp.finfo(covar.dtype).eps * covar.shape[0] * jnp.max(eigval)
            eigval = jnp.maximum(eigval, rcond)
            covar = eigvec @ jnp.diag(eigval) @ eigvec.T
            cholesky: jax.Array = jnp.linalg.cholesky(covar)
            if USE_INV:
                rcond = 10 * jnp.finfo(org_covar.dtype).eps * covar.shape[0]
                cholesky = jnp.triu(jnp.linalg.pinv(cholesky, rcond=rcond).T)
        cholesky = cholesky.astype(org_covar.dtype)
        return cholesky
    
    # make symmetric
    # covars = 0.5 * (covars + covars.swapaxes(-1, -2))
    covars_noised = 0.5 * (covars_noised + covars_noised.swapaxes(-1, -2))
    # [num_parallels, dataset_size]
    solved_means: jax.Array = jax.vmap(lambda cov, ys, m: jnp.linalg.solve(cov, ys - m)) \
                                      (covars_noised, datasets_y, means)
    # [num_parallels, dataset_size, dataset_size]
    cholesky: jax.Array = jnp.stack(list(map(_covar_cholesky, covars)))  #(covars_noised)
    # [num_parallels, dim]
    mean_x = jnp.mean(datasets_x, axis=1)[:, jnp.newaxis, :]
    length_scales = length_scales[:, jnp.newaxis, :]
    # x(..., dim) -> (..., dim)
    normalize_fn: Callable[[jax.Array], jax.Array] = lambda x: (x - mean_x) / length_scales
    # [num_parallels, dataset_size, dim]
    original_xs = normalize_fn(datasets_x)
    # [num_parallels, dataset_size]
    original_x2s = _last_dim_inner(original_xs)
    if jnp.any(jnp.isnan(means)) or jnp.any(jnp.isnan(out_scales)) or jnp.any(jnp.isnan(noises)) or jnp.any(
            jnp.isnan(solved_means)) or jnp.any(jnp.isnan(cholesky)) or jnp.any(jnp.isnan(original_xs)) or jnp.any(
                jnp.isnan(original_x2s)) or jnp.any(jnp.isnan(mean_x)) or jnp.any(jnp.isnan(length_scales)):
        print("[ERROR] NaN occurred!")
    return (means, out_scales, noises, solved_means, cholesky, original_xs, original_x2s), normalize_fn


@partial(jax.jit, static_argnames=["normalize_fn", "nu"])
def _gp_eval_full_out(inputs: jax.Array, normalize_fn: Callable[[jax.Array], jax.Array], nu: float, *args:
                      jax.Array) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """Computes the GP predicted means and covariances of the given inputs

    ### Args
    `inputs`: the inputs with dimension [num_parallels, *batch, dim(x)]
    
    Other inputs are the outputs of `_prepare_gp_eval`

    ### Returns
    The GP predicted means and covariances of the given inputs.
    Also gives intermediate values (raw_covariances = cholesky(covar_mat)^-1 @ matern).
    """
    (means, out_scales, noises, solved_means, cholesky, original_xs, original_x2s) = args
    num_parallels = means.shape[0]
    # dataset_size = solved_means.shape[1]
    dim = original_xs.shape[2]
    assert inputs.ndim >= 3
    assert inputs.shape[0] == num_parallels and inputs.shape[-1] == dim
    batch_shape = inputs.shape[1:-1]
    batch_size = math.prod(batch_shape)
    inputs = inputs.reshape(num_parallels, batch_size, dim)
    # Matern
    inputs = normalize_fn(inputs)
    if USE_FAST_DIST:
        x2 = original_x2s  # [num_parallels, dataset_size]
        y2 = _last_dim_inner(inputs)  # [num_parallels, batch_size]
        # [num_parallels, batch_size, dataset_size]
        xy = jax.vmap(lambda x, i: jnp.matmul(i, x.T))(original_xs, inputs)
        x2 = x2[:, jnp.newaxis, :]
        y2 = y2[:, :, jnp.newaxis]
        # [num_parallels, batch_size, dataset_size]
        matern = _matern_kernel_fast_dist(nu, x2, y2, xy)
    else:
        matern = _matern_kernel_slow_dist(nu, inputs, original_xs)
    matern = matern * out_scales[:, jnp.newaxis, jnp.newaxis]
    # covar
    # [num_parallels, batch_size, dataset_size]
    if USE_INV:
        print(f"[DEBUG] Using pseudo inverse rather than triangular solve")
        raw_covars = jax.vmap(jnp.matmul)(matern, cholesky)
    else:
        raw_covars = _triangular_solve(cholesky, matern, left=False, trans=True)
    # raw_covars = _least_square_solve(cholesky, matern, left=False, trans=True)  ## LS
    # [num_parallels, batch_size]
    covars = _last_dim_inner(raw_covars)
    # [num_parallels, batch_size]
    covars = out_scales[:, jnp.newaxis] + noises[:, jnp.newaxis] - covars
    # mean
    # mat(batch_size, dataset_size), sol(dataset_size), m() -> (batch_size)
    mean_fn = jax.vmap(lambda mat, sol, m: jnp.matmul(mat, sol) + m)
    # [num_parallels, batch_size]
    means = mean_fn(matern, solved_means, means)
    # return
    return means, covars, raw_covars


def get_gp_eval_fn(gp_net: MaternGaussianProcess, trained_params: Mapping[str, Mapping[str, Any]], datasets_x: jax.Array,
                   datasets_y: jax.Array) -> Callable[[jax.Array, jax.Array], Tuple[jax.Array, jax.Array]]:
    """Prepare the function that computes the GP predicted means and covariances of the given inputs

    ### Args
    `gp_net`: the trained GP model
    `trained_params`: the trained GP model parameters
    `datasets_x`: the original input dataset with dimension [num_parallels, dataset_size, dim]
    `datasets_y`: the original output dataset with dimension [num_parallels, dataset_size]

    ### Return
    The function that computes the GP predicted means and covariances of the given inputs
    """
    nu = gp_net.nu
    args, normalize_fn = _prepare_gp_eval(gp_net, trained_params, datasets_x, datasets_y)
    eval_fn = lambda in1: _gp_eval_full_out(in1, normalize_fn, nu, *args)[:2]
    return eval_fn


@partial(jax.jit, static_argnames=["normalize_fn", "nu"])
def _gp_eval_pair_diff(inputs1: jax.Array, inputs2: jax.Array, normalize_fn: Callable[[jax.Array], jax.Array], nu: float,
                       *args: jax.Array) -> Tuple[jax.Array, jax.Array]:
    """Computes the GP predicted means and covariances of the given input pairs' differences

    ### Args
    `inputs1`: the first inputs with dimension [num_parallels, *batch, dim(x)] to subtract from
    `inputs2`: the seconds inputs with dimension [num_parallels, *batch, dim(x)] to be subtracted
    
    Other inputs are the outputs of `_prepare_gp_eval`

    ### Returns
    The GP predicted means and covariances of the given input pairs' differences (inputs1 - inputs2).
    """
    (means, out_scales, noises, _, _, original_xs, _) = args
    num_parallels = means.shape[0]
    dim = original_xs.shape[2]
    assert inputs1.ndim >= 2
    assert inputs1.shape[0] == num_parallels and inputs1.shape[-1] == dim
    assert inputs1.shape == inputs2.shape
    batch_shape = inputs1.shape[1:-1]
    batch_size = math.prod(batch_shape)
    inputs1 = inputs1.reshape(num_parallels, batch_size, dim)
    inputs2 = inputs2.reshape(num_parallels, batch_size, dim)
    # get rho
    # [num_parallels, batch_size]
    distance = vmap_last_dim(jnp.linalg.norm, normalize_fn(inputs1) - normalize_fn(inputs2), last_dims_out_shape=())
    # [num_parallels, batch_size]
    rho = _matern_kernel(nu, distance) * out_scales[:, jnp.newaxis]
    # get eval outputs
    m1, c1, raw_covars1 = _gp_eval_full_out(inputs1, normalize_fn, nu, *args)
    m2, c2, raw_covars2 = _gp_eval_full_out(inputs2, normalize_fn, nu, *args)
    # [num_parallels, batch_size]
    covars_dot = _last_dim_inner2(raw_covars1, raw_covars2)
    # output
    means_diff = m1 - m2
    mask = jnp.abs(covars_dot - rho) >= c1 * c2
    rho = jnp.where(mask, c1 * c2, covars_dot - rho)
    covars_diff = c1 + c2 + 2 * rho
    covars_diff = jax.vmap(jnp.maximum)(covars_diff, noises**2)
    return means_diff.reshape(num_parallels, *batch_shape), covars_diff.reshape(num_parallels, *batch_shape)


def get_gp_eval_pair_diff_fn(gp_net: MaternGaussianProcess, trained_params: Mapping[str, Mapping[str, Any]],
                             datasets_x: jax.Array,
                             datasets_y: jax.Array) -> Callable[[jax.Array, jax.Array], Tuple[jax.Array, jax.Array]]:
    """Prepare the function that computes the GP predicted means and covariances of the given input pairs' differences

    ### Args
    `gp_net`: the trained GP model
    `trained_params`: the trained GP model parameters
    `datasets_x`: the original input dataset with dimension [num_parallels, dataset_size, dim]
    `datasets_y`: the original output dataset with dimension [num_parallels, dataset_size]

    ### Return
    The function that computes the GP predicted means and covariances of the given input pairs' differences
    """
    nu = gp_net.nu
    args, normalize_fn = _prepare_gp_eval(gp_net, trained_params, datasets_x, datasets_y)
    eval_fn: Callable[[jax.Array, jax.Array],
                      Tuple[jax.Array, jax.Array]] = lambda in1, in2: _gp_eval_pair_diff(in1, in2, normalize_fn, nu, *args)
    return eval_fn


class GenerativeModel(nn.Module):
    """
    The generative model for optimization
    
    ### Attributes:
    `drop_rate`: the dropout rate
    `activation_fn`: the activation function
    `dim_multipliers`: the tuple containing each hidden layer's size divided by the dimension
    `hidden_mins`: the tuple containing each hidden layer's minimum size
    """
    drop_rate: float = 1 / 128
    activation_fn: Callable[[jax.Array], jax.Array] = nn.tanh
    dim_multipliers: Tuple[int, ...] = (2, 4, 4, 4, 2)
    hidden_mins: Tuple[int, ...] = (128, 256, 256, 256, 128)
    dim: int = None
    
    def setup(self):
        assert len(self.dim_multipliers) > 0 and len(self.dim_multipliers) == len(self.hidden_mins)
        assert 0 <= self.drop_rate < 1
        assert all(m > 0 for m in self.dim_multipliers)
        assert all(m > 0 for m in self.hidden_mins)
    
    @nn.compact
    def __call__(self, xs: jax.Array, conds: jax.Array, training: bool = True) -> jax.Array:
        """Gives the inputs transformed by the generative model

        ### Args
        `xs`: the inputs with dimension [num_parallels, *batch, dim(x)]
        `conds`: the conditions with dimension [num_parallels, *batch, dim(condition)]

        ### Returns
        The transformed inputs.
        """
        assert xs.ndim >= 3
        dim = self.dim if self.dim else xs.shape[-1]
        for _mult, _min in zip(self.dim_multipliers, self.hidden_mins):
            xs = jnp.concatenate([xs, conds], axis=-1)
            xs = BatchDenseLayer(max(_mult * dim, _min))(xs)
            xs = self.activation_fn(xs)
            xs = nn.Dropout(rate=self.drop_rate, broadcast_dims=(0, ), deterministic=not training)(xs)
        xs = BatchDenseLayer(dim)(xs)
        xs = (jnp.arcsinh(xs) + 1.5) / 3
        return xs


@partial(jax.jit, static_argnames=["last_ndim"])
def _parallel_runs_mse(xs: jax.Array, ys: jax.Array, last_ndim: int = 1) -> jax.Array:
    if last_ndim == 1:
        inner = vmap_last_dim(lambda d: jnp.inner(d, d), xs - ys, last_dims_out_shape=())
    else:  # last_ndim > 1
        inner = vmap_last_dim(lambda d: jnp.sum(d * d), xs - ys, last_ndim=last_ndim, last_dims_out_shape=())
    remain_ndim = xs.ndim - last_ndim
    sum = jnp.sum(inner, axis=[i for i in range(1, remain_ndim)]) if remain_ndim > 1 else inner
    return sum / math.prod(xs.shape[1:])


class PairedGenerativeLoss(nn.Module):
    """
    The module for giving paired generative model loss
    
    ### Attributes:
    `eval_pair_diff`: the pair difference GP evaluation function
    `drop_rate`: the dropout rate
    `activation_fn`: the activation function
    `dim_multipliers`: the tuple containing each hidden layer's size divided by the dimension
    `hidden_mins`: the tuple containing each hidden layer's minimum size
    `cycle_scale`: the generative models' cyclic consistency loss scale
    `out_scale`: the generative models' output similarity loss scale
    `mll_scale`: the MLL loss scale for GP predictive outputs
    `mip_scale`: the MIP loss scale for GP predicted generative improves
    `mip_std_scale`: the standard deviation scale in the MIP loss
    """
    eval_pair_diff: Callable[[jax.Array, jax.Array], Tuple[jax.Array, jax.Array]] | None
    drop_rate: float = 1 / 128
    activation_fn: Callable[[jax.Array], jax.Array] = nn.tanh
    cycle_scale: float = 100
    out_scale: float = 1
    mll_scale: float = 1
    mip_scale: float = 0.1
    mip_std_scale: float = 1
    wide: bool = False
    deep: bool = False
    single_gen: bool = False
    gan: bool = False
    
    def setup(self) -> None:
        if not self.wide and not self.deep:
            dim_multipliers = (2, 4, 4, 4, 2)
            hidden_mins = (128, 256, 256, 256, 128)
        elif self.wide:
            dim_multipliers = (4, 6, 4)
            hidden_mins = (256, 384, 256)
        else:
            dim_multipliers = (2, 3, 3, 3, 3, 3, 2)
            hidden_mins = (128, 192, 192, 192, 192, 192, 128)
        win2lose = None if self.single_gen else GenerativeModel(self.drop_rate, self.activation_fn, dim_multipliers,
                                                                hidden_mins)
        lose2win = GenerativeModel(self.drop_rate, self.activation_fn, dim_multipliers, hidden_mins)
        if self.gan:
            self.gan_loss = sigmoid_binary_cross_entropy
            self.discrimitor_win = MLPPredictor(self.drop_rate, self.activation_fn)
            self.discrimitor_lose = MLPPredictor(self.drop_rate, self.activation_fn)
        self.win2lose = win2lose
        self.lose2win = lose2win
        self.mll = marginal_log_likelihood
        self.mse = _parallel_runs_mse
        self.mip = lambda mean, cov: self.mip_std_scale * jnp.mean(jnp.sqrt(cov) + mean, axis=1)
        
        del self.drop_rate, self.activation_fn
    
    def _get_nets(self):
        """
        Get the win to lose and lose to win generative models
        """
        return self.win2lose, self.lose2win
    
    def __call__(self,
                 wins_x: jax.Array,
                 loses_x: jax.Array,
                 wins_y: jax.Array,
                 loses_y: jax.Array,
                 conds: jax.Array,
                 training: bool = True) -> jax.Array:
        """Computes the paired generative losses

        ### Args
        `wins_x`: the winner inputs with dimension [num_parallels, *batch, dim(x)]
        `loses_x`: the winner inputs with dimension [num_parallels, *batch, dim(x)]
        `wins_y`: the winner outputs with dimension [num_parallels, *batch]
        `loses_y`: the winner outputs with dimension [num_parallels, *batch]

        ### Returns
        The paired generative losses: [lose cycle, win cycle, lose out, win out,
        win - lose MLL, lose - win MLL, G(lose) - win MIP, lose - G(win) MIP].
        """
        assert wins_x.ndim >= 3
        assert wins_x.shape == loses_x.shape and wins_y.shape == loses_y.shape
        num_parallels = wins_x.shape[0]
        dim = wins_x.shape[-1]
        batch_shape = wins_x.shape[1:-1]
        assert batch_shape == wins_y.shape[1:]
        batch_size = math.prod(batch_shape)
        wins_x = wins_x.reshape(num_parallels, batch_size, dim)
        loses_x = loses_x.reshape(num_parallels, batch_size, dim)
        wins_y = wins_y.reshape(num_parallels, batch_size)
        loses_y = loses_y.reshape(num_parallels, batch_size)
        if self.win2lose is not None:
            # generative model outputs
            fake_win = self.lose2win(loses_x, conds, training)
            fake_lose = self.win2lose(wins_x, conds, training)
            fake_win_x = loses_y + self.eval_pair_diff(fake_win, loses_x)[0]
            cycle_lose = self.win2lose(fake_win, conds, training)
            fake_lose_x = wins_y + self.eval_pair_diff(fake_lose, wins_x)[0]
            cycle_win = self.lose2win(fake_lose, conds, training)
            # losses
            win_cycle = self.cycle_scale * self.mse(cycle_win, wins_x)
            lose_cycle = self.cycle_scale * self.mse(cycle_lose, loses_x)
            win_out = self.out_scale * self.mse(fake_win, wins_x)
            lose_out = self.out_scale * self.mse(fake_lose, loses_x)
            if self.eval_pair_diff is not None:
                win_sub_lose_mll_loss = self.mll_scale * self.mll(*(self.eval_pair_diff(wins_x, cycle_lose) +
                                                                    (wins_y - loses_y, )))
                lose_sub_win_mll_loss = self.mll_scale * self.mll(*(self.eval_pair_diff(loses_x, cycle_win) +
                                                                    (loses_y - wins_y, )))
                genWin_sub_win_mip_loss = self.mip_scale * self.mip(*self.eval_pair_diff(fake_win, wins_x))
                ##lose_sub_genLose_mip_loss = 0 * self.mip_scale * self.mip(*self.eval_gp(loses_x, fake_lose))
                lose_sub_genLose_mip_loss = jnp.zeros(num_parallels)
            elif self.gan:
                dis_true_win = self.discrimitor_win(wins_x, training=training)
                dis_true_lose = self.discrimitor_lose(loses_x, training=training)
                dis_fake_win = self.discrimitor_win(fake_win, training=training)
                dis_fake_lose = self.discrimitor_lose(fake_lose, training=training)
                dis_fake_win_ng = self.discrimitor_win(jax.lax.stop_gradient(fake_win), training=training)
                dis_fake_lose_ng = self.discrimitor_lose(jax.lax.stop_gradient(fake_lose), training=training)
                
                genWin_sub_win_mip_loss = self.gan_loss(dis_true_win, jnp.ones_like(dis_true_win))
                genWin_sub_win_mip_loss += self.gan_loss(dis_true_lose, jnp.ones_like(dis_true_lose))
                genWin_sub_win_mip_loss = self.mll_scale * jnp.mean(genWin_sub_win_mip_loss, axis=1)
                
                win_sub_lose_mll_loss = self.gan_loss(dis_fake_win, jnp.zeros_like(dis_fake_win))
                win_sub_lose_mll_loss += self.gan_loss(dis_fake_lose, jnp.zeros_like(dis_fake_lose))
                win_sub_lose_mll_loss = -self.mll_scale * jnp.mean(win_sub_lose_mll_loss, axis=1)
                
                lose_sub_win_mll_loss = self.gan_loss(dis_fake_win_ng, jnp.zeros_like(dis_fake_win_ng))
                lose_sub_win_mll_loss += self.gan_loss(dis_fake_lose_ng, jnp.zeros_like(dis_fake_lose_ng))
                lose_sub_win_mll_loss = 2 * self.mll_scale * jnp.mean(lose_sub_win_mll_loss, axis=1)
                
                lose_sub_genLose_mip_loss = jnp.zeros(num_parallels)
            else:
                win_sub_lose_mll_loss = lose_sub_win_mll_loss = \
                    genWin_sub_win_mip_loss = lose_sub_genLose_mip_loss = jnp.zeros(num_parallels)
        else:
            fake_win = self.lose2win(loses_x, conds, training)
            win_out = self.out_scale * self.mse(fake_win, wins_x)
            if self.eval_pair_diff is not None:
                genWin_sub_win_mip_loss = self.mip_scale * self.mip(*self.eval_pair_diff(fake_win, wins_x))
            else:
                genWin_sub_win_mip_loss = jnp.zeros(num_parallels)
            lose_sub_genLose_mip_loss = win_sub_lose_mll_loss = lose_sub_win_mll_loss = \
                lose_cycle = win_cycle = lose_out = jnp.zeros(num_parallels)
        # return
        return jnp.stack([
            lose_cycle, win_cycle, lose_out, win_out, win_sub_lose_mll_loss, lose_sub_win_mll_loss, genWin_sub_win_mip_loss,
            lose_sub_genLose_mip_loss
        ])


def data_split(key: jax.Array,
               datasets_x: jax.Array,
               datasets_y: jax.Array,
               histories: Tuple[jax.Array, jax.Array] | None = None,
               portion: float = 0.1,
               sliding_window: float = 0.3,
               condition_dim: int = 16,
               same_norm: bool = False):
    """
    Split and normalize the given datasets
    
    ### Args
    `datasets_x`: the datasets of inputs with dimension [num_parallels, dataset_size, dim(x)]
    `datasets_y`: the datasets of outputs with dimension [num_parallels, dataset_size]
    `histories`: the datasets of inputs and outputs of last iteration, can be None
    `portion`: the portion of data that shall be splitted into the winner datasets
    `sliding_window`: when using `histories`, the history data with fitness values below
    `max(datasets_y) + sliding_window * std(datasets_y)` will be included to train GP

    ### Return
    `(GP input sets, GP output sets),
    (Winner input sets, Loser input sets, Winner output sets, Loser output sets, conditions),
    (de-normalization arguments for x, de-standardization arguments for y)`
    #### Note
    `Winner input set` is of size [num_parallels, portion * (1 - portion) * dataset_size**2, dim(x)], others alike
    """
    assert datasets_x.ndim == 3 and datasets_y.ndim == 2
    assert datasets_x.shape[:2] == datasets_y.shape
    num_parallels = datasets_x.shape[0]
    data_size = datasets_x.shape[1]
    win_len = math.floor(data_size * portion)
    if portion > 0:
        win_len = max(win_len, 1)
    assert win_len > 0
    _min_len = data_size
    
    # add history
    if histories is not None:
        assert histories[0].ndim == 3 and histories[1].ndim == 2
        # assert histories[0].shape[:2] == datasets_y.shape and \
        #        histories[1].shape == datasets_y.shape
        histories_x = histories[0]
        histories_y = histories[1]
        print(
            f"[DEBUG] Previous fitness: {print_with_prefix(jnp.mean(histories_y, axis=1))} ± {print_with_prefix(jnp.std(histories_y, axis=1))} ({print_with_prefix(jnp.min(histories_y, axis=1))} → {print_with_prefix(jnp.max(histories_y, axis=1))})"
        )
        print(
            f"[DEBUG] Current fitness: {print_with_prefix(jnp.mean(datasets_y, axis=1))} ± {print_with_prefix(jnp.std(datasets_y, axis=1))} ({print_with_prefix(jnp.min(datasets_y, axis=1))} → {print_with_prefix(jnp.max(datasets_y, axis=1))})"
        )
        # remove duplicate from history
        new_histories_x = []
        new_histories_y = []
        for i in range(num_parallels):
            hx = histories_x[i]
            dx = datasets_x[i]
            hy = histories_y[i]
            h_mask = jax.vmap(lambda a: ~jnp.any(jax.vmap(lambda b: jnp.all(a == b))(dx)))(hx)
            new_histories_x.append(hx[h_mask])
            new_histories_y.append(hy[h_mask])
        histories_x = new_histories_x
        histories_y = new_histories_y
        # find actual history size
        std_org = jnp.std(datasets_y, axis=1)
        max_org = jnp.max(datasets_y, axis=1)
        limit = sliding_window * std_org + max_org
        _nnz_all = []
        _min_len = min(map(len, histories_y))
        for y, l in zip(histories_y, limit):
            _nnz = jnp.count_nonzero(y <= l).item()
            _nnz_all.append(_nnz)
            _min_len = min(_min_len, _nnz)
        print(f"[DEBUG] All history sizes: {_nnz_all}")
        for i in range(num_parallels):
            histories_x[i], histories_y[i] = sort_select(histories_x[i], histories_y[i], num=_min_len)
        # form full datasets
        all_datasets_x = jnp.concatenate([datasets_x, jnp.asarray(histories_x)], axis=1)
        all_datasets_y = jnp.concatenate([datasets_y, jnp.asarray(histories_y)], axis=1)
    else:
        all_datasets_x = datasets_x
        all_datasets_y = datasets_y
    # standardize
    if same_norm:
        _, (denorm_x_max, denorm_x_min) = normalize(all_datasets_x, full_out=True)
        denorm_x_max = jnp.max(denorm_x_max, axis=0)
        denorm_x_min = jnp.min(denorm_x_min, axis=0)
        print(f"[INFO]  Force same normalization of x: {denorm_x_min} to {denorm_x_max}")
        denorm_x_max = jnp.stack([denorm_x_max] * num_parallels)
        denorm_x_min = jnp.stack([denorm_x_min] * num_parallels)
        denorm_x = (denorm_x_max, denorm_x_min)
        all_datasets_y, de_std_y = standardize(all_datasets_y, full_out=True)
    else:
        all_datasets_x, denorm_x = normalize(all_datasets_x, full_out=True)
        all_datasets_y, de_std_y = standardize(all_datasets_y, full_out=True)
    datasets_x = all_datasets_x[:, :data_size]
    datasets_y = all_datasets_y[:, :data_size]
    # get wins and loses
    datasets_x, datasets_y = jax.vmap(partial(sort_select, num=data_size)) \
                                     (datasets_x, datasets_y)
    wins_x = datasets_x[:, :win_len]
    wins_y = datasets_y[:, :win_len]
    loses_x = datasets_x[:, win_len:]
    loses_y = datasets_y[:, win_len:]
    
    # get tuple dataset
    def _tuple_fn(a: jax.Array, b: jax.Array) -> Tuple[jax.Array, jax.Array]:
        aa = jnp.repeat(a, len(b), axis=0)
        bb = jnp.tile(b, (len(a), 1) if a.ndim > 1 else len(a))
        return aa, bb
    
    cond_size = wins_x.shape[1] * loses_x.shape[1]
    cond = jax.random.normal(key, shape=(num_parallels, cond_size, condition_dim))
    wins_x, loses_x = jax.vmap(_tuple_fn)(wins_x, loses_x)
    wins_y, loses_y = jax.vmap(_tuple_fn)(wins_y, loses_y)
    # return
    return  (all_datasets_x, all_datasets_y), \
            (wins_x, loses_x, wins_y, loses_y, cond), \
            (denorm_x, de_std_y)


class MLPBase(nn.Module):
    """
    The MLP-based model
    
    ### Attributes:
    `drop_rate`: the dropout rate
    `activation_fn`: the activation function
    `dim_multipliers`: the tuple containing each hidden layer's size divided by the dimension
    `hidden_mins`: the tuple containing each hidden layer's minimum size
    """
    drop_rate: float = 1 / 128
    activation_fn: Callable[[jax.Array], jax.Array] = nn.tanh
    dim_multipliers: Tuple[float, ...] = (2, 1, 0.5, 0.25, 0.125)
    hidden_mins: Tuple[int, ...] = (128, 64, 32, 16, 8)
    last_fn: Callable[[jax.Array], jax.Array] = lambda x: x
    dim: int = None
    
    def setup(self):
        assert len(self.dim_multipliers) > 0 and len(self.dim_multipliers) == len(self.hidden_mins)
        assert 0 <= self.drop_rate < 1
        assert all(m > 0 for m in self.dim_multipliers)
        assert all(m > 0 for m in self.hidden_mins)
    
    @nn.compact
    def __call__(self, xs: jax.Array, training: bool = True) -> jax.Array:
        """Gives the inputs transformed by the generative model

        ### Args
        `xs`: the inputs with dimension [num_parallels, *batch, dim(x)]

        ### Returns
        The transformed inputs.
        """
        assert xs.ndim >= 3
        dim = self.dim if self.dim else xs.shape[-1]
        for _mult, _min in zip(self.dim_multipliers, self.hidden_mins):
            xs = BatchDenseLayer(max(round(_mult * dim), _min))(xs)
            xs = self.activation_fn(xs)
            xs = nn.Dropout(rate=self.drop_rate, broadcast_dims=(0, ), deterministic=not training)(xs)
        xs = self.last_fn(xs)
        return xs


class MLPPredictor(nn.Module):
    """
    The MLP-based prediction model for optimization
    
    ### Attributes:
    `drop_rate`: the dropout rate
    `activation_fn`: the activation function
    `dim_multipliers`: the tuple containing each hidden layer's size divided by the dimension
    `hidden_mins`: the tuple containing each hidden layer's minimum size
    """
    drop_rate: float = 1 / 128
    activation_fn: Callable[[jax.Array], jax.Array] = nn.tanh
    dim_multipliers: Tuple[float, ...] = (2, 1, 0.5, 0.25, 0.125)
    hidden_mins: Tuple[int, ...] = (128, 64, 32, 16, 8)
    
    @nn.compact
    def __call__(self, xs: jax.Array, training: bool = True) -> jax.Array:
        """Gives the inputs transformed by the generative model

        ### Args
        `xs`: the inputs with dimension [num_parallels, *batch, dim(x)]

        ### Returns
        The transformed inputs.
        """
        xs = MLPBase(self.drop_rate, self.activation_fn, self.dim_multipliers, self.hidden_mins)(xs, training)
        xs = BatchDenseLayer(1)(xs)
        xs = xs.reshape(xs.shape[:-1])
        return xs


class MLPPredictionLoss(nn.Module):
    """
    The MLP-based prediction MSE loss for optimization
    
    ### Attributes:
    `drop_rate`: the dropout rate
    `activation_fn`: the activation function
    `dim_multipliers`: the tuple containing each hidden layer's size divided by the dimension
    `hidden_mins`: the tuple containing each hidden layer's minimum size
    """
    drop_rate: float = 1 / 128
    activation_fn: Callable[[jax.Array], jax.Array] = nn.tanh
    dim_multipliers: Tuple[float, ...] = (2, 1, 0.5, 0.25, 0.125)
    hidden_mins: Tuple[int, ...] = (128, 64, 32, 16, 8)
    
    def setup(self):
        self.net = MLPPredictor(drop_rate=self.drop_rate,
                                activation_fn=self.activation_fn,
                                dim_multipliers=self.dim_multipliers,
                                hidden_mins=self.hidden_mins)
    
    def __call__(self, xs: jax.Array, targets: jax.Array, training: bool = True) -> jax.Array:
        """Gives the MSE loss value

        ### Args
        `xs`: the inputs with dimension [num_parallels, *batch, dim(x)]
        `targets`: the targets with dimension [num_parallels, *batch]

        ### Returns
        The MSE loss.
        """
        ys = self.net(xs, training=training)
        assert ys.shape == targets.shape
        return _parallel_runs_mse(ys, targets, last_ndim=ys.ndim - 1)


class VAELoss(nn.Module):
    dim: int
    drop_rate: float = 1 / 128
    activation_fn: Callable[[jax.Array], jax.Array] = nn.tanh
    dim_multipliers: Tuple[float, ...] = (2, 1, 0.5, 0.25, 0.125)
    hidden_mins: Tuple[int, ...] = (128, 64, 32, 16, 8)
    
    def setup(self):
        self.encoder = MLPBase(drop_rate=self.drop_rate,
                               activation_fn=self.activation_fn,
                               dim_multipliers=self.dim_multipliers,
                               hidden_mins=self.hidden_mins,
                               dim=self.dim)
        self.decoder = MLPBase(drop_rate=self.drop_rate,
                               activation_fn=self.activation_fn,
                               dim_multipliers=self.dim_multipliers[-2::-1] + (1, ),
                               hidden_mins=self.hidden_mins[-2::-1] + (1, ),
                               last_fn=lambda xs: (jnp.arcsinh(xs) + 1.5) / 3,
                               dim=self.dim)
        return super().setup()
    
    def _get_nets(self):
        return self.encoder, self.decoder
    
    def __call__(self, xs: jax.Array, training: bool = True) -> jax.Array:
        encoded = self.encoder(xs, training)
        encode_size = encoded.shape[-1] // 2
        assert encode_size * 2 == encoded.shape[-1]
        mean = encoded[..., :encode_size]
        log_var = encoded[..., encode_size:]
        key = self.make_rng('dropout')
        reparameterized = jax.random.normal(key, mean.shape) * jnp.exp(0.5 * log_var) + mean
        reconstruct_xs = self.decoder(reparameterized, training)
        loss_reconstruct = _parallel_runs_mse(reconstruct_xs, xs)
        loss_KL = -0.5 * jnp.mean(jnp.sum(1 + log_var - mean**2 - jnp.exp(log_var), axis=-1), axis=-1)
        return jnp.stack([loss_reconstruct, loss_KL])
