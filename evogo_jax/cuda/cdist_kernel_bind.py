from functools import partial, reduce
from typing import Tuple

import jax
import jax.numpy as jnp
from .build import gpu_ops
from jax import core, dtypes
from jax.core import ShapedArray
from jax.interpreters import mlir, xla, batching
from jax.interpreters.mlir import ir
from jax.lib import xla_client
from jaxlib.hlo_helpers import custom_call


# Create primitive for forward operation
_cdist_fwd_p = core.Primitive("cdist_fwd")
_cdist_fwd_p.multiple_results = False
_cdist_fwd_p.def_impl(partial(xla.apply_primitive, _cdist_fwd_p))

def cdist_fwd(A: jax.Array, B: jax.Array, p=2.0):
    dist = _cdist_fwd_p.bind(A, B, p=p)
    return dist, (A, B, dist)

def _cdist_fwd_batch(args, batch_axes, **kwargs):
  assert list(batch_axes) == [0, 0]
  res = _cdist_fwd_p.bind(*args, **kwargs)
  return res, 0

batching.primitive_batchers[_cdist_fwd_p] = _cdist_fwd_batch

# Create primitive for backward operation
_cdist_bwd_p = core.Primitive("cdist_bwd")
_cdist_bwd_p.multiple_results = True
_cdist_bwd_p.def_impl(partial(xla.apply_primitive, _cdist_bwd_p))

def cdist_bwd(p: float, res: Tuple[jax.Array, jax.Array, jax.Array], g: jax.Array):
    A, B, dist = res
    _, \
    grad_A, grad_B = _cdist_bwd_p.bind(A, B, dist, g, p=p)
    return grad_A, grad_B

def _cdist_bwd_batch(args, batch_axes, **kwargs):
  assert list(batch_axes) == [0, 0, 0, 0]
  outputs = _cdist_bwd_p.bind(*args, **kwargs)
  return outputs, (0, 0, 0)

batching.primitive_batchers[_cdist_bwd_p] = _cdist_bwd_batch


####################
# Lowering to MLIR #
####################


# Register functions defined in gpu_ops as custom call target for GPUs
for _name, _value in gpu_ops.get_cross_dist_registrations().items():
    xla_client.register_custom_call_target(_name, _value, platform="gpu")


def element_type_to_descriptor_type(element_type):
    _map = {
        # ir.BF16Type.get(): gpu_ops.ElementType.BF16,
        # ir.F16Type.get(): gpu_ops.ElementType.F16,
        ir.F32Type.get(): gpu_ops.ElementType.F32,
        ir.F64Type.get(): gpu_ops.ElementType.F64,
    }
    return _map.get(element_type)


def default_layouts(*shapes):
    return [range(len(shape) - 1, -1, -1) for shape in shapes]


def _cdist_fwd_cuda_lowering(ctx, A, B, p: float):
    a_type = ir.RankedTensorType(A.type)
    a_shape = a_type.shape
    b_type = ir.RankedTensorType(B.type)
    b_shape = b_type.shape
    element_type = a_type.element_type

    batches = a_shape[0] if len(a_shape) == 3 else 1
    d = a_shape[-1]
    m = a_shape[-2]
    n = b_shape[-2]
    result_shape = (batches, m, n) if len(a_shape) == 3 else (m, n)

    opaque = gpu_ops.create_cdist_descriptor(
        p, batches, d, m, n, element_type_to_descriptor_type(element_type)
    )
    out = custom_call(
        b"cross_dist_forward",
        result_types=[
            ir.RankedTensorType.get(result_shape, element_type),
        ],
        operands=[A, B],
        backend_config=opaque,
        operand_layouts=default_layouts(a_shape, b_shape),
        result_layouts=default_layouts(result_shape),
    ).results
    return out


mlir.register_lowering(
    _cdist_fwd_p,
    _cdist_fwd_cuda_lowering,
    platform="gpu",
)


def _cdist_bwd_cuda_lowering(ctx, A, B, dist, g, p: float):
    a_type = ir.RankedTensorType(A.type)
    a_shape = a_type.shape
    b_type = ir.RankedTensorType(B.type)
    b_shape = b_type.shape
    d_type = ir.RankedTensorType(dist.type)
    d_shape = d_type.shape
    element_type = a_type.element_type

    batches = a_shape[0] if len(a_shape) == 3 else 1
    d = a_shape[-1]
    m = a_shape[-2]
    n = b_shape[-2]
    temp_size = gpu_ops.get_temp_size(d, m, n)

    opaque = gpu_ops.create_cdist_descriptor(
        p, batches, d, m, n, element_type_to_descriptor_type(element_type)
    )
    out = custom_call(
        b"cross_dist_backward",
        result_types=[
            ir.RankedTensorType.get((batches, temp_size), element_type),
            ir.RankedTensorType.get(a_shape, element_type),
            ir.RankedTensorType.get(b_shape, element_type),
        ],
        operands=[A, B, dist, g],
        backend_config=opaque,
        operand_layouts=default_layouts(a_shape, b_shape, d_shape, d_shape),
        result_layouts=default_layouts(
            (batches, temp_size),
            a_shape, b_shape),
    ).results
    return out


mlir.register_lowering(
    _cdist_bwd_p,
    _cdist_bwd_cuda_lowering,
    platform="gpu",
)


#######################
# Abstract evaluation #
#######################


def _cdist_fwd_abstract(A: jax.Array, B: jax.Array, p: float):
    a_dtype = dtypes.canonicalize_dtype(A.dtype)
    b_dtype = dtypes.canonicalize_dtype(B.dtype)
    a_shape = A.shape
    b_shape = B.shape

    assert p > 0
    assert 2 <= A.ndim <= 3
    assert a_dtype == b_dtype and a_dtype in [jnp.float32, jnp.float64]
    assert A.ndim == B.ndim
    if A.ndim == 2:
        m = a_shape[0]
        n = b_shape[0]
        assert a_shape[1] == b_shape[1]
        return ShapedArray((m, n), a_dtype)
    else:
        b = a_shape[0]
        assert b == b_shape[0]
        m = a_shape[1]
        n = b_shape[1]
        assert a_shape[2] == b_shape[2]
        return ShapedArray((b, m, n), a_dtype)


_cdist_fwd_p.def_abstract_eval(_cdist_fwd_abstract)


def _cdist_bwd_abstract(
    A: jax.Array, B: jax.Array, dist: jax.Array, g: jax.Array, p: float
):
    a_dtype = dtypes.canonicalize_dtype(A.dtype)
    b_dtype = dtypes.canonicalize_dtype(B.dtype)
    d_dtype = dtypes.canonicalize_dtype(dist.dtype)
    g_dtype = dtypes.canonicalize_dtype(g.dtype)
    a_shape = A.shape
    b_shape = B.shape
    d_shape = dist.shape
    g_shape = g.shape

    assert p > 0
    assert 2 <= A.ndim <= 3
    assert (
        a_dtype == b_dtype
        and a_dtype == d_dtype
        and a_dtype == g_dtype
        and a_dtype in [jnp.float32, jnp.float64]
    )
    assert d_shape == g_shape
    assert A.ndim == B.ndim and A.ndim == dist.ndim and A.ndim == g.ndim
    if A.ndim == 2:
        b = 1
        m = a_shape[0]
        n = b_shape[0]
        d = a_shape[1]
        assert a_shape[1] == b_shape[1]
        assert d_shape == (m, n)
    else:
        b = a_shape[0]
        assert b == b_shape[0]
        m = a_shape[1]
        n = b_shape[1]
        d = a_shape[0]
        assert a_shape[2] == b_shape[2]
        assert d_shape == (b, m, n)
    temp_size = gpu_ops.get_temp_size(d, m, n)

    return (
        ShapedArray((b, temp_size), a_dtype),  # temp
        ShapedArray(a_shape, a_dtype),  # grad A
        ShapedArray(b_shape, b_dtype),  # trad B
    )


_cdist_bwd_p.def_abstract_eval(_cdist_bwd_abstract)


#######################################
# Top-level interface with custom vjp #
#######################################


@partial(jax.custom_vjp, nondiff_argnums=(2,))
def cdist(A: jax.Array, B: jax.Array, p=2.0) -> jax.Array:
    """The cross distance matrix function with custom backward function

    Args:
        A (jax.Array): The left array of size ([batches,] m, dim)
        B (jax.Array): The right array of size ([batches,] n, dim)
        p (float, optional): The power of the norm to use. Defaults to 2.0.

    Returns:
        jax.Array: The distance matrix of size ([batches,] m, n) containing the distances between each pair of A and B
    """
    output, _ = cdist_fwd(A, B, p=p)
    return output


cdist.defvjp(cdist_fwd, cdist_bwd)
