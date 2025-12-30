import math
import torch
from torch.library import triton_op, wrap_triton

import triton
import triton.language as tl

# cSpell:words cdna autotune cdiv ptrs


@triton.jit
def cdist_dot_kernel(
        # Pointers to matrices
        a_ptr, b_ptr, v_ptr, w_ptr,
        # Matrix dimensions
        M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
        # The stride variables represent how much to increase the ptr by when moving by 1
        # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
        # by to get the element one row down (A has M rows).
        stride_am, stride_ak,  #
        stride_bn, stride_bk,  #
        # Meta-parameters
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,  #
        # DIST_TYPE: tl.constexpr,  #
        # KERNEL_TYPE: tl.constexpr,  #
):
    """Kernel for computing the cross distance dot vector w = cdist(A, B) · v
    A has shape (M, K), B has shape (N, K), v has shape (N, 1), and w has shape (M, 1)
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    # See above `L2 Cache Optimizations` section for details.
    pid = tl.program_id(axis=0)
    pid_m = pid

    # ----------------------------------------------------------
    # Initialize vector w accumulator and dim M offsets
    w_accu = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    mask_am = offs_am < M
    offs_k = tl.arange(0, int(2 ** math.ceil(math.log2(K))))
    mask_k = offs_k < K
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    all_a = tl.load(a_ptrs, mask=mask_am[:, None], other=0.0)
    
    for pid_n in range(0, tl.cdiv(N, BLOCK_SIZE_N)):
        # ----------------------------------------------------------
        # Create pointers for the first blocks of A and B.
        # We will advance this pointer as we move in the K direction
        # and accumulate
        # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
        # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
        # See above `Pointer Arithmetic` section for details
        offs_k = tl.arange(0, BLOCK_SIZE_K)
        offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        mask_bn = offs_bn < N
        b_ptrs = b_ptr + (offs_bn[:, None] * stride_bn + offs_k[None, :] * stride_bk)
        
        # -----------------------------------------------------------
        # Iterate to compute a block of the C matrix.
        # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
        # of fp32 values for higher accuracy.
        # `accumulator` will be converted back to fp16 after the loop.
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
            # Load the next block of A and B, generate a mask by checking the K dimension.
            # If it is out of bounds, set it to 0.
            mask_k = offs_k < K - k * BLOCK_SIZE_K
            b = tl.load(b_ptrs, mask=mask_bn[:, None] & mask_k[None, :], other=0.0)
            a = all_a[:, (k * BLOCK_SIZE_K):min(K, (k + 1) * BLOCK_SIZE_K)]
            # We accumulate along the K dimension.
            # TODO: other distances
            # diff = a[:, None, :] - b[None, :, :]
            # diff = tl.sum(diff * diff, axis=-1)
            a2 = tl.sum(a * a, axis=-1)[:, None]
            b2 = tl.sum(b * b, axis=-1)[None, :]
            ab = tl.dot(a, tl.trans(b, 1, 0))
            diff = tl.maximum(a2 + b2 - 2 * ab, 0)
            accumulator = accumulator + diff
            # Advance the ptrs to the next K block.
            b_ptrs += BLOCK_SIZE_K * stride_bk

        # -----------------------------------------------------------
        # Compute the kernel function
        # TODO: other kernels
        c = tl.exp(-tl.sqrt(accumulator))
        
        # -----------------------------------------------------------
        # Compute the cross distance dot vector v
        v_ptrs = v_ptr + offs_bn
        v = tl.load(v_ptrs, mask=mask_bn, other=0.0)
        w = tl.sum(c * v[None, :], axis=1)
        w_accu = w_accu + w

    # -----------------------------------------------------------
    # Write back the block of the output vector w with masks.
    w_ptrs = w_ptr + offs_am
    tl.store(w_ptrs, w_accu, mask=mask_am)
    

def cdist_dot(A: torch.Tensor, B: torch.Tensor, v: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    # Check constraints.
    assert A.ndim == 2 and B.ndim == 2 and v.ndim == 1, "Incompatible number of dimensions"
    assert A.shape[1] == B.shape[1] and B.shape[0] == v.shape[0] and v.shape == w.shape, "Incompatible dimensions"
    assert A.is_contiguous() and B.is_contiguous() and v.is_contiguous(), "Matrices and vectors must be contiguous"
    M, K = A.shape
    N, _ = B.shape
    # 1D launch kernel where each block gets its own program.
    BLOCK_SIZE_M = 16
    wrap_triton(cdist_dot_kernel)[(triton.cdiv(M, BLOCK_SIZE_M),)](
        A, B, v, w,  #
        M, N, K,  #
        A.stride(0), A.stride(1),  #
        B.stride(0), B.stride(1),  #
        BLOCK_SIZE_M, 256, 32,
    )
    return w


@torch.compile
def cdist_dot_torch(A: torch.Tensor, B: torch.Tensor, v: torch.Tensor, w: torch.Tensor):
    C = torch.cdist(A, B)
    C = torch.exp(-C.sqrt())
    torch.matmul(C, v, out=w)
    return w


if __name__ == "__main__":
    import time
    
    torch.set_default_device("cuda")
    torch.manual_seed(42)
    M = 20000
    N = 20000
    K = 100
    A = torch.ones(M, K)
    B = A + 0.01
    v = torch.ones(N) * 0.01
    w = v.clone()
    w = cdist_dot(A, B, v, w)
    print(w)
    print(cdist_dot_torch(A, B, v, w))
    A = torch.rand(M, K)
    B = torch.rand(N, K)
    v = torch.rand(N)
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(100):
        w = cdist_dot(A, B, v, w)
    torch.cuda.synchronize()
    print(time.time() - t)
    t = time.time()
    for _ in range(100):
        w = cdist_dot_torch(A, B, v, w)
    torch.cuda.synchronize()
    print(time.time() - t)