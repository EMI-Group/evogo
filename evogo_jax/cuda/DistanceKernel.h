#pragma once

#include "helpers.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cooperative_groups.h"

#include <cmath>
#include <malloc.h>
#include <cassert>
#include <iostream>
#include <limits>


#define HEADER __forceinline__ __device__

#ifndef uint
using uint = unsigned int;
#endif // !uint

enum ElementType
{
	F32,
	F64
};

size_t getTempSize(int k, int m, int n);
void crossDist_forward(cudaStream_t stream, void **buffers, const char *opaque, size_t opaque_len);
void crossDist_backward(cudaStream_t stream, void **buffers, const char *opaque, size_t opaque_len);

struct DistDescriptor
{
	double p;
	int batches, dim, lenA, lenB;
	ElementType type;

	DistDescriptor(double p, int batches, int dim, int lenA, int lenB, const ElementType &type)
		: p(p), batches(batches), dim(dim), lenA(lenA), lenB(lenB), type(type)
	{
	}
};

template <typename T>
__host__ __device__ inline T copy_sign(const T number, const T sign)
{
#ifdef _MSC_VER
	return std::abs(number) * T(sign >= T(0) ? 1 : -1);
#else
	return std::copysign(number, sign);
#endif // _MSC_VER
}

template <typename T>
__host__ __device__ inline bool is_inf(const T number)
{
#ifdef _MSC_VER
	return std::abs(number) == std::numeric_limits<T>::infinity();
#else
	return std::isinf(number);
#endif // _MSC_VER
}

template <typename T>
__host__ __device__ inline bool is_nan(const T number)
{
#ifdef _MSC_VER
	return !(number == number);
#else
	return std::isnan(number);
#endif // _MSC_VER
}

template <typename F, typename... Args>
void for_each_argument_address(F f, Args &&...args)
{
	[](...) {}((f((void *)&std::forward<Args>(args)), 0)...);
}

template <typename... KernelParameters>
inline void cooperative_launch(const void* kernel_function,
							   dim3 grid_dim, dim3 block_dim, size_t sharedMem_size, cudaStream_t stream_id,
							   KernelParameters... parameters)
{
	void *arguments_ptrs[sizeof...(KernelParameters)];
	auto arg_index = sizeof...(KernelParameters) - 1;
	for_each_argument_address([&](void *x)
							  { arguments_ptrs[arg_index--] = x; },
							  parameters...);
	cudaLaunchCooperativeKernel(kernel_function,
								grid_dim, block_dim,
								arguments_ptrs,
								sharedMem_size, stream_id);
}