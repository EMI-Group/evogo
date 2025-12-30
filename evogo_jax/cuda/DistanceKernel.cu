#include "DistanceKernel.h"

template <typename T>
struct dists
{
	// General p norm
	struct p
	{
		static HEADER void inc(T &agg, const T diff, const T p)
		{
			agg += std::pow(std::abs(diff), p);
		}
		static HEADER void add(T &agg, const T otherAgg, const T p)
		{
			agg += otherAgg;
		}
		static HEADER T finish(const T agg, const T p)
		{
			return std::pow(agg, static_cast<T>(1) / p);
		}
		static HEADER T backward(const T diff, const T grad, const T dist, const T p)
		{
			return dist == 0 ? 0 : copy_sign(std::pow(std::abs(diff) / dist, p - 1) * grad, diff);
		}
	};

	// One norm
	struct one
	{
		static HEADER void inc(T &agg, const T diff, const T p)
		{
			agg += std::abs(diff);
		}
		static HEADER void add(T &agg, const T otherAgg, const T p)
		{
			agg += otherAgg;
		}
		static HEADER T finish(const T agg, const T p)
		{
			return agg;
		}
		static HEADER T backward(const T diff, const T grad, const T dist, const T p)
		{
			return grad * ((0 < diff) - (diff < 0)); // grad * sign(diff)
		}
	};

	// Special case backward when p is less than two
	struct lt_two
	{
		static HEADER T backward(const T diff, const T grad, const T dist, const T p)
		{
			return (dist == T{0} || (diff == T{0} && p < static_cast<T>(1))) ? T{0} : copy_sign(std::pow(std::abs(diff) / dist, p - 1), diff) * grad;
		}
	};

	// Two norm
	struct two
	{
		static HEADER void inc(T &agg, const T diff, const T p)
		{
			agg += diff * diff;
		}
		static HEADER void add(T &agg, const T otherAgg, const T p)
		{
			agg += otherAgg;
		}
		static HEADER T finish(const T agg, const T p)
		{
			return std::sqrt(agg);
		}
		static HEADER T backward(const T diff, const T grad, const T dist, const T p)
		{
			return dist == T{0} ? T{0} : grad * diff / dist;
		}
	};

	// Inf norm
	struct inf
	{
		static HEADER void inc(T &agg, const T diff, const T p)
		{
			if (std::abs(diff) > agg)
				agg = std::abs(diff);
		}
		static HEADER void add(T &agg, const T otherAgg, const T p)
		{
			if (otherAgg > agg)
				agg = otherAgg;
		}
		static HEADER T finish(const T agg, const T p)
		{
			return agg;
		}
		static HEADER T backward(const T diff, const T grad, const T dist, const T p)
		{
			return (std::abs(diff) == dist) * diff * grad;
		}
	};
};

template <typename T, uint ShareLd, uint Repeats, bool Transpose>
HEADER static void loadToShared(T *const shared, const T *const global, const T *const end, const uint ld, const bool noRest, const bool otherDimExceed)
{
	uint fromLD;
	if constexpr (Transpose)
	{
		fromLD = ld;
	}
	else
	{
		fromLD = 1;
	}
	if (noRest)
	{
#pragma unroll
		for (uint i = 0; i < Repeats; ++i)
		{
			shared[i * ShareLd] = global[i * ShareLd * fromLD];
		}
	}
	else
	{
#pragma unroll
		for (uint i = 0; i < Repeats; ++i)
		{
			shared[i * ShareLd] = (!otherDimExceed && (global + i * ShareLd * fromLD < end)) ? global[i * ShareLd * fromLD] : T{0};
		}
	}
}

template <typename T, uint Size, uint ZOffset, typename Dist>
HEADER static void saveDistToGlobal(const T (&local)[Size][Size], T *const global, const uint rows, const uint cols, const uint rowIdx, const uint colIdx, const bool noRest, const T p)
{
	if (noRest)
	{
#pragma unroll
		for (int i = 0; i < Size / 2; ++i)
		{
#pragma unroll
			for (int j = 0; j < Size / 2; ++j)
			{
				global[i * cols + j] = Dist::finish(local[i][j], p);
				global[i * cols + j + ZOffset] = Dist::finish(local[i][j + Size / 2], p);
				global[(i + ZOffset) * cols + j] = Dist::finish(local[i + Size / 2][j], p);
				global[(i + ZOffset) * cols + j + ZOffset] = Dist::finish(local[i + Size / 2][j + Size / 2], p);
			}
		}
	}
	else
	{
#pragma unroll
		for (int i = 0; i < Size / 2; ++i)
		{
#pragma unroll
			for (int j = 0; j < Size / 2; ++j)
			{
				if (rowIdx + i < rows && colIdx + j < cols)
					global[i * cols + j] = Dist::finish(local[i][j], p);
				if (rowIdx + i < rows && colIdx + j + ZOffset < cols)
					global[i * cols + j + ZOffset] = Dist::finish(local[i][j + Size / 2], p);
				if (rowIdx + i + ZOffset < rows && colIdx + j < cols)
					global[(i + ZOffset) * cols + j] = Dist::finish(local[i + Size / 2][j], p);
				if (rowIdx + i + ZOffset < rows && colIdx + j + ZOffset < cols)
					global[(i + ZOffset) * cols + j + ZOffset] = Dist::finish(local[i + Size / 2][j + Size / 2], p);
			}
		}
	}
}

template <typename T, uint Size>
HEADER static void saveToGlobal(const T (&local)[Size][Size], T *const global, const uint rows, const uint cols, const uint rowIdx, const uint colIdx, const bool noRest)
{
	if (noRest)
	{
#pragma unroll
		for (int i = 0; i < Size; ++i)
		{
#pragma unroll
			for (int j = 0; j < Size; ++j)
			{
				global[i * cols + j] = local[i][j];
			}
		}
	}
	else
	{
#pragma unroll
		for (int i = 0; i < Size; ++i)
		{
#pragma unroll
			for (int j = 0; j < Size; ++j)
			{
				if (rowIdx + i < rows && colIdx + j < cols)
					global[i * cols + j] = local[i][j];
			}
		}
	}
}

template <typename T, uint Size>
HEADER static void loadFromGlobal(T (&local)[Size][Size], const T *global, const uint rows, const uint cols, const uint rowIdx, const uint colIdx, const bool noRest, const T empty = T{0})
{
	if (noRest)
	{
#pragma unroll
		for (int i = 0; i < Size; ++i)
		{
#pragma unroll
			for (int j = 0; j < Size; ++j)
			{
				local[i][j] = global[i * cols + j];
			}
		}
	}
	else
	{
#pragma unroll
		for (int i = 0; i < Size; ++i)
		{
#pragma unroll
			for (int j = 0; j < Size; ++j)
			{
				local[i][j] = rowIdx + i < rows && colIdx + j < cols ? global[i * cols + j] : empty;
			}
		}
	}
}

constexpr uint WARP_SIZE = 32;

#ifndef __launch_bounds__
#define __launch_bounds__(...) __annotate__(launch_bounds(__VA_ARGS__))
#endif // !__launch_bounds__
#ifndef __CUDACC__
// Ensure Visual Studio recognizes CUDA syntax for __syncthreads()
extern "C" void __syncthreads();
#endif // !__CUDACC__

/*
A: M x K,  B: N x K,  out: M x N  (all row major)
Possible ThreadsPerBlock: 64, 256
	for the maximum allowed shared memory size for 128KiB shared memory per SM (Ampere+)
*/
template <typename T, typename Dist, int ThreadsPerBlock>
__global__ __launch_bounds__(ThreadsPerBlock, 512 / ThreadsPerBlock) void crossDistances(const int m, const int n, const int k, const T p, const T *A, const T *B, T *C)
{
	constexpr uint SMemCol = ThreadsPerBlock / WARP_SIZE, SMemRow = sqrtint_constexpr(ThreadsPerBlock) * SMemCol;
	constexpr uint SMemRowWarps = SMemRow / WARP_SIZE;
	constexpr uint SMemLD = (SMemRow + WARP_SIZE - 1) / WARP_SIZE * WARP_SIZE;

	const uint restK = k - k / SMemCol * SMemCol, restRowA = m - m / SMemRow * SMemRow, restRowB = n - n / SMemRow * SMemRow;
	const uint mDiv = m / SMemRow, nDiv = n / SMemRow, kDiv = k / SMemCol;

	const uint ty = threadIdx.x / WARP_SIZE, tx = threadIdx.x % WARP_SIZE; // ty: 0 ~ SCols - 1,   tid: 0 ~ 31
	constexpr uint SMemRowColDiv = SMemRow / SMemCol, ZOffset = SMemRowWarps * SMemRowColDiv;
	const uint smemOffA = (threadIdx.x / SMemRowColDiv) * SMemRowWarps;
	const uint smemOffB = (threadIdx.x % SMemRowColDiv) * SMemRowWarps;

	__shared__ __align__(16 * 1024) T shareA[SMemLD * SMemCol]; // row major shared memory for A
	__shared__ __align__(16 * 1024) T shareB[SMemLD * SMemCol]; // row major shared memory for B
	T dist[SMemCol][SMemCol] = {0};
	T panelA[SMemCol] = {0}, panelB[SMemCol] = {0};

	const uint fromOffsetA = blockIdx.y * SMemRow + tx, fromOffsetB = blockIdx.x * SMemRow + tx;
	const T *fromA = A + blockIdx.z * m * k + fromOffsetA * k + ty, *endA = A + (blockIdx.z + 1) * m * k;
	const T *fromB = B + blockIdx.z * n * k + fromOffsetB * k + ty, *endB = B + (blockIdx.z + 1) * n * k;
	const uint toSMemOffset = ty * SMemLD + tx;
	const uint outRowOffset = blockIdx.y * SMemRow + smemOffA, outColOffset = blockIdx.x * SMemRow + smemOffB;

	for (uint outerK = 0; outerK < (k + SMemCol - 1) / SMemCol; ++outerK)
	{
		// part1: global mem to shared mem
		const bool noRestK = (!restK || outerK < kDiv || (outerK == kDiv && ty < restK)), kExceed = outerK * SMemCol + ty >= k;
		loadToShared<T, WARP_SIZE, SMemRowWarps, true>(shareA + toSMemOffset, fromA, endA, k, noRestK && (!restRowA || blockIdx.y < mDiv), kExceed);
		loadToShared<T, WARP_SIZE, SMemRowWarps, true>(shareB + toSMemOffset, fromB, endB, k, noRestK && (!restRowB || blockIdx.x < nDiv), kExceed);
		fromA += SMemCol;
		fromB += SMemCol;
		__syncthreads();

		// part2: calculate via outer product
		// (SMemRowColDiv x SMemRowColDiv) threads, (SCols x SCols) per thread
		if (!restK || outerK < kDiv)
		{
#pragma unroll
			for (uint subK = 0; subK < SMemCol; ++subK)
			{
				const T *ptrA = shareA + subK * SMemLD + smemOffA;
				const T *ptrB = shareB + subK * SMemLD + smemOffB;
#pragma unroll
				// compile to two 128-bit (16B) word read, 1/4 warp per phase
				for (int i = 0; i < SMemRowWarps; ++i)
				{
					panelA[i] = ptrA[i];
					panelA[i + SMemRowWarps] = ptrA[i + ZOffset];
				}
#pragma unroll
				// compile to two 128-bit (16B) word read, 1/4 warp per phase
				for (int i = 0; i < SMemRowWarps; ++i)
				{
					panelB[i] = ptrB[i];
					panelB[i + SMemRowWarps] = ptrB[i + ZOffset];
				}
				// outer product
#pragma unroll
				for (int i = 0; i < SMemCol; ++i)
				{
#pragma unroll
					for (int j = 0; j < SMemCol; ++j)
					{
						Dist::inc(dist[i][j], panelA[i] - panelB[j], p);
					}
				}
			}
		}
		else
		{
			for (uint subK = 0; subK < restK; ++subK)
			{
				const T *ptrA = shareA + subK * SMemLD + smemOffA;
				const T *ptrB = shareB + subK * SMemLD + smemOffB;
#pragma unroll
				// compile to two 128-bit (16B) word read, 1/4 warp per phase
				for (int i = 0; i < SMemRowWarps; ++i)
				{
					panelA[i] = ptrA[i];
					panelA[i + SMemRowWarps] = ptrA[i + ZOffset];
				}
#pragma unroll
				// compile to two 128-bit (16B) word read, 1/4 warp per phase
				for (int i = 0; i < SMemRowWarps; ++i)
				{
					panelB[i] = ptrB[i];
					panelB[i + SMemRowWarps] = ptrB[i + ZOffset];
				}
				// outer product
#pragma unroll
				for (int i = 0; i < SMemCol; ++i)
				{
#pragma unroll
					for (int j = 0; j < SMemCol; ++j)
					{
						Dist::inc(dist[i][j], panelA[i] - panelB[j], p);
					}
				}
			}
		}
		__syncthreads();
	}

	// part3: save
	T *const distOut = C + blockIdx.z * m * n + outRowOffset * n + outColOffset;
	const bool noRestOut = (outRowOffset + ZOffset + SMemCol / 2 < m) && (outColOffset + ZOffset + SMemCol / 2 < n);
	saveDistToGlobal<T, SMemCol, ZOffset, Dist>(dist, distOut, m, n, outRowOffset, outColOffset, noRestOut, p);
}

template <typename T, typename Dist, uint ThreadsPerBlock, uint BlockSizeK>
__global__ __launch_bounds__(ThreadsPerBlock, 1) void crossDistances_backwardA_cg(const uint blocksN, const uint m, const uint n, const uint k, const T p, const T *A, const T *B, const T *C, const T *gradC, T *gradA)
{
	constexpr uint SCols = BlockSizeK / 8, BlockSizeM = 8192 / BlockSizeK;
	constexpr uint OutputLd = SCols * 2, BlockSizeN = BlockSizeK * 4, OutSize = 4;
	constexpr uint ReadPerThreadC = BlockSizeM * SCols / ThreadsPerBlock, ReadPerThreadB = BlockSizeK * SCols / ThreadsPerBlock;
	constexpr uint BlockMReadLd = BlockSizeM / ReadPerThreadC, BlockKReadLd = ReadPerThreadB >= 1 ? BlockSizeK / ReadPerThreadB : BlockSizeK;
	cooperative_groups::grid_group gg = cooperative_groups::this_grid();

	const uint bid = blockIdx.z;
	const uint tn = threadIdx.x / BlockMReadLd, tm = threadIdx.x % BlockMReadLd; // tn: 0~SCols-1, tm: 0~BlockSizeN/2-1
	const uint bm = blockIdx.y, bk = blockIdx.x;
	uint tnB, tkB;
	if constexpr (ReadPerThreadB >= 1)
	{
		tnB = threadIdx.x / (BlockSizeK / ReadPerThreadB);
		tkB = threadIdx.x % (BlockSizeK / ReadPerThreadB);
	}
	else
	{
		tnB = threadIdx.x / BlockSizeK;
		tkB = threadIdx.x % BlockSizeK;
	}
	const uint smemOffC = (threadIdx.x / OutputLd) * OutSize;
	const uint smemOffB = (threadIdx.x % OutputLd) * OutSize;
	const uint outRowOffset = bm * BlockSizeM + smemOffC, outColOffset = bk * BlockSizeK + smemOffB;
	const bool noRestOut = (outRowOffset + OutSize < m) && (outColOffset + OutSize < k);

	__shared__ __align__(16 * 1024) T shareC[BlockSizeM * SCols];	 // row major shared memory for C
	__shared__ __align__(16 * 1024) T shareGrad[BlockSizeM * SCols]; // row major shared memory for gradC
	__shared__ __align__(16 * 1024) T shareB[BlockSizeK * SCols];	 // row major shared memory for B

	T grads[OutSize][OutSize] = {0}, gradsTemp[OutSize][OutSize] = {0}, orgA[OutSize][OutSize] = {0};
	T panelC[OutSize] = {0}, panelGrad[OutSize] = {0}, panelB[OutSize] = {0};

	// part0: load A
	const T *fromA = A + (bid * m * k) + outRowOffset * k + outColOffset;
	loadFromGlobal<T, OutSize>(orgA, fromA, m, k, outRowOffset, outColOffset, noRestOut);

	for (uint bn = 0; bn < blocksN; bn++)
	{
		const uint fromOffsetRowC = bm * BlockSizeM + tm, fromOffsetColC = tn + bn * BlockSizeN;
		const uint fromOffsetRowB = tnB + bn * BlockSizeN, fromOffsetColB = bk * BlockSizeK + tkB;
		const uint toSMemCOffset = tn * BlockSizeM + tm;
		const uint toSMemBOffset = tnB * BlockSizeK + tkB;
		const T *fromC = C + (bid * m * n) + fromOffsetRowC * n + fromOffsetColC, *endC = C + (bid + 1) * m * n;
		const T *fromGrad = gradC + (bid * m * n) + fromOffsetRowC * n + fromOffsetColC, *endG = gradC + (bid + 1) * m * n;
		const T *fromB = B + (bid * n * k) + fromOffsetRowB * k + fromOffsetColB, *endB = B + (bid * n * k) + (fromOffsetRowB + 1) * k;
#pragma unroll
		for (int i = 0; i < OutSize; ++i)
		{
#pragma unroll
			for (int j = 0; j < OutSize; ++j)
			{
				gradsTemp[i][j] = 0;
			}
		}

#pragma unroll
		for (uint outerN = 0; outerN < BlockSizeN / SCols; ++outerN)
		{
			// part1: global mem to shared mem
			const bool noRestC = (fromOffsetRowC + (ReadPerThreadC - 1) * BlockMReadLd < m) && (bn < blocksN - 1);
			const bool cExceed = fromOffsetColC + outerN * SCols >= n, bExceed = fromOffsetRowB + outerN * SCols >= n;
			loadToShared<T, BlockMReadLd, ReadPerThreadC, true>(shareC + toSMemCOffset, fromC, endC, n, noRestC, cExceed);
			loadToShared<T, BlockMReadLd, ReadPerThreadC, true>(shareGrad + toSMemCOffset, fromGrad, endG, n, noRestC, cExceed);
			if constexpr (ReadPerThreadB >= 1)
			{
				const bool noRestB = (fromOffsetColB + (ReadPerThreadB - 1) * BlockKReadLd < k) && (bn < blocksN - 1);
				loadToShared<T, BlockKReadLd, ReadPerThreadB, false>(shareB + toSMemBOffset, fromB, endB, k, noRestB, bExceed);
			}
			else
			{
				if (tnB < SCols)
				{
					shareB[toSMemBOffset] = fromB < endB ? fromB[0] : T{0};
				}
			}
			fromC += SCols;
			fromGrad += SCols;
			fromB += SCols * k;
			endB += SCols * k;
			if (endB >= (B + (bid + 1) * n * k))
			{
				endB = B + (bid + 1) * n * k;
			}
			__syncthreads();

			// part2: calculate via outer product
			// (512/OutputLd x OutputLd) threads, (OutSize x OutSize) per thread
#pragma unroll
			for (int subN = 0; subN < SCols; ++subN)
			{
				const T *ptrC = shareC + subN * BlockSizeM + smemOffC;
				const T *ptrGC = shareGrad + subN * BlockSizeM + smemOffC;
				const T *ptrB = shareB + subN * BlockSizeK + smemOffB;
#pragma unroll
				for (int i = 0; i < OutSize; ++i)
				{
					panelC[i] = ptrC[i];
				}
#pragma unroll
				for (int i = 0; i < OutSize; ++i)
				{
					panelGrad[i] = ptrGC[i];
				}
#pragma unroll
				for (int i = 0; i < OutSize; ++i)
				{
					panelB[i] = ptrB[i];
				}
				// outer product
#pragma unroll
				for (int i = 0; i < OutSize; ++i)
				{
#pragma unroll
					for (int j = 0; j < OutSize; ++j)
					{
						gradsTemp[i][j] += Dist::backward(orgA[i][j] - panelB[j], panelGrad[i], panelC[i], p);
					}
				}
			}
			__syncthreads();
		}
#pragma unroll
		for (int i = 0; i < OutSize; ++i)
		{
#pragma unroll
			for (int j = 0; j < OutSize; ++j)
			{
				grads[i][j] += gradsTemp[i][j];
			}
		}
		// grid sync
		gg.sync();
	}
	// part3: save
	T *const gradOut = gradA + bid * m * k + outRowOffset * k + outColOffset;
	saveToGlobal(grads, gradOut, m, k, outRowOffset, outColOffset, noRestOut);
}

template <typename T, typename Dist, uint ThreadsPerBlock, uint BlockSizeK>
__global__ __launch_bounds__(ThreadsPerBlock, 1) void crossDistances_backwardB_cg(const uint blocksM, const uint m, const uint n, const uint k, const T p, const T *A, const T *B, const T *C, const T *gradC, T *gradB)
{
	constexpr uint SRows = BlockSizeK / 8, BlockSizeN = 8192 / BlockSizeK;
	constexpr uint OutputLd = SRows * 2, BlockSizeM = BlockSizeK * 4, OutSize = 4;
	constexpr uint ReadPerThreadC = BlockSizeN * SRows / ThreadsPerBlock, ReadPerThreadA = BlockSizeK * SRows / ThreadsPerBlock;
	constexpr uint BlockNReadLd = BlockSizeN / ReadPerThreadC, BlockKReadLd = ReadPerThreadA >= 1 ? BlockSizeK / ReadPerThreadA : BlockSizeK;
	cooperative_groups::grid_group gg = cooperative_groups::this_grid();

	const uint bid = blockIdx.z;
	const uint tm = threadIdx.x / BlockNReadLd, tn = threadIdx.x % BlockNReadLd; // tn: 0~SRows-1, tm: 0~BlockSizeN/2-1
	uint tmA, tkA;
	if constexpr (ReadPerThreadA >= 1)
	{
		tmA = threadIdx.x / (BlockSizeK / ReadPerThreadA);
		tkA = threadIdx.x % (BlockSizeK / ReadPerThreadA);
	}
	else
	{
		tmA = threadIdx.x / BlockSizeK;
		tkA = threadIdx.x % BlockSizeK;
	}
	const uint bn = blockIdx.y, bk = blockIdx.x;
	const uint smemOffC = (threadIdx.x / OutputLd) * OutSize;
	const uint smemOffA = (threadIdx.x % OutputLd) * OutSize;
	const uint outRowOffset = bn * BlockSizeN + smemOffC, outColOffset = bk * BlockSizeK + smemOffA;
	const bool noRestOut = (outRowOffset + OutSize < n) && (outColOffset + OutSize < k);

	__shared__ __align__(16 * 1024) T shareC[BlockSizeN * SRows];	 // row major shared memory for C
	__shared__ __align__(16 * 1024) T shareGrad[BlockSizeN * SRows]; // row major shared memory for gradC
	__shared__ __align__(16 * 1024) T shareA[BlockSizeK * SRows];	 // row major shared memory for A
	T grads[OutSize][OutSize] = {0}, gradsTemp[OutSize][OutSize] = {0}, orgB[OutSize][OutSize] = {0};
	T panelC[OutSize] = {0}, panelGrad[OutSize] = {0}, panelA[OutSize] = {0};

	// part0: load B
	const T* fromB = B + (bid * n * k) + outRowOffset * k + outColOffset;
	loadFromGlobal<T, OutSize>(orgB, fromB, n, k, outRowOffset, outColOffset, noRestOut);

	for (uint bm = 0; bm < blocksM; bm++)
	{
#pragma unroll
		for (int i = 0; i < OutSize; ++i)
		{
#pragma unroll
			for (int j = 0; j < OutSize; ++j)
			{
				gradsTemp[i][j] = 0;
			}
		}

		const uint fromOffsetColC = bn * BlockSizeN + tn, fromOffsetRowC = tm + bm * BlockSizeM;
		const uint fromOffsetRowA = tmA + bm * BlockSizeM, fromOffsetColA = bk * BlockSizeK + tkA;
		const uint toSMemCOffset = tm * BlockSizeN + tn;
		const uint toSMemAOffset = tmA * BlockSizeK + tkA;
		const T *fromC = C + (bid * m * n) + fromOffsetRowC * n + fromOffsetColC, *endC = C + (bid * m * n) + (fromOffsetRowC + 1) * n;
		const T *fromGrad = gradC + (bid * m * n) + fromOffsetRowC * n + fromOffsetColC, *endG = gradC + (bid * m * n) + (fromOffsetRowC + 1) * n;
		const T *fromA = A + (bid * m * k) + fromOffsetRowA * k + fromOffsetColA, *endA = A + (bid * m * k) + (fromOffsetRowA + 1) * k;

#pragma unroll
		for (uint outerM = 0; outerM < BlockSizeM / SRows; ++outerM)
		{
			// part1: global mem to shared mem
			const bool noRestC = (fromOffsetColC + (ReadPerThreadC - 1) * BlockNReadLd < n) && (bm < blocksM - 1);
			const bool cExceed = fromOffsetRowC + outerM * SRows >= m, aExceed = fromOffsetRowA + outerM * SRows >= m;
			loadToShared<T, BlockNReadLd, ReadPerThreadC, false>(shareC + toSMemCOffset, fromC, endC, n, noRestC, cExceed);
			loadToShared<T, BlockNReadLd, ReadPerThreadC, false>(shareGrad + toSMemCOffset, fromGrad, endG, n, noRestC, cExceed);
			if constexpr (ReadPerThreadA >= 1)
			{
				const bool noRestA = (fromOffsetColA + (ReadPerThreadA - 1) * BlockKReadLd < k) && (bm < blocksM - 1);
				loadToShared<T, BlockKReadLd, ReadPerThreadA, false>(shareA + toSMemAOffset, fromA, endA, k, noRestA, aExceed);
			}
			else
			{
				if (tmA < SRows)
				{
					shareA[toSMemAOffset] = fromA < endA ? fromA[0] : T{0};
				}
			}
			fromC += SRows * n;
			fromGrad += SRows * n;
			endC += SRows * n;
			endG += SRows * n;
			fromA += SRows * k;
			endA += SRows * k;
			if (endC >= (C + (bid + 1) * m * n))
			{
				endC = C + (bid + 1) * m * n;
				endG = gradC + (bid + 1) * m * n;
			}
			if (endA >= (A + (bid + 1) * m * k))
			{
				endA = A + (bid + 1) * m * k;
			}
			__syncthreads();

			// part2: calculate via outer product
			// (512/OutputLd x OutputLd) threads, (OutSize x OutSize) per thread
#pragma unroll
			for (int subM = 0; subM < SRows; ++subM)
			{
				const T *ptrC = shareC + subM * BlockSizeN + smemOffC;
				const T *ptrGC = shareGrad + subM * BlockSizeN + smemOffC;
				const T *ptrA = shareA + subM * BlockSizeK + smemOffA;
#pragma unroll
				for (int i = 0; i < OutSize; ++i)
				{
					panelC[i] = ptrC[i];
				}
#pragma unroll
				for (int i = 0; i < OutSize; ++i)
				{
					panelGrad[i] = ptrGC[i];
				}
#pragma unroll
				for (int i = 0; i < OutSize; ++i)
				{
					panelA[i] = ptrA[i];
				}
				// outer product
#pragma unroll
				for (int i = 0; i < OutSize; ++i)
				{
#pragma unroll
					for (int j = 0; j < OutSize; ++j)
					{
						gradsTemp[i][j] += Dist::backward(orgB[i][j] - panelA[j], panelGrad[i], panelC[i], p);
					}
				}
			}
			__syncthreads();
		}
#pragma unroll
		for (int i = 0; i < OutSize; ++i)
		{
#pragma unroll
			for (int j = 0; j < OutSize; ++j)
			{
				grads[i][j] += gradsTemp[i][j];
			}
		}
		gg.sync();
	}
	// part3: save
	T *const gradOut = gradB + bid * n * k + outRowOffset * k + outColOffset;
	saveToGlobal(grads, gradOut, n, k, outRowOffset, outColOffset, noRestOut);
}

template <typename T, typename Dist, uint ThreadsPerBlock, uint BlockSizeK>
__global__ __launch_bounds__(ThreadsPerBlock, 1) void crossDistances_backwardA(const uint b, const uint m, const uint n, const uint k, const T p, const T *A, const T *B, const T *C, const T *gradC, T *gradA)
{
	constexpr uint SCols = BlockSizeK / 8, BlockSizeM = 8192 / BlockSizeK;
	constexpr uint OutputLd = SCols * 2, BlockSizeN = BlockSizeK * 4, OutSize = 4;
	constexpr uint ReadPerThreadC = BlockSizeM * SCols / ThreadsPerBlock, ReadPerThreadB = BlockSizeK * SCols / ThreadsPerBlock;
	constexpr uint BlockMReadLd = BlockSizeM / ReadPerThreadC, BlockKReadLd = ReadPerThreadB >= 1 ? BlockSizeK / ReadPerThreadB : BlockSizeK;

	const uint blocksN = gridDim.z / b, bid = blockIdx.z / blocksN;
	const uint tn = threadIdx.x / BlockMReadLd, tm = threadIdx.x % BlockMReadLd; // tn: 0~SCols-1, tm: 0~BlockSizeN/2-1
	uint tnB, tkB;
	if constexpr (ReadPerThreadB >= 1)
	{
		tnB = threadIdx.x / (BlockSizeK / ReadPerThreadB);
		tkB = threadIdx.x % (BlockSizeK / ReadPerThreadB);
	}
	else
	{
		tnB = threadIdx.x / BlockSizeK;
		tkB = threadIdx.x % BlockSizeK;
	}
	const uint bn = blockIdx.z % blocksN, bm = blockIdx.y, bk = blockIdx.x;
	const uint smemOffC = (threadIdx.x / OutputLd) * OutSize;
	const uint smemOffB = (threadIdx.x % OutputLd) * OutSize;

	__shared__ __align__(16 * 1024) T shareC[BlockSizeM * SCols];	 // row major shared memory for C
	__shared__ __align__(16 * 1024) T shareGrad[BlockSizeM * SCols]; // row major shared memory for gradC
	__shared__ __align__(16 * 1024) T shareB[BlockSizeK * SCols];	 // row major shared memory for B
	T grads[OutSize][OutSize] = {0}, gradsTemp[OutSize][OutSize] = {0}, orgA[OutSize][OutSize] = {0};
	T panelC[OutSize] = {0}, panelGrad[OutSize] = {0}, panelB[OutSize] = {0};

	const uint fromOffsetRowC = bm * BlockSizeM + tm, fromOffsetColC = tn + bn * BlockSizeN;
	const uint fromOffsetRowB = tnB + bn * BlockSizeN, fromOffsetColB = bk * BlockSizeK + tkB;
	const uint toSMemCOffset = tn * BlockSizeM + tm;
	const uint toSMemBOffset = tnB * BlockSizeK + tkB;
	const uint outRowOffset = bm * BlockSizeM + smemOffC, outColOffset = bk * BlockSizeK + smemOffB;
	const T *fromC = C + (bid * m * n) + fromOffsetRowC * n + fromOffsetColC, *endC = C + (bid + 1) * m * n;
	const T *fromGrad = gradC + (bid * m * n) + fromOffsetRowC * n + fromOffsetColC, *endG = gradC + (bid + 1) * m * n;
	const T *fromB = B + (bid * n * k) + fromOffsetRowB * k + fromOffsetColB, *endB = B + (bid * n * k) + (fromOffsetRowB + 1) * k;
	const T *fromA = A + (bid * m * k) + outRowOffset * k + outColOffset;

	// part0: load A
	const bool noRestOut = (outRowOffset + OutSize < m) && (outColOffset + OutSize < k);
	loadFromGlobal<T, OutSize>(orgA, fromA, m, k, outRowOffset, outColOffset, noRestOut);

#pragma unroll
	for (uint outerN = 0; outerN < BlockSizeN / SCols; ++outerN)
	{
		// part1: global mem to shared mem
		const bool noRestC = (fromOffsetRowC + (ReadPerThreadC - 1) * BlockMReadLd < m) && (bn < blocksN - 1);
		const bool cExceed = fromOffsetColC + outerN * SCols >= n, bExceed = fromOffsetRowB + outerN * SCols >= n;
		loadToShared<T, BlockMReadLd, ReadPerThreadC, true>(shareC + toSMemCOffset, fromC, endC, n, noRestC, cExceed);
		loadToShared<T, BlockMReadLd, ReadPerThreadC, true>(shareGrad + toSMemCOffset, fromGrad, endG, n, noRestC, cExceed);
		if constexpr (ReadPerThreadB >= 1)
		{
			const bool noRestB = (fromOffsetColB + (ReadPerThreadB - 1) * BlockKReadLd < k) && (bn < blocksN - 1);
			loadToShared<T, BlockKReadLd, ReadPerThreadB, false>(shareB + toSMemBOffset, fromB, endB, k, noRestB, bExceed);
		}
		else
		{
			if (tnB < SCols)
			{
				shareB[toSMemBOffset] = fromB < endB ? fromB[0] : T{0};
			}
		}
		fromC += SCols;
		fromGrad += SCols;
		fromB += SCols * k;
		endB += SCols * k;
		if (endB >= (B + (bid + 1) * n * k))
		{
			endB = B + (bid + 1) * n * k;
		}
		__syncthreads();

		// part2: calculate via outer product
		// (512/OutputLd x OutputLd) threads, (OutSize x OutSize) per thread
#pragma unroll
		for (int i = 0; i < OutSize; ++i)
		{
#pragma unroll
			for (int j = 0; j < OutSize; ++j)
			{
				gradsTemp[i][j] = 0;
			}
		}
#pragma unroll
		for (int subN = 0; subN < SCols; ++subN)
		{
			const T *ptrC = shareC + subN * BlockSizeM + smemOffC;
			const T *ptrGC = shareGrad + subN * BlockSizeM + smemOffC;
			const T *ptrB = shareB + subN * BlockSizeK + smemOffB;
#pragma unroll
			for (int i = 0; i < OutSize; ++i)
			{
				panelC[i] = ptrC[i];
			}
#pragma unroll
			for (int i = 0; i < OutSize; ++i)
			{
				panelGrad[i] = ptrGC[i];
			}
#pragma unroll
			for (int i = 0; i < OutSize; ++i)
			{
				panelB[i] = ptrB[i];
			}
			// outer product
#pragma unroll
			for (int i = 0; i < OutSize; ++i)
			{
#pragma unroll
				for (int j = 0; j < OutSize; ++j)
				{
					gradsTemp[i][j] += Dist::backward(orgA[i][j] - panelB[j], panelGrad[i], panelC[i], p);
				}
			}
		}
#pragma unroll
		for (int i = 0; i < OutSize; ++i)
		{
#pragma unroll
			for (int j = 0; j < OutSize; ++j)
			{
				grads[i][j] += gradsTemp[i][j];
			}
		}
		__syncthreads();
	}

	// part3: save
	T *const gradOut = gradA + bid * m * k * blocksN + bn * m * k + outRowOffset * k + outColOffset;
	saveToGlobal(grads, gradOut, m, k, outRowOffset, outColOffset, noRestOut);
}

template <typename T, typename Dist, uint ThreadsPerBlock, uint BlockSizeK>
__global__ __launch_bounds__(ThreadsPerBlock, 1) void crossDistances_backwardB(const uint b, const uint m, const uint n, const uint k, const T p, const T *A, const T *B, const T *C, const T *gradC, T *gradB)
{
	constexpr uint SRows = BlockSizeK / 8, BlockSizeN = 8192 / BlockSizeK;
	constexpr uint OutputLd = SRows * 2, BlockSizeM = BlockSizeK * 4, OutSize = 4;
	constexpr uint ReadPerThreadC = BlockSizeN * SRows / ThreadsPerBlock, ReadPerThreadA = BlockSizeK * SRows / ThreadsPerBlock;
	constexpr uint BlockNReadLd = BlockSizeN / ReadPerThreadC, BlockKReadLd = ReadPerThreadA >= 1 ? BlockSizeK / ReadPerThreadA : BlockSizeK;

	const uint blocksM = gridDim.z / b, bid = blockIdx.z / blocksM;
	const uint tm = threadIdx.x / BlockNReadLd, tn = threadIdx.x % BlockNReadLd; // tn: 0~SRows-1, tm: 0~BlockSizeN/2-1
	uint tmA, tkA;
	if constexpr (ReadPerThreadA >= 1)
	{
		tmA = threadIdx.x / (BlockSizeK / ReadPerThreadA);
		tkA = threadIdx.x % (BlockSizeK / ReadPerThreadA);
	}
	else
	{
		tmA = threadIdx.x / BlockSizeK;
		tkA = threadIdx.x % BlockSizeK;
	}
	const uint bm = blockIdx.z % blocksM, bn = blockIdx.y, bk = blockIdx.x;
	const uint smemOffC = (threadIdx.x / OutputLd) * OutSize;
	const uint smemOffA = (threadIdx.x % OutputLd) * OutSize;

	__shared__ __align__(16 * 1024) T shareC[BlockSizeN * SRows];	 // row major shared memory for C
	__shared__ __align__(16 * 1024) T shareGrad[BlockSizeN * SRows]; // row major shared memory for gradC
	__shared__ __align__(16 * 1024) T shareA[BlockSizeK * SRows];	 // row major shared memory for A
	T grads[OutSize][OutSize] = {0}, gradsTemp[OutSize][OutSize] = {0}, orgB[OutSize][OutSize] = {0};
	T panelC[OutSize] = {0}, panelGrad[OutSize] = {0}, panelA[OutSize] = {0};

	const uint fromOffsetColC = bn * BlockSizeN + tn, fromOffsetRowC = tm + bm * BlockSizeM;
	const uint fromOffsetRowA = tmA + bm * BlockSizeM, fromOffsetColA = bk * BlockSizeK + tkA;
	const uint toSMemCOffset = tm * BlockSizeN + tn;
	const uint toSMemAOffset = tmA * BlockSizeK + tkA;
	const uint outRowOffset = bn * BlockSizeN + smemOffC, outColOffset = bk * BlockSizeK + smemOffA;
	const T *fromC = C + (bid * m * n) + fromOffsetRowC * n + fromOffsetColC, *endC = C + (bid * m * n) + (fromOffsetRowC + 1) * n;
	const T *fromGrad = gradC + (bid * m * n) + fromOffsetRowC * n + fromOffsetColC, *endG = gradC + (bid * m * n) + (fromOffsetRowC + 1) * n;
	const T *fromA = A + (bid * m * k) + fromOffsetRowA * k + fromOffsetColA, *endA = A + (bid * m * k) + (fromOffsetRowA + 1) * k;
	const T *fromB = B + (bid * n * k) + outRowOffset * k + outColOffset;

	// part0: load B
	const bool noRestOut = (outRowOffset + OutSize < n) && (outColOffset + OutSize < k);
	loadFromGlobal<T, OutSize>(orgB, fromB, n, k, outRowOffset, outColOffset, noRestOut);

#pragma unroll
	for (uint outerM = 0; outerM < BlockSizeM / SRows; ++outerM)
	{
		// part1: global mem to shared mem
		const bool noRestC = (fromOffsetColC + (ReadPerThreadC - 1) * BlockNReadLd < n) && (bm < blocksM - 1);
		const bool cExceed = fromOffsetRowC + outerM * SRows >= m, aExceed = fromOffsetRowA + outerM * SRows >= m;
		loadToShared<T, BlockNReadLd, ReadPerThreadC, false>(shareC + toSMemCOffset, fromC, endC, n, noRestC, cExceed);
		loadToShared<T, BlockNReadLd, ReadPerThreadC, false>(shareGrad + toSMemCOffset, fromGrad, endG, n, noRestC, cExceed);
		if constexpr (ReadPerThreadA >= 1)
		{
			const bool noRestA = (fromOffsetColA + (ReadPerThreadA - 1) * BlockKReadLd < k) && (bm < blocksM - 1);
			loadToShared<T, BlockKReadLd, ReadPerThreadA, false>(shareA + toSMemAOffset, fromA, endA, k, noRestA, aExceed);
		}
		else
		{
			if (tmA < SRows)
			{
				shareA[toSMemAOffset] = fromA < endA ? fromA[0] : T{0};
			}
		}
		fromC += SRows * n;
		fromGrad += SRows * n;
		endC += SRows * n;
		endG += SRows * n;
		fromA += SRows * k;
		endA += SRows * k;
		if (endC >= (C + (bid + 1) * m * n))
		{
			endC = C + (bid + 1) * m * n;
			endG = gradC + (bid + 1) * m * n;
		}
		if (endA >= (A + (bid + 1) * m * k))
		{
			endA = A + (bid + 1) * m * k;
		}
		__syncthreads();

		// part2: calculate via outer product
		// (512/OutputLd x OutputLd) threads, (OutSize x OutSize) per thread
#pragma unroll
		for (int i = 0; i < OutSize; ++i)
		{
#pragma unroll
			for (int j = 0; j < OutSize; ++j)
			{
				gradsTemp[i][j] = 0;
			}
		}
#pragma unroll
		for (int subM = 0; subM < SRows; ++subM)
		{
			const T *ptrC = shareC + subM * BlockSizeN + smemOffC;
			const T *ptrGC = shareGrad + subM * BlockSizeN + smemOffC;
			const T *ptrA = shareA + subM * BlockSizeK + smemOffA;
#pragma unroll
			for (int i = 0; i < OutSize; ++i)
			{
				panelC[i] = ptrC[i];
			}
#pragma unroll
			for (int i = 0; i < OutSize; ++i)
			{
				panelGrad[i] = ptrGC[i];
			}
#pragma unroll
			for (int i = 0; i < OutSize; ++i)
			{
				panelA[i] = ptrA[i];
			}
			// outer product
#pragma unroll
			for (int i = 0; i < OutSize; ++i)
			{
#pragma unroll
				for (int j = 0; j < OutSize; ++j)
				{
					gradsTemp[i][j] += Dist::backward(orgB[i][j] - panelA[j], panelGrad[i], panelC[i], p);
				}
			}
		}
#pragma unroll
		for (int i = 0; i < OutSize; ++i)
		{
#pragma unroll
			for (int j = 0; j < OutSize; ++j)
			{
				grads[i][j] += gradsTemp[i][j];
			}
		}
		__syncthreads();
	}

	// part3: save
	T *const gradOut = gradB + bid * n * k * blocksM + bm * n * k + outRowOffset * k + outColOffset;
	saveToGlobal(grads, gradOut, n, k, outRowOffset, outColOffset, noRestOut);
}

template <typename T, uint MaxRows>
__global__ void columnSum(const uint rows, const uint cols, const T *A, T *output)
{
	T colTemp[MaxRows / 2]{0};
	const uint idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx >= cols)
		return;
	const uint inputIdx = blockIdx.y * rows * cols + idx;
	for (uint i = 0; i < rows; i += 2)
	{
		colTemp[i / 2] = A[i * cols + inputIdx] + A[(i + 1) * cols + inputIdx];
	}
#pragma unroll
	for (uint step = 1; step <= MaxRows / 4; step *= 2)
	{
#pragma unroll
		for (uint i = 0; i < MaxRows / 2; i += step * 2)
		{
			colTemp[i] += colTemp[i + step];
		}
	}
	output[blockIdx.y * cols + idx] = colTemp[0];
}

template <typename T>
__global__ void columnSumNaive(const uint rows, const uint cols, const T *A, T *output)
{
	const uint idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx >= cols)
		return;
	const uint inputIdx = blockIdx.y * rows * cols + idx;
	T o{};
	for (uint i = 0; i < rows; ++i)
	{
		o += A[i * cols + inputIdx];
	}
	output[blockIdx.y * cols + idx] = o;
}

template <typename T>
inline void dist_forward_kernel(cudaStream_t stream, const DistDescriptor &d, const void *A, const void *B, void *C)
{
	dim3 gridForward((d.lenB + 128 - 1) / 128, (d.lenA + 128 - 1) / 128, d.batches);
	auto fn = crossDistances<T, typename dists<T>::p, 256>;
	if (d.p <= 0 || is_nan(d.p))
	{
		throw "Do not support norm of zero or negative number!";
	}
	else if (d.p == 1.0)
	{
		fn = crossDistances<T, typename dists<T>::one, 256>;
	}
	else if (d.p == 2.0)
	{
		fn = crossDistances<T, typename dists<T>::two, 256>;
	}
	else if (is_inf(d.p))
	{
		fn = crossDistances<T, typename dists<T>::inf, 256>;
	}
	fn<<<gridForward, 256, 0, stream>>>(d.lenA, d.lenB, d.dim, d.p, (const T *)A, (const T *)B, (T *)C);
}

void crossDist_forward(cudaStream_t stream, void **buffers, const char *opaque, size_t opaque_len)
{
	const DistDescriptor &d = *UnpackDescriptor<DistDescriptor>(opaque, opaque_len);
	// depth2leafProbs, const void* rouletteFuncs, const void* constSamples
	const void *A = (const void *)(buffers[0]);
	const void *B = (const void *)(buffers[1]);
	void *C = (void *)(buffers[2]);

#ifdef TEST
	Timer t;
	for (int i = 0; i < 1; i++)
	{
#endif
		switch (d.type)
		{
		case ElementType::F32:
			dist_forward_kernel<float>(stream, d, A, B, C);
			break;
		case ElementType::F64:
			dist_forward_kernel<double>(stream, d, A, B, C);
			break;
		default:
			throw "Unsupported data type";
			break;
		}
#ifdef TEST
	}
	auto err = cudaDeviceSynchronize();
	std::cout << "C++: " << t.elapsed() << std::endl;
	if (err != 0)
		std::cout << "Execution error of code " << (int)err << std::endl;
#endif
}

constexpr inline static int getBlockK(int k)
{
	int bk{};
	if ((k >= 64 && (k % 64 > 32 || k % 64 == 0)) || k / 64 >= 3)
		bk = 64;
	else if ((k >= 32 && (k % 32 > 16 || k % 32 == 0)) || k / 32 >= 3)
		bk = 32;
	else
		bk = 16;
	return bk;
}

size_t getTempSize(int k, int m, int n)
{
	size_t BlockSizeInner = getBlockK(k) * 4;
	size_t nBlocks = (n + BlockSizeInner - 1) / BlockSizeInner, mBlocks = (m + BlockSizeInner - 1) / BlockSizeInner;
	return std::max(nBlocks * m * k, mBlocks * n * k);
}

template <typename T, uint BlockSizeK>
inline void new_dist_backward_kernel(cudaStream_t stream, const DistDescriptor &d, const void *A, const void *B, const void *C, const void *gC, void *gA, void *gB)
{
	constexpr uint BlockSizeInner = BlockSizeK * 4, BlockSizeMN = 8192 / BlockSizeK;
	const uint nBlocks = (d.lenB + BlockSizeInner - 1) / BlockSizeInner, mBlocks = (d.lenA + BlockSizeInner - 1) / BlockSizeInner;
	const uint maxBlocks = std::max(nBlocks, mBlocks);
	dim3 gridBackwardA((d.dim + BlockSizeK - 1) / BlockSizeK, (d.lenA + BlockSizeMN - 1) / BlockSizeMN, d.batches);
	dim3 gridBackwardB((d.dim + BlockSizeK - 1) / BlockSizeK, (d.lenB + BlockSizeMN - 1) / BlockSizeMN, d.batches);

	auto fnA = crossDistances_backwardA_cg<T, typename dists<T>::p, 512, BlockSizeK>;
	auto fnB = crossDistances_backwardB_cg<T, typename dists<T>::p, 512, BlockSizeK>;
	if (d.p == 1.0)
	{
		fnA = crossDistances_backwardA_cg<T, typename dists<T>::one, 512, BlockSizeK>;
		fnB = crossDistances_backwardB_cg<T, typename dists<T>::one, 512, BlockSizeK>;
	}
	else if (d.p == 2.0)
	{
		fnA = crossDistances_backwardA_cg<T, typename dists<T>::two, 512, BlockSizeK>;
		fnB = crossDistances_backwardB_cg<T, typename dists<T>::two, 512, BlockSizeK>;
	}
	else if (is_inf(d.p))
	{
		fnA = crossDistances_backwardA_cg<T, typename dists<T>::inf, 512, BlockSizeK>;
		fnB = crossDistances_backwardB_cg<T, typename dists<T>::inf, 512, BlockSizeK>;
	}
	else if (d.p < 2)
	{
		fnA = crossDistances_backwardA_cg<T, typename dists<T>::lt_two, 512, BlockSizeK>;
		fnB = crossDistances_backwardB_cg<T, typename dists<T>::lt_two, 512, BlockSizeK>;
	}
#ifdef TEST
	std::cout << "p: " << d.p << "\tbatch: " << d.batches << "\tnBlocks: " << nBlocks << "\tmBlocks: " << mBlocks << "\td.lenA * d.dim: " << d.lenA * d.dim << "\td.lenB * d.dim: " << d.lenB * d.dim << std::endl;
#endif // TEST
	cooperative_launch((const void *)fnA,
					   gridBackwardA, dim3{512}, 0, stream,
					   nBlocks, d.lenA, d.lenB, d.dim, d.p, (const T *)A, (const T *)B, (const T *)C, (const T *)gC, (T *)gA);
#ifdef TEST
	cudaError_t err = cudaDeviceSynchronize();
	std::cout << err << "\t";
	T a[40]{};
	err = cudaMemcpy(&a, gA, sizeof(a), cudaMemcpyKind::cudaMemcpyDeviceToHost);
	for (uint i = 0; i < 40; i++)
	{
		std::cout << a[i] << "  ";
	}
	std::cout << std::endl;
#endif // TEST
	cooperative_launch((const void*)fnB,
					   gridBackwardB, dim3{512}, 0, stream,
					   mBlocks, d.lenA, d.lenB, d.dim, d.p, (const T *)A, (const T *)B, (const T *)C, (const T *)gC, (T *)gB);
#ifdef TEST
	err = cudaDeviceSynchronize();
	std::cout << err << "\t";
	T b[40]{};
	err = cudaMemcpy(&b, gB, sizeof(b), cudaMemcpyKind::cudaMemcpyDeviceToHost);
	std::cout << err << ":\t";
	for (uint i = 0; i < 40; i++)
	{
		std::cout << b[i] << "  ";
	}
	std::cout << std::endl;
#endif // TEST
}

template <typename T, uint BlockSizeK>
inline void dist_backward_kernel(cudaStream_t stream, const DistDescriptor &d, const void *A, const void *B, const void *C, const void *gC, void *temp, void *gA, void *gB)
{
	constexpr uint BlockSizeInner = BlockSizeK * 4, BlockSizeMN = 8192 / BlockSizeK;
	const uint nBlocks = (d.lenB + BlockSizeInner - 1) / BlockSizeInner, mBlocks = (d.lenA + BlockSizeInner - 1) / BlockSizeInner;
	const uint maxBlocks = std::max(nBlocks, mBlocks);
	dim3 gridBackwardA((d.dim + BlockSizeK - 1) / BlockSizeK, (d.lenA + BlockSizeMN - 1) / BlockSizeMN, nBlocks * d.batches);
	dim3 gridBackwardB((d.dim + BlockSizeK - 1) / BlockSizeK, (d.lenB + BlockSizeMN - 1) / BlockSizeMN, mBlocks * d.batches);

	////auto sumFn = columnSum<T, 64>;
	////if (maxBlocks <= 8)
	////	sumFn = columnSum<T, 8>;
	////if (maxBlocks <= 16)
	////	sumFn = columnSum<T, 16>;
	////if (maxBlocks <= 32)
	////	sumFn = columnSum<T, 32>;
	auto sumFn = columnSumNaive<T>;
	int gridSize{}, blockSize{};
	auto err = cudaOccupancyMaxPotentialBlockSize(&gridSize, &blockSize, sumFn, 0, 0);
	const int gridSizeA = std::max(gridSize, (int)(d.lenA * d.dim + blockSize - 1) / blockSize);
	const int gridSizeB = std::max(gridSize, (int)(d.lenB * d.dim + blockSize - 1) / blockSize);
	dim3 gridBackSumA(gridSizeA, d.batches), gridBackSumB(gridSizeB, d.batches);

	auto fnA = crossDistances_backwardA<T, typename dists<T>::p, 512, BlockSizeK>;
	auto fnB = crossDistances_backwardB<T, typename dists<T>::p, 512, BlockSizeK>;
	if (d.p == 1.0)
	{
		fnA = crossDistances_backwardA<T, typename dists<T>::one, 512, BlockSizeK>;
		fnB = crossDistances_backwardB<T, typename dists<T>::one, 512, BlockSizeK>;
	}
	else if (d.p == 2.0)
	{
		fnA = crossDistances_backwardA<T, typename dists<T>::two, 512, BlockSizeK>;
		fnB = crossDistances_backwardB<T, typename dists<T>::two, 512, BlockSizeK>;
	}
	else if (is_inf(d.p))
	{
		fnA = crossDistances_backwardA<T, typename dists<T>::inf, 512, BlockSizeK>;
		fnB = crossDistances_backwardB<T, typename dists<T>::inf, 512, BlockSizeK>;
	}
	else if (d.p < 2)
	{
		fnA = crossDistances_backwardA<T, typename dists<T>::lt_two, 512, BlockSizeK>;
		fnB = crossDistances_backwardB<T, typename dists<T>::lt_two, 512, BlockSizeK>;
	}
#ifdef TEST
	std::cout << "p: " << d.p << "\tbatch: " << d.batches << "\tblockSize: " << blockSize << "\tgridSizeA: " << gridSizeA << "\tgridSizeB: " << gridSizeB << "\tnBlocks: " << nBlocks << "\tmBlocks: " << mBlocks << "\td.lenA * d.dim: " << d.lenA * d.dim << "\td.lenB * d.dim: " << d.lenB * d.dim << std::endl;
#endif // TEST
	fnA<<<gridBackwardA, 512, 0, stream>>>(d.batches, d.lenA, d.lenB, d.dim, d.p, (const T *)A, (const T *)B, (const T *)C, (const T *)gC, (T *)temp);
#ifdef TEST
	err = cudaDeviceSynchronize();
	std::cout << err << "\t";
	T a{};
	err = cudaMemcpy(&a, gA, sizeof(T), cudaMemcpyKind::cudaMemcpyDeviceToHost);
	std::cout << err << ":\t" << a << std::endl;
#endif // TEST
	sumFn<<<gridBackSumA, blockSize, 0, stream>>>(nBlocks, d.lenA * d.dim, (const T *)temp, (T *)gA);
#ifdef TEST
	err = cudaDeviceSynchronize();
	std::cout << err << "\t" << std::endl;
	err = cudaMemcpy(&a, gA, sizeof(T), cudaMemcpyKind::cudaMemcpyDeviceToHost);
	std::cout << err << ":\t" << a << std::endl;
#endif // TEST
	fnB<<<gridBackwardB, 512, 0, stream>>>(d.batches, d.lenA, d.lenB, d.dim, d.p, (const T *)A, (const T *)B, (const T *)C, (const T *)gC, (T *)temp);
#ifdef TEST
	err = cudaDeviceSynchronize();
	std::cout << err << "\t";
	T b[40]{};
	err = cudaMemcpy2D(&b, sizeof(T), temp, sizeof(T) * d.lenB * d.dim, sizeof(T), mBlocks, cudaMemcpyKind::cudaMemcpyDeviceToHost);
	std::cout << err << ":\t";
	for (uint i = 0; i < mBlocks; i++)
	{
		std::cout << b[i] << "  ";
	}
	std::cout << std::endl;
#endif // TEST
	sumFn<<<gridBackSumB, blockSize, 0, stream>>>(mBlocks, d.lenB * d.dim, (const T *)temp, (T *)gB);
#ifdef TEST
	err = cudaDeviceSynchronize();
	std::cout << err << "\t";
	err = cudaMemcpy(&a, gB, sizeof(T), cudaMemcpyKind::cudaMemcpyDeviceToHost);
	std::cout << err << ":\t" << a << std::endl;
#endif // TEST
}

void crossDist_backward(cudaStream_t stream, void **buffers, const char *opaque, size_t opaque_len)
{
	const DistDescriptor &d = *UnpackDescriptor<DistDescriptor>(opaque, opaque_len);
	// depth2leafProbs, const void* rouletteFuncs, const void* constSamples
	const void *A = (const void *)(buffers[0]);
	const void *B = (const void *)(buffers[1]);
	const void *C = (const void *)(buffers[2]);
	const void *gC = (const void *)(buffers[3]);
	void *temp = (void *)(buffers[4]);
	void *gA = (void *)(buffers[5]);
	void *gB = (void *)(buffers[6]);
	// void *gA = (void *)(buffers[4]);
	// void *gB = (void *)(buffers[5]);

	if (d.p <= 0 || is_nan(d.p))
	{
		std::cerr << "Do not support norm of zero or negative number!" << std::endl;
		return;
	}

#ifdef TEST
	Timer t;
	for (int i = 0; i < 1; i++)
	{
#endif
		auto fn = dist_backward_kernel<float, 64>;
		switch (d.type)
		{
		case ElementType::F32:
			switch (getBlockK(d.dim))
			{
			case 64:
				fn = dist_backward_kernel<float, 64>;
				break;
			case 32:
				fn = dist_backward_kernel<float, 32>;
				break;
			default:
				fn = dist_backward_kernel<float, 16>;
				break;
			}
			break;
		case ElementType::F64:
			switch (getBlockK(d.dim))
			{
			case 64:
				fn = dist_backward_kernel<double, 64>;
				break;
			case 32:
				fn = dist_backward_kernel<double, 32>;
				break;
			default:
				fn = dist_backward_kernel<double, 16>;
				break;
			}
			break;
		default:
			std::cerr << "Unsupported data type!" << std::endl;
			return;
		}
		fn(stream, d, A, B, C, gC, temp, gA, gB);
		// fn(stream, d, A, B, C, gC, gA, gB);
#ifdef TEST
	}
	auto err = cudaDeviceSynchronize();
	std::cout << "C++: " << t.elapsed() << std::endl;
	if (err != 0)
		std::cout << "Execution error of code " << (int)err << std::endl;
#endif
}

#ifdef TEST
int main()
{
	constexpr uint m = 1024, n = 1024, k = 64;
	constexpr uint BlockSizeK = getBlockK(k), BlockSizeInner = BlockSizeK * 4;
	const uint nBlocks = (n + BlockSizeInner - 1) / BlockSizeInner, mBlocks = (m + BlockSizeInner - 1) / BlockSizeInner;
	using dist = dists<float>::two;

	float *dA, *dB, *dC, *dG, *dGA_, *dGB_, *dGA, *dGB, *A = new float[m * k], *B = new float[n * k], *C = new float[m * n], *G = new float[m * n];
	cudaMalloc(&dA, (size_t)m * k * sizeof(float));
	cudaMalloc(&dB, (size_t)n * k * sizeof(float));
	cudaMalloc(&dC, (size_t)m * n * sizeof(float));
	cudaMalloc(&dG, (size_t)m * n * sizeof(float));
	cudaMalloc(&dGA_, (size_t)nBlocks * m * k * sizeof(float));
	cudaMalloc(&dGB_, (size_t)mBlocks * n * k * sizeof(float));
	cudaMalloc(&dGA, (size_t)m * k * sizeof(float));
	cudaMalloc(&dGB, (size_t)n * k * sizeof(float));
	srand(1000);
	for (uint i = 0; i < m; i++)
	{
		for (uint j = 0; j < k; j++)
		{
			// A[i * k + j] = j + std::sqrt((double)i);
			A[i * k + j] = rand() / (float)RAND_MAX;
		}
	}
	for (uint i = 0; i < n; i++)
	{
		for (uint j = 0; j < k; j++)
		{
			// B[i * k + j] = 2 * j + (float)i;
			B[i * k + j] = rand() / (float)RAND_MAX;
		}
	}
	for (uint i = 0; i < m; i++)
	{
		for (uint j = 0; j < n; j++)
		{
			G[i * n + j] = std::sin((double)i) + std::cos((double)j);
		}
	}
	cudaMemcpy(dA, A, (size_t)m * k * sizeof(float), cudaMemcpyKind::cudaMemcpyHostToDevice);
	cudaMemcpy(dB, B, (size_t)n * k * sizeof(float), cudaMemcpyKind::cudaMemcpyHostToDevice);
	cudaMemcpy(dG, G, (size_t)m * n * sizeof(float), cudaMemcpyKind::cudaMemcpyHostToDevice);
	cudaMemset(dC, 0, (size_t)m * n * sizeof(float));
	//cudaMemset(dGA_, 0, (size_t)nBlocks * m * k * sizeof(float));
	//cudaMemset(dGB_, 0, (size_t)mBlocks * n * k * sizeof(float));

	auto d = DistDescriptor(2, 1, k, m, n, ElementType::F32);
	dist_forward_kernel<float>(NULL, d, dA, dB, dC);
	dist_backward_kernel<float, BlockSizeK>(NULL, d, dA, dB, dC, dC, dGA, dGB);
}
#endif // TEST