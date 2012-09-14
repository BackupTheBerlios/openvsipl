/* Copyright (c) 2009 by CodeSourcery.  All rights reserved. */

/** @file    trans.cu
    @author  Don McCoy
    @date    2009-06-09
    @brief   VSIPL++ Library: CUDA Kernel for matrix transpose
*/

/***********************************************************************
  Included Files
***********************************************************************/

#include <cuComplex.h>

#include "util.hpp"


/***********************************************************************
  Device Kernels -- Each thread computes one element
***********************************************************************/

__global__ void 
k_transpose_ss(float const* input, float* output, size_t rows, size_t cols)
{
  int const tx = threadIdx.x;
  int const ty = threadIdx.y;
  int const bx = blockIdx.x;
  int const by = blockIdx.y;
  int const row = __mul24(blockDim.y, by) + ty;
  int const col = __mul24(blockDim.x, bx) + tx;

  int const idx_in = __mul24(col, rows) + row;
  int const idx_out = __mul24(row, cols) + col;
  output[idx_out] = input[idx_in];
}

__global__ void 
k_transpose_cc(cuComplex const* input, cuComplex* output, size_t rows, size_t cols)
{
  int const tx = threadIdx.x;
  int const ty = threadIdx.y;
  int const bx = blockIdx.x;
  int const by = blockIdx.y;
  int const row = __mul24(blockDim.y, by) + ty;
  int const col = __mul24(blockDim.x, bx) + tx;

  int const idx_in = __mul24(col, rows) + row;
  int const idx_out = __mul24(row, cols) + col;
  output[idx_out] = input[idx_in];
}

__global__ void 
k_copy_ss(float const* input, float* output, size_t rows, size_t cols)
{
  int const tx = threadIdx.x;
  int const ty = threadIdx.y;
  int const bx = blockIdx.x;
  int const by = blockIdx.y;
  int const row = __mul24(blockDim.y, by) + ty;
  int const col = __mul24(blockDim.x, bx) + tx;

  int const idx = __mul24(row, cols) + col;
  output[idx] = input[idx];
}

__global__ void 
k_copy_cc(cuComplex const* input, cuComplex* output, size_t rows, size_t cols)
{
  int const tx = threadIdx.x;
  int const ty = threadIdx.y;
  int const bx = blockIdx.x;
  int const by = blockIdx.y;
  int const row = __mul24(blockDim.y, by) + ty;
  int const col = __mul24(blockDim.x, bx) + tx;

  int const idx = __mul24(row, cols) + col;
  output[idx] = input[idx];
}


/***********************************************************************
  Definitions
***********************************************************************/

namespace vsip
{
namespace impl
{
namespace cuda
{

void
transpose_ss(
  float const* input,
  float*       output,
  size_t       rows,
  size_t       cols)
{
  dim3 threads = calculate_thread_block_size(rows, cols);
  dim3 blocks(cols/threads.x, rows/threads.y);

  k_transpose_ss<<<blocks, threads>>>(input, output, rows, cols);
  cudaThreadSynchronize();
}


void
transpose_cc(
  cuComplex const* input,
  cuComplex*       output,
  size_t           rows,
  size_t           cols)
{
  dim3 threads = calculate_thread_block_size(rows, cols);
  dim3 blocks(cols/threads.x, rows/threads.y);

  k_transpose_cc<<<blocks, threads>>>(input, output, rows, cols);
  cudaThreadSynchronize();
}

void
copy_ss(
  float const* input,
  float*       output,
  size_t       rows,
  size_t       cols)
{
  dim3 threads = calculate_thread_block_size(rows, cols);
  dim3 blocks(cols/threads.x, rows/threads.y);

  k_copy_ss<<<blocks, threads>>>(input, output, rows, cols);
  cudaThreadSynchronize();
}


void
copy_cc(
  cuComplex const* input,
  cuComplex*       output,
  size_t           rows,
  size_t           cols)
{
  dim3 threads = calculate_thread_block_size(rows, cols);
  dim3 blocks(cols/threads.x, rows/threads.y);

  k_copy_cc<<<blocks, threads>>>(input, output, rows, cols);
  cudaThreadSynchronize();
}

} // namespace vsip::impl::cuda
} // namespace vsip::impl
} // namespace vsip
