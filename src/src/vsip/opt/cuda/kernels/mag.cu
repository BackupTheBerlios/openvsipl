/* Copyright (c) 2009 by CodeSourcery.  All rights reserved. */

/** @file    mag.cu
    @author  Don McCoy
    @date    2009-06-21
    @brief   VSIPL++ Library: CUDA kernel for complex magnitude.
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
k_mag_cs(cuComplex const* in, float* out, size_t rows, size_t cols)
{
  // Using the two-layer coordinate system (grid + thread block),
  // determine which row and column of the matrix to compute.
  int const ty = threadIdx.y;
  int const tx = threadIdx.x;
  int const by = blockIdx.y;
  int const bx = blockIdx.x;
  int const row = __mul24(blockDim.y, by) + ty;
  int const col = __mul24(blockDim.x, bx) + tx;

  // Threads outside the bounds of the matrix do no work.  This makes
  // it possible to always divide the thread blocks into efficiently-
  // sized pieces.
  //
  if ((row < rows) && (col < cols))
  { 
    // Compute the magnitude of each value and write it to
    // the output matrix.
    int const idx = __mul24(col, rows) + row;
    out[idx] = __fsqrt_rn(in[idx].x * in[idx].x +
                          in[idx].y * in[idx].y);
  }
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
mag_cs(
  cuComplex const* in,
  float*           out,
  size_t           rows,
  size_t           cols)
{
  dim3 grid, threads;
  distribute_matrix(rows, cols, grid, threads);

  k_mag_cs<<<grid, threads>>>(in, out, rows, cols);
  cudaThreadSynchronize();
}

} // namespace vsip::impl::cuda
} // namespace vsip::impl
} // namespace vsip
