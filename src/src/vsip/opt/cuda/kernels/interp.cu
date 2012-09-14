/* Copyright (c) 2009 by CodeSourcery.  All rights reserved. */

/** @file    interp.cu
    @author  Don McCoy
    @date    2009-07-27
    @brief   VSIPL++ Library: CUDA kernels used soley for testing.
               May also be used as a template for new kernels.
*/

/***********************************************************************
  Included Files
***********************************************************************/

#include <stdio.h>
#include <cuComplex.h>

#include "util.hpp"


/***********************************************************************
  Device functions (callable only via kernels)
***********************************************************************/

// scalar-complex multiply
//   c = a * b   where b and c are complex and a is real
__device__ void scmul(cuComplex& c, float a, cuComplex b)
{
  c.x = a * b.x;
  c.y = a * b.y;
}


/***********************************************************************
  Device Kernels -- Each thread computes one element
***********************************************************************/

__global__ void 
k_interpolate(
  unsigned int const* indices,      // m x n
  float const*        window,       // m x n x I
  cuComplex const*    in,           // m x n
  cuComplex*          out,          // m x nx
  size_t              depth,        // I
  size_t              wstride,      // I + pad bytes
  size_t              cols_in,      // n
  size_t              cols_out,     // nx
  size_t              rows,         // m
  bool                shift_input)  // swap left and right halves of input if requested
{
  // The problem is broken up such that each thread computes an
  // entire row of the output matrix.  This is done to avoid 
  // dealing with the fact that each output value has contributions 
  // from multiple nearby input values (multiplied by different
  // weight values) that must be summed.  
  int const tx = threadIdx.x;
  int const bx = blockIdx.x;
  int const row = __mul24(blockDim.x, bx) + tx;

  // set pointers to the correct row for this thread
  indices += row * cols_in;
  window += row * cols_in * wstride;
  in += row * cols_in;
  out += row * cols_out;

  // Threads outside the bounds of the matrix do no work.  This makes
  // it possible to always divide the thread blocks into efficiently-
  // sized pieces.
  //
  if (row < rows)
  { 
    // zero the output
    for (int i = 0; i < cols_out; ++i)
    {
      out[i].x = 0.f;
      out[i].y = 0.f;
    }

    for (int i = 0; i < cols_in; ++i)
    {
      int ikxcols = indices[i];
      int i_shift = shift_input ? (i + cols_in/2) % cols_in : i;
      float const* pw = window + __mul24(i, wstride);
      
      for (int h = 0; h < depth; ++h)
      {
        cuComplex tmp;
        scmul(tmp, pw[h], in[i_shift]);
	out[ikxcols + h].x += tmp.x;
	out[ikxcols + h].y += tmp.y;
      }
    }
  }
}

__global__ void 
k_freq_domian_fftshift(cuComplex const* input, cuComplex* output, size_t rows, size_t cols)
{
  int const tx = threadIdx.x;
  int const ty = threadIdx.y;
  int const bx = blockIdx.x;
  int const by = blockIdx.y;
  int const row = __mul24(blockDim.y, by) + ty;
  int const col = __mul24(blockDim.x, bx) + tx;

  // Threads outside the bounds of the matrix do no work.  This makes
  // it possible to always divide the thread blocks into efficiently-
  // sized pieces.
  //
  if ((row < rows) && (col < cols))
  { 
    int const idx = __mul24(row, cols) + col;

    if ((row & 1) == (col & 1))
      scmul(output[idx], -1, input[idx]);
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
interpolate(
  unsigned int const* indices,  // m x n
  float const*        window,   // m x n x I
  cuComplex const*    in,       // m x n
  cuComplex*          out,      // m x nx
  size_t              depth,    // I
  size_t              wstride,  // I + pad bytes (if any)
  size_t              cols_in,  // n
  size_t              cols_out, // nx
  size_t              rows)     // m
{
  dim3 grid, threads;

  // Call kernel, distributing by row, with flag set to false so the input 
  // is not shifted.
  distribute_vector(rows, grid, threads);
  k_interpolate<<<grid, threads>>>(indices, window, in, out, 
    depth, wstride, cols_in, cols_out, rows, false);
  cudaThreadSynchronize();
}

void
interpolate_with_shift(
  unsigned int const* indices,  // m x n
  float const*        window,   // m x n x I
  cuComplex const*    in,       // m x n
  cuComplex*          out,      // m x nx
  size_t              depth,    // I
  size_t              wstride,  // I + pad bytes (if any)
  size_t              cols_in,  // n
  size_t              cols_out, // nx
  size_t              rows)     // m
{
  dim3 grid, threads;

  // Call kernel with flag set to shift the input (fftshift/freqswap)
  distribute_vector(rows, grid, threads);
  k_interpolate<<<grid, threads>>>(indices, window, in, out, 
    depth, wstride, cols_in, cols_out, rows, true);
  cudaThreadSynchronize();


  // Perform a second fftshift, but in the frequency domain.  Combining
  // this kernel with the above results in a kernel that is too large
  // to be launched.  Keeping this part separate has the added advantage
  // of being able to distribute one element per thread instead of one
  // row per thread.
  distribute_matrix(rows, cols_out, grid, threads);
  k_freq_domian_fftshift<<<grid, threads>>>(out, out, rows, cols_out);
  cudaThreadSynchronize();
}

} // namespace vsip::impl::cuda
} // namespace vsip::impl
} // namespace vsip
