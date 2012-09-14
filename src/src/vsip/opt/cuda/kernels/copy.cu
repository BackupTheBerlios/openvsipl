/* Copyright (c) 2009 by CodeSourcery.  All rights reserved. */

/** @file    copy.cu
    @author  Don McCoy
    @date    2009-03-10
    @brief   VSIPL++ Library: CUDA Kernel for memory copy for benchmarking only
*/

/***********************************************************************
  Included Files
***********************************************************************/

#include <cuda_runtime.h>
#include "util.hpp"


/***********************************************************************
  Device Kernels -- Each thread computes 'size' elements
***********************************************************************/

#define MAX_SHARED  (16*1024)
#define THREADS  512

// size must be less than or equal to MAX_SHARED/sizeof(float)
__global__ void
dev2shared(float* dev, size_t size)
{
  // shared memory
  // the size is determined by the host application
  extern  __shared__  float sdata[];

  dev += size * threadIdx.x;
  float* dst = sdata + size * threadIdx.x;

  for (size_t i = 0; i < size; ++i)
    dst[i] = dev[i];
  __syncthreads();
}


__global__ void 
dev2dev(float const* input, float* output, size_t size)
{
  int const tx = threadIdx.x;
  int const bx = blockIdx.x;

  int const idx = __mul24(blockDim.x, bx) + tx;
  if (idx < size)
    output[idx] = input[idx];
}


__global__ void 
zeroes2dev(float* inout, size_t size)
{
  int const tx = threadIdx.x;
  int const bx = blockIdx.x;

  int const idx = __mul24(blockDim.x, bx) + tx;
  if (idx < size)
    inout[idx] = 0.f;
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
copy_device_to_shared(float* src, size_t size)
{
  // When the size exceeds the maximum amount of shared memory,
  // break the transfer up into pieces.  In a realistic application
  // one would never exceed the maximum like this, but this is for 
  // performance testing purposes only.
  while (size) 
  {
    int count = (size > MAX_SHARED) ? MAX_SHARED : size;

    dev2shared<<<THREADS, 1, MAX_SHARED>>>(src, count/THREADS);
    cudaThreadSynchronize();

    src += count;
    size -= count;
  }
}

void
copy_device_to_device(float const* src, float* dest, size_t size)
{
  dim3 grid, threads;
  distribute_vector(size, grid, threads);

  dev2dev<<<grid, threads>>>(src, dest, size);
  cudaThreadSynchronize();
}

void
copy_zeroes_to_device(float* dest, size_t size)
{
  dim3 grid, threads;
  distribute_vector(size, grid, threads);

  zeroes2dev<<<grid, threads>>>(dest, size);
  cudaThreadSynchronize();
}

} // namespace vsip::impl::cuda
} // namespace vsip::impl
} // namespace vsip
