/* Copyright (c) 2009 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/

/** @file    tests/distrib.cpp
    @author  Don McCoy
    @date    2009-06-21
    @brief   VSIPL++ Library: Test grid and thread block distribution
               utility routines.
*/

/***********************************************************************
  Included Files
***********************************************************************/

#include <iostream>
#include <cuComplex.h>
#include <cuda_runtime_api.h>

#include <vsip/initfin.hpp>
#include <vsip/matrix.hpp>
#include <vsip/opt/cuda/bindings.hpp>
#include <vsip/opt/cuda/gpu_block.hpp>
#include <vsip/opt/cuda/kernels.hpp>
#include <vsip/opt/cuda/kernels/util.hpp>

#include <vsip_csl/test.hpp>


using namespace std;
using namespace vsip;


// These values are hardware-dependent, but must be known at compile
// time.  Currently, we build for compute capability 1.1, so these
// definitions shadow the ones being tested and may have to be adjusted
// if we compile for a higher version (and/or) if we determine the 
// compute capability dynamically

#define MAX_THREADS_PER_BLOCK     512
#define MAX_THREADS_PER_BLOCK_X    32
#define MAX_THREADS_PER_BLOCK_Y    16

#define MAX_GRID_DIM_X  65535
#define MAX_GRID_DIM_Y  65535



/***********************************************************************
  Definitions
***********************************************************************/

void
test_elementwise_vector_distribution(
  size_t size,
  size_t grid_x,
  size_t threads_x)
{
  using vsip::impl::cuda::Gpu_block;

  // Get the distribution parameters (the same ones used when invoking
  // kernels) and verify they match the expected values)
  dim3 grid;
  dim3 threads;
  distribute_vector(size, grid, threads);

  test_assert(grid.x == grid_x);
  test_assert(threads.x == threads_x);

  // Create a block and obtain a pointer to the data on the GPU.  Add guard
  // values that contain a known value that may later be checked to see that no
  // thread wrote to those locations.
  size_t guard_values = 16;
  size_t guard = size_t(-1);
  Gpu_block<1, size_t> vector(size + guard_values, guard);
  Vector<size_t, Gpu_block<1, size_t> > view(vector);


  // Use scoping to prevent the pointer from being accessed after
  // the device is released.
  {
    vector.device_request(vsip::impl::SYNC_IN);
    size_t* s = vector.device_data();
    
    // Run the kernel that checks the distribution by filling each element 
    // of the vector with the linear offset in memory for that element.
    impl::cuda::check_distrib(s, size);
    test_assert(cudaSuccess == cudaGetLastError());
    
    vector.device_release();
  }


  // Check the result.
  for (index_type i = 0; i < size; ++i)
    test_assert(size_t(i) == vector.get(i));

  // Verify guard row is untouched
  for (index_type j = size; j < size + guard_values; ++j)
    test_assert(guard == vector.get(j));
}


void
test_elementwise_matrix_distribution(
  size_t rows, size_t cols,
  size_t grid_y, size_t grid_x,
  size_t threads_y, size_t threads_x)
{
  using vsip::impl::cuda::Gpu_block;

  // Get the distribution parameters (the same ones used when invoking
  // kernels) and verify they match the expected values)
  dim3 grid;
  dim3 threads;
  distribute_matrix(rows, cols, grid, threads);

  test_assert(grid.x == grid_x);
  test_assert(grid.y == grid_y);
  test_assert(threads.x == threads_x);
  test_assert(threads.y == threads_y);

  // Create a block and obtain a pointer to the data on the GPU.  Add a guard
  // row that contains known values that may later be checked to see that no
  // thread wrote to those locations.
  size_t guard = size_t(-1);
  Gpu_block<2, size_t> matrix(Domain<2>(rows + 1, cols), guard);
  Matrix<size_t, Gpu_block<2, size_t> > view(matrix);


  // Use scoping to prevent the pointer from being accessed after
  // the device is released.
  {
    matrix.device_request(vsip::impl::SYNC_IN);
    size_t* m = matrix.device_data();
    
    // Run the kernel that checks the distribution by filling each element 
    // of the matrix with the linear offset in memory for that element.
    impl::cuda::check_distrib(m, rows, cols);
    test_assert(cudaSuccess == cudaGetLastError());
    
    matrix.device_release();
  }


  // Check the result.
  for (index_type i = 0; i < rows; ++i)
    for (index_type j = 0; j < cols; ++j)
      test_assert(size_t(i * cols + j) == matrix.get(i, j));

  // Verify guard row is untouched
  for (index_type j = 0; j < cols; ++j)
    test_assert(guard == matrix.get(rows, j));
}


template <typename T>
void
test_elementwise_kernel_invocation(size_t rows, size_t cols, bool should_succeed)
{
  // dummy allocation
  T* m = NULL;
  test_assert(cudaSuccess == cudaMalloc((void**)&m, 256*sizeof(T)));

  // call kernel that does nothing, just to see if it works for the given size
  vsip::impl::cuda::null_kernel(m, rows, cols);

  if (should_succeed)  test_assert(cudaSuccess == cudaGetLastError());
  else                 test_assert(cudaSuccess != cudaGetLastError());

  cudaFree(m);
}



void
test_elementwise_limits()
{
  // Check that matrices too large for cuda to handle are 
  // recognized.
  size_t max_rows = MAX_THREADS_PER_BLOCK_Y * MAX_GRID_DIM_Y;
  size_t max_cols = MAX_THREADS_PER_BLOCK_X * MAX_GRID_DIM_X;
  test_assert(matrix_is_distributable(max_rows, max_cols));
  test_assert(!matrix_is_distributable(max_rows + 1, max_cols));
  test_assert(!matrix_is_distributable(max_rows    , max_cols + 1));
  test_assert(!matrix_is_distributable(max_rows + 1, max_cols + 1));

  // Call kernels of various sizes and pass a true/false expectation
  // as to whether or not the kernel call should succeed.
  test_elementwise_kernel_invocation<float>(max_rows + 1, max_cols, false);
  test_elementwise_kernel_invocation<float>(max_rows, max_cols + 1, false);
  test_elementwise_kernel_invocation<complex<float> >(max_rows + 1, max_cols, false);
  test_elementwise_kernel_invocation<complex<float> >(max_rows, max_cols + 1, false);

  // The cases that should succeed take about 30 seconds to run, therefore 
  // skip these.  [Note: they passed once on Tesla hardware.]
  //
  // The "known failures" above are more important in that they 
  // will tell us if those size limits do not match our expectations.
#if 0
  test_elementwise_kernel_invocation<float>(max_rows, max_cols, true);
  test_elementwise_kernel_invocation<complex<float> >(max_rows, max_cols, true);
#endif
}


void
test_vector()
{
  // for the standard elementwise vector distribution, the thread count
  // is kept at the maximum of 512
  //
  //                                   input  grid threads                       
  test_elementwise_vector_distribution(  37,   1,   512);
  test_elementwise_vector_distribution( 512,   1,   512);
  test_elementwise_vector_distribution( 513,   2,   512);
  test_elementwise_vector_distribution(1025,   3,   512);
}


void
test_matrix()
{
  // for the standard elementwise matrix distribution, the thread count
  // is kept at the maximum of 512 by using a block size of 16 x 32
  //
  //                                   input    grid   threads                       
  test_elementwise_matrix_distribution(15, 15,  1, 1,  16, 32);
  test_elementwise_matrix_distribution(16, 32,  1, 1,  16, 32);
  test_elementwise_matrix_distribution(17, 33,  2, 2,  16, 32);
  test_elementwise_matrix_distribution(15, 33,  1, 2,  16, 32);
  test_elementwise_matrix_distribution(32, 31,  2, 1,  16, 32);
  test_elementwise_matrix_distribution(32, 64,  2, 2,  16, 32);
  test_elementwise_matrix_distribution(33, 65,  3, 3,  16, 32);

  // see that large elementwise distributions are handled properly
  test_elementwise_limits();
}


/***********************************************************************
  Main
***********************************************************************/

int
main(int argc, char** argv)
{
  vsipl init(argc, argv);

  test_vector();
  test_matrix();

  return 0;
}
