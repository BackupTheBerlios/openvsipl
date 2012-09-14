/* Copyright (c) 2009 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    vsip/opt/cuda/inter.hpp
    @author  Don McCoy
    @date    2009-07-27
    @brief   VSIPL++ Library: User-defined kernel for polar to rectangular
               interpolation for SSAR images.
*/

#ifndef VSIP_OPT_CUDA_INTERP_HPP
#define VSIP_OPT_CUDA_INTERP_HPP

#if VSIP_IMPL_REF_IMPL
# error "vsip/opt files cannot be used as part of the reference impl."
#endif

/***********************************************************************
  Included Files
***********************************************************************/

#include <vsip/support.hpp>
#include <vsip/matrix.hpp>
#include <vsip/tensor.hpp>
#include <vsip/opt/cuda/bindings.hpp>
#include <vsip/opt/cuda/device_memory.hpp>


/***********************************************************************
  Definitions
***********************************************************************/

namespace vsip
{

namespace impl
{

namespace cuda
{

template <typename IT,
	  typename T,
	  typename Block1,
	  typename Block2,
	  typename Block3,
	  typename Block4>
void
interpolate(
  const_Matrix<IT, Block1>	   indices,  // n x m
  Tensor<T, Block2>                window,   // n x m x I
  const_Matrix<complex<T>, Block3> in,       // n x m
  Matrix<complex<T>, Block4>       out,      // nx x m
  length_type                      depth,
  length_type                      padded_depth)
{
  // All blocks must have the same dimension ordering
  typedef typename Block_layout<Block1>::order_type order1_type;
  typedef typename Block_layout<Block2>::order_type order2_type;
  typedef typename Block_layout<Block3>::order_type order3_type;
  typedef typename Block_layout<Block4>::order_type order4_type;
  assert(order1_type::impl_dim0 == order2_type::impl_dim0);
  assert(order1_type::impl_dim0 == order3_type::impl_dim0);
  assert(order1_type::impl_dim0 == order4_type::impl_dim0);
  assert(order1_type::impl_dim1 == order2_type::impl_dim1);
  assert(order1_type::impl_dim1 == order3_type::impl_dim1);
  assert(order1_type::impl_dim1 == order4_type::impl_dim1);

  Device_memory<Block1> dev_indices(indices.block(), impl::SYNC_IN);
  Device_memory<Block2> dev_window(window.block(), impl::SYNC_IN);
  Device_memory<Block3> dev_in(in.block(), impl::SYNC_IN);
  Device_memory<Block4> dev_out(out.block(), impl::SYNC_OUT);

  size_t rows_in = in.size(0);
  size_t rows_out = out.size(0);
  size_t cols = in.size(1);
  assert(cols == out.size(1));

  interpolate(
    dev_indices.data(),
    dev_window.data(),
    reinterpret_cast<cuComplex const*>(dev_in.data()),
    reinterpret_cast<cuComplex*>(dev_out.data()),
    depth,
    padded_depth,
    rows_in,
    rows_out,
    cols);
}


template <typename IT,
	  typename T,
	  typename Block1,
	  typename Block2,
	  typename Block3,
	  typename Block4>
void
interpolate_with_shift(
  const_Matrix<IT, Block1>	   indices,  // n x m
  Tensor<T, Block2>                window,   // n x m x I
  const_Matrix<complex<T>, Block3> in,       // n x m
  Matrix<complex<T>, Block4>       out,      // nx x m
  length_type                      depth,
  length_type                      padded_depth)
{
  // All blocks must have the same dimension ordering
  typedef typename Block_layout<Block1>::order_type order1_type;
  typedef typename Block_layout<Block2>::order_type order2_type;
  typedef typename Block_layout<Block3>::order_type order3_type;
  typedef typename Block_layout<Block4>::order_type order4_type;
  assert(order1_type::impl_dim0 == order2_type::impl_dim0);
  assert(order1_type::impl_dim0 == order3_type::impl_dim0);
  assert(order1_type::impl_dim0 == order4_type::impl_dim0);
  assert(order1_type::impl_dim1 == order2_type::impl_dim1);
  assert(order1_type::impl_dim1 == order3_type::impl_dim1);
  assert(order1_type::impl_dim1 == order4_type::impl_dim1);

  Device_memory<Block1> dev_indices(indices.block(), impl::SYNC_IN);
  Device_memory<Block2> dev_window(window.block(), impl::SYNC_IN);
  Device_memory<Block3> dev_in(in.block(), impl::SYNC_IN);
  Device_memory<Block4> dev_out(out.block(), impl::SYNC_OUT);

  size_t rows_in = in.size(0);
  size_t rows_out = out.size(0);
  size_t cols = in.size(1);
  assert(cols == out.size(1));

  interpolate_with_shift(
    dev_indices.data(),
    dev_window.data(),
    reinterpret_cast<cuComplex const*>(dev_in.data()),
    reinterpret_cast<cuComplex*>(dev_out.data()),
    depth,
    padded_depth,
    rows_in,
    rows_out,
    cols);
}


} // namespace vsip::impl::cuda
} // namespace vsip::impl
} // namespace vsip

#endif // VSIP_OPT_CUDA_INTERP_HPP
