/* Copyright (c) 2009 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    vsip/opt/cuda/matvec.hpp
    @author  Don McCoy
    @date    2009-02-05
    @brief   VSIPL++ Library: CUDA-based BLAS evaluators 
*/

#ifndef VSIP_OPT_CUDA_MATVEC_HPP
#define VSIP_OPT_CUDA_MATVEC_HPP

#if VSIP_IMPL_REF_IMPL
# error "vsip/opt files cannot be used as part of the reference impl."
#endif

/***********************************************************************
  Included Files
***********************************************************************/

#include <vsip/opt/dispatch.hpp>
#include <vsip/opt/cuda/bindings.hpp>
#include <vsip/opt/cuda/device_memory.hpp>


/***********************************************************************
  Declarations
***********************************************************************/

namespace vsip_csl
{
namespace dispatcher
{

/// CUDA evaluator for vector-vector dot-product (non-conjugated).
template <typename T,
          typename Block0,
          typename Block1>
struct Evaluator<op::prod_vv_dot, be::cuda,
                 T(Block0 const&, Block1 const&)>
{
  static bool const ct_valid =
    impl::cuda::Cuda_traits<T>::valid &&
    impl::Type_equal<T, typename Block0::value_type>::value &&
    impl::Type_equal<T, typename Block1::value_type>::value &&
    // check that direct access is supported
    impl::Ext_data_cost<Block0>::value == 0 &&
    impl::Ext_data_cost<Block1>::value == 0 &&
    // check that format is interleaved.
    !impl::Is_split_block<Block0>::value &&
    !impl::Is_split_block<Block1>::value;

  static bool rt_valid(Block0 const& a, Block1 const& b) 
  { 
    impl::Ext_data<Block0> ext_a(const_cast<Block0&>(a));
    impl::Ext_data<Block1> ext_b(const_cast<Block1&>(b));
    
    // check that data is unit stride
    return ((ext_a.stride(0) == 1) && (ext_b.stride(0) == 1));
  }

  static T exec(Block0 const& a, Block1 const& b)
  {
    assert(a.size(1, 0) == b.size(1, 0));

    impl::cuda::Device_memory<Block0 const> dev_a(a);
    impl::cuda::Device_memory<Block1 const> dev_b(b);

    T r = impl::cuda::dot(a.size(1, 0),
			  dev_a.data(), a.impl_stride(1, 0),
			  dev_b.data(), b.impl_stride(1, 0));
    ASSERT_CUBLAS_OK();

    return r;
  }
};


/// CUDA evaluator for vector-vector dot-product (conjugated).
template <typename Block0, typename Block1>
struct Evaluator<op::prod_vv_dot, be::cuda,
  typename Block0::value_type(Block0 const&, expr::Unary<expr::op::Conj, Block1, true> const&)>
{
  typedef typename Block0::value_type value_type;

  static bool const ct_valid = 
    impl::cuda::Cuda_traits<value_type>::valid &&
    impl::Type_equal<value_type, typename Block1::value_type>::value &&
    // check that direct access is supported
    impl::Ext_data_cost<Block0>::value == 0 &&
    impl::Ext_data_cost<Block1>::value == 0 &&
    // check that format is interleaved.
    !impl::Is_split_block<Block0>::value &&
    !impl::Is_split_block<Block1>::value;

  static bool rt_valid(Block0 const &a,
		       expr::Unary<expr::op::Conj, Block1, true> const &b)
  {
    impl::Ext_data<Block0> ext_a(const_cast<Block0&>(a));
    impl::Ext_data<Block1> ext_b(const_cast<Block1&>(b.arg()));
    
    // check that data is unit stride
    return ((ext_a.stride(0) == 1) && (ext_b.stride(0) == 1));
  }

  static value_type exec(Block0 const &a, 
			 expr::Unary<expr::op::Conj, Block1, true> const &b)
  {
    assert(a.size(1, 0) == b.size(1, 0));

    impl::cuda::Device_memory<Block0 const> dev_a(a);
    impl::cuda::Device_memory<Block1 const> dev_b(b.arg());

    value_type r = impl::cuda::dotc(a.size(1, 0),
				    dev_b.data(), b.arg().impl_stride(1, 0), 
				    dev_a.data(), a.impl_stride(1, 0));
    ASSERT_CUBLAS_OK();
    // Note:
    //   BLAS    cdotc(x, y)  => conj(x) * y, while 
    //   VSIPL++ cvjdot(x, y) => x * conj(y)

    return r;
  }
};


} // namespace vsip_csl::dispatcher
} // namespace vsip_csl

#endif // VSIP_OPT_CUDA_MATVEC_HPP
