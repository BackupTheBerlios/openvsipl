/* Copyright (c) 2009 by CodeSourcery.  All rights reserved. */

/** @file    vsip/opt/cuda/eval_fftmul.hpp
    @author  Don McCoy
    @date    2009-07-20
    @brief   VSIPL++ Library: Evaluator for an FFTM followed by
               multiplication with another matrix
*/

#ifndef VSIP_OPT_CUDA_EVAL_FFTMUL_HPP
#define VSIP_OPT_CUDA_EVAL_FFTMUL_HPP

/***********************************************************************
  Included Files
***********************************************************************/

#include <vsip/core/fft.hpp>
#include <vsip/core/metaprogramming.hpp>
#include <vsip/opt/expr/assign_fwd.hpp>
#include <vsip/opt/cuda/block_unwrapper.hpp>
#include <vsip/opt/cuda/device_memory.hpp>


/***********************************************************************
  Declarations
***********************************************************************/

namespace vsip_csl
{
namespace dispatcher
{

template <typename DstBlock,
          typename LBlock,
          typename RBlock,
          template <typename> class F>
struct Evaluator<op::assign<2>, be::cuda,
  void(DstBlock &,
       expr::Binary<expr::op::Mult, 
         LBlock,
         expr::Unary<F,
           RBlock,
           false> const,
         true> const &)>
{
  static char const* name() { return "Cuda_tag"; }

  typedef
  expr::Binary<expr::op::Mult, 
    LBlock,
    expr::Unary<F, 
      RBlock,
      false> const,
    true> const
  SrcBlock;

  typedef std::complex<float> C;
  typedef typename impl::fft::workspace<2u, C, C>  workspace_type;
  typedef typename impl::fft::fftm<C, C, 0, -1>  be_fwd_row_type;
  typedef typename impl::fft::fftm<C, C, 1, -1>  be_fwd_col_type;
  typedef typename impl::fft::fftm<C, C, 0, 1>  be_inv_row_type;
  typedef typename impl::fft::fftm<C, C, 1, 1>  be_inv_col_type;
  typedef typename expr::op::fft<2u, 
    be_fwd_row_type, workspace_type>::Functor<RBlock> fftm_fwd_row_functor_type;
  typedef typename expr::op::fft<2u, 
    be_fwd_col_type, workspace_type>::Functor<RBlock> fftm_fwd_col_functor_type;
  typedef typename expr::op::fft<2u, 
    be_inv_row_type, workspace_type>::Functor<RBlock> fftm_inv_row_functor_type;
  typedef typename expr::op::fft<2u, 
    be_inv_col_type, workspace_type>::Functor<RBlock> fftm_inv_col_functor_type;

  typedef F<RBlock> unary_functor_type;

  typedef typename DstBlock::value_type dst_value_type;
  typedef typename SrcBlock::value_type src_value_type;

  typedef typename impl::Block_layout<DstBlock>::order_type dst_order_type;
  typedef typename impl::Block_layout<LBlock>::order_type l_order_type;
  typedef typename impl::Block_layout<RBlock>::order_type r_order_type;

  static bool const ct_valid =
    // the unary functor must be fftm
    (impl::Type_equal<fftm_fwd_row_functor_type, unary_functor_type>::value ||
     impl::Type_equal<fftm_fwd_col_functor_type, unary_functor_type>::value ||
     impl::Type_equal<fftm_inv_row_functor_type, unary_functor_type>::value ||
     impl::Type_equal<fftm_inv_col_functor_type, unary_functor_type>::value) &&
    // only complex is presently handled
    impl::Type_equal<dst_value_type, std::complex<float> >::value &&
    impl::Type_equal<src_value_type, std::complex<float> >::value &&
    // source types must be the same as the result type
    impl::Type_equal<dst_value_type, typename LBlock::value_type>::value &&
    impl::Type_equal<dst_value_type, typename RBlock::value_type>::value &&
    // check that direct access is supported
    impl::Ext_data_cost<DstBlock>::value == 0 &&
    impl::Ext_data_cost<LBlock>::value == 0 &&
    impl::Ext_data_cost<RBlock>::value == 0 &&
    // complex split is not supported presently
    !impl::Is_split_block<DstBlock>::value &&
    !impl::Is_split_block<LBlock>::value &&
    !impl::Is_split_block<RBlock>::value &&
    // dimension ordering must be the same
    impl::Type_equal<dst_order_type, l_order_type>::value &&
    impl::Type_equal<dst_order_type, r_order_type>::value;

  static bool rt_valid(DstBlock& dst, SrcBlock const& src)
  {
    impl::Ext_data<DstBlock> ext_dst(dst);
    impl::Ext_data<LBlock>   ext_l(src.arg1());
    impl::Ext_data<RBlock>   ext_r(src.arg2().arg());

    dimension_type const dim0 = dst_order_type::impl_dim0;           
    dimension_type const dim1 = dst_order_type::impl_dim1;           

    return
      (ext_dst.stride(dim1) == 1) &&
      (ext_l.stride(dim1) == 1) &&
      (ext_r.stride(dim1) == 1) &&
      (ext_dst.stride(dim0) == static_cast<stride_type>(ext_dst.size(dim1))) &&
      (ext_l.stride(dim0) == static_cast<stride_type>(ext_l.size(dim1))) &&
      (ext_r.stride(dim0) == static_cast<stride_type>(ext_r.size(dim1)));
  }
  
  static void exec(DstBlock& dst, SrcBlock const& src)
  {
    // compute the Fftm, store the temporary result in the output block
    src.arg2().apply(dst);

    // complete by multiplying by the other input and returning
    impl::cuda::Device_memory<DstBlock> dev_dst(dst, impl::SYNC_INOUT);
    impl::cuda::Device_memory<LBlock const> dev_l(src.arg1());

    dimension_type const dim0 = dst_order_type::impl_dim0;
    dimension_type const dim1 = dst_order_type::impl_dim1;

    // matrix-matrix multiply with scaling
    impl::cuda::mmmuls(
      dev_l.data(),    // input
      dev_dst.data(),  // input
      dev_dst.data(),  // output
      1.0f,            // scale factor
      dev_dst.size(dim0), dev_dst.size(dim1));
  }
};

} // namespace vsip::dispatcher

} // namespace vsip

#endif // VSIP_OPT_CUDA_EVAL_FFTMUL_HPP
