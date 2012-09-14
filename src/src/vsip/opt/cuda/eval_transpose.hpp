/* Copyright (c) 2009 by CodeSourcery.  All rights reserved. */

/** @file    vsip/opt/cuda/eval_transpose.hpp
    @author  Don McCoy
    @date    2009-06-10
    @brief   VSIPL++ Library: CUDA evaluator for vector-matrix multiply.

*/

#ifndef VSIP_OPT_CUDA_EVAL_TRANSPOSE_HPP
#define VSIP_OPT_CUDA_EVAL_TRANSPOSE_HPP

#include <vsip/opt/expr/assign_fwd.hpp>
#include <vsip/opt/cuda/bindings.hpp>
#include <vsip/opt/cuda/block_unwrapper.hpp>
#include <vsip/opt/cuda/device_memory.hpp>


namespace vsip_csl
{
namespace dispatcher
{    

/// Evaluator for copy and transpose using CUDA
///
/// Dispatches cases where the dimension ordering matches the 
/// requested orientation to the SPU's (row-major/by-row and 
/// col-major/by-col).  The other cases are re-dispatched.
template <typename DstBlock, typename SrcBlock>
struct Evaluator<op::assign<2>, be::cuda, void(DstBlock &, SrcBlock const &)>
{
  typedef typename impl::cuda::Block_unwrapper<SrcBlock>::block_type src_block_type;

  typedef typename impl::Block_layout<SrcBlock>::order_type src_order_type;
  typedef typename impl::Block_layout<DstBlock>::order_type dst_order_type;

  typedef typename DstBlock::value_type dst_value_type;
  typedef typename SrcBlock::value_type src_value_type;

  static char const* name()
  {
    char s = impl::Type_equal<src_order_type, row2_type>::value ? 'r' : 'c';
    char d = impl::Type_equal<dst_order_type, row2_type>::value ? 'r' : 'c';
    if      (s == 'r' && d == 'r')    return "Expr_CUDA_Trans (rr copy)";
    else if (s == 'r' && d == 'c')    return "Expr_CUDA_Trans (rc trans)";
    else if (s == 'c' && d == 'r')    return "Expr_CUDA_Trans (cr trans)";
    else /* (s == 'c' && d == 'c') */ return "Expr_CUDA_Trans (cc copy)";
  }

  static bool const is_rhs_expr   = impl::Is_expr_block<SrcBlock>::value;

  static bool const is_lhs_split  = impl::Is_split_block<DstBlock>::value;
  static bool const is_rhs_split  = impl::Is_split_block<SrcBlock>::value;

  static int const  lhs_cost      = impl::Ext_data_cost<DstBlock>::value;
  static int const  rhs_cost      = impl::Ext_data_cost<SrcBlock>::value;

  static bool const ct_valid =
    // check that types are equal
    impl::Type_equal<src_value_type, dst_value_type>::value &&
    // check that CUDA supports this data type
    impl::cuda::Cuda_traits<src_value_type>::valid &&
    // check that the source block is not an expression
    !is_rhs_expr &&
    // check that direct access is supported
    lhs_cost == 0 && rhs_cost == 0 &&
    // check complex layout is not split (either real or interleaved are ok)
    !is_lhs_split &&
    !is_rhs_split;

  static bool rt_valid(DstBlock& dst, SrcBlock const& src)
  { 
    // Both source and destination blocks must be dense
    impl::Ext_data<DstBlock> dev_dst(reinterpret_cast<DstBlock const&>(dst));
    impl::Ext_data<SrcBlock> dev_src(src);

    dimension_type const s_dim0 = src_order_type::impl_dim0;
    dimension_type const s_dim1 = src_order_type::impl_dim1;
    dimension_type const d_dim0 = dst_order_type::impl_dim0;
    dimension_type const d_dim1 = dst_order_type::impl_dim1;

    return 
      (dev_dst.stride(d_dim1) == 1) &&
      (dev_src.stride(s_dim1) == 1) &&
      (dev_dst.stride(d_dim0) == static_cast<stride_type>(dev_dst.size(d_dim1))) &&
      (dev_src.stride(s_dim0) == static_cast<stride_type>(dev_src.size(s_dim1)));
  }

  static void exec(DstBlock& dst, src_block_type const& src, row2_type, row2_type)
  {
    impl::cuda::Device_memory<DstBlock> dev_dst(dst, impl::SYNC_OUT);
    impl::cuda::Device_memory<src_block_type const> dev_src(src);

    impl::cuda::copy(dev_src.data(),
		     dev_dst.data(),
		     dev_dst.size(0), dev_dst.size(1) );
  }

  static void exec(DstBlock& dst, src_block_type const& src, col2_type, col2_type)
  {
    impl::cuda::Device_memory<DstBlock> dev_dst(dst, impl::SYNC_OUT);
    impl::cuda::Device_memory<src_block_type const> dev_src(src);

    impl::cuda::copy(dev_src.data(),
		     dev_dst.data(),
		     dev_dst.size(0), dev_dst.size(1) );
  }

  static void exec(DstBlock& dst, src_block_type const& src, col2_type, row2_type)
  {
    impl::cuda::Device_memory<DstBlock> dev_dst(dst, impl::SYNC_OUT);
    impl::cuda::Device_memory<src_block_type const> dev_src(src);

    impl::cuda::transpose(dev_src.data(),
			  dev_dst.data(),
			  dev_dst.size(1), dev_dst.size(0) );
  }

  static void exec(DstBlock& dst, src_block_type const& src, row2_type, col2_type)
  {
    impl::cuda::Device_memory<DstBlock> dev_dst(dst, impl::SYNC_OUT);
    impl::cuda::Device_memory<src_block_type const> dev_src(src);

    impl::cuda::transpose(dev_src.data(),
			  dev_dst.data(),
			  dev_dst.size(0), dev_dst.size(1) );
  }

  static void exec(DstBlock& dst, SrcBlock const& src)
  {
    src_block_type const &src_block = impl::cuda::Block_unwrapper<SrcBlock>::underlying_block(src);

    exec(dst, src_block, dst_order_type(), src_order_type());
  }
};

} // namespace vsip_csl::dispatcher
} // namespace vsip_csl

#endif // VSIP_OPT_CUDA_EVAL_TRANSPOSE_HPP
