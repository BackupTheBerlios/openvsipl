/* Copyright (c) 2009 by CodeSourcery.  All rights reserved. */

/** @file    vsip/opt/cuda/eval_vmmul.hpp
    @author  Don McCoy
    @date    2009-04-07
    @brief   VSIPL++ Library: CUDA evaluator for vector-matrix multiply.

*/

#ifndef VSIP_OPT_CUDA_EVAL_VMMUL_HPP
#define VSIP_OPT_CUDA_EVAL_VMMUL_HPP

#include <vsip/opt/expr/assign_fwd.hpp>
#include <vsip/opt/cuda/device_memory.hpp>
#include <vsip/opt/cuda/kernels.hpp>

namespace vsip_csl
{
namespace dispatcher
{

/// Evaluator for vector-matrix multiply.
///
/// Dispatches cases where the dimension ordering matches the 
/// requested orientation to the SPU's (row-major/by-row and 
/// col-major/by-col).  The other cases are re-dispatched.
template <typename DstBlock,
	  typename VBlock,
	  typename MBlock,
	  dimension_type SD>
struct Evaluator<op::assign<2>, be::cuda,
		 void(DstBlock &, expr::Vmmul<SD, VBlock, MBlock> const &)>
{
  static char const* name() { return "CUDA_vmmul"; }

  typedef expr::Vmmul<SD, VBlock, MBlock> SrcBlock;
  typedef typename SrcBlock::value_type src_type;
  typedef typename DstBlock::value_type dst_type;
  typedef typename VBlock::value_type v_type;
  typedef typename MBlock::value_type m_type;
  typedef typename impl::Block_layout<DstBlock>::layout_type dst_lp;
  typedef typename impl::Block_layout<VBlock>::layout_type vblock_lp;
  typedef typename impl::Block_layout<MBlock>::layout_type mblock_lp;
  typedef typename impl::Block_layout<DstBlock>::order_type order_type;
  typedef typename impl::Block_layout<MBlock>::order_type src_order_type;

  static bool const ct_valid = 
    // inputs must not be expression blocks
    impl::Is_expr_block<VBlock>::value == 0 &&
    impl::Is_expr_block<MBlock>::value == 0 &&
    // split complex not supported
    impl::Is_split_block<DstBlock>::value == 0 &&
    impl::Is_split_block<VBlock>::value == 0 &&
    impl::Is_split_block<MBlock>::value == 0 &&
    // ensure value types are supported
    impl::cuda::Cuda_traits<dst_type>::valid &&
    impl::cuda::Cuda_traits<v_type>::valid &&
    impl::cuda::Cuda_traits<m_type>::valid &&
    // result type must match type expected (determined by promotion)
    impl::Type_equal<dst_type, src_type>::value &&
    // check that direct access is supported
    impl::Ext_data_cost<DstBlock>::value == 0 &&
    impl::Ext_data_cost<VBlock>::value == 0 &&
    impl::Ext_data_cost<MBlock>::value == 0 &&
    // dimension ordering must be the same
    impl::Type_equal<order_type, src_order_type>::value;

  static bool rt_valid(DstBlock& dst, SrcBlock const& src)
  {
    VBlock const& vblock = src.get_vblk();
    MBlock const& mblock = src.get_mblk();

    impl::Ext_data<DstBlock, dst_lp>  ext_dst(dst, impl::SYNC_OUT);
    impl::Ext_data<VBlock, vblock_lp> ext_v(vblock, impl::SYNC_IN);
    impl::Ext_data<MBlock, mblock_lp> ext_m(mblock, impl::SYNC_IN);

    if ((SD == row && impl::Type_equal<order_type, row2_type>::value) ||
        (SD == col && impl::Type_equal<order_type, col2_type>::value))
    {
      dimension_type const axis = SD == row ? 1 : 0;
      length_type dst_stride = static_cast<length_type>(abs(ext_dst.stride(axis == 0)));
      length_type m_stride = static_cast<length_type>(abs(ext_m.stride(axis == 0)));
      return 
        // make sure blocks are dense (major stride == minor size)
        (ext_dst.size(axis) == dst_stride) &&
        (ext_m.size(axis) == m_stride) &&
        // ensure unit stride along the dimension opposite the chosen one
	(ext_dst.stride(axis) == 1) &&
	(ext_m.stride(axis) == 1) &&
	(ext_v.stride(0) == 1);
    }
    else
    {
      dimension_type const axis = SD == row ? 0 : 1;
      length_type dst_stride = static_cast<length_type>(abs(ext_dst.stride(axis == 0)));
      length_type m_stride = static_cast<length_type>(abs(ext_m.stride(axis == 0)));
      return 
        // make sure blocks are dense (major stride == minor size)
        (ext_dst.size(axis) == dst_stride) &&
        (ext_m.size(axis) == m_stride) &&
        // ensure unit stride along the same dimension as the chosen one
	(ext_dst.stride(axis) == 1) &&
	(ext_m.stride(axis) == 1) &&
	(ext_v.stride(0) == 1);
    }
  }
  
  static void exec(DstBlock& dst, SrcBlock const& src)
  {
    VBlock const& vblock = src.get_vblk();
    MBlock const& mblock = src.get_mblk();

    impl::cuda::Device_memory<DstBlock> dev_dst(dst, impl::SYNC_OUT);
    impl::cuda::Device_memory<VBlock const> dev_v(vblock);
    impl::cuda::Device_memory<MBlock const> dev_m(mblock);

    // The ct_valid check above ensures that the order taken 
    // matches the storage order if reaches this point.
    if (SD == row && impl::Type_equal<order_type, row2_type>::value)
    {
      impl::cuda::vmmul_row(dev_v.data(),
			    dev_m.data(),
			    dev_dst.data(),
			    dst.size(2, 0),    // number of rows
			    dst.size(2, 1));   // length of each row
    }
    else if (SD == col && impl::Type_equal<order_type, row2_type>::value)
    {
      impl::cuda::vmmul_col(dev_v.data(),
			    dev_m.data(),
			    dev_dst.data(),
			    dst.size(2, 0),    // number of rows
			    dst.size(2, 1));   // length of each row
    }
    else if (SD == col && impl::Type_equal<order_type, col2_type>::value)
    {
      impl::cuda::vmmul_row(dev_v.data(),
			    dev_m.data(),
			    dev_dst.data(),
			    dst.size(2, 1),    // number of cols
			    dst.size(2, 0));   // length of each col
    }
    else // if (SD == row && impl::Type_equal<order_type, col2_type>::value)
    {
      impl::cuda::vmmul_col(dev_v.data(),
			    dev_m.data(),
			    dev_dst.data(),
			    dst.size(2, 1),    // number of cols
			    dst.size(2, 0));   // length of each col
    }
  }
};

} // namespace vsip_csl::dispatcher
} // namespace vsip_csl

#endif // VSIP_OPT_CUDA_EVAL_VMMUL_HPP
