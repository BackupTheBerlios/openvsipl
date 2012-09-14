/* Copyright (c) 2006 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    vsip/opt/cbe/ppu/bindings.hpp
    @author  Stefan Seefeld
    @date    2006-12-29
    @brief   VSIPL++ Library: Wrappers and traits to bridge with IBMs CBE SDK.
*/

#ifndef VSIP_OPT_CBE_PPU_BINDINGS_HPP
#define VSIP_OPT_CBE_PPU_BINDINGS_HPP

#if VSIP_IMPL_REF_IMPL
# error "vsip/opt files cannot be used as part of the reference impl."
#endif

#include <vsip/opt/expr/assign_fwd.hpp>
#include <vsip/core/adjust_layout.hpp>
#include <vsip/opt/cbe/vmmul_params.h>
#include <vsip/opt/cbe/ppu/task_manager.hpp>
#include <vsip/opt/cbe/ppu/util.hpp>

namespace vsip
{
namespace impl
{
namespace cbe
{

typedef complex<float> CF;

template <typename T> void vmul(T const* A, T const* B, T* R, length_type len);
template <typename T> void vmul(std::pair<T*, T*> const& A,
				std::pair<T*, T*> const& B,
				std::pair<T*, T*> const& R,
				length_type              len);
template <typename T> void vadd(T const* A, T const* B, T* R, length_type len);
template <typename T> void vadd(std::pair<T*, T*> const& A,
				std::pair<T*, T*> const& B,
				std::pair<T*, T*> const& R,
				length_type              len);

template <template <typename, typename> class Operator,
	  typename A, bool AIsSplit,
	  typename B, bool BIsSplit,
	  typename C, bool CIsSplit>
struct Is_op_supported { static bool const value = false;};

template <template <typename, typename> class O>
struct Is_op_supported<O, CF, false, CF, false, CF, false>
{
  static bool const value = true;
};

template <template <typename, typename> class O>
struct Is_op_supported<O, CF, true, CF, true, CF, true>
{
  static bool const value = true;
};

template <template <typename, typename> class O>
struct Is_op_supported<O, float, false, float, false, float, false>
{
  static bool const value = true;
};


template <typename T> void
vmmul_row(T const* V, T const* M, T* R, 
      stride_type m_stride, stride_type r_stride,
      length_type length, length_type lines);

template <typename T>
void vmmul_row(
  T const* V,
  std::pair<T*, T*> const& M,
  std::pair<T*, T*> const&             R,
  stride_type m_stride, stride_type r_stride, 
  length_type lines, length_type length);

template <typename T>
void vmmul_row(
  std::pair<T*, T*> const& V,
  std::pair<T*, T*> const& M,
  std::pair<T*, T*> const&             R,
  stride_type m_stride, stride_type r_stride, 
  length_type lines, length_type length);

template <typename T> void
vmmul_col(T const* V, T const* M, T* R, 
      stride_type m_stride, stride_type r_stride,
      length_type length, length_type lines);

template <typename T>
void vmmul_col(
  T const* V,
  std::pair<T*, T*> const& M,
  std::pair<T*, T*> const&             R,
  stride_type m_stride, stride_type r_stride, 
  length_type lines, length_type length);

template <typename T>
void vmmul_col(
  std::pair<T*, T*> const& V,
  std::pair<T*, T*> const& M,
  std::pair<T*, T*> const&             R,
  stride_type m_stride, stride_type r_stride, 
  length_type lines, length_type length);



// Vmmul cases supported by CBE backends

template <dimension_type RowCol,
	  typename       VType,
	  bool           VIsSplit,
	  typename       MType,
	  bool           MIsSplit,
	  typename       DstType,
	  bool           DstIsSplit>
struct Is_vmmul_supported
{ static bool const value = false; };

// vmmul_row(C, C, C)
template <>
struct Is_vmmul_supported<row, CF, false, CF, false, CF, false>
{ static bool const value = true; };

// vmmul_row(Z, Z, Z)
template <>
struct Is_vmmul_supported<row, CF, true, CF, true, CF, true>
{ static bool const value = true; };

// vmmul_col(R, Z, Z)
template <bool NA>
struct Is_vmmul_supported<col, float, NA, CF, true, CF, true>
{ static bool const value = true; };

// vmmul_col(Z, Z, Z)
template <>
struct Is_vmmul_supported<col, CF, true, CF, true, CF, true>
{ static bool const value = true; };

template <template <typename, typename> class Operator,
	  typename DstBlock,
	  typename LBlock,
	  typename RBlock>
struct Evaluator
{
  typedef expr::Binary<Operator, LBlock, RBlock, true> SrcBlock;

  typedef typename Adjust_layout_dim<
      1, typename Block_layout<DstBlock>::layout_type>::type
    dst_lp;

  typedef typename Adjust_layout_dim<
      1, typename Block_layout<LBlock>::layout_type>::type
    lblock_lp;

  typedef typename Adjust_layout_dim<
      1, typename Block_layout<RBlock>::layout_type>::type
    rblock_lp;

  static bool const ct_valid = 
    !Is_expr_block<LBlock>::value &&
    !Is_expr_block<RBlock>::value &&
    Is_op_supported<Operator,
		    typename DstBlock::value_type,
		    impl::Is_split_block<DstBlock>::value,
		    typename LBlock::value_type,
		    impl::Is_split_block<LBlock>::value,
		    typename RBlock::value_type,
		    impl::Is_split_block<RBlock>::value>::value &&
     // check that direct access is supported
     Ext_data_cost<DstBlock>::value == 0 &&
     Ext_data_cost<LBlock>::value == 0 &&
     Ext_data_cost<RBlock>::value == 0;

  static length_type tunable_threshold()
  {
    typedef typename DstBlock::value_type T;

    if (VSIP_IMPL_TUNE_MODE)
      return 0;
    // Compare interleaved vmul -2 --svpp-num-spes {0,8}.
    else if (Type_equal<Operator<T, T>,
	     expr::op::Mult<complex<float>, complex<float> > >::value)
      return 16384;
    // Compare vmul -1 --svpp-num-spes {0,8}.
    else if (Type_equal<Operator<T, T>, expr::op::Mult<float, float> >::value)
      return 65536;
    else
      return 0;
  }

  static bool rt_valid(DstBlock& dst, SrcBlock const& src)
  {
    // check if all data is unit stride
    Ext_data<DstBlock, dst_lp>    ext_dst(dst,       SYNC_OUT);
    Ext_data<LBlock,   lblock_lp> ext_l(src.arg1(),  SYNC_IN);
    Ext_data<RBlock,   rblock_lp> ext_r(src.arg2(), SYNC_IN);
    return ext_dst.size(0) >= tunable_threshold() &&
           ext_dst.stride(0) == 1 &&
	   ext_l.stride(0) == 1   &&
	   ext_r.stride(0) == 1   &&
	   is_dma_addr_ok(ext_dst.data()) &&
	   is_dma_addr_ok(ext_l.data())   &&
	   is_dma_addr_ok(ext_r.data())   &&
           Task_manager::instance()->num_spes() > 0;
  }
};

} // namespace vsip::impl::cbe
} // namespace vsip::impl
} // namespace vsip

namespace vsip_csl
{
namespace dispatcher
{

#define VSIP_IMPL_CBE_PPU_VV_EXPR(OP, FUN)				\
template <typename LHS, typename LBlock, typename RBlock>		\
struct Evaluator<op::assign<1>, be::cbe_sdk,				\
		 void(LHS &,				                \
		      expr::Binary<OP, LBlock, RBlock, true> const &)>	\
  : impl::cbe::Evaluator<OP, LHS, LBlock, RBlock>                       \
{									\
  static char const* name() { return "Expr_CBE_SDK_VV-" #FUN; }		\
									\
  typedef expr::Binary<OP, LBlock, RBlock, true> RHS;		        \
  									\
  typedef typename impl::Adjust_layout_dim<				\
  1, typename impl::Block_layout<LHS>::layout_type>::type		\
    dst_lp;								\
  									\
  typedef typename impl::Adjust_layout_dim<				\
  1, typename impl::Block_layout<LBlock>::layout_type>::type		\
    lblock_lp;								\
  									\
  typedef typename impl::Adjust_layout_dim<				\
  1, typename impl::Block_layout<RBlock>::layout_type>::type		\
    rblock_lp;								\
  									\
  static void exec(LHS &lhs, RHS const &rhs)			        \
  {									\
    impl::Ext_data<LHS, dst_lp> ext_dst(lhs, impl::SYNC_OUT);           \
    impl::Ext_data<LBlock, lblock_lp> ext_l(rhs.arg1(), impl::SYNC_IN);	\
    impl::Ext_data<RBlock, rblock_lp> ext_r(rhs.arg2(), impl::SYNC_IN); \
									\
    FUN(ext_l.data(), ext_r.data(), ext_dst.data(), lhs.size());	\
  } 									\
};

VSIP_IMPL_CBE_PPU_VV_EXPR(expr::op::Add,  impl::cbe::vadd)
// VSIP_IMPL_CBE_PPU_VV_EXPR(expr::op::Sub,  cbe::vsub)
VSIP_IMPL_CBE_PPU_VV_EXPR(expr::op::Mult,  impl::cbe::vmul)
// VSIP_IMPL_CBE_PPU_VV_EXPR(expr::op::Div,  cbe::vdiv)

#undef VSIP_IMPL_CBE_PPU_VV_EXPR



/// Evaluator for vector-matrix multiply.

/// Dispatches cases where the dimension ordering matches the 
/// requested orientation to the SPU's (row-major/by-row and 
/// col-major/by-col).  The other cases are re-dispatched.
template <typename LHS, typename VBlock, typename MBlock, dimension_type D>
struct Evaluator<op::assign<2>, be::cbe_sdk,
		 void(LHS &, expr::Vmmul<D, VBlock, MBlock> const &)>
{
  static char const* name() { return "Cbe_Sdk_Vmmul"; }

  typedef expr::Vmmul<D, VBlock, MBlock> RHS;
  typedef typename LHS::value_type lhs_value_type;
  typedef typename VBlock::value_type v_value_type;
  typedef typename MBlock::value_type m_value_type;
  typedef typename impl::Block_layout<LHS>::layout_type lhs_lp;
  typedef typename impl::Block_layout<VBlock>::layout_type vblock_lp;
  typedef typename impl::Block_layout<MBlock>::layout_type mblock_lp;
  typedef typename impl::Block_layout<LHS>::order_type lhs_order_type;
  typedef typename impl::Block_layout<MBlock>::order_type src_order_type;

  static bool const is_row_vmmul =
    (D == row && impl::Type_equal<lhs_order_type, row2_type>::value ||
     D == col && impl::Type_equal<lhs_order_type, col2_type>::value);

  static bool const ct_valid = 
    !impl::Is_expr_block<VBlock>::value &&
    !impl::Is_expr_block<MBlock>::value &&
    impl::cbe::Is_vmmul_supported<is_row_vmmul ? row : col,
			    v_value_type,
			    impl::Is_split_block<VBlock>::value,
			    m_value_type,
			    impl::Is_split_block<MBlock>::value,
			    lhs_value_type,
			    impl::Is_split_block<LHS>::value>::value &&
     // check that direct access is supported
    impl::Ext_data_cost<LHS>::value == 0 &&
    impl::Ext_data_cost<VBlock>::value == 0 &&
    impl::Ext_data_cost<MBlock>::value == 0 &&
    impl::Type_equal<lhs_order_type, src_order_type>::value;

  static bool rt_valid(LHS &lhs, RHS const &rhs)
  {
    VBlock const& vblock = rhs.get_vblk();
    MBlock const& mblock = rhs.get_mblk();

    impl::Ext_data<LHS, lhs_lp> ext_dst(lhs, impl::SYNC_OUT);
    impl::Ext_data<VBlock, vblock_lp> ext_v(vblock, impl::SYNC_IN);
    impl::Ext_data<MBlock, mblock_lp> ext_m(mblock, impl::SYNC_IN);

    if (is_row_vmmul)
    {
      dimension_type const axis = D == row ? 1 : 0;
      length_type size = lhs.size(2, axis);

      return 
	// (large sizes are broken down)
	(size >= VSIP_IMPL_MIN_VMMUL_SIZE) && 
	(ext_dst.stride(axis) == 1) &&
	(ext_m.stride(axis)   == 1) &&
	(ext_v.stride(0) == 1) &&
	impl::cbe::is_dma_addr_ok(ext_dst.data()) &&
	impl::cbe::is_dma_addr_ok(ext_v.data()) &&
	impl::cbe::is_dma_addr_ok(ext_m.data()) &&
	impl::cbe::is_dma_stride_ok<lhs_value_type>(ext_dst.stride(D == row ? 0 : 1)) &&
	impl::cbe::is_dma_stride_ok<m_value_type>(ext_m.stride(D == row ? 0 : 1)) &&
	// (non-granular sizes handled)
	impl::cbe::Task_manager::instance()->num_spes() > 0;
    }
    else
    {
      dimension_type const axis = D == row ? 0 : 1;
      length_type size = lhs.size(2, axis);

      return 
	// (large sizes are broken down)
	(size >= VSIP_IMPL_MIN_VMMUL_SIZE) && 
	(ext_dst.stride(axis) == 1) &&
	(ext_m.stride(axis)   == 1) &&
	(ext_v.stride(0) == 1) &&
	impl::cbe::is_dma_addr_ok(ext_dst.data()) &&
	// (V doesn't need to be DMA aligned)
	impl::cbe::is_dma_addr_ok(ext_m.data()) &&
	impl::cbe::is_dma_stride_ok<lhs_value_type>(ext_dst.stride(D == row ? 1 : 0)) &&
	impl::cbe::is_dma_stride_ok<m_value_type>(ext_m.stride(D == row ? 1 : 0)) &&
	impl::cbe::is_dma_size_ok(size * sizeof(v_value_type)) &&
	impl::cbe::Task_manager::instance()->num_spes() > 0;
    }
  }
  
  static void exec(LHS &lhs, RHS const &rhs)
  {
    VBlock const& vblock = rhs.get_vblk();
    MBlock const& mblock = rhs.get_mblk();

    Matrix<lhs_value_type, LHS> m_dst(lhs);
    const_Vector<lhs_value_type, VBlock>  v(const_cast<VBlock&>(vblock));
    const_Matrix<lhs_value_type, MBlock>  m(const_cast<MBlock&>(mblock));

    impl::Ext_data<LHS, lhs_lp> ext_dst(lhs, impl::SYNC_OUT);
    impl::Ext_data<VBlock, vblock_lp> ext_v(vblock, impl::SYNC_IN);
    impl::Ext_data<MBlock, mblock_lp> ext_m(mblock, impl::SYNC_IN);

    // The ct_valid check above ensures that the order taken 
    // matches the storage order if reaches this point.
    if (D == row && impl::Type_equal<lhs_order_type, row2_type>::value)
    {
      impl::cbe::vmmul_row(ext_v.data(),
			   ext_m.data(),
			   ext_dst.data(),
			   ext_m.stride(0),   // elements between rows of source matrix
			   ext_dst.stride(0), // elements between rows of destination matrix
			   lhs.size(2, 0),    // number of rows
			   lhs.size(2, 1));   // length of each row
    }
    else if (D == col && impl::Type_equal<lhs_order_type, row2_type>::value)
    {
      impl::cbe::vmmul_col(ext_v.data(),
			   ext_m.data(),
			   ext_dst.data(),
			   ext_m.stride(0),   // elements between rows of source matrix
			   ext_dst.stride(0), // elements between rows of destination matrix
			   lhs.size(2, 0),    // number of rows
			   lhs.size(2, 1));   // length of each row
    }
    else if (D == col && impl::Type_equal<lhs_order_type, col2_type>::value)
    {
      impl::cbe::vmmul_row(ext_v.data(),
			   ext_m.data(),
			   ext_dst.data(),
			   ext_m.stride(1),   // elements between cols of source matrix
			   ext_dst.stride(1), // elements between cols of destination matrix
			   lhs.size(2, 1),    // number of cols
			   lhs.size(2, 0));   // length of each col
    }
    else // if (D == row && Type_equal<order_type, col2_type>::value)
    {
      impl::cbe::vmmul_col(ext_v.data(),
			   ext_m.data(),
			   ext_dst.data(),
			   ext_m.stride(1),   // elements between cols of source matrix
			   ext_dst.stride(1), // elements between cols of destination matrix
			   lhs.size(2, 1),    // number of cols
			   lhs.size(2, 0));   // length of each col
    }
  }
};

} // namespace vsip_csl::dispatcher
} // namespace vsip_csl

#endif
