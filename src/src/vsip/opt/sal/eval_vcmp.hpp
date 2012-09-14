/* Copyright (c) 2006 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    vsip/opt/sal/eval_vcmp.hpp
    @author  Jules Bergmann
    @date    2006-10-26
    @brief   VSIPL++ Library: Dispatch for Mercury SAL -- vector comparisons.
*/

#ifndef VSIP_OPT_SAL_EVAL_VCMP_HPP
#define VSIP_OPT_SAL_EVAL_VCMP_HPP

#if VSIP_IMPL_REF_IMPL
# error "vsip/opt files cannot be used as part of the reference impl."
#endif

#include <vsip/opt/expr/assign_fwd.hpp>
#include <vsip/core/expr/fns_elementwise.hpp>
#include <vsip/core/extdata.hpp>
#include <vsip/core/metaprogramming.hpp>
#include <vsip/opt/sal/eval_util.hpp>
#include <vsip/core/adjust_layout.hpp>
#include <vsip/opt/sal/is_op_supported.hpp>

namespace vsip_csl
{
namespace dispatcher
{

///  Threshold Expressions
/// Optimize threshold expression: ite(A > B, VAL1, VAL0)
#define VSIP_IMPL_SAL_VCMP_EXPR(FUNCTOR, OPTOKEN, FUN, VAL1, VAL0)	\
template <typename DstBlock,						\
	  typename T,							\
	  typename Block2,						\
	  typename Block1>						\
struct Evaluator<op::assign<1>, be::mercury_sal,			\
  void(DstBlock &,							\
       expr::Ternary<expr::op::Ite,					\
	   expr::Binary<FUNCTOR, Block1, Block2, true> const,           \
	   expr::Scalar<1, T>,				                \
	   expr::Scalar<1, T>,				                \
       true> const &)>				                        \
{									\
  static char const* name() { return "Expr_SAL_vsmp-" # FUNCTOR; }	\
									\
  typedef expr::Ternary<expr::op::Ite,					\
    expr::Binary<FUNCTOR, Block1, Block2, true> const,		        \
    expr::Scalar<1, T>,					                \
    expr::Scalar<1, T>,						        \
    true>					                        \
	SrcBlock;							\
									\
  typedef typename DstBlock::value_type dst_type;			\
									\
  typedef typename impl::sal::Effective_value_type<DstBlock>::type  eff_d_t; \
  typedef typename impl::sal::Effective_value_type<Block1, T>::type eff_1_t;	\
  typedef typename impl::sal::Effective_value_type<Block2, T>::type eff_2_t;	\
									\
  typedef typename impl::Adjust_layout_dim<				\
      1, typename impl::Block_layout<DstBlock>::layout_type>::type		\
    dst_lp;								\
  									\
  typedef typename impl::Adjust_layout_dim<					\
      1, typename impl::Block_layout<Block1>::layout_type>::type		\
    block1_lp;								\
  									\
  typedef typename impl::Adjust_layout_dim<					\
      1, typename impl::Block_layout<Block2>::layout_type>::type		\
    block2_lp;								\
  									\
  static bool const ct_valid =						\
     impl::sal::Is_op2_supported<OPTOKEN, eff_1_t, eff_2_t, eff_d_t>::value &&\
     /* check that direct access is supported */			\
     impl::Ext_data_cost<DstBlock>::value == 0 &&				\
     impl::Ext_data_cost<Block1>::value == 0 &&				\
     impl::Ext_data_cost<Block2>::value == 0;					\
									\
  static bool rt_valid(DstBlock&, SrcBlock const& src)			\
  {									\
    return src.arg2().value() == T(VAL1) &&				\
      src.arg3().value() == T(VAL0);					\
  }									\
									\
  static void exec(DstBlock& dst, SrcBlock const& src)			\
  {									\
    using namespace impl;                                               \
    typedef expr::Scalar<1, T> sb_type;				        \
    sal::Ext_wrapper<DstBlock, dst_lp>  ext_dst(dst, SYNC_OUT);       	\
    sal::Ext_wrapper<Block1, block1_lp> ext_A(src.arg1().arg1(), SYNC_IN);\
    sal::Ext_wrapper<Block2, block2_lp> ext_B(src.arg1().arg2(), SYNC_IN);\
									\
    FUN(typename sal::Ext_wrapper<Block1, block1_lp>::sal_type(ext_A),	\
	typename sal::Ext_wrapper<Block2, block2_lp>::sal_type(ext_B),	\
	typename sal::Ext_wrapper<DstBlock, dst_lp>::sal_type(ext_dst),	\
	dst.size());							\
  }									\
};

VSIP_IMPL_SAL_VCMP_EXPR(expr::op::Eq, impl::sal::veq_token, impl::sal::lveq, 1, 0)
VSIP_IMPL_SAL_VCMP_EXPR(expr::op::Ne, impl::sal::vne_token, impl::sal::lvne, 1, 0)
VSIP_IMPL_SAL_VCMP_EXPR(expr::op::Gt, impl::sal::vgt_token, impl::sal::lvgt, 1, 0)
VSIP_IMPL_SAL_VCMP_EXPR(expr::op::Ge, impl::sal::vge_token, impl::sal::lvge, 1, 0)
VSIP_IMPL_SAL_VCMP_EXPR(expr::op::Lt, impl::sal::vlt_token, impl::sal::lvlt, 1, 0)
VSIP_IMPL_SAL_VCMP_EXPR(expr::op::Le, impl::sal::vle_token, impl::sal::lvle, 1, 0)

} // namespace vsip::impl
} // namespace vsip

#endif // VSIP_OPT_SAL_EVAL_VCMP_HPP
