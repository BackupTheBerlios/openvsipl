/* Copyright (c) 2009 by CodeSourcery.  All rights reserved. */

/** @file    vsip/opt/cuda/eval_elementwise.hpp
    @author  Don McCoy
    @date    2009-06-23
    @brief   VSIPL++ Library: CUDA evaluators for unary and binary expressions.

*/

#ifndef VSIP_OPT_CUDA_EVAL_ELEMENTWISE_HPP
#define VSIP_OPT_CUDA_EVAL_ELEMENTWISE_HPP

#include <vsip/opt/expr/assign_fwd.hpp>
#include <vsip/opt/cuda/bindings.hpp>
#include <vsip/opt/cuda/block_unwrapper.hpp>
#include <vsip/opt/cuda/device_memory.hpp>


namespace vsip_csl
{
namespace dispatcher
{

// M_CR_EXPR   for matrix expressions taking a complex input block and 
//             producing a real output block
#define VSIP_IMPL_CUDA_M_CR_EXPR(OP, FUN)                               \
template <typename LHS, typename Block>                                 \
struct Evaluator<op::assign<2>, be::cuda,				\
		 void(LHS &, expr::Unary<OP, Block, true> const &)>     \
{                                                                       \
  static char const* name() { return "Expr_CUDA_M_CR-" #FUN; }          \
                                                                        \
  typedef expr::Unary<OP, Block, true>  RHS;		                \
  typedef typename impl::cuda::Block_unwrapper<Block>::block_type       \
    src_block_type;                                                     \
                                                                        \
  static bool rt_valid(LHS &lhs, RHS const &rhs)                        \
  {                                                                     \
    /* check if all data is unit stride */                              \
    impl::Ext_data<LHS> ext_dst(lhs, impl::SYNC_OUT);		        \
    impl::Ext_data<Block> ext_src(rhs.arg(), impl::SYNC_IN);            \
    return (ext_dst.stride(1) == 1 && ext_src.stride(1) == 1);          \
  }                                                                     \
                                                                        \
  static bool const ct_valid =                                          \
    /* must be complex */                                               \
    impl::Is_complex<typename Block::value_type>::value &&              \
    /* cannot be an expression block */                                 \
    !impl::Is_expr_block<Block>::value &&                               \
    /* both destination and source block types must be supported */     \
                                                                        \
    impl::cuda::Cuda_traits<typename LHS::value_type>::valid &&         \
    impl::cuda::Cuda_traits<typename Block::value_type>::valid &&       \
    /* check that direct access is supported */                         \
    impl::Ext_data_cost<LHS>::value == 0 &&                             \
    impl::Ext_data_cost<Block>::value == 0 &&                           \
    /* complex split is not supported presently */                      \
    !impl::Is_split_block<LHS>::value &&                                \
    !impl::Is_split_block<Block>::value;                                \
                                                                        \
  static void exec(LHS &lhs, RHS const &rhs)                            \
  {                                                                     \
    src_block_type const& src_block =                                   \
      impl::cuda::Block_unwrapper<Block>::underlying_block(rhs.arg());  \
    impl::cuda::Device_memory<LHS> dev_dst(lhs, impl::SYNC_OUT);        \
    impl::cuda::Device_memory<src_block_type const> dev_src(src_block); \
                                                                        \
    FUN(dev_src.data(),                                                 \
	dev_dst.data(),							\
	dev_dst.size(0), dev_dst.size(1));				\
  }                                                                     \
};

//
// Definition list for all CUDA-supported unary operations:
//
VSIP_IMPL_CUDA_M_CR_EXPR(expr::op::Mag, impl::cuda::cmag)

#undef VSIP_IMPL_CUDA_M_CR_EXPR


} // namespace vsip_csl::dispatcher
} // namespace vsip_csl

#endif // VSIP_OPT_CUDA_EVAL_ELEMENTWISE_HPP

 
