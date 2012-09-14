/* Copyright (c) 2006 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    vsip/opt/expr/eval_dense.hpp
    @author  Jules Bergmann
    @date    2006-06-05
    @brief   VSIPL++ Library: Evaluate a dense multi-dimensional expression
                              as a vector expression.
*/

#ifndef VSIP_OPT_EXPR_EVAL_DENSE_HPP
#define VSIP_OPT_EXPR_EVAL_DENSE_HPP

#if VSIP_IMPL_REF_IMPL
# error "vsip/opt files cannot be used as part of the reference impl."
#endif

#include <vsip/opt/expr/assign_fwd.hpp>

namespace vsip
{
namespace impl
{



/// redim_get and redim_put are helper functions for Redim_block
/// They allow a single Redim_block class definition to reduce both
/// 2-dimensional and 3-dimensional blocks.
template <typename BlockT>
typename BlockT::value_type
redim_get(BlockT const& blk, index_type l_idx, Int_type<2>)
{
  typedef typename Block_layout<BlockT>::order_type order_type;

  dimension_type dim[2];
  index_type     idx[2];
  dim[0] = order_type::impl_dim0;
  dim[1] = order_type::impl_dim1;

  for (dimension_type d=2; d-->0;)
  {
    idx[dim[d]] = l_idx % blk.size(2, dim[d]);
    l_idx /= blk.size(2, dim[d]);
  }

  return blk.get(idx[0], idx[1]);
}

template <typename BlockT>
void
redim_put(BlockT &blk,
	  index_type l_idx,
	  typename BlockT::value_type value,
	  Int_type<2>)
{
  typedef typename Block_layout<BlockT>::order_type order_type;

  dimension_type dim[2];
  index_type     idx[2];
  dim[0] = order_type::impl_dim0;
  dim[1] = order_type::impl_dim1;

  for (dimension_type d=2; d-->0;)
  {
    idx[dim[d]] = l_idx % blk.size(2, dim[d]);
    l_idx /= blk.size(2, dim[d]);
  }

  blk.put(idx[0], idx[1], value);
}



template <typename BlockT>
typename BlockT::value_type
redim_get(BlockT const& blk, index_type l_idx, Int_type<3>)
{
  typedef typename Block_layout<BlockT>::order_type order_type;

  dimension_type dim[3];
  index_type     idx[3];
  dim[0] = order_type::impl_dim0;
  dim[1] = order_type::impl_dim1;
  dim[2] = order_type::impl_dim2;

  for (dimension_type d=3; d-->0;)
  {
    idx[dim[d]] = l_idx % blk.size(3, dim[d]);
    l_idx /= blk.size(3, dim[d]);
  }

  return blk.get(idx[0], idx[1], idx[2]);
}



template <typename BlockT>
void
redim_put(BlockT &blk,
	  index_type l_idx,
	  typename BlockT::value_type value,
	  Int_type<3>)
{
  typedef typename Block_layout<BlockT>::order_type order_type;

  dimension_type dim[3];
  index_type     idx[3];
  dim[0] = order_type::impl_dim0;
  dim[1] = order_type::impl_dim1;
  dim[2] = order_type::impl_dim2;

  for (dimension_type d=3; d-->0;)
  {
    idx[dim[d]] = l_idx % blk.size(3, dim[d]);
    l_idx /= blk.size(3, dim[d]);
  }

  blk.put(idx[0], idx[1], idx[2], value);
}


/// Redimension block.
///
/// Provides a 1-dimensional view of a multidimensional block.
/// Intended for use when a multidimensional block refers to dense
/// data, but does not support 1,x-dimensional access (for example
/// a Sliced_block).  Redim_block's direct data interface requires
/// underlying block to be dense, but get/put work regardless of the
/// layout.
template <typename       BlockT,
	  dimension_type OrigDim>
class Redim_block
  : Compile_time_assert<OrigDim == 2 || OrigDim == 3>
{
  // Compile-time values and typedefs.
public:
  static dimension_type const dim = 1;

  typedef typename BlockT::value_type           value_type;
  typedef typename BlockT::reference_type       reference_type;
  typedef typename BlockT::const_reference_type const_reference_type;
  typedef typename BlockT::map_type             map_type;

  typedef typename Block_layout<BlockT>::order_type raw_order_type;

  // Constructors
public:
  Redim_block(BlockT& block)
    : blk_(&block)
  {}

  Redim_block(Redim_block const& rb) VSIP_NOTHROW
    : blk_(&*rb.blk_)
  {}

  ~Redim_block() VSIP_NOTHROW {}

  // Accessors
public:
  value_type get(index_type idx) const VSIP_NOTHROW
  {
    return redim_get(*blk_, idx, Int_type<OrigDim>());
  }

  void put(index_type idx, value_type val) const VSIP_NOTHROW
  {
    redim_put(*blk_, idx, val, Int_type<OrigDim>());
  }

  length_type size() const VSIP_NOTHROW
  { return blk_->size(); }

  length_type size(dimension_type D, dimension_type d) const VSIP_NOTHROW
  {
    assert(D == 1 && d == 0);
    return blk_->size();
  }

  map_type const& map() const VSIP_NOTHROW
  { return blk_->map_; }


  // Reference-counting (nop since Redim_block is held by-value).
public:
  void increment_count() const VSIP_NOTHROW {}
  void decrement_count() const VSIP_NOTHROW {}


  // Support Direct_data interface.
public:
  typedef Storage<typename Block_layout<BlockT>::complex_type, value_type>
		storage_type;
  typedef typename storage_type::type       data_type;
  typedef typename storage_type::const_type const_data_type;

  data_type       impl_data()       VSIP_NOTHROW
  { return blk_->impl_data(); }

  const_data_type impl_data() const VSIP_NOTHROW
  { return blk_->impl_data(); }

  stride_type impl_stride(dimension_type total_dim, dimension_type d)
     const VSIP_NOTHROW
  {
    // Force 1-dimensional access.  This should only be forced
    // when it makes sense of course.

    assert(total_dim == 1 && d == 0);
    return OrigDim == 2 ? blk_->impl_stride(2, raw_order_type::impl_dim1)
                        : blk_->impl_stride(3, raw_order_type::impl_dim2);
  }


  // Member data.
private:
  typename View_block_storage<BlockT>::type blk_;
};



template <typename       BlockT,
	  dimension_type Dim>
struct Block_layout<Redim_block<BlockT, Dim> >
{
  // Dimension: 1
  // Access   : Same
  // Order    : row1_type
  // Stride   : Stride_unit if parent Stride_unit*
  //            Stride_unknown otherwise
  // Cmplx    : Same

public:
  static dimension_type const dim = 1;

  typedef typename Block_layout<BlockT>::access_type access_type;
  typedef row1_type                                 order_type;
  typedef typename ITE_Type<
    Block_layout<BlockT>::pack_type::is_ct_unit_stride,
    As_type<Stride_unit>, As_type<Stride_unknown> >::type pack_type;
  typedef typename Block_layout<BlockT>::complex_type complex_type;

  typedef Layout<dim, order_type, pack_type, complex_type> layout_type;
};

// Store Redim_block by-value.
template <typename BlockT, dimension_type Dim>
struct View_block_storage<Redim_block<BlockT, Dim> >
  : By_value_block_storage<Redim_block<BlockT, Dim> >
{};



/// Expression reduction to determine if an expression consists of
/// dense data at the leaves (either blocks with stride_unit_dense 
/// packing, subviews that are dense, or scalar_blocks).  Check is
/// done at runtime, checking for gaps in highest-dimension stride.
struct Reduce_is_expr_dense
{
  template <dimension_type            Dim0,
	    typename                  T>
  bool
  apply(expr::Scalar<Dim0, T> const&)
  {
    return true;
  }

  template <template <typename> class O, typename B>
  bool
  apply(expr::Unary<O, B, true> const &b) { return apply(b.arg());}

  template <template <typename, typename> class Operation,
	    typename LBlock,
	    typename RBlock>
  bool
  apply(expr::Binary<Operation, LBlock, RBlock, true> const& blk)
  {
    return apply(blk.arg1()) && apply(blk.arg2());
  }

  template <template <typename, typename, typename> class Operation,
	    typename Block1,
	    typename Block2,
	    typename Block3>
  bool
  apply(expr::Ternary<Operation, Block1, Block2, Block3, true> const& blk)
  {
    return apply(blk.arg1()) && apply(blk.arg2()) && apply(blk.arg3());
  }

  // Leaf combine function.
  template <typename BlockT>
  bool
  apply(BlockT const&, Bool_type<false>) const
  {
    return false;
  }

  // Leaf combine function.
  template <typename BlockT>
  bool
  apply(BlockT const& block, Bool_type<true>) const
  {
    typedef typename Block_layout<BlockT>::order_type order_type;

    Ext_data<BlockT> ext(block, SYNC_IN);

    if (Block_layout<BlockT>::dim == 1)
      return ext.stride(0) == 1;
    else if (Block_layout<BlockT>::dim == 2)
      return (ext.stride(order_type::impl_dim0) ==
	      static_cast<stride_type>(ext.size(order_type::impl_dim1)))
	     && ext.stride(order_type::impl_dim1) == 1;
    else if (Block_layout<BlockT>::dim == 3)
      return (ext.stride(order_type::impl_dim0) ==
	      static_cast<stride_type>(ext.size(order_type::impl_dim1) *
				       ext.size(order_type::impl_dim2)))
             && ext.stride(order_type::impl_dim2) == 1;
    else return false;
  }

  // Leaf combine function.
  template <typename BlockT>
  bool
  apply(BlockT const& block) const
  {
    return apply(block, Bool_type<Ext_data_cost<BlockT>::value == 0>());
  }
};

/// Helper function to apply Reduce_is_expr_dense reduction.
template <typename BlockT>
bool
is_expr_dense(BlockT& blk)
{
  Reduce_is_expr_dense obj;
  return obj.apply(blk);
}



/// Reduction to redimension an expression from x-dimensional (where x > 1)
/// to 1-dimensional.
///
/// Transform expression block dimensions to 1, keeps dense blocks
/// (which are 1,x-dimensional) as is, wraps other blocks with Redim_block.
template <dimension_type NewDim>
class Redim_expr
{
public:
  template <typename BlockT>
  struct leaf_node
  {
    typedef Redim_block<BlockT, Block_layout<BlockT>::dim> type;
  };

  template <dimension_type Dim0,
	    typename       T,
	    typename       OrderT,
	    typename       MapT>
  struct leaf_node<Dense<Dim0, T, OrderT, MapT> >
  {
    typedef Dense<Dim0, T, OrderT, MapT> type;
  };

  template <dimension_type Dim0,
	    typename       T,
	    typename       OrderT>
  struct leaf_node<Dense<Dim0, T, OrderT, Local_or_global_map<Dim0> > >
  {
    typedef Dense<Dim0, T, OrderT, Local_map> type;
  };

  template <dimension_type Dim0,
	    typename       T>
  struct leaf_node<expr::Scalar<Dim0, T> >
  {
    typedef expr::Scalar<NewDim, T> type;
  };

  template <template <typename> class O, typename B>
  struct unary_node
  {
    typedef expr::Unary<O, B, true> const type;
  };

  template <template <typename, typename> class Operation,
	    typename LBlock,
	    typename RBlock>
  struct binary_node
  {
    typedef expr::Binary<Operation, LBlock, RBlock, true> const type;
  };

  template <template <typename, typename, typename> class Operation,
	    typename Block1,
	    typename Block2,
	    typename Block3>
  struct ternary_node
  {
    typedef expr::Ternary<Operation, Block1, Block2, Block3, true> const type;
  };

  template <typename BlockT>
  struct transform
  {
    typedef typename leaf_node<BlockT>::type type;
  };

  template <template <typename> class O, typename B>
  struct transform<expr::Unary<O, B, true> const>
  {
    typedef typename unary_node<O, typename transform<B>::type>::type
    type;
  };

  template <template <typename, typename> class Operation,
	    typename LBlock,
	    typename RBlock>
  struct transform<expr::Binary<Operation, LBlock, RBlock, true> const>
  {
    typedef typename binary_node<Operation,
				 typename transform<LBlock>::type,
				 typename transform<RBlock>::type>
    ::type type;
  };

  template <template <typename, typename, typename> class Operation,
	    typename Block1,
	    typename Block2,
	    typename Block3>
  struct transform<expr::Ternary<Operation, Block1, Block2, Block3, true> const>
  {
    typedef typename ternary_node<Operation,
				  typename transform<Block1>::type,
				  typename transform<Block2>::type,
				  typename transform<Block3>::type>
    ::type type;
  };


  template <template <typename> class O, typename B>
  typename transform<expr::Unary<O, B, true> const>::type
  apply(expr::Unary<O, B, true> const &b)
  {
    typedef typename transform<expr::Unary<O, B, true> const>::type
      block_type;
    return block_type(b.operation(), apply(const_cast<B&>(b.arg())));
  }

  template <template <typename, typename> class Operation,
	    typename LBlock,
	    typename RBlock>
  typename transform<expr::Binary<Operation, LBlock, RBlock, true> const>::type
  apply(expr::Binary<Operation, LBlock, RBlock, true> const& blk)
  {
    typedef typename
      transform<expr::Binary<Operation, LBlock, RBlock, true> const>::type
        block_type;
    return block_type(apply(const_cast<LBlock&>(blk.arg1())),
		      apply(const_cast<RBlock&>(blk.arg2())));
  }

  template <template <typename, typename, typename> class Operation,
	    typename Block1,
	    typename Block2,
	    typename Block3>
  typename transform<expr::Ternary<Operation, Block1, Block2, Block3, true> const>::type
  apply(expr::Ternary<Operation, Block1, Block2, Block3, true> const& blk)
  {
    typedef typename
      transform<expr::Ternary<Operation, Block1, Block2, Block3, true> const>::type
        block_type;
    return block_type(apply(const_cast<Block1&>(blk.arg1())),
		      apply(const_cast<Block2&>(blk.arg2())),
		      apply(const_cast<Block3&>(blk.arg3())));
  }

  // Leaf combine function for Dense.
  template <dimension_type Dim0,
	    typename       T,
	    typename       OrderT,
	    typename       MapT>
  // typename transform<Dense<Dim0, T, OrderT, MapT> >::type&
  Dense<Dim0, T, OrderT, MapT>&
  apply(Dense<Dim0, T, OrderT, MapT>& block) const
  {
    return block;
  }

  template <dimension_type Dim0,
	    typename       T,
	    typename       OrderT>
  // typename transform<Dense<Dim0, T, OrderT, Local_or_global_map<Dim0> > >::type&
  Dense<Dim0, T, OrderT, Local_map>&
  apply(Dense<Dim0, T, OrderT, Local_or_global_map<Dim0> >& block) const
  {
    return block.get_local_block();
  }

  // Leaf combine function for expr::Scalar.
  template <dimension_type Dim0,
	    typename       T>
  typename transform<expr::Scalar<Dim0, T> >::type
  apply(expr::Scalar<Dim0, T> & block) const
  {
    return expr::Scalar<NewDim, T>(block.value());
  }


  // Leaf combine function.
  template <typename BlockT>
  typename transform<BlockT>::type
  apply(BlockT& block) const
  {
    typedef typename transform<BlockT>::type block_type;
    return block_type(block);
  }

  // Constructors.
public:
  Redim_expr() {}
};



/// Reduction to check if all leaf blocks have dimension-ordering
/// equivalent to OrderT.
template <typename OrderT>
struct Reduce_is_same_dim_order
{
public:
  template <typename BlockT>
  struct leaf_node
  {
    typedef Bool_type<Type_equal<typename Block_layout<BlockT>::order_type,
				 OrderT>::value> type;
  };

  template <dimension_type Dim0,
	    typename       T>
  struct leaf_node<expr::Scalar<Dim0, T> >
  {
    typedef Bool_type<true> type;
  };

  template <template <typename> class Operation,
	    typename Block>
  struct unary_node
  {
    typedef Block type;
  };

  template <template <typename, typename> class Operation,
	    typename LBlock,
	    typename RBlock>
  struct binary_node
  {
    typedef Bool_type<LBlock::value && RBlock::value> type;
  };

  template <template <typename, typename, typename> class Operation,
	    typename Block1,
	    typename Block2,
	    typename Block3>
  struct ternary_node
  {
    typedef Bool_type<Block1::value && Block2::value && Block3::value> type;
  };

  template <typename Block>
  struct transform
  {
    typedef typename leaf_node<Block>::type type;
  };

  template <template <typename> class O, typename B>
  struct transform<expr::Unary<O, B, true> const>
  {
    typedef typename unary_node<O, typename transform<B>::type>::type
    type;
  };

  template <template <typename, typename> class Operation,
	    typename LBlock,
	    typename RBlock>
  struct transform<expr::Binary<Operation, LBlock, RBlock, true> const>
  {
    typedef typename binary_node<Operation,
				 typename transform<LBlock>::type,
				 typename transform<RBlock>::type>
    ::type type;
  };

  template <template <typename, typename, typename> class Operation,
	    typename Block1,
	    typename Block2,
	    typename Block3>
  struct transform<expr::Ternary<Operation, Block1, Block2, Block3, true> const>
  {
    typedef typename ternary_node<Operation,
				  typename transform<Block1>::type,
				  typename transform<Block2>::type,
				  typename transform<Block3>::type>
    ::type type;
  };
};


template <typename OrderT,
	  typename BlockT>
struct Is_same_dim_order
{
  static bool const value =
    Reduce_is_same_dim_order<OrderT>::template transform<BlockT>::type::value;
};



/// Reduction to determine if all leaf blocks of an expression support
/// direct access (cost == 0).
struct Reduce_is_expr_direct_access
{
public:
  template <typename BlockT>
  struct leaf_node
  {
    typedef Bool_type<Ext_data_cost<BlockT>::value == 0> type;
  };

  template <dimension_type Dim0,
	    typename       T>
  struct leaf_node<expr::Scalar<Dim0, T> >
  {
    typedef Bool_type<true> type;
  };

  template <template <typename> class Operation,
	    typename Block>
  struct unary_node
  {
    typedef Block type;
  };

  template <template <typename, typename> class Operation,
	    typename LBlock,
	    typename RBlock>
  struct binary_node
  {
    typedef Bool_type<LBlock::value && RBlock::value> type;
  };

  template <template <typename, typename, typename> class Operation,
	    typename Block1,
	    typename Block2,
	    typename Block3>
  struct ternary_node
  {
    typedef Bool_type<Block1::value && Block2::value && Block3::value> type;
  };

  template <typename Block>
  struct transform
  {
    typedef typename leaf_node<Block>::type type;
  };

  template <template <typename> class O, typename B>
  struct transform<expr::Unary<O, B, true> const>
  {
    typedef typename unary_node<O, typename transform<B>::type>::type
    type;
  };

  template <template <typename, typename> class Operation,
	    typename LBlock,
	    typename RBlock>
  struct transform<expr::Binary<Operation, LBlock, RBlock, true> const>
  {
    typedef typename binary_node<Operation,
				 typename transform<LBlock>::type,
				 typename transform<RBlock>::type>
    ::type type;
  };

  template <template <typename, typename, typename> class Operation,
	    typename Block1,
	    typename Block2,
	    typename Block3>
  struct transform<expr::Ternary<Operation, Block1, Block2, Block3, true> const>
  {
    typedef typename ternary_node<Operation,
				  typename transform<Block1>::type,
				  typename transform<Block2>::type,
				  typename transform<Block3>::type>
				::type type;
  };
};


template <typename BlockT>
struct Is_expr_direct_access
{
  static bool const value =
    Reduce_is_expr_direct_access::template transform<BlockT>::type::value;
};

} // namespace vsip::impl
} // namespace vsip

namespace vsip_csl
{
namespace dispatcher
{

/// Evaluator to convert dense multi-dimensional expressions into
/// 1 dimensional expressions.
template <dimension_type D, typename LHS, typename RHS>
struct Evaluator<op::assign<D>, be::dense_expr, void(LHS &, RHS const &)>
{
  static char const *name() { return "be::dense_expr";}

  static bool const ct_valid =
    D > 1 &&
    impl::Ext_data_cost<LHS>::value == 0 &&
    impl::Is_expr_direct_access<RHS>::value &&
    impl::Is_same_dim_order<typename impl::Block_layout<LHS>::order_type,
			    RHS>::value;

  static bool rt_valid(LHS &lhs, RHS const &rhs)
  { return impl::is_expr_dense(lhs) && impl::is_expr_dense(rhs);}

  static void exec(LHS &lhs, RHS const &rhs)
  {
    using impl::Redim_expr;
    typedef typename Redim_expr<1>::template transform<RHS>::type
      new_rhs_type;
    typedef typename Redim_expr<1>::template transform<LHS>::type
      new_lhs_type;

    Redim_expr<1> redim;

    // Create an lvalue that dispatch() below can write into.
    typename impl::View_block_storage<new_lhs_type>::plain_type
      new_lhs = redim.apply(const_cast<LHS&>(lhs));

    vsip_csl::dispatch<op::assign<1>, void, new_lhs_type &, new_rhs_type const &>
      (new_lhs, redim.apply(const_cast<RHS&>(rhs)));
  }
};

} // namespace vsip_csl::dispatcher
} // namespace vsip_csl

#endif
