/* Copyright (c) 2005, 2006 by CodeSourcery.  All rights reserved. */

/** @file    vsip/core/reductions/reductions.hpp
    @author  Jules Bergmann
    @date    2005-07-12
    @brief   VSIPL++ Library: Reduction functions.
	     [math.fns.reductions].

*/

#ifndef VSIP_CORE_REDUCTIONS_REDUCTIONS_HPP
#define VSIP_CORE_REDUCTIONS_REDUCTIONS_HPP

/***********************************************************************
  Included Files
***********************************************************************/

#include <vsip/support.hpp>
#include <vsip/vector.hpp>
#include <vsip/matrix.hpp>
#include <vsip/tensor.hpp>
#include <vsip/core/reductions/functors.hpp>
#include <vsip/core/parallel/services.hpp>
#include <vsip/core/dispatch.hpp>
#if !VSIP_IMPL_REF_IMPL
#  include <vsip/opt/dispatch.hpp>
#  ifdef VSIP_IMPL_HAVE_SAL
#    include <vsip/opt/sal/eval_reductions.hpp>
#  endif
#endif
#if VSIP_IMPL_HAVE_CVSIP
#  include <vsip/core/cvsip/eval_reductions.hpp>
#endif



/***********************************************************************
  Declarations
***********************************************************************/

namespace vsip_csl
{
namespace dispatcher
{

/***********************************************************************
  Generic evaluators.
***********************************************************************/

#ifndef VSIP_IMPL_REF_IMPL
template<template <typename> class ReduceT>
struct List<op::reduce<ReduceT> >
{
  typedef Make_type_list<be::cvsip, be::mercury_sal, be::generic>::type type;
};
#endif


/// Generic evaluator for vector reductions.
template <template <typename> class ReduceT,
	  typename                  T,
	  typename                  Block>
struct Evaluator<op::reduce<ReduceT>, be::generic, 
                 void(T&, Block const&, row1_type, impl::Int_type<1>)>
{
  static bool const ct_valid = true;
  static bool rt_valid(T&, Block const&, row1_type, impl::Int_type<1>)
  { return true; }

  static void exec(T& r, Block const& a, row1_type, impl::Int_type<1>)
  {
    typedef typename Block::value_type VT;
    typename ReduceT<VT>::accum_type state = ReduceT<VT>::initial();

    length_type length = a.size(1, 0);
    PRAGMA_IVDEP
    for (index_type i=0; i<length; ++i)
    {
      state = ReduceT<VT>::update(state, a.get(i));
      if (ReduceT<VT>::done(state)) break;
    }

    r = ReduceT<VT>::value(state, length);
  }
};


/// Generic evaluator for matrix reductions (tuple<0, 1, 2>).
template <template <typename> class ReduceT,
	  typename                  T,
	  typename                  Block>
struct Evaluator<op::reduce<ReduceT>, be::generic,
                 void(T&, Block const&, row2_type, impl::Int_type<2>)>
{
  static bool const ct_valid = true;
  static bool rt_valid(T&, Block const&, row2_type, impl::Int_type<2>)
  { return true; }

  static void exec(T& r, Block const& a, row2_type, impl::Int_type<2>)
  {
    typedef typename Block::value_type VT;
    typename ReduceT<T>::accum_type state = ReduceT<VT>::initial();

    length_type length_i = a.size(2, 0);
    length_type length_j = a.size(2, 1);

    for (index_type i=0; i<length_i; ++i)
      PRAGMA_IVDEP
      for (index_type j=0; j<length_j; ++j)
      {
	state = ReduceT<VT>::update(state, a.get(i, j));
	if (ReduceT<VT>::done(state)) break;
      }

    r = ReduceT<VT>::value(state, length_i*length_j);
  }
};


/// Generic evaluator for matrix reductions (tuple<2, 1, 0>).
template <template <typename> class ReduceT,
	  typename                  T,
	  typename                  Block>
struct Evaluator<op::reduce<ReduceT>, be::generic,
                 void(T&, Block const&, col2_type, impl::Int_type<2>)>
{
  static bool const ct_valid = true;
  static bool rt_valid(T&, Block const&, col2_type, impl::Int_type<2>)
  { return true; }

  static void exec(T& r, Block const& a, col2_type, impl::Int_type<2>)
  {
    typedef typename Block::value_type VT;
    typename ReduceT<T>::accum_type state = ReduceT<VT>::initial();

    length_type length_i = a.size(2, 0);
    length_type length_j = a.size(2, 1);

    for (index_type j=0; j<length_j; ++j)
    PRAGMA_IVDEP
    for (index_type i=0; i<length_i; ++i)
    {
      state = ReduceT<VT>::update(state, a.get(i, j));
      if (ReduceT<VT>::done(state)) break;
    }

    r = ReduceT<VT>::value(state, length_i*length_j);
  }
};


/// Generic evaluator for tensor reductions (tuple<0, 1, 2>).
template <template <typename> class ReduceT,
	  typename                  T,
	  typename                  Block>
struct Evaluator<op::reduce<ReduceT>, be::generic,
                 void(T&, Block const&, tuple<0, 1, 2>, impl::Int_type<3>)>
{
  static bool const ct_valid = true;
  static bool rt_valid(T&, Block const&, tuple<0, 1, 2>, impl::Int_type<3>)
  { return true; }

  static void exec(T& r, Block const& a, tuple<0, 1, 2>, impl::Int_type<3>)
  {
    typedef typename Block::value_type VT;
    typename ReduceT<T>::accum_type state = ReduceT<VT>::initial();

    length_type length_0 = a.size(3, 0);
    length_type length_1 = a.size(3, 1);
    length_type length_2 = a.size(3, 2);

    for (index_type i=0; i<length_0; ++i)
    for (index_type j=0; j<length_1; ++j)
    for (index_type k=0; k<length_2; ++k)
    {
      state = ReduceT<VT>::update(state, a.get(i, j, k));
      if (ReduceT<VT>::done(state)) break;
    }

    r = ReduceT<VT>::value(state, length_0*length_1*length_2);
  }
};


/// Generic evaluator for tensor reductions (tuple<0, 2, 1>).
template <template <typename> class ReduceT,
	  typename                  T,
	  typename                  Block>
struct Evaluator<op::reduce<ReduceT>, be::generic,
                 void(T&, Block const&, tuple<0, 2, 1>, impl::Int_type<3>)>
{
  static bool const ct_valid = true;
  static bool rt_valid(T&, Block const&, tuple<0, 2, 1>, impl::Int_type<3>)
  { return true; }

  static void exec(T& r, Block const& a, tuple<0, 2, 1>, impl::Int_type<3>)
  {
    typedef typename Block::value_type VT;
    typename ReduceT<T>::accum_type state = ReduceT<VT>::initial();

    length_type length_0 = a.size(3, 0);
    length_type length_1 = a.size(3, 1);
    length_type length_2 = a.size(3, 2);

    for (index_type i=0; i<length_0; ++i)
    for (index_type k=0; k<length_2; ++k)
    for (index_type j=0; j<length_1; ++j)
    {
      state = ReduceT<VT>::update(state, a.get(i, j, k));
      if (ReduceT<VT>::done(state)) break;
    }

    r = ReduceT<VT>::value(state, length_0*length_1*length_2);
  }
};


/// Generic evaluator for tensor reductions (tuple<1, 0, 2>).
template <template <typename> class ReduceT,
	  typename                  T,
	  typename                  Block>
struct Evaluator<op::reduce<ReduceT>, be::generic,
                 void(T&, Block const&, tuple<1, 0, 2>, impl::Int_type<3>)>
{
  static bool const ct_valid = true;
  static bool rt_valid(T&, Block const&, tuple<1, 0, 2>, impl::Int_type<3>)
  { return true; }

  static void exec(T& r, Block const& a, tuple<1, 0, 2>, impl::Int_type<3>)
  {
    typedef typename Block::value_type VT;
    typename ReduceT<T>::accum_type state = ReduceT<VT>::initial();

    length_type length_0 = a.size(3, 0);
    length_type length_1 = a.size(3, 1);
    length_type length_2 = a.size(3, 2);

    for (index_type j=0; j<length_1; ++j)
    for (index_type i=0; i<length_0; ++i)
    for (index_type k=0; k<length_2; ++k)
    {
      state = ReduceT<VT>::update(state, a.get(i, j, k));
      if (ReduceT<VT>::done(state)) break;
    }

    r = ReduceT<VT>::value(state, length_0*length_1*length_2);
  }
};


/// Generic evaluator for tensor reductions (tuple<1, 2, 0>).
template <template <typename> class ReduceT,
	  typename                  T,
	  typename                  Block>
struct Evaluator<op::reduce<ReduceT>, be::generic,
                 void(T&, Block const&, tuple<1, 2, 0>, impl::Int_type<3>)>
{
  static bool const ct_valid = true;
  static bool rt_valid(T&, Block const&, tuple<1, 2, 0>, impl::Int_type<3>)
  { return true; }

  static void exec(T& r, Block const& a, tuple<1, 2, 0>, impl::Int_type<3>)
  {
    typedef typename Block::value_type VT;
    typename ReduceT<T>::accum_type state = ReduceT<VT>::initial();

    length_type length_0 = a.size(3, 0);
    length_type length_1 = a.size(3, 1);
    length_type length_2 = a.size(3, 2);

    for (index_type j=0; j<length_1; ++j)
    for (index_type k=0; k<length_2; ++k)
    for (index_type i=0; i<length_0; ++i)
    {
      state = ReduceT<VT>::update(state, a.get(i, j, k));
      if (ReduceT<VT>::done(state)) break;
    }

    r = ReduceT<VT>::value(state, length_0*length_1*length_2);
  }
};


/// Generic evaluator for tensor reductions (tuple<2, 0, 1>).
template <template <typename> class ReduceT,
	  typename                  T,
	  typename                  Block>
struct Evaluator<op::reduce<ReduceT>, be::generic,
                 void(T&, Block const&, tuple<2, 0, 1>, impl::Int_type<3>)>
{
  static bool const ct_valid = true;
  static bool rt_valid(T&, Block const&, tuple<2, 0, 1>, impl::Int_type<3>)
  { return true; }

  static void exec(T& r, Block const& a, tuple<2, 0, 1>, impl::Int_type<3>)
  {
    typedef typename Block::value_type VT;
    typename ReduceT<T>::accum_type state = ReduceT<VT>::initial();

    length_type length_0 = a.size(3, 0);
    length_type length_1 = a.size(3, 1);
    length_type length_2 = a.size(3, 2);

    for (index_type k=0; k<length_2; ++k)
    for (index_type i=0; i<length_0; ++i)
    for (index_type j=0; j<length_1; ++j)
    {
      state = ReduceT<VT>::update(state, a.get(i, j, k));
      if (ReduceT<VT>::done(state)) break;
    }

    r = ReduceT<VT>::value(state, length_0*length_1*length_2);
  }
};


/// Generic evaluator for tensor reductions (tuple<2, 1, 0>).
template <template <typename> class ReduceT,
	  typename                  T,
	  typename                  Block>
struct Evaluator<op::reduce<ReduceT>, be::generic,
                 void(T&, Block const&, tuple<2, 1, 0>, impl::Int_type<3>)>
{
  static bool const ct_valid = true;
  static bool rt_valid(T&, Block const&, tuple<2, 1, 0>, impl::Int_type<3>)
  { return true; }

  static void exec(T& r, Block const& a, tuple<2, 1, 0>, impl::Int_type<3>)
  {
    typedef typename Block::value_type VT;
    typename ReduceT<T>::accum_type state = ReduceT<VT>::initial();

    length_type length_0 = a.size(3, 0);
    length_type length_1 = a.size(3, 1);
    length_type length_2 = a.size(3, 2);

    for (index_type k=0; k<length_2; ++k)
    for (index_type j=0; j<length_1; ++j)
    for (index_type i=0; i<length_0; ++i)
    {
      state = ReduceT<VT>::update(state, a.get(i, j, k));
      if (ReduceT<VT>::done(state)) break;
    }

    r = ReduceT<VT>::value(state, length_0*length_1*length_2);
  }
};



/***********************************************************************
  Parallel evaluators.
***********************************************************************/

#if !VSIP_IMPL_REF_IMPL
template <template <typename> class ReduceT,
          typename                  T,
	  typename                  Block,
	  typename                  OrderT,
	  int                       Dim>
struct Par_reduction_eval_base
{
  static bool const ct_valid = 
    !impl::Is_local_map<typename Block::map_type>::value &&
    impl::Reduction_supported<ReduceT<typename Block::value_type>::rtype,
			      typename Block::value_type>::value;

  static bool rt_valid(T&, Block const&, OrderT, impl::Int_type<Dim>)
  { return true; }
};


template <typename                  T,
	  typename                  Block,
	  typename                  OrderT,
	  int                       Dim>
struct Evaluator<op::reduce<impl::Mean_value>, be::parallel,
                 void(T&, Block const&, OrderT, impl::Int_type<Dim>)>
  : Par_reduction_eval_base<impl::Mean_value, T, Block, OrderT, Dim>
{
  static void exec(T& r, Block const& a, OrderT, impl::Int_type<Dim>)
  {
    typedef typename Block::value_type VT;
    T l_r;
    typedef typename impl::Distributed_local_block<Block>::type local_block_type;
    typedef typename impl::Block_layout<local_block_type>::order_type order_type;
    typedef impl::Int_type<Dim>                                       dim_type;
    typedef impl::Mean_value<VT>                                      reduce_type;
    typedef typename Block::map_type                            map_type;

    dispatch<op::reduce<impl::Sum_value>, void,
             typename impl::Sum_value<VT>::result_type&,
             local_block_type const&,
             order_type,
             dim_type>
      (l_r, get_local_block(a), order_type(), dim_type());

    if (!impl::Type_equal<map_type, Global_map<Block::dim> >::value)
	r = a.map().impl_comm().allreduce(reduce_type::rtype, l_r);
    else
      r = l_r;

    r /= static_cast<typename reduce_type::accum_type>(a.size());
  }
};



template <typename                  T,
	  typename                  Block,
	  typename                  OrderT,
	  int                       Dim>
struct Evaluator<op::reduce<impl::Mean_magsq_value>, be::parallel,
                 void(T&, Block const&, OrderT, impl::Int_type<Dim>)>
  : Par_reduction_eval_base<impl::Mean_magsq_value, T, Block, OrderT, Dim>
{
  static void exec(T& r, Block const& a, OrderT, impl::Int_type<Dim>)
  {
    typedef typename Block::value_type VT;
    T l_r;
    typedef typename impl::Distributed_local_block<Block>::type local_block_type;
    typedef typename impl::Block_layout<local_block_type>::order_type order_type;
    typedef impl::Int_type<Dim>                                       dim_type;
    typedef impl::Mean_magsq_value<VT>                                reduce_type;
    typedef typename Block::map_type                            map_type;

    dispatch<op::reduce<impl::Sum_magsq_value>, void,
      typename impl::Sum_magsq_value<VT>::result_type&,
             local_block_type const&,
             order_type,
             dim_type>
      (l_r, get_local_block(a), order_type(), dim_type());

    if (!impl::Type_equal<map_type, Global_map<Block::dim> >::value)
      r = a.map().impl_comm().allreduce(reduce_type::rtype, l_r);
    else
      r = l_r;

    r /= static_cast<typename reduce_type::accum_type>(a.size());
  }
};


template <template <typename> class ReduceT,
	  typename                  T,
	  typename                  Block,
	  typename                  OrderT,
	  int                       Dim>
struct Evaluator<op::reduce<ReduceT>, be::parallel, 
                 void(T&, Block const&, OrderT, impl::Int_type<Dim>)>
  : Par_reduction_eval_base<ReduceT, T, Block, OrderT, Dim>
{
  static void exec(T& r, Block const& a, OrderT, impl::Int_type<Dim>)
  {
    typedef typename Block::value_type VT;
    T l_r = T();
    typedef typename impl::Distributed_local_block<Block>::type local_block_type;
    typedef typename impl::Block_layout<local_block_type>::order_type order_type;
    typedef impl::Int_type<Dim>                                       dim_type;
    typedef ReduceT<VT>                                         reduce_type;
    typedef typename Block::map_type                            map_type;

    dispatch<op::reduce<ReduceT>, void,
             typename ReduceT<VT>::result_type&,
	     local_block_type const&,
             order_type,
             dim_type>
      (l_r, get_local_block(a), order_type(), dim_type());

    if (!impl::Type_equal<map_type, Global_map<Block::dim> >::value)
      r = a.map().impl_comm().allreduce(ReduceT<T>::rtype, l_r);
    else
      r = l_r;
  }
};
#endif

} // namespace vsip_csl::dispatcher
} // namespace vsip_csl

namespace vsip
{
namespace impl
{

template <template <typename> class ReduceT,
	  typename                  ViewT>
typename ReduceT<typename ViewT::value_type>::result_type
reduce(ViewT v)
{
  using namespace vsip_csl::dispatcher;

  typedef typename ViewT::value_type T;
  typedef typename ReduceT<T>::result_type result_type;
  typedef typename ViewT::block_type block_type;
  typedef typename Block_layout<typename ViewT::block_type>::order_type
    order_type;
  typedef Int_type<ViewT::dim> dim_type;

  result_type r;

#if VSIP_IMPL_REF_IMPL
  Evaluator<op::reduce<ReduceT>, be::cvsip, 
    void(result_type&, block_type const&, order_type, dim_type)>::
    exec(r, v.block(), order_type(), dim_type());
#else

  typedef Make_type_list<be::parallel,
                         be::cvsip,
                         be::mercury_sal, 
                         be::generic>::type list_type;

  Dispatcher<op::reduce<ReduceT>, 
    void(result_type&, block_type const&, order_type, dim_type), list_type>::
    dispatch(r, v.block(), order_type(), dim_type());
#endif

  return r;
}

} // namespace vsip::impl



/***********************************************************************
  API Reduction Functions
***********************************************************************/

template <typename                            T,
	  template <typename, typename> class ViewT,
	  typename                            BlockT>
typename impl::All_true<T>::result_type
alltrue(ViewT<T, BlockT> v)
{
  typedef typename impl::Block_layout<BlockT>::order_type order_type;
  return impl::reduce<impl::All_true>(v);
}



template <typename                            T,
	  template <typename, typename> class ViewT,
	  typename                            BlockT>
typename impl::Any_true<T>::result_type
anytrue(ViewT<T, BlockT> v)
{
  typedef typename impl::Block_layout<BlockT>::order_type order_type;
  return impl::reduce<impl::Any_true>(v);
}



template <typename                            T,
	  template <typename, typename> class ViewT,
	  typename                            BlockT>
typename impl::Mean_value<T>::result_type
meanval(ViewT<T, BlockT> v)
{
  typedef typename impl::Block_layout<BlockT>::order_type order_type;
  return impl::reduce<impl::Mean_value>(v);
}



// Note: meansqval computes the mean of the magnitude square

template <typename                            T,
	  template <typename, typename> class ViewT,
	  typename                            BlockT>
typename impl::Mean_magsq_value<T>::result_type
meansqval(ViewT<T, BlockT> v)
{
  typedef typename impl::Block_layout<BlockT>::order_type order_type;
  return impl::reduce<impl::Mean_magsq_value>(v);
}



template <typename                            T,
	  template <typename, typename> class ViewT,
	  typename                            BlockT>
typename impl::Sum_value<T>::result_type
sumval(ViewT<T, BlockT> v)
{
  typedef typename impl::Block_layout<BlockT>::order_type order_type;
  return impl::reduce<impl::Sum_value>(v);
}



template <typename                            T,
	  template <typename, typename> class ViewT,
	  typename                            BlockT>
typename impl::Sum_sq_value<T>::result_type
sumsqval(ViewT<T, BlockT> v)
{
  typedef typename impl::Block_layout<BlockT>::order_type order_type;
  return impl::reduce<impl::Sum_sq_value>(v);
}



} // namespace vsip

#endif // VSIP_CORE_REDUCTIONS_REDUCTIONS_HPP
