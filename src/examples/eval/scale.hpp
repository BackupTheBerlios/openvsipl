/* Copyright (c) 2009 by CodeSourcery.  All rights reserved. */

/** @file    scale.hpp
    @author  Stefan Seefeld
    @date    2009-05-03
    @brief   Example scale function using return block optimization.
*/

#include <vsip/core/config.hpp>
#include <vsip/vector.hpp>
#include <vsip_csl/expr/functor.hpp>

namespace example
{
using namespace vsip;
using vsip_csl::expr::Unary;
using vsip_csl::expr::Unary_functor;
using vsip_csl::lazy_Vector;

// Scale implements a call operator that scales its input
// argument, and returns it by reference.
template <typename ArgumentBlockType>
struct Scale : Unary_functor<ArgumentBlockType>
{
  Scale(ArgumentBlockType const &a, typename ArgumentBlockType::value_type s)
    : Unary_functor<ArgumentBlockType>(a), value(s) {}
  template <typename ResultBlockType>
  void apply(ResultBlockType &r) const
  {
    ArgumentBlockType const &a = this->arg();
    for (index_type i = 0; i != r.size(); ++i)
      r.put(i, a.get(i) * value);
  }

  typename ArgumentBlockType::value_type value;
};

// scale is a return-block optimised function returning an expression.
template <typename T, typename BlockType>
lazy_Vector<T, Unary<Scale, BlockType> const>
scale(const_Vector<T, BlockType> input, T value)
{
  Scale<BlockType> s(input.block(), value);
  Unary<Scale, BlockType> block(s);
  return lazy_Vector<T, Unary<Scale, BlockType> const>(block);
}
}
