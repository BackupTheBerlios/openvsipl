/* Copyright (c) 2009 by CodeSourcery.  All rights reserved. */

/** @file    vsip/core/c++0x.hpp
    @author  Stefan Seefeld
    @date    2009-07-06
    @brief   VSIPL++ Library: C++0x bits that we can't use from libstdc++ just yet.
*/

#ifndef VSIP_CORE_CXX0X_HPP
#define VSIP_CORE_CXX0X_HPP

namespace vsip
{
namespace impl
{

template <bool B, typename T = void>
struct enable_if_c { typedef T type;};

template <typename T>
struct enable_if_c<false, T> {};

/// Define a nested type if some predicate holds.
template <typename C, typename T = void>
struct enable_if : public enable_if_c<C::value, T> {};

/// A conditional expression, but for types. If true, first, if false, second.
template<bool B, typename Iftrue, typename Iffalse>
struct conditional
{
  typedef Iftrue type;
};

template<typename Iftrue, typename Iffalse>
struct conditional<false, Iftrue, Iffalse>
{
  typedef Iffalse type;
};


} // namespace vsip::impl
} // namespace vsip

#endif
