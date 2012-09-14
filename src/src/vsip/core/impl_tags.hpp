/* Copyright (c) 2006 by CodeSourcery.  All rights reserved. */

/** @file    vsip/core/impl_tags.hpp
    @author  Jules Bergmann
    @date    2006-12-06
    @brief   VSIPL++ Library: Implementation Tags.

*/

#ifndef VSIP_CORE_IMPL_TAGS_HPP
#define VSIP_CORE_IMPL_TAGS_HPP

/***********************************************************************
  Declarations
***********************************************************************/

namespace vsip
{

namespace impl
{

// Implementation tags.
//
// Each implementation (generic, BLAS, IPP, etc) has a unique
// implementation tag.
//

struct Intel_ipp_tag {};	// Intel IPP Library
struct Mercury_sal_tag {};	// Mercury SAL Library
struct Cbe_sdk_tag {};          // IBM CBE SDK.
struct Cml_tag {};              // IBM Cell Math Library
struct Simd_builtin_tag {};	// Builtin SIMD routines (non loop fusion)

struct Lapack_tag {};		// LAPACK implementation (ATLAS, MKL, etc)
struct Generic_tag {};		// Generic implementation.
struct Cvsip_tag {};		// C-VSIPL library.

} // namespace vsip::impl
} // namespace vsip

#endif // VSIP_CORE_IMPL_TAGS_HPP
