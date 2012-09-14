/* Copyright (c) 2009 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    vsip_csl/dda.hpp
    @author  Stefan Seefeld
    @date    2009-29-30
    @brief   Sourcery VSIPL++: Direct Data Access API. 
*/

#ifndef VSIP_CSL_DDA_HPP
#define VSIP_CSL_DDA_HPP

#include <vsip/core/extdata.hpp>

namespace vsip_csl
{
namespace dda
{
  using vsip::impl::Stride_unknown;
  using vsip::impl::Stride_unit;
  using vsip::impl::Stride_unit_dense;
  using vsip::impl::Stride_unit_align;

  using vsip::impl::Layout;
  using vsip::impl::Rt_layout;
  using vsip::impl::Applied_layout;

  using vsip::impl::Desired_block_layout;

  using vsip::impl::sync_action_type;
  using vsip::impl::SYNC_IN;
  using vsip::impl::SYNC_OUT;
  using vsip::impl::SYNC_INOUT;
  using vsip::impl::SYNC_IN_NOPRESERVE;

  using vsip::impl::Ext_data;
}
}

#endif
