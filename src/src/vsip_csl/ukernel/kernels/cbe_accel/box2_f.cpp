/* Copyright (c) 2008 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    vsip_csl/ukernel/kernels/cbe_accel/box2_f.cpp
    @author  Jules Bergmann
    @date    2008-08-08
    @brief   VSIPL++ Library: 2D Box filter ukernel.
*/

/***********************************************************************
  Included Files
***********************************************************************/

#include <vsip_csl/ukernel/kernels/cbe_accel/box2_f.hpp>

typedef Box2_kernel kernel_type;

#include <vsip_csl/ukernel/cbe_accel/alf_base.hpp>
