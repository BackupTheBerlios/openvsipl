/* Copyright (c) 2008 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    vsip_csl/ukernel/kernels/cbe_accel/cfft_f.hpp
    @author  Jules Bergmann
    @date    2008-08-08
    @brief   VSIPL++ Library: Inter-complex fastconv ukernel.
*/

/***********************************************************************
  Included Files
***********************************************************************/

#include <vsip_csl/ukernel/kernels/cbe_accel/cfft_f.hpp>

typedef Fft_kernel kernel_type;

char Fft_kernel::buf1[FFT_BUF1_SIZE_BYTES] __attribute((aligned(128)));
char Fft_kernel::buf2[FFT_BUF2_SIZE_BYTES] __attribute((aligned(128)));

#include <vsip_csl/ukernel/cbe_accel/alf_base.hpp>
