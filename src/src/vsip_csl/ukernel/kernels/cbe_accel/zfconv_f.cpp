/* Copyright (c) 2008 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    vsip_csl/ukernel/kernels/cbe_accel/zfconv_f.hpp
    @author  Jules Bergmann
    @date    2008-08-21
    @brief   VSIPL++ Library: Inter-complex fastconv ukernel.
*/

/***********************************************************************
  Included Files
***********************************************************************/

#include <vsip_csl/ukernel/kernels/params/fused_param.hpp>

#include <vsip_csl/ukernel/kernels/cbe_accel/fused_kernel.hpp>
#include <vsip_csl/ukernel/kernels/cbe_accel/zfft_f.hpp>
#include <vsip_csl/ukernel/kernels/cbe_accel/zvmmul_f.hpp>

typedef Fused_kernel<Fft_kernel, Zvmmul_kernel, Fft_kernel>
	kernel_type;

char Fft_kernel::buf1[2*MAX_FFT_1D_SIZE*sizeof(float)]
     __attribute((aligned(128)));
char Fft_kernel::buf2[1*MAX_FFT_1D_SIZE*sizeof(float)+128]
     __attribute((aligned(128)));

#include <vsip_csl/ukernel/cbe_accel/alf_base.hpp>
