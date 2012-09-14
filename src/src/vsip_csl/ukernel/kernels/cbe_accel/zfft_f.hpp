/* Copyright (c) 2008 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    vsip_csl/ukernel/kernels/cbe_accel/zfft_f.hpp
    @author  Jules Bergmann
    @date    2008-06-12
    @brief   VSIPL++ Library: CBE ukernel for split-complex float FFT.
*/
#ifndef VSIP_CSL_UKERNEL_KERNELS_CBE_ACCEL_ZFFT_F_HPP
#define VSIP_CSL_UKERNEL_KERNELS_CBE_ACCEL_ZFFT_F_HPP

/***********************************************************************
  Included Files
***********************************************************************/

#include <cstdio>
#include <utility>
#include <cassert>
#include <spu_intrinsics.h>

#include <cml.h>
#include <cml_core.h>
#include <vsip_csl/ukernel/cbe_accel/ukernel.hpp>
#include <vsip_csl/ukernel/kernels/params/fft_param.hpp>

#define MIN_FFT_1D_SIZE	  32
#define MAX_FFT_1D_SIZE	  4096

#define DUMP_STACK(TEXT)                                                \


/***********************************************************************
  Definitions
***********************************************************************/

struct Fft_kernel : Spu_kernel
{
  typedef std::pair<float*, float*> in0_type;
  typedef std::pair<float*, float*> out0_type;

  static unsigned int const in_argc  = 1;
  static unsigned int const out_argc = 1;
  typedef Uk_fft_params param_type;

  Fft_kernel()
  {}

  void init(param_type& params)
  {
    size  = params.size;
    dir   = params.dir;
    scale = params.scale;

    int rt = cml_fft1d_setup_f(&fft, CML_FFT_CC, size, buf2);
    assert(rt && fft != NULL);
  }

  void compute(
    in0_type  const& in,
    out0_type const& out,
    Pinfo const&     p_in,
    Pinfo const&     p_out)
  {
    cml_zzfft1d_op_f(fft,
		     in.first, in.second,
		     out.first, out.second,
		     dir, (float*)buf1);

    if (scale != 1.f)
      cml_core_rzsvmul1_f(scale, out.first, out.second,
			  out.first, out.second, size);
  }

  // Member data
  size_t      size;
  int         dir;
  float       scale;

  fft1d_f*    fft;

  static char buf1[2*MAX_FFT_1D_SIZE*sizeof(float)];
  static char buf2[1*MAX_FFT_1D_SIZE*sizeof(float)+128];
};

#endif
