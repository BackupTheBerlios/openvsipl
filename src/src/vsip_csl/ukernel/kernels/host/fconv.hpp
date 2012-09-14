/* Copyright (c) 2008 by CodeSourcery.  All rights reserved. */

/** @file    vsip_csl/ukernel/kernels/host/fconv.hpp
    @author  Jules Bergmann
    @date    2008-06-27
    @brief   VSIPL++ Library: Vector-matrix element-wise multiply Ukernel
*/

#ifndef VSIP_CSL_UKERNEL_KERNELS_HOST_FCONV_HPP
#define VSIP_CSL_UKERNEL_KERNELS_HOST_FCONV_HPP

/***********************************************************************
  Included Files
***********************************************************************/

#include <vsip_csl/ukernel/host/ukernel.hpp>
#include <vsip_csl/ukernel/kernels/params/fused_param.hpp>
#include <vsip_csl/ukernel/kernels/params/fft_param.hpp>
#include <vsip_csl/ukernel/kernels/params/vmmul_param.hpp>

namespace vsip_csl
{
namespace ukernel
{

/***********************************************************************
  Definitions
***********************************************************************/

// Host-side vector elementwise copy ukernel.

class Fconv : public Host_kernel
{
  // Parameters.
  //  - 'tag_type' is used to select the appropriate kernel (via
  //    Task_manager's Task_map)
  //  - in_argc and out_argc describe the number of input and output
  //    streams.
  //  - param_type (inherited) defaults to 'Empty_params'.
public:
  static unsigned int const pre_argc = 1;
  static unsigned int const in_argc  = 1;
  static unsigned int const out_argc = 1;

  typedef Uk_fused_params<Uk_fft_params,
			  Uk_vmmul_params,
			  Uk_fft_params>
		param_type;

  // Host-side ukernel object initialization.
  //
  // Streaming pattern divides linear vector into blocks, with minimum
  // size 256, maximum size 1024.  (Blocksize_sdist has an implicit
  // quantum of 16 elements).
  Fconv(unsigned int size)
    : size_(size),
      pre_sp_(Whole_sdist()),
      io_sp_(Blocksize_sdist(1), Whole_sdist())
  {}

  // Host-side compute kernel.  Used if accelerator is not available.
  template <typename View0, typename View1, typename View2>
  void compute( View0 in0, View1 in1, View2 out)
  {
    out = in0 * in1;
  }

  // Queury API:
  // - fill_params() fills the parameter block to be passed to the
  //   accelerators.
  // - in_spatt()/out_spatt() allow VSIPL++ to determine streaming
  //   pattern for user-kernel.  Since both input and output have same
  //   streaming pattern, simply return 'sp'
  void fill_params(param_type& param) const
  {
    param.k1_params.size  = size_;
    param.k1_params.dir   = -1;
    param.k1_params.scale = 1.f;
    param.k2_params.size  = size_;
    param.k3_params.size  = size_;
    param.k3_params.dir   = +1;
    param.k3_params.scale = 1.f / size_;
  }

  Stream_pattern const& in_spatt(vsip::index_type i) const
  { return (i == 0) ? pre_sp_ : io_sp_;}

  Stream_pattern const& out_spatt(vsip::index_type) const
  { return io_sp_;}

  // Member data.
  //
  // 'sp' is the stream pattern.
private:
  unsigned int size_;
  Stream_pattern pre_sp_;
  Stream_pattern io_sp_;
};

}
}

DEFINE_UKERNEL_TASK(vsip_csl::ukernel::Fconv,
		    void(std::complex<float>*, std::complex<float>*,
			 std::complex<float>*),
		    "uk_plugin", cfconv_f)

DEFINE_UKERNEL_TASK(vsip_csl::ukernel::Fconv,
		    void(std::pair<float*,float*>, std::pair<float*,float*>,
			 std::pair<float*,float*>),
		    "uk_plugin", zfconv_f)

#endif
