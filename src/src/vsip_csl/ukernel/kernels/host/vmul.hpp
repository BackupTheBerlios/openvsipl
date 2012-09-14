/* Copyright (c) 2008 by CodeSourcery.  All rights reserved. */

/** @file    vsip_csl/ukernel/kernels/host/vmul.hpp
    @author  Jules Bergmann
    @date    2008-06-24
    @brief   VSIPL++ Library: Vector element-wise multiply Ukernel
*/

#ifndef VSIP_CSL_UKERNEL_KERNELS_HOST_VMUL_HPP
#define VSIP_CSL_UKERNEL_KERNELS_HOST_VMUL_HPP

/***********************************************************************
  Included Files
***********************************************************************/

#include <vsip_csl/ukernel/host/ukernel.hpp>

namespace vsip_csl
{
namespace ukernel
{

/***********************************************************************
  Definitions
***********************************************************************/

// Host-side vector elementwise copy ukernel.
class Vmul : public Host_kernel
{
  // Parameters.
  //  - 'tag_type' is used to select the appropriate kernel (via
  //    Task_manager's Task_map)
  //  - in_argc and out_argc describe the number of input and output
  //    streams.
  //  - param_type (inherited) defaults to 'Empty_params'.
public:
  static unsigned int const in_argc  = 2;
  static unsigned int const out_argc = 1;

  // Host-side ukernel object initialization.
  //
  // Streaming pattern divides linear vector into blocks, with minimum
  // size 256, maximum size 1024.  (Blocksize_sdist has an implicit
  // quantum of 16 elements).
  Vmul()
    : sp_(Blocksize_sdist(1024, 256))
  {}

  // Host-side compute kernel.  Used if accelerator is not available.
  template <typename View0, typename View1, typename View2>
  void compute(View0 in0, View1 in1, View2 out)
  {
    out = in0 * in1;
  }

  Stream_pattern const& in_spatt(vsip::index_type) const
  { return sp_;}

  Stream_pattern const& out_spatt(vsip::index_type) const
  { return sp_;}

private:
  Stream_pattern sp_;	
};

}
}

DEFINE_UKERNEL_TASK(vsip_csl::ukernel::Vmul,
		    void(float*, float*, float*), "uk_plugin", vmul_f)
DEFINE_UKERNEL_TASK(vsip_csl::ukernel::Vmul,
		    void(std::complex<float>*, std::complex<float>*, std::complex<float>*),
		    "uk_plugin", cvmul_f)
DEFINE_UKERNEL_TASK(vsip_csl::ukernel::Vmul,
                    void(std::pair<float*, float*>, std::pair<float*, float*>, std::pair<float*, float*>),
		    "uk_plugin", zvmul_f)

#endif
