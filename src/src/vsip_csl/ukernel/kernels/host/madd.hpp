/* Copyright (c) 2008 by CodeSourcery.  All rights reserved. */

/** @file    vsip_csl/ukernel/kernels/host/madd.hpp
    @author  Don McCoy
    @date    2008-08-26
    @brief   VSIPL++ Library: User-defined kernel for multiply-add.
*/

#ifndef VSIP_CSL_UKERNEL_KERNELS_HOST_MADD_HPP
#define VSIP_CSL_UKERNEL_KERNELS_HOST_MADD_HPP

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

// Host-side vector elementwise multiply-add ukernel.
class Madd : public Host_kernel
{
  // Parameters.
  //  - 'tag_type' is used to select the appropriate kernel (via
  //    Task_manager's Task_map)
  //  - in_argc and out_argc describe the number of input and output
  //    streams.
  //  - param_type (inherited) defaults to 'Empty_params'.
public:
  static unsigned int const in_argc  = 3;
  static unsigned int const out_argc = 1;

  // Host-side ukernel object initialization.
  //
  // Streaming pattern divides matrix into whole, single rows.

  Madd()
    : sp_(Blocksize_sdist(1), Whole_sdist())
  {}



  // Host-side compute kernel.  Used if accelerator is not available.
  template <typename View0, typename View1, typename View2, typename View3>
  void compute(View0 in0, View1 in1, View2 in2, View3 out)
  {
    out = in0 * in1 + in2;
  }


  Stream_pattern const& in_spatt(vsip::index_type i) const
  { return sp_; }

  Stream_pattern const& out_spatt(vsip::index_type) const
  { return sp_; }

private:
  Stream_pattern sp_;
};

}
}

DEFINE_UKERNEL_TASK(vsip_csl::ukernel::Madd,
  void(float*, float*, float*, float*),
  "uk_plugin", madd_f)

DEFINE_UKERNEL_TASK(vsip_csl::ukernel::Madd,
  void(std::complex<float>*, std::complex<float>*, std::complex<float>*, 
    std::complex<float>*),
  "uk_plugin", cmadd_f)

DEFINE_UKERNEL_TASK(vsip_csl::ukernel::Madd,
  void(float*, std::complex<float>*, std::complex<float>*, 
    std::complex<float>*),
  "uk_plugin", scmadd_f)

#endif
