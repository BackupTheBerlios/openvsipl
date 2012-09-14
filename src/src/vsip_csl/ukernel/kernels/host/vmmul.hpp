/* Copyright (c) 2008 by CodeSourcery.  All rights reserved. */

/** @file    vsip_csl/ukernel/kernels/host/vmmul.hpp
    @author  Jules Bergmann
    @date    2008-06-27
    @brief   VSIPL++ Library: Vector-matrix element-wise multiply Ukernel
*/

#ifndef VSIP_CSL_UKERNEL_KERNELS_HOST_VMMUL_HPP
#define VSIP_CSL_UKERNEL_KERNELS_HOST_VMMUL_HPP

/***********************************************************************
  Included Files
***********************************************************************/

#include <vsip_csl/ukernel/host/ukernel.hpp>
#include <vsip_csl/ukernel/kernels/params/vmmul_param.hpp>

namespace vsip_csl
{
namespace ukernel
{

/***********************************************************************
  Definitions
***********************************************************************/

// Host-side vector elementwise copy ukernel.
class Vmmul : public Host_kernel
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

  typedef Uk_vmmul_params param_type;

  // Host-side ukernel object initialization.
  //
  // Streaming pattern divides linear vector into blocks, with minimum
  // size 256, maximum size 1024.  (Blocksize_sdist has an implicit
  // quantum of 16 elements).
  Vmmul(unsigned int size)
    : size_(size),
      pre_sp_(Whole_sdist()),
      io_sp_(Blocksize_sdist(1), Whole_sdist())
  {}

  // Host-side compute kernel.  Used if accelerator is not available.
  template <typename View0, typename View1, typename View2>
  void compute(View0 in0, View1 in1, View2 out)
  {
    out = in0 * in1;
  }

  void fill_params(param_type& param) const
  {
    param.size  = size_;
  }

  Stream_pattern const& in_spatt(vsip::index_type i) const
  { return (i == 0) ? pre_sp_ : io_sp_; }

  Stream_pattern const& out_spatt(vsip::index_type) const
  { return io_sp_; }

private:
  unsigned int size_;
  Stream_pattern pre_sp_;	
  Stream_pattern io_sp_;	
};

}
}

DEFINE_UKERNEL_TASK(vsip_csl::ukernel::Vmmul,
  void(std::complex<float>*, std::complex<float>*, std::complex<float>*),
  "uk_plugin", cvmmul_f)

DEFINE_UKERNEL_TASK(vsip_csl::ukernel::Vmmul,
  void(std::pair<float*,float*>, std::pair<float*,float*>,
       std::pair<float*,float*>),
  "uk_plugin", zvmmul_f)

#endif
