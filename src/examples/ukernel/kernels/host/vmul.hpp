/* Copyright (c) 2008 by CodeSourcery.  All rights reserved. */

/** @file    kernels/host/vmul.hpp
    @author  Jules Bergmann, Stefan Seefeld
    @date    2008-12-16
    @brief   VSIPL++ Library: Vector element-wise multiply Ukernel
*/

#ifndef KERNELS_HOST_VMUL_HPP
#define KERNELS_HOST_VMUL_HPP

#include <vsip_csl/ukernel/host/ukernel.hpp>

namespace example
{

// Host-side vector elementwise multiply ukernel.
class Vmul : public vsip_csl::ukernel::Host_kernel
{
  // Parameters:
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
  Vmul() : sp_(vsip_csl::ukernel::Blocksize_sdist(1024, 256)) {}

  // Host-side compute kernel.  Used if accelerator is not available.
  template <typename View0, typename View1, typename View2>
  void compute(View0 in0, View1 in1, View2 out)
  {
    out = in0 * in1;
  }

  // Query API:  in_spatt()/out_spatt() allow VSIPL++ to determine
  // streaming pattern for user-kernel.  Since both input and output
  // have same streaming pattern, simply return 'sp'
  vsip_csl::ukernel::Stream_pattern const& in_spatt(vsip::index_type) const
  { return sp_;}

  vsip_csl::ukernel::Stream_pattern const& out_spatt(vsip::index_type) const
  { return sp_;}

private:
  vsip_csl::ukernel::Stream_pattern sp_;	
};
}

#endif
