/* Copyright (c) 2008 by CodeSourcery.  All rights reserved. */

/** @file    vsip_csl/ukernel/kernels/host/box2.hpp
    @author  Jules Bergmann
    @date    2008-08-01
    @brief   VSIPL++ Library: Box2 Ukernel
*/

#ifndef VSIP_CSL_UKERNEL_KERNELS_HOST_BOX2_HPP
#define VSIP_CSL_UKERNEL_KERNELS_HOST_BOX2_HPP

/***********************************************************************
  Included Files
***********************************************************************/

#include <vsip_csl/ukernel/host/ukernel.hpp>
#include <vsip_csl/ukernel/kernels/params/box2_param.hpp>

namespace vsip_csl
{
namespace ukernel
{

/***********************************************************************
  Definitions
***********************************************************************/

// Host-side vector elementwise copy ukernel.
class Box2 : public Host_kernel
{
  // Parameters.
  //  - 'tag_type' is used to select the appropriate kernel (via
  //    Task_manager's Task_map)
  //  - in_argc and out_argc describe the number of input and output
  //    streams.
  //  - param_type (inherited) defaults to 'Empty_params'.
public:
  static unsigned int const in_argc  = 1;
  static unsigned int const out_argc = 1;
  typedef Uk_box2_params param_type;

  // Host-side ukernel object initialization.
  //
  // Streaming pattern divides linear vector into blocks, with minimum
  // size 256, maximum size 1024.  (Blocksize_sdist has an implicit
  // quantum of 16 elements).
  Box2(int overlap)
    : in_sp_(Blockoverlap_sdist(16, 16, overlap, overlap, 1, 1),
	     Blockoverlap_sdist(64, 16, overlap, overlap, 1, 1)),
      out_sp_(Blocksize_sdist(16, 16, 16),
	      Blocksize_sdist(64, 64, 16)),
      overlap0_(overlap),
      overlap1_(overlap)
  {}

  void fill_params(param_type& param) const
  {
    param.overlap0 = overlap0_;
    param.overlap1 = overlap1_;
  }


  // Host-side compute kernel.  Used if accelerator is not available.
  template <typename View1, typename View2>
  void compute(View1 in, View2 out)
  {
    out = in;
  }


  // Queury API:  in_spatt()/out_spatt() allow VSIPL++ to determine
  // streaming pattern for user-kernel.  Since both input and output
  // have same streaming pattern, simply return 'sp'
  Stream_pattern const& in_spatt(vsip::index_type) const
  { return in_sp_; }

  Stream_pattern const& out_spatt(vsip::index_type) const
  { return out_sp_; }

private:
  Stream_pattern in_sp_;	
  Stream_pattern out_sp_;	
  int overlap0_;
  int overlap1_;
};

}
}

DEFINE_UKERNEL_TASK(vsip_csl::ukernel::Box2, void(float*, float*), "uk_plugin", box2_f)

#endif
