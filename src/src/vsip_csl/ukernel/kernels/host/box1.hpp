/* Copyright (c) 2008 by CodeSourcery.  All rights reserved. */

/** @file    vsip_csl/ukernel/kernels/host/box1.hpp
    @author  Jules Bergmann
    @date    2008-08-01
    @brief   VSIPL++ Library: Box1 Ukernel
*/

#ifndef VSIP_CSL_UKERNEL_KERNELS_HOST_BOX1_HPP
#define VSIP_CSL_UKERNEL_KERNELS_HOST_BOX1_HPP

/***********************************************************************
  Included Files
***********************************************************************/

#include <vsip_csl/ukernel/host/ukernel.hpp>
#include <vsip_csl/ukernel/kernels/params/box1_param.hpp>

namespace vsip_csl
{
namespace ukernel
{

/***********************************************************************
  Definitions
***********************************************************************/

// Host-side vector elementwise copy ukernel.
class Box1 : public Host_kernel
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
  typedef Box1_params param_type;

  // Host-side ukernel object initialization.
  //
  // Streaming pattern divides linear vector into blocks, with minimum
  // size 256, maximum size 1024.  (Blocksize_sdist has an implicit
  // quantum of 16 elements).
  Box1(int overlap, int same_support)
    : in_sp_(same_support ?
	     Blockoverlap_sdist(32, 32, overlap, overlap, 0, 0) :
	     Blockoverlap_sdist(32, overlap)),
      out_sp_(Blocksize_sdist(32)),
      overlap_(overlap)
  {}

  void fill_params(param_type& param) const
  {
    param.overlap = overlap_;
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
  vsip::impl::ukernel::Stream_pattern const& in_spatt(vsip::index_type) const
  { return in_sp_; }

  vsip::impl::ukernel::Stream_pattern const& out_spatt(vsip::index_type) const
  { return out_sp_; }

private:
  Stream_pattern in_sp_;	
  Stream_pattern out_sp_;	
  int            overlap_;
};

}
}

DEFINE_UKERNEL_TASK(vsip_csl::ukernel::Box1, void(float*, float*), "uk_plugin", box1_f)

#endif
