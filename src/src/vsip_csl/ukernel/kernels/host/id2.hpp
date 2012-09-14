/* Copyright (c) 2008 by CodeSourcery.  All rights reserved. */

/** @file    vsip_csl/ukernel/kernels/host/id2.hpp
    @author  Jules Bergmann
    @date    2008-07-29
    @brief   VSIPL++ Library: ID2 Ukernel
*/

#ifndef VSIP_CSL_UKERNEL_KERNELS_HOST_ID2_HPP
#define VSIP_CSL_UKERNEL_KERNELS_HOST_ID2_HPP

/***********************************************************************
  Included Files
***********************************************************************/

#include <vsip_csl/ukernel/host/ukernel.hpp>
#include <vsip_csl/ukernel/kernels/params/id2_param.hpp>

namespace vsip_csl
{
namespace ukernel
{


/***********************************************************************
  Definitions
***********************************************************************/

Stream_pattern
spatt_shape(int shape)
{
  switch (shape)
  {
  case 1:
    return Stream_pattern(Blocksize_sdist(32),
		          Blocksize_sdist(32));
  case 2:
    return Stream_pattern(Blocksize_sdist(2),
			  Blocksize_sdist(32));
  case 3:
    return Stream_pattern(Blocksize_sdist(2),
			  Blocksize_sdist(16));
  case 4:
    return Stream_pattern(Blocksize_sdist(4),
			  Blocksize_sdist(32));
  case 5:
    return Stream_pattern(Blocksize_sdist(1),
			  Blocksize_sdist(32));
  default:
    return Stream_pattern(Blocksize_sdist(1),
		          Blocksize_sdist(1024));
  }
}


// Host-side vector elementwise copy ukernel.
class Id2 : public Host_kernel
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
  typedef Uk_id2_params param_type;

  // Host-side ukernel object initialization.
  //
  // Streaming pattern divides linear vector into blocks, with minimum
  // size 256, maximum size 1024.  (Blocksize_sdist has an implicit
  // quantum of 16 elements).
  Id2(int shape, unsigned int rows, unsigned int cols)
    : sp_(spatt_shape(shape)),
      rows_(rows),
      cols_(cols)
  {}

  void fill_params(param_type& param) const
  {
    param.rows  = rows_;
    param.cols  = cols_;
  }

  // Host-side compute kernel.  Used if accelerator is not available.
  template <typename View1, typename View2>
  void compute(View1 in, View2 out)
  {
    out = in;
  }


  // Query API:  in_spatt()/out_spatt() allow VSIPL++ to determine
  // streaming pattern for user-kernel.  Since both input and output
  // have same streaming pattern, simply return 'sp'
  Stream_pattern const& in_spatt(vsip::index_type) const
  { return sp_; }

  Stream_pattern const& out_spatt(vsip::index_type) const
  { return sp_; }

private:
  Stream_pattern sp_;
  unsigned int rows_;
  unsigned int cols_;
};

}
}

DEFINE_UKERNEL_TASK(vsip_csl::ukernel::Id2, void(float*, float*), "uk_plugin", id2_f)

#endif
