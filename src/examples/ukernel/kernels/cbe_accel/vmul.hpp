/* Copyright (c) 2009 by CodeSourcery.  All rights reserved. */

/** @file    kernels/cbe_accel/vmul.hpp
    @author  Jules Bergmann, Stefan Seefeld
    @date    2009-06-23
    @brief   Standalone vmul ukernel example.
*/

#include <utility>
#include <cml.h>
#include <cml_core.h>
#include <vsip_csl/ukernel/cbe_accel/ukernel.hpp>

namespace example
{
struct Vmul_kernel : Spu_kernel
{
  typedef float *in0_type;
  typedef float *in1_type;
  typedef float *out0_type;

  static unsigned int const in_argc  = 2;
  static unsigned int const out_argc = 1;

  static bool const in_place = true;

  void compute(in0_type in0, in1_type in1, out0_type out,
	       Pinfo const &p_in0, Pinfo const &p_in1, Pinfo const &p_out)
  {
    cml_vmul1_f(in0, in1, out, p_out.l_total_size);
  }
};
}
