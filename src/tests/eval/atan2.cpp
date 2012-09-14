/* Copyright (c) 2005, 2006 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/

#define VSIP_PROFILE_MASK 0xffff

#include <vsip/initfin.hpp>
#include <vsip_csl/test.hpp>
#include <vsip_csl/test_dispatch.hpp>
#include <vsip/core/expr/fns_elementwise.hpp>

using namespace vsip;
using namespace vsip_csl;


void
test()
{
  Vector<float> a(16);
  Vector<float> b(16);
  Vector<float> c(16);
  clear_dispatch_trace();
  c = atan2(a,b);
  char const *expected_trace[] = {"Expr_Loop 1D atan2(S,S) 16"};
  validate_dispatch_trace(expected_trace);
}


int main(int argc, char **argv)
{
  vsip::vsipl library(argc, argv);
  vsip::impl::profile::prof->set_mode(vsip::impl::profile::pm_trace);
  test();
}
