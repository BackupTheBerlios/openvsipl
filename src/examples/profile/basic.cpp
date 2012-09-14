/* Copyright (c) 2009 by CodeSourcery.  All rights reserved. */

// Defining these will turn on profiling support 
// for the 'user' category.
#define VSIP_PROFILE_USER 1

#include <vsip/initfin.hpp>
#include <vsip/support.hpp>
#include <vsip/vector.hpp>
#include <vsip_csl/profile.hpp>

using namespace vsip;
using namespace vsip_csl;

int
main(int argc, char **argv)
{
  vsipl init(argc, argv);
  {
    int const op_count = 10;
    profile::Scope<profile::user> scope("Scope 1", op_count);
    sleep(2);
  }
  profile::event<profile::user>("An event");
  {
    int const op_count = 20;
    profile::Scope<profile::user> scope("Scope 2", op_count);
    sleep(1);
  }
  Vector<float> v(8);
  return 0;
}
