/* Copyright (c) 2009 by CodeSourcery.  All rights reserved. */

/** @file    interpolate.cpp
    @author  Stefan Seefeld
    @date    2009-05-03
    @brief   Example interpolate function using return block optimization.
*/

#include <vsip/initfin.hpp>
#include "scale.hpp"
#include "interpolate.hpp"

using namespace example;

int 
main(int argc, char **argv)
{
  vsipl init(argc, argv);
  Vector<float> a(8, 2.);
  Vector<float> b = interpolate(scale(a, 2.f), 32);
}
