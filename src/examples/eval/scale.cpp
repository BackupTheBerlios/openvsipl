/* Copyright (c) 2009 by CodeSourcery.  All rights reserved. */

/** @file    scale.cpp
    @author  Stefan Seefeld
    @date    2009-05-03
    @brief   Example scale function using return block optimization.
*/

#include <vsip/initfin.hpp>
#include "scale.hpp"

using namespace example;

int 
main(int argc, char **argv)
{
  vsipl init(argc, argv);
  Vector<float> a(8, 2.);
  Vector<float> b = scale(a, 2.f);
}
