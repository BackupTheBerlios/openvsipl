/* Copyright (c) 2009 by CodeSourcery.  All rights reserved. */

/** @file    ukernel/vmul.cpp
    @author  Jules Bergmann, Stefan Seefeld
    @date    2008-12-16
    @brief   VSIPL++ Library: Demonstrate standalone vmul ukernel
*/

#include <vsip/initfin.hpp>
#include <vsip/support.hpp>
#include <vsip/matrix.hpp>
#include <vsip/signal.hpp>
#include <vsip/random.hpp>
#include <vsip_csl/ukernel/host/ukernel.hpp>
#include <vsip_csl/test.hpp>
#include <vsip_csl/output.hpp>
#include <kernels/host/vmul.hpp>


DEFINE_UKERNEL_TASK(example::Vmul, void(float*, float*, float*), ".", vmul)

using namespace vsip;
using namespace vsip_csl;


template <typename Kernel, typename T>
void
run_ukernel(length_type size)
{
  Kernel kernel;

  vsip_csl::ukernel::Ukernel<Kernel> uk(kernel);

  Vector<T> in0(size);
  Vector<T> in1(size);
  Vector<T> out(size);

  Rand<T> gen(0, 0);

  in0 = gen.randu(size);
  in1 = gen.randu(size);

  in0(Domain<1>(16)) = ramp<T>(0, 1, 16);
  in1(Domain<1>(16)) = ramp<T>(0, 0.1, 16);

  uk(in0, in1, out);

  for (index_type i=0; i<size; ++i)
  {
    if (!equal(in0.get(i) * in1.get(i), out.get(i)))
    {
      std::cerr << "Error:" << std::endl;
      std::cerr << "index " << i << ": "
		<< in0.get(i) << " * "
		<< in1.get(i) << " = "
		<< in0.get(i) * in1.get(i) << "  vs  "
		<< out.get(i)
		<< std::endl;
    }
  }
}

int
main(int argc, char** argv)
{
  vsipl init(argc, argv);
  run_ukernel<example::Vmul, float>(1024+32);
  return 0;
}
