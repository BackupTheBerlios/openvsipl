/* Copyright (c) 2009 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    benchmarks/cuda/unary.cpp
    @author  Don McCoy
    @date    2009-06-26
    @brief   VSIPL++ Library: Benchmark for CUDA-based unary functions.
*/


/***********************************************************************
  Included Files
***********************************************************************/

#include <iostream>
#include <ostream>

#include <cuda_runtime.h>
#include <cuComplex.h>

#include <vsip/initfin.hpp>
#include <vsip/support.hpp>
#include <vsip/math.hpp>
#include <vsip/core/expr/fns_elementwise.hpp>
#include <vsip/core/profile.hpp>
#include <vsip/opt/cuda/kernels.hpp>
#include <vsip/opt/cuda/device_memory.hpp>

#include <vsip_csl/test.hpp>
#include "loop.hpp"

using namespace std;
using namespace vsip;
using vsip_csl::equal;
namespace expr = vsip_csl::expr;

#define DEBUG 0

/***********************************************************************
  Declarations
***********************************************************************/

// Functors from fns_elementwise.hpp are used to differentiate the
// benchmark types (instead of the struct *_tag; declarations)


/***********************************************************************
  Unary expression test harness
***********************************************************************/

template <typename T1,
          typename T2,
          class F>
struct t_unary_base : Benchmark_base
{
  typedef Dense<2, T1, row2_type>  src_block_t;
  typedef Dense<2, T2, row2_type>  dst_block_t;

  typedef void(unary_functor_type)(T1 const*, T2*, length_type, length_type);

  t_unary_base(unary_functor_type f)
    : functor_(f)
  {}

  char const* what() 
  { 
    ostringstream out;
    out << "CUDA t_unary<..., " << F::name() << ">";
    return out.str().c_str();
  }
  
  void exec(length_type rows, length_type cols, length_type loop, float& time)
  {
    Matrix<T1, src_block_t>   A(rows, cols);
    Matrix<T2, dst_block_t>   Z(rows, cols, T2());
    for (index_type i = 0; i < rows; ++i)
      for (index_type j = 0; j < cols; ++j)
        A.put(i, j, T1(i * cols + j, -0.5));

    // Scoping is used to control the lifetime of Ext_data<> objects.  These 
    // must be destroyed before accessing data through the view again.
    {     
      impl::cuda::Device_memory<src_block_t const> dev_a(A.block());
      impl::cuda::Device_memory<dst_block_t> dev_z(Z.block());
      T1 const* pA = dev_a.data();
      T2* pZ = dev_z.data();

      // Benchmark the operation
      vsip::impl::profile::Timer t1;
      t1.start();
      for (index_type l = 0; l < loop; ++l)
      {
        this->functor_(pA, pZ, rows, cols);
        cudaThreadSynchronize();
      }
      t1.stop();
      time = t1.delta();
    }

    // validate results
    for (index_type i = 0; i < rows; ++i)
      for (index_type j = 0; j < cols; ++j)
      {
#if DEBUG
        if (!equal(F::apply(A.get(i, j)), Z.get(i, j)))
        {
          cout << "ERROR: at location " << i << ", " << j << endl
                    << "       expected: " << F::apply(A.get(i, j)) << endl
                    << "       got     : " << Z.get(i, j) << endl;
        }
#endif
        test_assert(equal(F::apply(A.get(i, j)), Z.get(i, j)));
      }
  }

private:
  // data members
  unary_functor_type* functor_;
};




/***********************************************************************
  Row size constant
***********************************************************************/

template <typename T1, typename T2, class F>
struct t_unary_rows_fixed : public t_unary_base<T1, T2, F>
{
  int ops_per_point(length_type)  { return rows_ * ops_per_element_; }
  int riob_per_point(length_type) { return rows_ * sizeof(T1); }
  int wiob_per_point(length_type) { return rows_ * sizeof(T2); }
  int mem_per_point(length_type)  { return rows_ * (sizeof(T1) + sizeof(T2)); }

  void operator()(vsip::length_type size, vsip::length_type loop, float& time)
  {
    this->exec(rows_, size, loop, time);
  }

  typedef typename t_unary_base<T1, T2, F>::unary_functor_type unary_functor_type;

  t_unary_rows_fixed(unary_functor_type func, vsip::length_type rows, int ops)
    : t_unary_base<T1, T2, F>(func), rows_(rows), ops_per_element_(ops)
  {}

// Member data
  vsip::length_type rows_;
  int ops_per_element_;
};


/***********************************************************************
  Column size constant
***********************************************************************/

template <typename T1, typename T2, class F>
struct t_unary_cols_fixed : public t_unary_base<T1, T2, F>
{
  int ops_per_point(length_type)  { return cols_ * ops_per_element_; }
  int riob_per_point(length_type) { return cols_ * sizeof(T1); }
  int wiob_per_point(length_type) { return cols_ * sizeof(T2); }
  int mem_per_point(length_type)  { return cols_ * (sizeof(T1) + sizeof(T2)); }

  void operator()(vsip::length_type size, vsip::length_type loop, float& time)
  {
    this->exec(size, cols_, loop, time);
  }

  typedef typename t_unary_base<T1, T2, F>::unary_functor_type unary_functor_type;

  t_unary_cols_fixed(unary_functor_type func, vsip::length_type cols, int ops)
    : t_unary_base<T1, T2, F>(func), cols_(cols), ops_per_element_(ops)
  {}

// Member data
  vsip::length_type cols_;
  int ops_per_element_;
};



/***********************************************************************
  Benchmark Driver
***********************************************************************/

void
defaults(Loop1P& loop)
{
  loop.param_["rows"] = "64";
  loop.param_["size"] = "2048";
}



int
test(Loop1P& loop, int what)
{
  typedef float F;
  typedef complex<float> C;

  length_type rows  = atoi(loop.param_["rows"].c_str());
  length_type size  = atoi(loop.param_["size"].c_str());

  cout << "rows: " << rows << "  size: " << size << endl;
  
  switch (what)
  {
    // Template parameters are:  
    //     <Input Type, Output Type, Functor>
    // Constructor parameters are:  
    //     (function handler, size of the fixed-dimension, operation count per element)

    // sweep # columns
  case  1: loop(t_unary_rows_fixed<C, F, expr::op::Mag<C> >(impl::cuda::cmag, rows, 1)); break;

    // sweep # rows
  case 11: loop(t_unary_cols_fixed<C, F, expr::op::Mag<C> >(impl::cuda::cmag, size, 1)); break;

    // help
  default:
    cout
      << "CUDA unary expressions -- fixed rows\n"
      << "   -1 -- complex magnitude\n"
      ;
    cout
      << "CUDA unary expressions -- fixed columns\n"
      << "  -11 -- complex magnitude\n"
      ;
    return 0;
  }
  return 1;
}
