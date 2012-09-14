/* Copyright (c) 2009 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/

/** @file    tests/gpu_block.cpp
    @author  Don McCoy
    @date    2009-07-17
    @brief   VSIPL++ Library: Test various functions using Gpu_blocks
*/

/***********************************************************************
  Included Files
***********************************************************************/

#include <iostream>

#include <vsip/initfin.hpp>
#include <vsip/support.hpp>
#include <vsip/vector.hpp>
#include <vsip/random.hpp>
#include <vsip/opt/cuda/gpu_block.hpp>

#include <vsip_csl/test.hpp>

using namespace std;
using namespace vsip;
using namespace impl;

#ifndef DEBUG
#define DEBUG 0
#endif

#if DEBUG
#include <vsip_csl/output.hpp>
using namespace vsip_csl;
#endif


/***********************************************************************
  Definitions
***********************************************************************/


/// GPU Block Tests
///
template <typename T>
class test_gpu_matrix
{
  typedef std::complex<T> complex_type;

  typedef cuda::Gpu_block<2, complex_type> block_type;
  typedef Matrix<complex_type, block_type> matrix_type;

public:
  test_gpu_matrix(length_type rows, length_type cols)
    : rows_(rows), cols_(cols),
      m_in_(rows, cols, complex_type()),
      m_tmp_(rows, cols, complex_type()),
      m_out_(rows, cols, complex_type())
  {}

  void fftm_identity()
  {
    Rand<> gen(0, 0);
    m_in_ = gen.randu(rows_, cols_);

    typedef Fftm<complex_type, complex_type, row, fft_fwd, by_value> fwd_fftm_type;
    typedef Fftm<complex_type, complex_type, row, fft_inv, by_value> inv_fftm_type;
    fwd_fftm_type fwd_fftm(Domain<2>(rows_, cols_), 1.0f);    
    inv_fftm_type inv_fftm(Domain<2>(rows_, cols_), 1.0f/cols_);
    
    m_tmp_ = fwd_fftm(m_in_);
    m_out_ = inv_fftm(m_tmp_);

#if DEBUG
    cout << m_in_.row(0) << endl;
    cout << m_tmp_.row(0) << endl;
    cout << m_out_.row(0) << endl;
#endif

    test_assert(vsip_csl::view_equal(m_in_, m_out_));
  }

  void exec()
  {
    fftm_identity();
  }

private: 
  length_type rows_;
  length_type cols_;
  matrix_type m_in_;
  matrix_type m_tmp_;
  matrix_type m_out_;
};



/***********************************************************************
  Main
***********************************************************************/

int
main(int argc, char** argv)
{
  vsipl init(argc, argv);

  using cuda::Gpu_block;


  /// Matrix tests using Gpu_blocks

  test_gpu_matrix<float>(10, 20).exec();

  return 0;
}
