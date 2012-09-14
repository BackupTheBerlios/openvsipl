/* Copyright (c) 2009 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/

/** @file    tests/gpu_memory.cpp
    @author  Don McCoy
    @date    2009-04-08
    @brief   VSIPL++ Library: Test GPU memory management functions for CUDA
*/

/***********************************************************************
  Included Files
***********************************************************************/

#include <iostream>

#include <vsip/initfin.hpp>
#include <vsip/support.hpp>
#include <vsip/vector.hpp>
# include <vsip/opt/cuda/device_memory.hpp>
# include <vsip/opt/cuda/gpu_block.hpp>
# include <vsip/opt/cuda/kernels.hpp>

#include <vsip_csl/test.hpp>

using namespace std;
using namespace vsip;
using namespace impl;


/***********************************************************************
  Definitions
***********************************************************************/


template <dimension_type Dim, 
          typename T, 
          typename Block1,
          typename Block2>
struct check_block;

template <typename T, 
          typename Block1,
          typename Block2>
struct check_block<1, T, Block1, Block2>
{
  bool
  equal(Block1& a, Block2& b)
  {
    if (a.size() != b.size()) return false;
    
    for (vsip::length_type i = 0; i != a.size(); ++i)
      if (!vsip_csl::equal(a.get(i), b.get(i)))
        return false;
    return true;
  }
};

template <typename T, 
          typename Block1,
          typename Block2>
struct check_block<2, T, Block1, Block2>
{
  bool
  equal(Block1& a, Block2& b)
  {
    if (a.size() != b.size()) return false;
    
    for (vsip::length_type i = 0; i != a.size(2, 0); ++i)
      for (vsip::length_type j = 0; j != a.size(2, 1); ++j)
        if (!vsip_csl::equal(a.get(i, j), b.get(i, j)))
          return false;
    return true;
  }
};

template <typename T, 
          typename Block1,
          typename Block2>
struct check_block<3, T, Block1, Block2>
{
  bool
  equal(Block1& a, Block2& b)
  {
    if (a.size() != b.size()) return false;
    
    for (vsip::length_type i = 0; i != a.size(3, 0); ++i)
      for (vsip::length_type j = 0; j != a.size(3, 1); ++j)
        for (vsip::length_type k = 0; k != a.size(3, 2); ++k)
          if (!vsip_csl::equal(a.get(i, j, k), b.get(i, j, k)))
            return false;
    return true;
  }
};


template <dimension_type Dim, 
          typename T, 
          typename Block>
struct fill_block;

template <typename T, 
          typename Block>
struct fill_block<1, T, Block>
{
  void
  ramp(Block& a)
  {
    for (length_type i = 0; i < a.size(); ++i)
      a.put(i, T(i));
  }
};

template <typename T, 
          typename Block>
struct fill_block<2, T, Block>
{
  void
  ramp(Block& a)
  {
    for (length_type i = 0; i < a.size(2, 0); ++i)
      for (length_type j = 0; j < a.size(2, 1); ++j)
        a.put(i, j, T(i * a.size(2, 1) + j));
  }
};

template <typename T, 
          typename Block>
struct fill_block<3, T, Block>
{
  void
  ramp(Block& a)
  {
    for (length_type i = 0; i < a.size(3, 0); ++i)
      for (length_type j = 0; j < a.size(3, 1); ++j)
        for (length_type k = 0; k < a.size(3, 2); ++k)
          a.put(i, j, k, T((i * a.size(3, 2) + j) * a.size(3, 1) + k));
  }
};




/// GPU Block Tests
///
template <dimension_type Dim,
          typename       T,
          typename       Order>
class test_gpu_block
{
  typedef cuda::Gpu_block<Dim, T, Order> block_type;
  typedef cuda::Gpu_block<Dim, T, Order> const const_block_type;

public:
  test_gpu_block(Domain<Dim> const& dom, T const val)
    : value_(val), in_block_(dom), in_const_block_(dom, val), out_block_(dom)
  {
    fill_block<Dim, T, block_type>().ramp(in_block_);
  }

  void exec()
  {
    // Test by creating and filling input blocks, copying to
    // an output block and verifying the data.  This tests the
    // block's ability to automatically copy data over to its 
    // corresponding GPU buffer space.

    // Check the initialized block.  The device must be requested
    // before a data pointer is obtained.  Likewise, the device
    // should be released when the caller no longer needs a pointer
    // to the GPU data.
    out_block_.device_request(SYNC_OUT);
    in_const_block_.device_request(SYNC_IN);
    cuda::zero_device_memory(out_block_.device_data(), out_block_.size());
    cuda::copy_device_memory(
      in_const_block_.device_data(), 
      out_block_.device_data(), 
      in_const_block_.size());
    in_const_block_.device_release();
    test_assert((check_block<Dim, T, const_block_type, block_type>().
        equal(in_const_block_, out_block_)));

    // Now check the un-initialized block
    cuda::zero_device_memory(out_block_.device_data(), out_block_.size());
    in_block_.device_request(SYNC_INOUT);
    cuda::copy_device_memory(
      in_block_.device_data(), 
      out_block_.device_data(), 
      in_block_.size());
    in_block_.device_release();
    test_assert((check_block<Dim, T, block_type, block_type>().
        equal(in_block_, out_block_)));
  }

private: 
  T value_;
  block_type in_block_;
  const_block_type in_const_block_;
  block_type out_block_;
};



/// Device Memory Tests
///
template <dimension_type Dim,
          typename       T,
          typename       Block>
class test_device_memory
{
  typedef Block block_type;
  typedef Block const const_block_type;

public:
  test_device_memory(Domain<Dim> const& dom, T const val)
    : value_(val), in_block_(dom), in_const_block_(dom, val), 
      out_block_(dom)
  {
    fill_block<Dim, T, block_type>().ramp(in_block_);
  }

  void exec()
  {
    // Test by walking tnhrough the various valid combinations of const vs. 
    // non-const blocks and the three different sync_action_types. The
    // constraints on these are
    // 
    //   input must be SYNC_IN or SYNC_INOUT
    //   if the input is const, then it defaults to SYNC_IN 
    //   output must be SYNC_OUT or SYNC_INOUT and may not be const
    //
    const_test(SYNC_INOUT);
    test_assert((check_block<Dim, T, const_block_type, block_type>().
        equal(in_const_block_, out_block_)));

    const_test(SYNC_OUT);
    test_assert((check_block<Dim, T, const_block_type, block_type>().
        equal(in_const_block_, out_block_)));

    nonconst_test(SYNC_IN, SYNC_INOUT);
    test_assert((check_block<Dim, T, block_type, block_type>().
        equal(in_block_, out_block_)));

    nonconst_test(SYNC_INOUT, SYNC_INOUT);
    test_assert((check_block<Dim, T, block_type, block_type>().
        equal(in_block_, out_block_)));

    nonconst_test(SYNC_IN, SYNC_OUT);
    test_assert((check_block<Dim, T, block_type, block_type>().
        equal(in_block_, out_block_)));

    nonconst_test(SYNC_INOUT, SYNC_OUT);
    test_assert((check_block<Dim, T, block_type, block_type>().
        equal(in_block_, out_block_)));

  }

private:
  void
  const_test(sync_action_type sync_out)
  {
    length_type const total_size = in_block_.size();
    cuda::Device_memory<const_block_type> dev_in(in_const_block_);
    cuda::Device_memory<block_type> dev_out(out_block_, sync_out);
    cuda::zero_device_memory(dev_out.data(), total_size);
    cuda::copy_device_memory(
      dev_in.data(),
      dev_out.data(),
      total_size);
  }

  void
  nonconst_test(sync_action_type sync_in, sync_action_type sync_out)
  {
    length_type const total_size = in_block_.size();
    cuda::Device_memory<block_type> dev_in(in_block_, sync_in);
    cuda::Device_memory<block_type> dev_out(out_block_, sync_out);
    cuda::zero_device_memory(dev_out.data(), total_size);
    cuda::copy_device_memory(
      dev_in.data(),
      dev_out.data(),
      total_size);
  }

private: 
  T value_;
  Block in_block_;
  Block const in_const_block_;
  Block out_block_;
};



template <typename T,
          typename Order>
void 
test_gpu_cases(T initial_value)
{
  test_gpu_block<1, T, Order>(Domain<1>(16),       initial_value).exec();
  test_gpu_block<1, T, Order>(Domain<1>(2048),     initial_value).exec();
  test_gpu_block<2, T, Order>(Domain<2>(10, 16),   initial_value).exec();
  test_gpu_block<2, T, Order>(Domain<2>(32, 4096), initial_value).exec();
  test_gpu_block<3, T, Order>(Domain<3>(10, 16, 8),   initial_value).exec();
  test_gpu_block<3, T, Order>(Domain<3>(32, 1024, 4), initial_value).exec();
}



/***********************************************************************
  Main
***********************************************************************/

int
main(int argc, char** argv)
{
  vsipl init(argc, argv);

  using cuda::Gpu_block;

  typedef float                F;
  typedef std::complex<float>  C;


  //// GPU block tests ////

  test_gpu_cases<F, row2_type>(F(2));
  test_gpu_cases<C, row2_type>(C(2, -1));
  test_gpu_cases<F, col2_type>(F(3));
  test_gpu_cases<C, col2_type>(C(3, -2));


  //// Device Memory interface tests ////

  // test scalar floats
  test_device_memory<1, F, Dense<1, F> >(Domain<1>(32),       F(2)).exec();
  test_device_memory<1, F, Dense<1, F> >(Domain<1>(57),       F(3)).exec();
  test_device_memory<1, F, Dense<1, F> >(Domain<1>(4096),     F(4)).exec();
  test_device_memory<2, F, Dense<2, F> >(Domain<2>(10, 15),   F(2)).exec();
  test_device_memory<2, F, Dense<2, F> >(Domain<2>(128, 256), F(3)).exec();
  test_device_memory<2, F, Dense<2, F> >(Domain<2>(16, 8192), F(4)).exec();
  test_device_memory<3, F, Dense<3, F> >(Domain<3>(4, 16, 8),    F(2)).exec();
  test_device_memory<3, F, Dense<3, F> >(Domain<3>(10, 73, 9),   F(3)).exec();
  test_device_memory<3, F, Dense<3, F> >(Domain<3>(16, 128, 32), F(4)).exec();

  // test complex floats
  test_device_memory<1, C, Dense<1, C> >(Domain<1>(32),       C(2, -1)).exec();
  test_device_memory<1, C, Dense<1, C> >(Domain<1>(57),       C(3, -1)).exec();
  test_device_memory<1, C, Dense<1, C> >(Domain<1>(4096),     C(4, -1)).exec();
  test_device_memory<2, C, Dense<2, C> >(Domain<2>(10, 15),   C(2, -1)).exec();
  test_device_memory<2, C, Dense<2, C> >(Domain<2>(128, 256), C(3, -1)).exec();
  test_device_memory<2, C, Dense<2, C> >(Domain<2>(16, 8192), C(4, -1)).exec();
  test_device_memory<3, C, Dense<3, C> >(Domain<3>(4, 16, 8),    C(2, -1)).exec();
  test_device_memory<3, C, Dense<3, C> >(Domain<3>(10, 73, 9),   C(3, -1)).exec();
  test_device_memory<3, C, Dense<3, C> >(Domain<3>(16, 128, 32), C(4, -1)).exec();


  // test other block types
  test_device_memory<1, F, Gpu_block<1, F> >(Domain<1>(32),      F(2)).exec();
  test_device_memory<1, C, Gpu_block<1, C> >(Domain<1>(128),     C(2, -1)).exec();
  test_device_memory<2, F, Gpu_block<2, F> >(Domain<2>(10, 32),  F(2)).exec();
  test_device_memory<2, C, Gpu_block<2, C> >(Domain<2>(16, 128), C(2, -1)).exec();
  test_device_memory<3, F, Gpu_block<3, F> >(Domain<3>(10, 16, 4), F(2)).exec();
  test_device_memory<3, C, Gpu_block<3, C> >(Domain<3>(16, 64, 8), C(2, -1)).exec();

  test_device_memory<1, F, Fast_block<1, F> >(Domain<1>(32),      F(2)).exec();
  test_device_memory<1, C, Fast_block<1, C> >(Domain<1>(128),     C(2, -1)).exec();
  test_device_memory<2, F, Fast_block<2, F> >(Domain<2>(10, 32),  F(2)).exec();
  test_device_memory<2, C, Fast_block<2, C> >(Domain<2>(16, 128), C(2, -1)).exec();
  test_device_memory<3, F, Fast_block<3, F> >(Domain<3>(10, 16, 4), F(3)).exec();
  test_device_memory<3, C, Fast_block<3, C> >(Domain<3>(16, 64, 8), C(3, -1)).exec();

  return 0;
}
