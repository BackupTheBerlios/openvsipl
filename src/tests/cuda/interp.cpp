/* Copyright (c) 2009 by CodeSourcery.  All rights reserved. */

/** @file    tests/cuda/interp.cpp
    @author  Don McCoy
    @date    2009-08-17
    @brief   VSIPL++ Library: CUDA-based test for polar to rectangular
               interpolation for SSAR images.
*/

/***********************************************************************
  Included Files
***********************************************************************/

#include <vsip/initfin.hpp>
#include <vsip/support.hpp>
#include <vsip/matrix.hpp>
#include <vsip/signal.hpp>
#include <vsip/random.hpp>
#include <vsip/opt/cuda/gpu_block.hpp>
#include <vsip/opt/cuda/interp.hpp>

#include <vsip_csl/test.hpp>
#include <vsip_csl/output.hpp>

using namespace std;
using namespace vsip;
using namespace vsip_csl;


#define DBG_SHOW_IO      0
#define DBG_SHOW_ERRORS  0


namespace ref
{

template <typename IT,
	  typename T,
	  typename Block1,
	  typename Block2,
	  typename Block3,
	  typename Block4>
void
interpolate(
  const_Matrix<IT, Block1>	   indices,  // n x m
  Tensor<T, Block2>                window,   // n x m x I
  const_Matrix<complex<T>, Block3> in,       // n x m
  Matrix<complex<T>, Block4>       out,      // nx x m
  length_type                      depth,
  bool                             shift)    // perform shift also
{
  length_type n = indices.size(0);
  length_type m = indices.size(1);
  length_type nx = out.size(0);
  length_type I = depth; // window.size(2) may include padding
  assert(n == in.size(0));
  assert(m == in.size(1));
  assert(m == out.size(1));
  assert(window.size(0) == n);
  assert(window.size(1) == m);

  out = complex<T>(0);

  for (index_type j = 0; j < m; ++j)
  {
    for (index_type i = 0; i < n; ++i)
    {
      index_type ikxrows = indices.get(i, j);
      index_type i_shift = shift ? (i + n/2) % n : i;
      for (index_type h = 0; h < I; ++h)
      {
        out.put(ikxrows + h, j, out.get(ikxrows + h, j) + 
          (in.get(i_shift, j) * window.get(i, j, h)));
      }
    }
    if (shift)
      out.col(j)(Domain<1>(j%2, 2, nx/2 + ((nx%2) ? (j+1)%2 : 0))) *= T(-1);
  }
}

} // namespace ref



/***********************************************************************
  Definitions
***********************************************************************/

template <typename T>
void
test_interp_kernel(length_type rows, length_type cols, length_type depth, 
  bool with_shift)
{
  typedef uint32_t I;
  typedef std::complex<T>  C;
  typedef vsip::impl::cuda::Gpu_block<2, C, col2_type> inout_block_type;
  typedef vsip::impl::cuda::Gpu_block<2, I, col2_type> index_block_type;
  typedef vsip::impl::cuda::Gpu_block<3, T, col2_type> window_block_type;

  std::cout << rows << " x " << cols << " x " << depth 
            << "  shift: " << (with_shift ? "true" : "false") << std::endl;

  length_type padded_depth = depth;
  if (padded_depth % 4 != 0)
    padded_depth += (4 - (padded_depth % 4));

  Matrix<I, index_block_type> indices(rows, cols);
  Tensor<T, window_block_type> window(rows, cols, padded_depth);
  Matrix<C, inout_block_type> input(rows, cols);
  // filled with non-zero values to ensure all are overwritten
  Matrix<C, inout_block_type> out(rows + depth - 1, cols, C(-4, 4));
  Matrix<C, inout_block_type> ref(rows + depth - 1, cols, C(4, -4));

  // set up input data, weights and indices
  Rand<C> gen(0, 0);
  input = gen.randu(rows, cols);

  Rand<T> gen_real(1, 0);
  for (index_type k = 0; k < depth; ++k)
    window(whole_domain, whole_domain, k) = gen_real.randu(rows, cols);

  // The size of the output is determined by the way the indices are
  // set up.  Here, they are mapped one-to-one, so the output ends up
  // being larger by an amount determined by the depth of the window
  // function used.
  for (index_type i = 0; i < rows; ++i)
    indices.row(i) = i;


  // Compute reference output image
  ref::interpolate(indices, window, input, ref, depth, with_shift);


  // Compute output image using custom kernel
  if (with_shift)
    vsip::impl::cuda::interpolate_with_shift(indices, window, input, out, depth, padded_depth);
  else
    vsip::impl::cuda::interpolate(indices, window, input, out, depth, padded_depth);


  // verify results
#if  DBG_SHOW_IO
  cout << "window = " << endl << window.template transpose<2, 0, 1>() << endl;
  cout << "indices = " << endl << indices << endl;
  cout << "input = " << endl << input << endl;
  cout << "ref = " << endl << ref << endl;
  cout << "out = " << endl << out << endl;
#endif

#if DBG_SHOW_ERRORS
  int count = 0;
  for (index_type i = 0; i < out.size(0); ++i)
    for (index_type j = 0; j < out.size(1); ++j)
    {
      if (!equal(out.get(i, j), ref.get(i, j)))
      {
        cout << "[" << i << ", " << j << "] : " << out.get(i, j) << " != " << ref.get(i, j) 
             << "    " << ref.get(i, j) - out.get(i, j) << endl;
        ++count;
      }
      if (count >= 10)  // show first few errors only
        break;
    }
#endif

  test_assert(view_equal(out, ref));
}



/***********************************************************************
  Main
***********************************************************************/

int
main(int argc, char** argv)
{
  vsipl init(argc, argv);

// This kernel is presently only implemented for interleaved complex
#if !VSIP_IMPL_PREFER_SPLIT_COMPLEX

  // parameters are:
  //  input rows, input cols
  //  window width
  //  perform fftshift (freqswap) in first dimension
  test_interp_kernel<float>(8, 4, 13, false);
  test_interp_kernel<float>(128, 256, 16, false);
  test_interp_kernel<float>(1144, 1072, 17, false);

  test_interp_kernel<float>(8, 4, 13, true);
  test_interp_kernel<float>(128, 256, 16, true);
  test_interp_kernel<float>(1144, 1072, 17, true);
#endif

  return 0;
}
