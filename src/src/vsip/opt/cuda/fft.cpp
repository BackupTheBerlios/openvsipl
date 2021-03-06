/* Copyright (c) 2009 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    vsip/opt/cuda/fft.cpp
    @author  Don McCoy
    @date    2009-02-26
    @brief   VSIPL++ Library: FFT wrappers and traits to bridge with 
             NVidia's CUDA FFT library.
*/

/***********************************************************************
  Included Files
***********************************************************************/

#include <complex>
#include <cufft.h>
#include <cuda_runtime.h>

#include <vsip/core/config.hpp>
#include <vsip/support.hpp>
#include <vsip/domain.hpp>
#include <vsip/core/equal.hpp>
#include <vsip/core/fft/backend.hpp>
#include <vsip/core/fft/util.hpp>
#include <vsip/core/fns_scalar.hpp>
#include <vsip/core/metaprogramming.hpp>
#include <vsip/opt/cuda/bindings.hpp>
#include <vsip/opt/cuda/device_memory.hpp>
#include <vsip/opt/cuda/fft.hpp>


/***********************************************************************
  Declarations
***********************************************************************/

namespace vsip
{
namespace impl
{
namespace cuda
{


// CUDA's type naming convention:
//   C type:         typedef:
//
//   float           cufftReal
//   complex<float>  cufftComplex


/// The base class mini 'driver' for both 1-D FFTs and FFTMs
template <typename  T>
struct Driver_base_1d
{
  Driver_base_1d() 
    : plan_(), dev_in_(), dev_out_(), insize_(), outsize_(), 
      rows_(), cols_(), scale_()
  {}

  typedef typename Scalar_of<T>::type scalar_type;

  /// 1-D Driver initialization function
  ///   fft_type   Specifies real or complex FFT
  ///   n          Number of columns (FFT length)
  ///   m          Number of rows (defaults to one for single FFTs)
  void init(cufftType fft_type, size_t n, size_t m = 1, 
    scalar_type scale = 1.f)  VSIP_THROW((std::bad_alloc))
  {
    // Determine buffer memory requirements
    size_t n2 = n / 2 + 1;
    if (fft_type == CUFFT_R2C)
    {
      insize_ = m * n * sizeof(cufftReal);
      outsize_ = m * n2 * sizeof(cufftComplex);
    }
    else if (fft_type == CUFFT_C2R)
    {
      insize_ = m * n2 * sizeof(cufftComplex);
      outsize_ = m * n * sizeof(cufftReal);
    }
    else if (fft_type == CUFFT_C2C)
    {
      insize_ = m * n * sizeof(cufftComplex);
      outsize_ = m * n * sizeof(cufftComplex);
    }

    // Create a plan
    cufftResult result = 
      cufftPlan1d(&plan_, n, fft_type, m);
    ASSERT_CUFFT_OK(result);
    if (result != CUFFT_SUCCESS)
      VSIP_IMPL_THROW(std::bad_alloc());

    // Save the sizes and scale factor
    rows_ = m;
    cols_ = n;
    scale_ = scale;
  }

  void fini()  VSIP_NOTHROW
  {
    cufftResult result = cufftDestroy(plan_);
    ASSERT_CUFFT_OK(result);
  }


  /// real->complex
  void forward(T const* in, std::complex<T>* out)  VSIP_NOTHROW
  {
    cufftResult result = cufftExecR2C(plan_, 
      reinterpret_cast<cufftReal*>(const_cast<T*>(in)),
      reinterpret_cast<cufftComplex*>(out));
    ASSERT_CUFFT_OK(result);

    if(!almost_equal(scale_, scalar_type(1.)))
      smmul_sc(scale_,
        reinterpret_cast<cufftComplex const*>(out),
        reinterpret_cast<cufftComplex*>(out),
        rows_, cols_);
  }

  /// complex->real
  void forward(std::complex<T> const* in, T* out)  VSIP_NOTHROW
  {
    cufftResult result = cufftExecC2R(plan_, 
      reinterpret_cast<cufftComplex*>(const_cast<std::complex<T>*>(in)),
      reinterpret_cast<cufftReal*>(out));
    ASSERT_CUFFT_OK(result);

    if(!almost_equal(scale_, scalar_type(1.)))
      smmul_ss(scale_,
        out,
        out,
        rows_, cols_);
  }

  /// complex->complex
  void forward(T const* in, T* out)  VSIP_NOTHROW
  {
    cufftResult result = cufftExecC2C(plan_, 
      reinterpret_cast<cufftComplex*>(const_cast<T*>(in)),
      reinterpret_cast<cufftComplex*>(out), CUFFT_FORWARD);
    ASSERT_CUFFT_OK(result);

    if(!almost_equal(scale_, scalar_type(1.)))
      smmul_sc(scale_,
        reinterpret_cast<cufftComplex const*>(out),
        reinterpret_cast<cufftComplex*>(out),
        rows_, cols_);
  }

  /// complex->complex
  void inverse(T const* in, T* out)  VSIP_NOTHROW
  {
    cufftResult result = cufftExecC2C(plan_, 
      reinterpret_cast<cufftComplex*>(const_cast<T*>(in)),
      reinterpret_cast<cufftComplex*>(out), CUFFT_INVERSE);
    ASSERT_CUFFT_OK(result);

    if(!almost_equal(scale_, scalar_type(1.)))
      smmul_sc(scale_,
        reinterpret_cast<cufftComplex const*>(out),
        reinterpret_cast<cufftComplex*>(out),
        rows_, cols_);
  }

  cufftHandle plan_;
  void* dev_in_;
  void* dev_out_;
  size_t insize_;
  size_t outsize_;
  size_t rows_;
  size_t cols_;
  scalar_type scale_;
};




template <dimension_type D, //< Dimension
          typename       T> //< Type
struct Driver;


template <typename  T>
struct Driver<1, T>
  : Driver_base_1d<T>
{
  // CUDA FFT type is R2C, C2R or C2C
  Driver(cufftType fft_type, Domain<1> const &dom, length_type mult = 1, 
    typename Scalar_of<T>::type scale = 1.f) 
  {
    this->init(fft_type, dom.size(), mult, scale);
  }
  ~Driver() { this->fini();}
};




template <dimension_type D, //< Dimension
	  typename I,       //< Input type
	  typename O,       //< Output type
	  int A,            //< Axis
          int E>            //< Exponent
class impl;

// 1D real -> complex FFT
template <typename T, int A, int E>
class impl<1, T, std::complex<T>, A, E>
  : public fft::backend<1, T, std::complex<T>, A, E>,
    private Driver<1, T>
{
  typedef T rtype;
  typedef std::complex<rtype> ctype;
  typedef std::pair<rtype*, rtype*> ztype;

public:
  impl(Domain<1> const &dom, rtype scale)
    : Driver<1, T>(CUFFT_R2C, dom, 1, scale)
  {}
  virtual char const* name() { return "fft-cuda-1D-real-forward"; }
  virtual bool supports_scale() { return true;}
  virtual bool supports_cuda_memory() { return true;}
  virtual void by_reference(rtype *in, stride_type in_s,
			    ctype *out, stride_type out_s,
			    length_type /*l*/)
  {
    assert(in_s == 1 && out_s == 1 && (E == -1));
    this->forward(in, out);
  }
  virtual void by_reference(rtype */*in*/, stride_type /*in_s*/,
			    ztype /*out*/, stride_type /*out_s*/,
			    length_type /*l*/)
  {}
};

// 1D complex -> real FFT
template <typename T, int A, int E>
class impl<1, std::complex<T>, T, A, E>
  : public fft::backend<1, std::complex<T>, T, A, E>,
    private Driver<1, T>
{
  typedef T rtype;
  typedef std::complex<rtype> ctype;
  typedef std::pair<rtype*, rtype*> ztype;

public:
  impl(Domain<1> const &dom, rtype scale)
    : Driver<1, T>(CUFFT_C2R, dom, 1, scale)
  {}
  virtual char const* name() { return "fft-cuda-1D-real-inverse"; }
  virtual bool supports_scale() { return true;}
  virtual bool supports_cuda_memory() { return true;}
  virtual void by_reference(ctype *in, stride_type in_s,
			    rtype *out, stride_type out_s,
			    length_type /*l*/)
  {
    assert(in_s == 1 && out_s == 1 && (E == 1));
    this->forward(in, out);
  }
  virtual void by_reference(ztype, stride_type,
			    rtype*, stride_type,
			    length_type)
  {}
};

// 1D complex -> complex FFT
template <typename T, int A, int E>
class impl<1, std::complex<T>, std::complex<T>, A, E>
  : public fft::backend<1, std::complex<T>, std::complex<T>, A, E>,
    private Driver<1, std::complex<T> >
{
  typedef T rtype;
  typedef std::complex<rtype> ctype;
  typedef std::pair<rtype*, rtype*> ztype;

public:
  impl(Domain<1> const &dom, rtype scale)
    : Driver<1, std::complex<T> >(CUFFT_C2C, dom, 1, scale)
  {}
  virtual char const* name() { return "fft-cuda-1D-complex"; }
  virtual bool supports_scale() { return true;}
  virtual bool supports_cuda_memory() { return true;}
  virtual void in_place(ctype *inout, stride_type s, length_type /*l*/)
  {
    assert(s == 1);
    if (E == -1) this->forward(inout, inout);
    else         this->inverse(inout, inout);
  }
  virtual void in_place(ztype, stride_type, length_type)
  {
  }
  virtual void by_reference(ctype *in, stride_type in_s,
			    ctype *out, stride_type out_s,
			    length_type /*l*/)
  {
    assert(in_s == 1 && out_s == 1);
    if (E == -1) this->forward(in, out);
    else         this->inverse(in, out);
  }
  virtual void by_reference(ztype, stride_type,
			    ztype, stride_type,
			    length_type)
  {}
};




template <typename I, //< Input type
	  typename O, //< Output type
	  int A,      //< Axis
          int E>      //< Exponent
class fftm;

// real -> complex FFTM
template <typename T, int A>
class fftm<T, std::complex<T>, A, -1>
  : public fft::fftm<T, std::complex<T>, A, -1>,
    private Driver<1, T>
{
  typedef T rtype;
  typedef std::complex<rtype> ctype;
  typedef std::pair<rtype*, rtype*> ztype;

public:
  fftm(Domain<2> const &dom, rtype scale)
    : Driver<1, T>(CUFFT_R2C, dom[A], dom[1 - A].size(), scale),
      mult_(dom[1 - A].size())
  {}

  virtual char const* name() { return "fftm-cuda-real-forward"; }
  virtual bool supports_scale() { return true;}
  virtual bool supports_cuda_memory() { return true;}
  virtual void by_reference(rtype *in,
			    stride_type in_r_stride, stride_type in_c_stride,
			    ctype *out,
			    stride_type out_r_stride, stride_type out_c_stride,
			    length_type rows, length_type cols)
  {
    if (A == 0)
    {
      assert(cols == mult_);
      assert(in_r_stride == 1 && out_r_stride == 1);
      assert((in_c_stride > 0) && (static_cast<length_type>(in_c_stride) >= rows));
      assert((out_c_stride > 0) && (static_cast<length_type>(out_c_stride) >= rows/2 + 1));
    }
    else
    {   
      assert(rows == mult_);
      assert(in_c_stride == 1 && out_c_stride == 1);
      assert((in_r_stride > 0) && (static_cast<length_type>(in_r_stride) >= cols));
      assert((out_r_stride > 0) && (static_cast<length_type>(out_r_stride) >= cols/2 + 1));
    }
    this->forward(in, out);
  }
  virtual void by_reference(rtype*, stride_type, stride_type,
			    ztype, stride_type, stride_type,
			    length_type, length_type)
  {}

  length_type mult_;
};

// complex -> real FFTM
template <typename T, int A>
class fftm<std::complex<T>, T, A, 1>
  : public fft::fftm<std::complex<T>, T, A, 1>,
    private Driver<1, T>
{
  typedef T rtype;
  typedef std::complex<rtype> ctype;
  typedef std::pair<rtype*, rtype*> ztype;

public:
  fftm(Domain<2> const &dom, rtype scale)
    : Driver<1, T>(CUFFT_C2R, dom[A], dom[1 - A].size(), scale),
      mult_(dom[1 - A].size())
  {}

  virtual char const* name() { return "fftm-cuda-real-inverse"; }
  virtual bool supports_scale() { return true;}
  virtual bool supports_cuda_memory() { return true;}
  virtual void by_reference(ctype *in,
			    stride_type in_r_stride, stride_type in_c_stride,
			    rtype *out,
			    stride_type out_r_stride, stride_type out_c_stride,
			    length_type rows, length_type cols)
  {
    if (A == 0)
    {
      assert(cols == mult_);
      assert(in_r_stride == 1 && out_r_stride == 1);
      assert((in_c_stride > 0) && (static_cast<length_type>(in_c_stride) >= rows/2 + 1));
      assert((out_c_stride > 0) && (static_cast<length_type>(out_c_stride) >= rows));
    }
    else
    {   
      assert(rows == mult_);
      assert(in_c_stride == 1 && out_c_stride == 1);
      assert((in_r_stride > 0) && (static_cast<length_type>(in_r_stride) >= cols/2 + 1));
      assert((out_r_stride > 0) && (static_cast<length_type>(out_r_stride) >= cols));
    }

    this->forward(in, out);
  }
  virtual void by_reference(ztype, stride_type, stride_type,
			    rtype *, stride_type, stride_type,
			    length_type, length_type)
  {}

  length_type mult_;
};


// complex -> complex FFTM
template <typename T, int A, int E>
class fftm<std::complex<T>, std::complex<T>, A, E>
  : public fft::fftm<std::complex<T>, std::complex<T>, A, E>,
    private Driver<1, std::complex<T> >
{
  typedef T rtype;
  typedef std::complex<rtype> ctype;
  typedef std::pair<rtype*, rtype*> ztype;

public:
  fftm(Domain<2> const &dom, rtype scale)
    : Driver<1, std::complex<T> >(CUFFT_C2C, dom[A], dom[1 - A].size(), scale),
      mult_(dom[1 - A].size())
  {}

  virtual char const* name() { return "fftm-cuda-complex"; }
  virtual bool supports_scale() { return true;}
  virtual bool supports_cuda_memory() { return true;}
  virtual void in_place(ctype *inout,
			stride_type r_stride, stride_type c_stride,
			length_type rows, length_type cols)
  {
    if (A == 0)
    {
      assert(cols == mult_);
      assert(r_stride == 1);
      assert((c_stride > 0) && (static_cast<length_type>(c_stride) >= rows));
    }
    else
    {   
      assert(rows == mult_);
      assert(c_stride == 1);
      assert((r_stride > 0) && (static_cast<length_type>(r_stride) >= cols));
    }

    if (E == -1) this->forward(inout, inout);
    else         this->inverse(inout, inout);
  }

  virtual void in_place(ztype, stride_type, stride_type,
			length_type, length_type)
  {}

  virtual void by_reference(ctype *in,
			    stride_type in_r_stride, stride_type in_c_stride,
			    ctype *out,
			    stride_type out_r_stride, stride_type out_c_stride,
			    length_type rows, length_type cols)
  {
    if (A == 0)
    {
      assert(cols == mult_);
      assert(in_r_stride == 1 && out_r_stride == 1);
      assert((in_c_stride > 0) && (static_cast<length_type>(in_c_stride) >= rows));
      assert((out_c_stride > 0) && (static_cast<length_type>(out_c_stride) >= rows));
    }
    else
    {   
      assert(rows == mult_);
      assert(in_c_stride == 1 && out_c_stride == 1);
      assert((in_r_stride > 0) && (static_cast<length_type>(in_r_stride) >= cols));
      assert((out_r_stride > 0) && (static_cast<length_type>(out_r_stride) >= cols));
    }

    if (E == -1) this->forward(in, out);
    else         this->inverse(in, out);
  }
  virtual void by_reference(ztype, stride_type, stride_type,
			    ztype, stride_type, stride_type,
			    length_type, length_type)
  {}

  length_type mult_;
};



// FFT create() functions

#define VSIPL_IMPL_PROVIDE(D, I, O, A, E)		\
template <>                                             \
std::auto_ptr<fft::backend<D, I, O, A, E> >		\
create(Domain<D> const &dom,                            \
       fft::backend<D, I, O, A, E>::scalar_type scale)  \
{                                                       \
  return std::auto_ptr<fft::backend<D, I, O, A, E> >	\
    (new impl<D, I, O, A, E>(dom, scale));              \
}

// Note: Double precision is not supported by CUDA 2.1
VSIPL_IMPL_PROVIDE(1, float, std::complex<float>, 0, -1)
VSIPL_IMPL_PROVIDE(1, std::complex<float>, float, 0, 1)
VSIPL_IMPL_PROVIDE(1, std::complex<float>, std::complex<float>, 0, -1)
VSIPL_IMPL_PROVIDE(1, std::complex<float>, std::complex<float>, 0, 1)

#undef VSIPL_IMPL_PROVIDE



// FFTM create() functions

#define VSIPL_IMPL_PROVIDE(I, O, A, E)                  \
template <>                                             \
std::auto_ptr<fft::fftm<I, O, A, E> >		        \
create(Domain<2> const &dom,                            \
  vsip::impl::Scalar_of<I>::type scale)                 \
{                                                       \
  return std::auto_ptr<fft::fftm<I, O, A, E> >          \
    (new fftm<I, O, A, E>(dom, scale));                 \
}

VSIPL_IMPL_PROVIDE(float, std::complex<float>, 0, -1)
VSIPL_IMPL_PROVIDE(float, std::complex<float>, 1, -1)
VSIPL_IMPL_PROVIDE(std::complex<float>, float, 0, 1)
VSIPL_IMPL_PROVIDE(std::complex<float>, float, 1, 1)
VSIPL_IMPL_PROVIDE(std::complex<float>, std::complex<float>, 0, -1)
VSIPL_IMPL_PROVIDE(std::complex<float>, std::complex<float>, 1, -1)
VSIPL_IMPL_PROVIDE(std::complex<float>, std::complex<float>, 0, 1)
VSIPL_IMPL_PROVIDE(std::complex<float>, std::complex<float>, 1, 1)

#undef VSIPL_IMPL_PROVIDE


} // namespace vsip::impl::cuda
} // namespace vsip::impl
} // namespace vsip
