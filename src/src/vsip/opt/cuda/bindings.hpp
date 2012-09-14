/* Copyright (c) 2009 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    vsip/opt/cuda/bindings.hpp
    @author  Don McCoy
    @date    2009-02-05
    @brief   VSIPL++ Library: Bindings for CUDA's BLAS functions and
               for custom CUDA kernels.
*/

#ifndef VSIP_OPT_CUDA_BINDINGS_HPP
#define VSIP_OPT_CUDA_BINDINGS_HPP

#if VSIP_IMPL_REF_IMPL
# error "vsip/opt files cannot be used as part of the reference impl."
#endif

/***********************************************************************
  Included Files
***********************************************************************/

#include <complex>

#include <vsip/opt/expr/assign_fwd.hpp>
#include <vsip/support.hpp>
#include <vsip/opt/cuda/kernels.hpp>

#ifndef NDEBUG
#include <iostream>
#define ASSERT_CUDA_OK()					\
{								\
  cudaError_t error = cudaGetLastError();			\
  if (error != cudaSuccess)					\
  {								\
    std::cerr << "CUDA problem encountered (error "		\
	      << error << ")" << std::endl;			\
    std::cerr << cudaGetErrorString(error) << std::endl;	\
  }								\
  assert(error == cudaSuccess);					\
}

#define ASSERT_CUBLAS_OK()					\
{								\
  cublasStatus status = cublasGetError();			\
  if (status != 0)						\
    std::cerr << "CUBLAS problem encountered (error "		\
	      << status << ")" << std::endl;			\
  assert(status == 0);						\
}

#define ASSERT_CUFFT_OK(result)					\
{								\
  if (result != 0)						\
    std::cerr << "CUFFT problem encountered (error "		\
	      << result << ")" << std::endl;			\
  assert(result == 0);						\
}

#else
#define ASSERT_CUDA_OK()
#define ASSERT_CUBLAS_OK()
#define ASSERT_CUFFT_OK(r)

#endif // CUDA_DEBUG


/***********************************************************************
  Declarations
***********************************************************************/

extern "C"
{
#if !defined(CUBLAS_H_)
// Prototypes for CUBLAS functions called directly (see cublas.h)
// 
typedef unsigned int cublasStatus;

float cublasSdot(int n, 
		 const float *x, int incx, 
		 const float *y, int incy);
cuComplex cublasCdotu(int n, 
                      const cuComplex *x, int incx, 
                      const cuComplex *y, int incy);
cuComplex cublasCdotc(int n, 
                      const cuComplex *x, int incx, 
                      const cuComplex *y, int incy);
cublasStatus cublasGetError();
#endif // !defined(CUBLAS_H_)

#if !defined(__CUDA_RUNTIME_API_H__)
// From cuda_runtime_api.h
//
enum cudaError
{ 
  cudaSuccess = 0
};
typedef cudaError cudaError_t;

enum cudaMemcpyKind
{
  cudaMemcpyHostToHost = 0,
  cudaMemcpyHostToDevice,
  cudaMemcpyDeviceToHost,
  cudaMemcpyDeviceToDevice
};

cudaError_t cudaMemcpy  (void *dst, const void *src, size_t count, enum cudaMemcpyKind kind);
cudaError_t cudaMemcpy2D(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind);
cudaError_t cudaMalloc(void **devPtr, size_t size);
cudaError_t cudaFree(void *devPtr);
cudaError_t cudaGetLastError(void);
const char* cudaGetErrorString(cudaError_t error);
cudaError_t cudaThreadSynchronize(void);

#endif // !defined(__CUDA_RUNTIME_API_H__)
}


namespace vsip
{

namespace impl
{

namespace cuda
{

// Functions to interface with CUDA
//
void initialize(int& argc, char**&argv);
void finalize();


//
// C++ --> C interface functions
//

// cuda::dot()
#define VSIP_IMPL_CUBLAS_DOT(T, CUDA_T, VPPFCN, CUBLASFCN)      \
inline T                                                        \
VPPFCN(int n,                                                   \
    const T* x, int incx,                                       \
    const T* y, int incy)                                       \
{                                                               \
  if (incx < 0) x += incx * (n-1);                              \
  if (incy < 0) y += incy * (n-1);                              \
  return CUBLASFCN(n, (const CUDA_T*)x, incx,			\
		      (const CUDA_T*)y, incy);			\
}

#define VSIP_IMPL_CUBLAS_DOTC(T, CUDA_T, VPPFCN, CUBLASFCN)     \
inline T                                                        \
VPPFCN(int n,                                                   \
    const T* x, int incx,                                       \
    const T* y, int incy)                                       \
{                                                               \
  if (incx < 0) x += incx * (n-1);                              \
  if (incy < 0) y += incy * (n-1);                              \
  CUDA_T r = CUBLASFCN(n, (const CUDA_T*)x, incx,		\
		          (const CUDA_T*)y, incy);		\
  return T(r.x, r.y);                                           \
}

// Note: CUDA functions return the C99 Complex type.  The way the
// return value is handled when converting back to the C++ type relies
// on a GNU extension and may not work with all compilers.
VSIP_IMPL_CUBLAS_DOT (float,               float,     dot,  cublasSdot)
VSIP_IMPL_CUBLAS_DOTC(std::complex<float>, cuComplex, dot,  cublasCdotu)
VSIP_IMPL_CUBLAS_DOTC(std::complex<float>, cuComplex, dotc, cublasCdotc)
#undef VSIP_IMPL_CUBLAS_DOT
#undef VSIP_IMPL_CUBLAS_DOTC


// Wrapper functions used for vmmul serial expression evaluator.
// These functions convert parameters to the proper (standard C) types
// and call the appropriate (non-overloaded) kernel entry point.

inline
void 
transpose(
  float const* input,
  float*       output,
  size_t       rows,
  size_t       cols)
{
  transpose_ss(input, output, rows, cols);
}

inline
void
transpose(
  std::complex<float> const* input,
  std::complex<float>*       output,
  size_t                     rows,
  size_t                     cols)
{
  transpose_cc(
    reinterpret_cast<cuComplex const*>(input),
    reinterpret_cast<cuComplex*>(output),
    rows, cols);
}

inline
void 
copy(
  float const* input,
  float*       output,
  size_t       rows,
  size_t       cols)
{
  copy_ss(input, output, rows, cols);
}

inline
void
copy(
  std::complex<float> const* input,
  std::complex<float>*       output,
  size_t                     rows,
  size_t                     cols)
{
  copy_cc(
    reinterpret_cast<cuComplex const*>(input),
    reinterpret_cast<cuComplex*>(output),
    rows, cols);
}

inline
void 
fftshift(
  float const*   input,
  float*         output,
  size_t         rows,
  size_t         cols,
  dimension_type in_major_dim, 
  dimension_type out_major_dim)
{
  fftshift_s(input, output, rows, cols, in_major_dim, out_major_dim);
}

inline
void
fftshift(
  std::complex<float> const* input,
  std::complex<float>*       output,
  size_t                     rows,
  size_t                     cols,
  dimension_type             in_major_dim, 
  dimension_type             out_major_dim)
{
  fftshift_c(
    reinterpret_cast<cuComplex const*>(input),
    reinterpret_cast<cuComplex*>(output),
    rows, cols, in_major_dim, out_major_dim);
}

inline
void 
vmmul_row(
  float const* kernel,  
  float const* input,
  float*       output,
  size_t       rows,
  size_t       cols)
{
  vmmul_row_ss(kernel, input, output, rows, cols);
}

inline
void 
vmmul_row(
  float const*               kernel,  
  std::complex<float> const* input,
  std::complex<float>*       output,
  size_t                     rows,
  size_t                     cols)
{
  vmmul_row_sc(
    kernel,
    reinterpret_cast<cuComplex const*>(input),
    reinterpret_cast<cuComplex*>(output),
    rows, cols);
}

inline
void 
vmmul_row(
  std::complex<float> const* kernel,  
  float const*               input,
  std::complex<float>*       output,
  size_t                     rows,
  size_t                     cols)
{
  vmmul_row_cs(
    reinterpret_cast<cuComplex const*>(kernel),
    input,
    reinterpret_cast<cuComplex*>(output),
    rows, cols);
}

inline
void 
vmmul_row(
  std::complex<float> const* kernel,  
  std::complex<float> const* input,
  std::complex<float>*       output,
  size_t                     rows,
  size_t                     cols)
{
  vmmul_row_cc(
    reinterpret_cast<cuComplex const*>(kernel),
    reinterpret_cast<cuComplex const*>(input),
    reinterpret_cast<cuComplex*>(output),
    rows, cols);
}


inline
void 
vmmul_col(
  float const* kernel,  
  float const* input,
  float*       output,
  size_t       rows,
  size_t       cols)
{
  vmmul_col_ss(kernel, input, output, rows, cols);
}

inline
void 
vmmul_col(
  float const*               kernel,  
  std::complex<float> const* input,
  std::complex<float>*       output,
  size_t                     rows,
  size_t                     cols)
{
  vmmul_col_sc(
    kernel,
    reinterpret_cast<cuComplex const*>(input),
    reinterpret_cast<cuComplex*>(output),
    rows, cols);
}

inline
void 
vmmul_col(
  std::complex<float> const* kernel,  
  float const*               input,
  std::complex<float>*       output,
  size_t                     rows,
  size_t                     cols)
{
  vmmul_col_cs(
    reinterpret_cast<cuComplex const*>(kernel),
    input,
    reinterpret_cast<cuComplex*>(output),
    rows, cols);
}

inline
void 
vmmul_col(
  std::complex<float> const* kernel,  
  std::complex<float> const* input,
  std::complex<float>*       output,
  size_t                     rows,
  size_t                     cols)
{
  vmmul_col_cc(
    reinterpret_cast<cuComplex const*>(kernel),
    reinterpret_cast<cuComplex const*>(input),
    reinterpret_cast<cuComplex*>(output),
    rows, cols);
}


inline
void 
mmmuls(
  std::complex<float> const* kernel,  
  std::complex<float> const* input,
  std::complex<float>*       output,
  float                      scale,
  size_t                     rows,
  size_t                     cols)
{
  mmmuls_cc(
    reinterpret_cast<cuComplex const*>(kernel),
    reinterpret_cast<cuComplex const*>(input),
    reinterpret_cast<cuComplex*>(output),
    scale,
    rows, cols);
}


inline
void
cmag(
  std::complex<float> const* input,
  float*                     output,
  size_t                     rows,
  size_t                     cols)
{
  mag_cs(
    reinterpret_cast<cuComplex const*>(input),
    output,
    rows, cols);
}



/***********************************************************************
  kernels used for testing only
***********************************************************************/

inline
void
copy_device_memory(
  float const* src, 
  float* dest, 
  size_t size)
{
  copy_device_to_device(src, dest, size);
}

inline
void
copy_device_memory(
  std::complex<float> const* src, 
  std::complex<float>* dest, 
  size_t size)
{
  copy_device_to_device(
    reinterpret_cast<float const*>(src),
    reinterpret_cast<float*>(dest),
    size * 2);
}


inline
void
zero_device_memory(
  float* dest, 
  size_t size)
{
  copy_zeroes_to_device(dest, size);
}

inline
void
zero_device_memory(
  std::complex<float>* dest, 
  size_t size)
{
  copy_zeroes_to_device(
    reinterpret_cast<float*>(dest),
    size * 2);
}


inline
void
null_kernel(
  float* inout,
  size_t rows,
  size_t cols)
{
  null_s(
    reinterpret_cast<float*>(inout),
    rows, cols);
}

inline
void
null_kernel(
  std::complex<float>* inout, 
  size_t rows,
  size_t cols)
{
  null_c(
    reinterpret_cast<cuComplex*>(inout),
    rows, cols);
}





/// Copy-block class used when direct data access is possible
/// and the dimension ordering is not determined until runtime.
///
///  :Dim:  The block dimension.
///
///  :ExtT:  A class such as Rt_ext_data with size() and stride() members
///          and a data() member that returns an Rt_pointer<> object.
template <dimension_type Dim,
          typename ExtT>
struct copy_rt_ext_data;

template <typename ExtT>
struct copy_rt_ext_data<1, ExtT>
{
  typedef typename ExtT::value_type T;
  static void host_to_dev(ExtT const& src_ext, T* dest, Rt_tuple const& order 
    __attribute__((unused)))
  {
    assert(order.impl_dim0 == 0);
    cudaMemcpy(dest, src_ext.data().as_inter(), src_ext.size(0) * sizeof(T), 
      cudaMemcpyHostToDevice);
    ASSERT_CUDA_OK();
  }
  static void dev_to_host(ExtT& dest_ext, T const* src, Rt_tuple const& order 
    __attribute__((unused)))
  {
    assert(order.impl_dim0 == 0);
    cudaMemcpy(dest_ext.data().as_inter(), src, dest_ext.size(0) * sizeof(T), 
      cudaMemcpyDeviceToHost);
    ASSERT_CUDA_OK();
  }
};

template <typename ExtT>
struct copy_rt_ext_data<2, ExtT>
{
  typedef typename ExtT::value_type T;
  static void host_to_dev(ExtT const& src_ext, T* dest, Rt_tuple const& order)
  {
    dimension_type const dim0 = order.impl_dim0;
    dimension_type const dim1 = order.impl_dim1; 
    assert((dim0 == vsip::dim0 && dim1 == vsip::dim1) ||
           (dim0 == vsip::dim1 && dim1 == vsip::dim0));
    // Note: data on the GPU is stored in dense format
    if (dim0 == vsip::dim0)  // row major
      cudaMemcpy2D( 
        dest, src_ext.size(1) * sizeof(T),
        src_ext.data().as_inter(), src_ext.stride(0) * sizeof(T),
        src_ext.size(1) * sizeof(T), src_ext.size(0),
        cudaMemcpyHostToDevice);
    else                     // column major
      cudaMemcpy2D( 
        dest, src_ext.size(0) * sizeof(T),
        src_ext.data().as_inter(), src_ext.stride(1) * sizeof(T),
        src_ext.size(0) * sizeof(T), src_ext.size(1),
        cudaMemcpyHostToDevice);
    ASSERT_CUDA_OK();
  }
  static void dev_to_host(ExtT& dest_ext, T const* src, Rt_tuple const& order) 
  {
    dimension_type const dim0 = order.impl_dim0;
    dimension_type const dim1 = order.impl_dim1; 
    assert((dim0 == vsip::dim0 && dim1 == vsip::dim1) ||
           (dim0 == vsip::dim1 && dim1 == vsip::dim0));
    // Note: data on the GPU is stored in dense format
    if (dim0 == vsip::dim0)  // row major
      cudaMemcpy2D( 
        dest_ext.data().as_inter(), dest_ext.stride(0) * sizeof(T),
        src, dest_ext.size(1) * sizeof(T),
        dest_ext.size(1) * sizeof(T), dest_ext.size(0),
        cudaMemcpyDeviceToHost);
    else                     // column major
      cudaMemcpy2D( 
        dest_ext.data().as_inter(), dest_ext.stride(1) * sizeof(T),
        src, dest_ext.size(0) * sizeof(T),
        dest_ext.size(0) * sizeof(T), dest_ext.size(1),
        cudaMemcpyDeviceToHost);
    ASSERT_CUDA_OK();
  }
};

template <typename ExtT>
struct copy_rt_ext_data<3, ExtT>
{
  typedef typename ExtT::value_type T;
  static void host_to_dev(ExtT const& src_ext, T* dest, Rt_tuple const& order 
    __attribute__((unused)))
  {
    // At the present time, strided copies with three dimensional blocks
    // are not supported.
    assert(src_ext.stride(order.impl_dim0) == 
      static_cast<stride_type>(src_ext.size(order.impl_dim1)) *
      static_cast<stride_type>(src_ext.size(order.impl_dim2)));
    assert(src_ext.stride(order.impl_dim1) == 
      static_cast<stride_type>(src_ext.size(order.impl_dim2)));
    assert(src_ext.stride(order.impl_dim2) == 1);

    size_t size = src_ext.size(0) * src_ext.size(1) * src_ext.size(2) * sizeof(T);
    cudaMemcpy(dest, src_ext.data().as_inter(), size, cudaMemcpyHostToDevice);
    ASSERT_CUDA_OK();
  }
  static void dev_to_host(ExtT& dest_ext, T const* src, Rt_tuple const& order 
    __attribute__((unused)))
  {
    assert(dest_ext.stride(order.impl_dim0) == 
      static_cast<stride_type>(dest_ext.size(order.impl_dim1)) *
      static_cast<stride_type>(dest_ext.size(order.impl_dim2)));
    assert(dest_ext.stride(order.impl_dim1) == 
      static_cast<stride_type>(dest_ext.size(order.impl_dim2)));
    assert(dest_ext.stride(order.impl_dim2) == 1);

    size_t size = dest_ext.size(0) * dest_ext.size(1) * dest_ext.size(2) * sizeof(T);
    cudaMemcpy(dest_ext.data().as_inter(), src, size, cudaMemcpyDeviceToHost);
    ASSERT_CUDA_OK();
  }
};

/// External interface for copy_block<> template member function
/// that copies from CPU host memory to GPU device memory.
template <typename ExtT>
inline void
copy_host_to_dev(ExtT const& src_ext,
                 typename ExtT::value_type* dest,
                 Rt_tuple const& order = Rt_tuple())
{
  copy_rt_ext_data<ExtT::ext_type::dim, ExtT>::host_to_dev(src_ext, dest, order);
}

/// External interface for copy_block<> template member function
/// that copies from GPU device memory to CPU host memory.
template <typename ExtT>
inline void
copy_dev_to_host(ExtT& dest_ext,
                 typename ExtT::value_type const* src,
                 Rt_tuple const& order = Rt_tuple())
{
  copy_rt_ext_data<ExtT::ext_type::dim, ExtT>::dev_to_host(dest_ext, src, order);
}


 
/// CUDA capabilities known at compile time are expressed as traits.
///
/// Note that support for double precision is included in CUDA, but
/// not present on all hardware.  The correct way to fill out these
/// traits would be with an inquiry run at configure time, or an
/// option for cross compiling that forces it to assume one way or
/// the other.
///
template <typename T>
struct Cuda_traits
{
  static bool const valid = false;
};

template <>
struct Cuda_traits<float>
{
  static bool const valid = true;
  static char const trans = 't';
};

template <>
struct Cuda_traits<double>
{
  static bool const valid = false;
  static char const trans = 't';
};

template <>
struct Cuda_traits<std::complex<float> >
{
  static bool const valid = true;
  static char const trans = 'c';
};

template <>
struct Cuda_traits<std::complex<double> >
{
  static bool const valid = false;
  static char const trans = 'c';
};

} // namespace vsip::impl::cuda
} // namespace vsip::impl
} // namespace vsip

#endif // VSIP_OPT_CUDA_BINDINGS_HPP
