/* Copyright (c) 2009 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    vsip/opt/cuda/kernels.hpp
    @author  Don McCoy
    @date    2009-03-11
    @brief   VSIPL++ Library: Custom CUDA kernels
*/

#ifndef VSIP_OPT_CUDA_KERNELS_HPP
#define VSIP_OPT_CUDA_KERNELS_HPP

#if VSIP_IMPL_REF_IMPL
# error "vsip/opt files cannot be used as part of the reference impl."
#endif

/***********************************************************************
  Included Files
***********************************************************************/

// These definitions are made available for internal headers so that
// user applications do not need to have CUDA headers installed. 
// Include cublas.h, cufft.h and other CUDA header only in .cpp files, 
// and before other internal headers.

#if !defined(CU_COMPLEX_H_)
typedef struct { float x, y; } float2;
typedef float2 cuComplex;
#endif

#if !defined(_CUFFT_H_)
typedef cuComplex cufftComplex;
#endif


/***********************************************************************
  Declarations
***********************************************************************/

namespace vsip
{

namespace impl
{

namespace cuda
{

// Host-side functions
//
// These are simple wrappers around the actual CUDA kernels.  Note that 
// input data resides in device memory allocated with cudaMalloc() and 
// filled from the host using cudaMemcpy().  Upon return, data may be 
// copied back to host or operated on by other kernels.

/// Matrix transpose, scalar, S --> S
///   Note: the output is of size 'rows' x 'cols'
void
transpose_ss(
  float const* input,
  float*       output,
  size_t       rows,
  size_t       cols);

/// Matrix transpose, complex, C --> C
///   Note: the output is of size 'rows' x 'cols'
void
transpose_cc(
  cuComplex const* input,
  cuComplex*       output,
  size_t           rows,
  size_t           cols);

/// Matrix copy, scalar, S --> S
void
copy_ss(
  float const* input,
  float*       output,
  size_t       rows,
  size_t       cols);

/// Matrix copy, scalar, C --> C
void
copy_cc(
  cuComplex const* input,
  cuComplex*       output,
  size_t           rows,
  size_t           cols);

/// Swaps quadrants of a matrix, scalar, S --> S
void
fftshift_s(
  float const* input,
  float*       output,
  size_t       rows,
  size_t       cols,
  unsigned int in_major_dim, 
  unsigned int out_major_dim);

/// Swaps quadrants of a matrix, complex, C --> C
void
fftshift_c(
  cuComplex const* input,
  cuComplex*       output,
  size_t           rows,
  size_t           cols,
  unsigned int     in_major_dim, 
  unsigned int     out_major_dim);


/// Scalar-Matrix elementwise multiply, s * S --> S
void
smmul_ss(
  float const  scale,
  float const* input,
  float*       output,
  size_t       rows,
  size_t       cols);

/// Scalar-Matrix elementwise multiply, s * C --> C
void
smmul_sc(
  float const      scale,
  cuComplex const* input,
  cuComplex*       output,
  size_t           rows,
  size_t           cols);


/// Vector-Matrix multiply, row-wise, S * S --> S
void 
vmmul_row_ss(
  float const* kernel,  
  float const* input,
  float*       output,
  size_t       rows,
  size_t       cols);

/// Vector-Matrix multiply, row-wise, S * C --> C
void 
vmmul_row_sc(
  float const*     kernel,
  cuComplex const* input,
  cuComplex*       output,
  size_t           rows,
  size_t           cols);

/// Vector-Matrix multiply, row-wise, C * S --> C
void 
vmmul_row_cs(
  cuComplex const* kernel,  
  float const*     input,
  cuComplex*       output,
  size_t           rows,
  size_t           cols);

/// Vector-Matrix multiply, row-wise, C * C --> C
void 
vmmul_row_cc(
  cuComplex const* kernel,  
  cuComplex const* input,
  cuComplex*       output,
  size_t           rows,
  size_t           cols);


/// Vector-Matrix multiply, column-wise, S * S --> S
void 
vmmul_col_ss(
  float const* kernel,  
  float const* input,
  float*       output,
  size_t       rows,
  size_t       cols);

/// Vector-Matrix multiply, column-wise, S * C --> C
void 
vmmul_col_sc(
  float const*     kernel,
  cuComplex const* input,
  cuComplex*       output,
  size_t           rows,
  size_t           cols);

/// Vector-Matrix multiply, column-wise, C * S --> C
void 
vmmul_col_cs(
  cuComplex const* kernel,  
  float const*     input,
  cuComplex*       output,
  size_t           rows,
  size_t           cols);

/// Vector-Matrix multiply, column-wise, C * C --> C
void 
vmmul_col_cc(
  cuComplex const* kernel,  
  cuComplex const* input,
  cuComplex*       output,
  size_t           rows,
  size_t           cols);


/// Vector-Matrix multiply, with scale, row-wise, C * C --> C
void 
vmmuls_row_cc(
  cuComplex const* kernel,  
  cuComplex const* input,
  cuComplex*       output,
  float            scale,
  size_t           rows,
  size_t           cols);

/// Vector-Matrix multiply, with scale, column-wise, C * C --> C
void 
vmmuls_col_cc(
  cuComplex const* kernel,  
  cuComplex const* input,
  cuComplex*       output,
  float            scale,
  size_t           rows,
  size_t           cols);


/// Matrix-Matrix multiply, with scale, row-wise, C * C --> C
void 
mmmuls_cc(
  cuComplex const* kernel,  
  cuComplex const* input,
  cuComplex*       output,
  float            scale,
  size_t           rows,
  size_t           cols);

/// 
void
mag_cs(
  cuComplex const* in,
  float*           out,
  size_t           rows,
  size_t           cols);

void
interpolate(
  unsigned int const* indices,  // n x m
  float const*        window,   // n x m x I
  cuComplex const*    in,       // n x m
  cuComplex*          out,      // nx x m
  size_t              depth,    // I
  size_t              wstride,  // I + pad bytes
  size_t              rows_in,  // n
  size_t              rows_out, // nx
  size_t              cols);    // m

void
interpolate_with_shift(
  unsigned int const* indices,  // n x m
  float const*        window,   // n x m x I
  cuComplex const*    in,       // n x m
  cuComplex*          out,      // nx x m
  size_t              depth,    // I
  size_t              wstride,  // I + pad bytes
  size_t              rows_in,  // n
  size_t              rows_out, // nx
  size_t              cols);    // m


/// Memory copy, from device (global) to shared (fast, on-core) memory
void
copy_device_to_shared(
  float* src, 
  size_t size);

/// Memory copy, from device to device memory
void
copy_device_to_device(
  float const* src, 
  float* dest, 
  size_t size);

/// Memory fill device with zeroes
void
copy_zeroes_to_device(
  float* dest, 
  size_t size);


/// Null kernels --- for testing only ---
void
null_s(
  float* inout,
  size_t rows,
  size_t cols);

/// Null kernels --- for testing only ---
void
null_c(
  cuComplex* inout,
  size_t     rows,
  size_t     cols);

/// The check distribution kernel is for testing only.  Each thread
/// writes the linear offset of each element to that element.
void
check_distrib(
  size_t* inout,
  size_t size);

/// The check distribution kernel is for testing only.  Each thread
/// writes the linear offset of each element to that element.
void
check_distrib(
  size_t* inout,
  size_t size);

/// The check distribution kernel is for testing only.  Each thread
/// computes a unique row/col offset and converts it into a linear
/// index that is then written into the buffer
void
check_distrib(
  size_t* inout,
  size_t rows,
  size_t cols);


} // namespace vsip::impl::cuda
} // namespace vsip::impl
} // namespace vsip

#endif // VSIP_OPT_CUDA_KERNELS_HPP
