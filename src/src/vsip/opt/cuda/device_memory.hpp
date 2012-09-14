/* Copyright (c) 2009 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    vsip/opt/cuda/device_memory.hpp
    @author  Don McCoy
    @date    2009-04-05
    @brief   VSIPL++ Library: Device (on GPU) memory class for interfacing
               with CUDA-allocated memory.
*/

#ifndef VSIP_OPT_CUDA_DEVICE_MEMORY_HPP
#define VSIP_OPT_CUDA_DEVICE_MEMORY_HPP

#if VSIP_IMPL_REF_IMPL
# error "vsip/opt files cannot be used as part of the reference impl."
#endif

/***********************************************************************
  Included Files
***********************************************************************/

#include <vsip/support.hpp>
#include <vsip/opt/cuda/bindings.hpp>
#include <vsip/opt/cuda/device_storage.hpp>
#include <vsip/opt/cuda/gpu_block.hpp>
#include <vsip/opt/rt_extdata.hpp>

#ifndef CUDA_DEBUG
#define CUDA_DEBUG  0
#endif

#if CUDA_DEBUG
# include <iostream>
# define DB_SHOW(str, sync)             \
  std::cout << "Dev_mem: "              \
  << (str) << ", "                      \
  << (sync & SYNC_IN ? "IN " : "")      \
  << (sync & SYNC_OUT ? "OUT " : "")    \
  << std::endl;
#else
# define DB_SHOW(str, sync)
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

/// Helper class used to allocate/deallocate memory and copy data
/// to and from GPU memory space.
template <typename Block>
class Device_memory : public Non_copyable
{
  typedef typename Block::value_type  T;

  // this is the desired layout policy (not the Block's)
  typedef Rt_layout<Block_layout<Block>::dim> layout_type;

public:
  static dimension_type const Dim = Block_layout<Block>::dim;

  /// Allocates device (GPU) memory to be used for either input or output 
  /// to GPU kernel functions.
  ///
  ///   :src_block:  Block encapsulating storage in host memory.
  ///
  ///   :sync:  A copy from host to device occurs if SYNC_IN or SYNC_INOUT 
  ///           is used.  A copy from device to host memory occurs upon
  ///           destruction if SYNC_OUT or SYNC_INOUT is used.
  /// 
  ///   :layout:  Pointer to the runtime layout describing how the data
  ///             should be arranged prior to transfer to or from GPU
  ///             memory.  The default will utilize the current block's
  ///             layout.
  ///
  Device_memory(Block& src_block, sync_action_type sync = SYNC_INOUT, 
    layout_type* layout = NULL)
    : sync_(sync), 
      storage_(src_block.size()),
      desired_rtl_(layout != NULL ? *layout : block_layout<Dim>(src_block)),
      ext_src_(src_block, desired_rtl_, sync_),
      host_(ext_src_.data().as_inter())
  {
    DB_SHOW("Block", sync_);
    if (sync_ & SYNC_IN)
      copy_host_to_dev(ext_src_, storage_.data(), desired_rtl_.order);
  }

  /// Host/GPU memory helper class destructor.  Frees all resources after
  /// copying device data back to the host (if SYNC_IN is not specified
  /// in the constructor).
  ~Device_memory()
  {
    DB_SHOW("~    ", sync_);
    // Data must be copied back because the storage on the GPU
    // is about to be freed.  The exception is if SYNC_IN is used,
    // meaning the block was input only and therefore does not
    // need to be copied back.
    if (sync_ != SYNC_IN)
      copy_dev_to_host(ext_src_, storage_.data(), desired_rtl_.order);
  }

  /// Returns the buffer address (in GPU memory space).
  T* data() { return storage_.data(); }

  /// Returns the stride of the data in the requested layout for
  /// the given dimension.
  stride_type stride(dimension_type d) const
  {
    assert(d < Dim);
    return ext_src_.stride(d);
  }

  /// Returns the size of the data in the requested layout for
  /// the given dimension.
  length_type size(dimension_type d) const
  {
    assert(d < Dim);
    return ext_src_.size(d);
  }
    
private:
  sync_action_type sync_;
  Device_storage<T> storage_;
  layout_type desired_rtl_;
  Rt_ext_data<Block> ext_src_;
  T* host_;
};


/// Specialization for const blocks
template <typename Block>
class Device_memory<Block const> : public Non_copyable
{
  typedef typename Block::value_type  T;

  // this is the desired layout policy (not the Block's)
  typedef Rt_layout<Block_layout<Block>::dim> layout_type;

public:
  static dimension_type const Dim = Block_layout<Block>::dim;

  /// Allocates device (GPU) memory to be used for input to GPU kernel 
  /// functions.
  ///
  ///   :src_block:  Block encapsulating storage in host memory.
  ///
  ///   :sync:  Must be set to SYNC_IN for constant block types.  A copy 
  ///           from host to device always occurs.
  /// 
  ///   :layout:  Pointer to the runtime layout describing how the data
  ///             should be arranged prior to transfer to or from GPU
  ///             memory.  The default will utilize the current block's
  ///             layout.
  ///
  Device_memory(Block const& src_block, sync_action_type sync = SYNC_IN,
    layout_type* layout = NULL)
    : sync_(SYNC_IN),
      storage_(src_block.size()),
      desired_rtl_(layout != NULL ? *layout : block_layout<Dim>(src_block)),
      ext_src_(src_block, desired_rtl_, sync_),
      host_(ext_src_.data().as_inter())
  {
    DB_SHOW("Block const", sync);
    // Provided for compatibility.  Must be set to the default.
    assert(sync == SYNC_IN);

    copy_host_to_dev(ext_src_, storage_.data(), desired_rtl_.order);
  }
  
  /// Constant block class destructor.  No data is copied.
  ~Device_memory()
  {
    DB_SHOW("~ const    ", SYNC_IN);  
  }

  /// Returns the buffer address (in GPU memory space).
  T* data() { return storage_.data(); }
    
  /// Returns the stride of the data in the requested layout for
  /// the given dimension.
  stride_type stride(dimension_type d) const
  {
    assert(d < Dim);
    return ext_src_.stride(d);
  }

  /// Returns the size of the data in the requested layout for
  /// the given dimension.
  length_type size(dimension_type d) const
  {
    assert(d < Dim);
    return ext_src_.size(d);
  }

private:
  sync_action_type sync_;
  Device_storage<T> storage_;
  layout_type desired_rtl_;
  Rt_ext_data<typename Proper_type_of<Block>::type> ext_src_;
  T const* host_;
};



/// Specialization for Gpu_block 
template <dimension_type Dim,
          typename       T,
          typename       Order>
class Device_memory<Gpu_block<Dim, T, Order> > : public Non_copyable
{
  typedef Gpu_block<Dim, T, Order> block_type;

  // this is the desired layout policy (not the Block's)
  typedef Rt_layout<Dim> layout_type;

public:

  /// Specialization utilizing a Gpu_block for input or output to GPU kernel
  /// functions.
  ///
  ///   :src_block:  Block encapsulating storage in both host memory and
  ///                device memory.
  ///
  ///   :sync:  The value SYNC_IN or SYNC_INOUT will cause the input block 
  ///           to be copied from host to device memory, if the data is
  ///           not already valid on the GPU.
  /// 
  ///   :layout:  This parameter is ignored for Gpu_blocks.
  ///
  Device_memory(block_type& src_block, sync_action_type sync = SYNC_INOUT,
    layout_type* layout __attribute__((unused)) = NULL)
    : block_(src_block), 
      sync_(sync)
  {
    DB_SHOW("Gpu_block", sync);

    // GPU-based blocks are restricted to dense, either row or column-major 
    // layout (i.e. tuple<0, 1, 2> or tuple<1, 0, 2>).
    assert((layout != NULL ? layout->pack == stride_unit_dense : 1));
    assert((layout != NULL ? layout->order.impl_dim2 == Rt_tuple().impl_dim2 : 1));
    assert((layout != NULL ? layout->complex == cmplx_inter_fmt : 1));

    // Obtain pointer to device, but only after requesting that the
    // block make the device pointer reference valid data (meaning, copy 
    // data from host to device memory i.f.f necessary).
    block_.device_request(sync);
  }

  /// Host/GPU memory helper class destructor.
  ~Device_memory()
  {
    DB_SHOW("~        ", sync_);

    // Inform the underlying block that the pointer to device memory is
    // no longer needed.
    block_.device_release();
  }

  /// Returns the buffer address (in GPU memory space).  Data is first copied 
  /// to the GPU if the host copy has been updated since the block was created.
  T* data() { return block_.device_data(); }
    
  /// Returns the stride of the data in the requested layout for
  /// the given dimension.
  stride_type stride(dimension_type d) const
  {
    assert(d < Dim);
    return block_.impl_stride(Dim, d);
  }

  /// Returns the size of the data in the requested layout for
  /// the given dimension.
  length_type size(dimension_type d) const
  {
    assert(d < Dim);
    return block_.size(Dim, d);
  }

private:
  block_type& block_;
  sync_action_type sync_;
};


/// Specialization for Gpu_block const
template <dimension_type Dim,
          typename       T,
          typename       Order>
class Device_memory<Gpu_block<Dim, T, Order> const> : public Non_copyable
{
  typedef Gpu_block<Dim, T, Order> const block_type;

  // this is the desired layout policy (not the Block's)
  typedef Rt_layout<Dim> layout_type;

public:
  /// Specialization utilizing a constant Gpu_block for input to GPU kernel
  /// functions.
  ///
  ///   :src_block:  Block encapsulating storage in both host memory and
  ///                device memory.
  ///
  ///   :sync:  Must be set to SYNC_IN for constant blocks.
  /// 
  ///   :layout:  This parameter is ignored for Gpu_blocks.
  ///
  Device_memory(block_type& src_block, sync_action_type sync = SYNC_IN,
    layout_type* layout __attribute__((unused)) = NULL)
    : block_(src_block)
  {
    DB_SHOW("Gpu_block const", sync);
    // The parameter is provided for compatibility only.  Must be set
    // to the default.
    assert(sync == SYNC_IN);

    // GPU-based blocks are restricted to dense, either row or column-major 
    // layout (i.e. tuple<0, 1, 2> or tuple<1, 0, 2>).
    assert((layout != NULL ? layout->pack == stride_unit_dense : 1));
    assert((layout != NULL ? layout->order.impl_dim2 == Rt_tuple().impl_dim2 : 1));
    assert((layout != NULL ? layout->complex == cmplx_inter_fmt : 1));

    // Obtain pointer to device, but only after requesting that the
    // block make the device pointer reference valid data (meaning, copy 
    // data from host to device memory i.f.f necessary).
    block_.device_request(SYNC_IN);
  }
  
  /// Host/GPU memory helper class destructor.  No data is copied back.
  ~Device_memory() 
  {
    DB_SHOW("~ const        ", SYNC_IN);

    // Inform the underlying block that the pointer to device memory is
    // no longer needed.
    block_.device_release();
  }

  /// Returns the buffer address (in GPU memory space).
  T const* data() { return block_.device_data(); }

  /// Returns the stride of the data in the requested layout for
  /// the given dimension.
  stride_type stride(dimension_type d) const
  {
    assert(d < Dim);
    return block_.impl_stride(Dim, d);
  }

  /// Returns the size of the data in the requested layout for
  /// the given dimension.
  length_type size(dimension_type d) const
  {
    assert(d < Dim);
    return block_.size(Dim, d);
  }

private:
  block_type& block_;
};


} // namespace vsip::impl::cuda
} // namespace vsip::impl
} // namespace vsip

#endif // VSIP_OPT_CUDA_DEVICE_MEMORY_HPP
