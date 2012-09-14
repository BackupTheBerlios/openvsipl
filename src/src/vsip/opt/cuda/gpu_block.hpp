/* Copyright (c) 2009 by CodeSourcery.  All rights reserved. 

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    vsip/opt/cuda/gpu_block.hpp
    @author  Don McCoy
    @date    2009-04-05
    @brief   VSIPL++ Library: GPU block class for use with CUDA
*/

#ifndef VSIP_OPT_GPU_BLOCK_HPP
#define VSIP_OPT_GPU_BLOCK_HPP

#if VSIP_IMPL_REF_IMPL
# error "vsip/opt files cannot be used as part of the reference impl."
#endif

/***********************************************************************
  Included Files
***********************************************************************/
#include <stdexcept>
#include <string>
#include <utility>

#include <vsip/support.hpp>
#include <vsip/domain.hpp>

#include <vsip/core/refcount.hpp>
#include <vsip/core/layout.hpp>
#include <vsip/core/extdata.hpp>
#include <vsip/core/block_traits.hpp>
#include <vsip/core/metaprogramming.hpp>
#include <vsip/core/noncopyable.hpp>

#include <vsip/opt/cuda/bindings.hpp>
#include <vsip/opt/cuda/device_storage.hpp>

#ifndef CUDA_DEBUG
#define CUDA_DEBUG  0
#endif

#if CUDA_DEBUG
# include <iostream>
# define GPU_DB_SHOW(str, sync)         \
  std::cout << "Gpu_block: "            \
  << (str) << ", "                      \
  << (sync & SYNC_IN ? "IN " : "")      \
  << (sync & SYNC_OUT ? "OUT " : "")    \
  << std::endl;
#else
# define GPU_DB_SHOW(str, sync)
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

enum memory_state_type
{
  cpu_and_device_memory_valid,
  device_memory_invalid,
  host_memory_invalid
};


/// Lightweight class used to copy data between host and a Gpu_block.  
///  
///   :Dim:   The dimension of the block.  Must be 1 or 2.
///
///   :Order: The dimension order.  Only row2_type and col2_type
///           are supported.
///
///   :T:     The data type of the block.
template <dimension_type Dim,
          typename       Order, 
          typename       T>
struct copy_block;


/// Implementation class for Gpu_block.
///
/// Gpu_block_impl implements a multi-dimiensional block that is
/// adapted by Gpu_block specializations for each dimension.
///
/// Requires:
///   Dim to be block dimension,
///   T to be a value type,
///   Order to be the dimension-ordering of the block
template <dimension_type Dim   = 1,
          typename       T     = VSIP_DEFAULT_VALUE_TYPE,
          typename       Order = row2_type,
          typename       Map   = Local_map>
class Gpu_block_impl
  : public vsip::impl::Ref_count<Gpu_block_impl<Dim, T, Order> >,
    public Non_copyable
{
  // Compile-time values and types.
public:
  static dimension_type const dim = Dim;

  typedef T        value_type;
  typedef T&       reference_type;
  typedef T const& const_reference_type;

  typedef Order                   order_type;
  typedef Map                     map_type;

  // Enable Direct_data access to data.
  template <typename, typename, typename>
  friend class impl::data_access::Low_level_data_access;

  // Direct access is also needed for copying GPU data
  template <dimension_type, typename, typename>
  friend struct copy_block;

  // Implementation types.
public:
  typedef Layout<Dim, Order, Stride_unit_dense, Cmplx_inter_fmt>
                                                layout_type;
  typedef impl::Applied_layout<layout_type>     applied_layout_type;
  typedef Allocated_storage<typename layout_type::complex_type, T> storage_type;
  typedef Device_storage<T>               device_storage_type;

  // Constructors and destructor.
public:
  Gpu_block_impl(Domain<Dim> const& dom, Map const& = Map());
  Gpu_block_impl(Domain<Dim> const& dom, T value, Map const& = Map());
  ~Gpu_block_impl()
    { storage_.deallocate(layout_.total_size()); }

  // Data accessors.
public:

  /// 1-dimensional accessors
  T get(index_type idx) const
  {
    assert(idx < size());
    return storage_.get(idx);
  }

  void put(index_type idx, T val)
  {
    assert(idx < size());
    storage_.put(idx, val);
    memory_state_ = device_memory_invalid;
  }

protected:
  /// Dim-dimensional accessors
  T get(Index<Dim> const& idx) const
  {
    for (dimension_type d=0; d<Dim; ++d)
      assert(idx[d] < layout_.size(d));
    return storage_.get(layout_.index(idx));
  }

  void put(Index<Dim> const& idx, T val)
  {
    for (dimension_type d=0; d<Dim; ++d)
      assert(idx[d] < layout_.size(d));
    storage_.put(layout_.index(idx), val);
    memory_state_ = device_memory_invalid;
  }


  // Accessors.
public:
  length_type size() const;
  length_type size(dimension_type D, dimension_type d) const;
  Map const& map() const { return map_;}

  // 
public:
  void device_request(sync_action_type sync) const
  {
    // Here, the caller can indicate that data needs to be copied
    // from the host to the device, but it will only do this if
    // the data has been altered on the CPU (as indicated by the
    // memory state flags).
    //
    // If the block is used for output only, no data is copied
    // from the host to the device.
    if ((sync & SYNC_IN) && device_memory_is_invalid())
      copy_host_to_dev();
    GPU_DB_SHOW(" dev_request", sync);
  }
  void device_release() const
  {
    // This function is provided for symmetry with the device_request()
    // method.  This may be used if later we decide to use a locking
    // mechanism to enforce a proper method of access.  Or, if it 
    // becomes necessary to take some action based on the calling 
    // function no longer needing access to the block (or even possibly
    // just for accounting purposes).
    GPU_DB_SHOW(" dev_release", 0);
  }

  // Support Direct_data interface.
public:
  typedef typename storage_type::type       data_type;
  typedef typename storage_type::const_type const_data_type;

  data_type       impl_host_data()       { return storage_.data(); }
  const_data_type impl_host_data() const { return storage_.data(); }

  stride_type impl_stride(dimension_type D, dimension_type d) const;

  // GPU memory interface
  T*       impl_device_data()       { return device_storage_.data(); }
  T const* impl_device_data() const { return device_storage_.data(); }

protected:
  bool device_memory_is_invalid() const { return memory_state_ == device_memory_invalid; }
  bool host_memory_is_invalid() const   { return memory_state_ == host_memory_invalid; }

  void copy_host_to_dev() const
  {
    GPU_DB_SHOW(" copy_h2d", 0);
    copy_block<Dim, order_type, T>::host_to_dev(*this);
    memory_state_ = cpu_and_device_memory_valid;
  }

  void copy_dev_to_host() const
  {
    GPU_DB_SHOW(" copy_d2h", 0);
    copy_block<Dim, order_type, T>::dev_to_host(*this);
    memory_state_ = cpu_and_device_memory_valid;
  }


  // Member Data
private:
  applied_layout_type layout_;
  mutable storage_type        storage_;
  mutable device_storage_type device_storage_;
  map_type            map_;

protected:
  mutable memory_state_type   memory_state_;
};



/// General template declaration for Gpu_block block class.
///
/// Requires:
///   Dim to be block dimension,
///   T to be a value type,
///   Order to be the dimension-ordering of the block
template <dimension_type Dim   = 1,
          typename       T     = VSIP_DEFAULT_VALUE_TYPE,
          typename       Order = row2_type,
          typename       Map   = Local_map>
class Gpu_block;


/// Gpu_block specialization for 1-dimension block.
template <typename       T,
          typename       Order,
          typename       Map>
class Gpu_block<1, T, Order, Map>
  : public Gpu_block_impl<1, T, Order, Map>
{
  typedef Gpu_block_impl<1, T, Order, Map> base_t;
  typedef Gpu_block<1, T, Order, Map> block_type;
  typedef typename block_type::order_type order_type;

  // Constructors.
public:
  Gpu_block(Domain<1> const& dom, Local_map const& map = Local_map())
    : base_t(dom, map)
  {
    // The base class does not initialize the host memory.
    // GPU memory is marked as invalid.
  }

  Gpu_block(Domain<1> const& dom, T value, Local_map const& map = Local_map())
    : base_t(dom, value, map)
  {
    // The base class initializes the host memory and marks
    // the GPU memory as invalid.
  }

  /// Updates host memory from GPU memory (if necessary).
  inline 
  void device_flush() 
  { 
    if (base_t::host_memory_is_invalid())
      base_t::copy_dev_to_host();
  }

  /// Returns a ponter to CPU memory, but only after updating it
  /// if necessary
  inline
  T* impl_data()
  { 
    device_flush();

    // Since a non-const pointer is returned, the CPU data must be
    // presumed altered, even if it is not.  Set the flag that 
    // indicates data must be copied back.
    base_t::memory_state_ = device_memory_invalid;
    return base_t::impl_host_data();
  }

  inline
  T const* impl_data() const
  { 
    if (base_t::host_memory_is_invalid())
      base_t::copy_dev_to_host();

    return base_t::impl_host_data();
  }

  /// Returns a pointer to GPU memory.  It is the caller's responsibility
  /// to ensure that data is valid on the device (or not) as appropriate.
  /// E.g. inputs should be copied, but for output blocks, having junk
  /// data in the device buffer is fine, because it will be overwritten.
  ///
  /// The function device_request(sync) should be called prior to 
  /// device_data() in order to satisfy the above criteria.
  T* device_data() 
  { 
    // Since a non-const pointer is returned, the GPU data must be
    // presumed altered, even if it is not.  Set the flag that 
    // indicates data must be copied back.
    base_t::memory_state_ = host_memory_invalid;
    return base_t::impl_device_data(); 
  }

  T const* device_data() const 
  { 
    return base_t::impl_device_data(); 
  }

  /// 1-dimensional data accessors
  T get(Index<1> idx) const
  {
    // Verify data is on the CPU is valid before returning values
    if (base_t::host_memory_is_invalid())
      base_t::copy_dev_to_host();

    return base_t::get(idx);
  }

  // put is inherited
};



/// Gpu_block specialization for 2-dimension block.
template <typename       T,
          typename       Order,
          typename       Map>
class Gpu_block<2, T, Order, Map>
  : public Gpu_block_impl<2, T, Order, Map>
{
  typedef Gpu_block_impl<2, T, Order, Map> base_t;
  typedef Gpu_block<2, T, Order, Map> block_type;
  typedef typename block_type::order_type order_type;

  // Constructors.
public:
  Gpu_block(Domain<2> const& dom, Local_map const& map = Local_map())
    : base_t(dom, map)
  {}

  Gpu_block(Domain<2> const& dom, T value, Local_map const& map = Local_map())
    : base_t(dom, value, map)
  {}

  /// Updates host memory from GPU memory (if necessary).
  void device_flush() 
  { 
    if (base_t::host_memory_is_invalid())
      base_t::copy_dev_to_host();
  }


  /// Returns a ponter to CPU memory, but only after updating it
  /// if necessary
  inline
  T* impl_data()
  { 
    device_flush();

    // Since a non-const pointer is returned, the CPU data must be
    // presumed altered, even if it is not.  Set the flag that 
    // indicates data must be copied back.
    base_t::memory_state_ = device_memory_invalid;
    return base_t::impl_host_data();
  }

  inline
  T const* impl_data() const
  { 
    if (base_t::host_memory_is_invalid())
      base_t::copy_dev_to_host();

    return base_t::impl_host_data();
  }

  /// Returns a pointer to GPU memory.  It is the caller's responsibility
  /// to ensure that data is valid on the device (or not) as appropriate.
  /// E.g. inputs should be copied, but for output blocks, having junk
  /// data in the device buffer is fine, because it will be overwritten.
  ///
  /// The function device_request(sync) should be called prior to 
  /// device_data() in order to satisfy the above criteria.
  T* device_data() 
  { 
    // Since a non-const pointer is returned, the GPU data must be
    // presumed altered, even if it is not.  Set the flag that 
    // indicates data must be copied back.
    base_t::memory_state_ = host_memory_invalid;
    return base_t::impl_device_data(); 
  }

  T const* device_data() const 
  { 
    return base_t::impl_device_data(); 
  }

  /// 2-dimensional data accessors.
  T get(Index<2> idx) const
  {
    // Verify data is on the CPU is valid before returning values
    if (base_t::host_memory_is_invalid())
      base_t::copy_dev_to_host();

    return base_t::get(idx);
  }

  T get(index_type idx0, index_type idx1) const
  { 
    // Verify data is on the CPU is valid before returning values
    if (base_t::host_memory_is_invalid())
      base_t::copy_dev_to_host();

    return base_t::get(Index<2>(idx0, idx1)); 
  }

  void put(index_type idx0, index_type idx1, T val)
  { 
    base_t::put(Index<2>(idx0, idx1), val); 
  }
};



/// Gpu_block specialization for 3-dimensional blocks.
template <typename       T,
          typename       Order,
          typename       Map>
class Gpu_block<3, T, Order, Map>
  : public Gpu_block_impl<3, T, Order, Map>
{
  typedef Gpu_block_impl<3, T, Order, Map> base_t;
  typedef Gpu_block<3, T, Order, Map> block_type;
  typedef typename block_type::order_type order_type;

  // Constructors.
public:
  Gpu_block(Domain<3> const& dom, Local_map const& map = Local_map())
    : base_t(dom, map)
  {}

  Gpu_block(Domain<3> const& dom, T value, Local_map const& map = Local_map())
    : base_t(dom, value, map)
  {}

  /// Updates host memory from GPU memory (if necessary).
  void device_flush() 
  { 
    if (base_t::host_memory_is_invalid())
      base_t::copy_dev_to_host();
  }


  /// Returns a ponter to CPU memory, but only after updating it
  /// if necessary
  inline
  T* impl_data()
  { 
    device_flush();

    // Since a non-const pointer is returned, the CPU data must be
    // presumed altered, even if it is not.  Set the flag that 
    // indicates data must be copied back.
    base_t::memory_state_ = device_memory_invalid;
    return base_t::impl_host_data();
  }

  inline
  T const* impl_data() const
  { 
    if (base_t::host_memory_is_invalid())
      base_t::copy_dev_to_host();

    return base_t::impl_host_data();
  }

  /// Returns a pointer to GPU memory.  It is the caller's responsibility
  /// to ensure that data is valid on the device (or not) as appropriate.
  /// E.g. inputs should be copied, but for output blocks, having junk
  /// data in the device buffer is fine, because it will be overwritten.
  ///
  /// The function device_request(sync) should be called prior to 
  /// device_data() in order to satisfy the above criteria.
  T* device_data() 
  { 
    // Since a non-const pointer is returned, the GPU data must be
    // presumed altered, even if it is not.  Set the flag that 
    // indicates data must be copied back.
    base_t::memory_state_ = host_memory_invalid;
    return base_t::impl_device_data(); 
  }

  T const* device_data() const 
  { 
    return base_t::impl_device_data(); 
  }

  /// 3-dimensional data accessors.
  T get(Index<3> idx) const
  {
    // Verify data is on the CPU is valid before returning values
    if (base_t::host_memory_is_invalid())
      base_t::copy_dev_to_host();

    return base_t::get(idx);
  }

  T get(index_type idx0, index_type idx1, index_type idx2) const
  { 
    // Verify data is on the CPU is valid before returning values
    if (base_t::host_memory_is_invalid())
      base_t::copy_dev_to_host();

    return base_t::get(Index<3>(idx0, idx1, idx2)); 
  }

  void put(index_type idx0, index_type idx1, index_type idx2, T val)
  { 
    base_t::put(Index<3>(idx0, idx1, idx2), val); 
  }
};



/***********************************************************************
  Definitions
***********************************************************************/

/// Construct a Gpu_block_impl.
template <dimension_type Dim,
          typename       T,
          typename       Order,
          typename       Map>
inline
Gpu_block_impl<Dim, T, Order, Map>::Gpu_block_impl(
  Domain<Dim> const& dom,
  Map const&         map)
  : layout_ (dom),
    storage_(layout_.total_size()),
    device_storage_(layout_.total_size()),
    map_    (map),
    memory_state_(device_memory_invalid)
{
  // These checks ensure that only supported template paramters are 
  // passed in from the base class constructor.  As support for these
  // are added, these checks may be removed.
  typedef Local_map                 map_type;
  VSIP_IMPL_STATIC_ASSERT((Type_equal<map_type, Local_map>::value));
}



/// Construct a Gpu_block_impl with initialized data.
template <dimension_type Dim,
          typename       T,
          typename       Order,
          typename       Map>
inline
Gpu_block_impl<Dim, T, Order, Map>::Gpu_block_impl(
  Domain<Dim> const& dom,
  T                  val,
  Map const&         map)
  : layout_ (dom),
    storage_(layout_.total_size(), val),
    device_storage_(layout_.total_size()),
    map_    (map),
    memory_state_(device_memory_invalid)
{
  // These checks ensure that only supported template paramters are 
  // passed in from the base class constructor.  As support for these
  // are added, these checks may be removed.
  typedef Local_map                 map_type;
  VSIP_IMPL_STATIC_ASSERT((Type_equal<map_type, Local_map>::value));
}



/// Return the total size of block.
template <dimension_type Dim,
          typename       T,
          typename       Order,
          typename       Map>
inline
length_type
Gpu_block_impl<Dim, T, Order, Map>::size() const
{
  length_type retval = layout_.size(0);
  for (dimension_type d=1; d<Dim; ++d)
    retval *= layout_.size(d);
  return retval;
}



/// Return the size of the block in a specific dimension.
///
/// Requires:
///   block_dim selects which block-dimensionality (block_dim <= 2).
///   d is the dimension of interest (0 <= d < block_dim).
/// Returns:
///   The size of dimension d.
template <dimension_type Dim,
          typename       T,
          typename       Order,
          typename       Map>
inline
length_type
Gpu_block_impl<Dim, T, Order, Map>::size(
  dimension_type block_dim,
  dimension_type d) const
{
  assert((block_dim == 1 || block_dim == Dim) && (d < block_dim));

  if (block_dim == 1)
    return size();
  else
    return layout_.size(d);
}



/// Return the stride of the block in a specific dimension.
/// Requires:
///   block_dim selects which block-dimensionality (block_dim == 1 or 2).
///   d is the dimension of interest (0 <= d < block_dim).
/// Returns:
///   The stride of dimension d.
template <dimension_type Dim,
          typename       T,
          typename       Order,
          typename       Map>
inline
stride_type
Gpu_block_impl<Dim, T, Order, Map>::impl_stride(
  dimension_type block_dim,
  dimension_type d) const
{
  assert(block_dim == 1 || block_dim == Dim);
  assert(d < Dim);

  if (block_dim == 1)
    return 1;
  else
    return layout_.stride(d);
}


/// Specialization for Dim == 1
template <typename Order, typename T>
struct copy_block<1, Order, T>
{
  static inline void host_to_dev(Gpu_block_impl<1, T, Order> const& block)
  {
    T const* src = block.storage_.data();
    T* dest = block.device_storage_.data();
    cudaMemcpy(dest, src, block.size() * sizeof(T), cudaMemcpyHostToDevice);
    ASSERT_CUDA_OK();
  }
  static inline void dev_to_host(Gpu_block_impl<1, T, Order> const& block)
  {
    T const* src = block.device_storage_.data();
    T* dest = block.storage_.data();
    cudaMemcpy(dest, src, block.size() * sizeof(T), cudaMemcpyDeviceToHost);
    ASSERT_CUDA_OK();
  }
};

/// Specialization for Dim == 2, row-major
template <typename T>
struct copy_block<2, row2_type, T>
{
  static void host_to_dev(Gpu_block_impl<2, T, row2_type> const& block)
  {
    T const* src = block.storage_.data();
    T* dest = block.device_storage_.data();
    cudaMemcpy2D( 
      dest, block.size(2, 1) * sizeof(T),
      src, block.impl_stride(2, 0) * sizeof(T),
      block.size(2, 1) * sizeof(T), block.size(2, 0),
      cudaMemcpyHostToDevice);
    ASSERT_CUDA_OK();
  }
  static void dev_to_host(Gpu_block_impl<2, T, row2_type> const& block)
  {
    T const* src = block.device_storage_.data();
    T* dest = block.storage_.data();
    cudaMemcpy2D( 
      dest, block.impl_stride(2, 0) * sizeof(T),
      src, block.size(2, 1) * sizeof(T),
      block.size(2, 1) * sizeof(T), block.size(2, 0),
      cudaMemcpyDeviceToHost);
    ASSERT_CUDA_OK();
  }
};

/// Specialization for Dim == 2, col-major
template <typename T>
struct copy_block<2, col2_type, T>
{
  static void host_to_dev(Gpu_block_impl<2, T, col2_type> const& block)
  {
    T const* src = block.storage_.data();
    T* dest = block.device_storage_.data();
    cudaMemcpy2D( 
      dest, block.size(2, 0) * sizeof(T),
      src, block.impl_stride(2, 1) * sizeof(T),
      block.size(2, 0) * sizeof(T), block.size(2, 1),
      cudaMemcpyHostToDevice);
    ASSERT_CUDA_OK();
  }
  static void dev_to_host(Gpu_block_impl<2, T, col2_type> const& block)
  {
    T const* src = block.device_storage_.data();
    T* dest = block.storage_.data();
    cudaMemcpy2D( 
      dest, block.impl_stride(2, 1) * sizeof(T),
      src, block.size(2, 0) * sizeof(T),
      block.size(2, 0) * sizeof(T), block.size(2, 1),
      cudaMemcpyDeviceToHost);
    ASSERT_CUDA_OK();
  }
};

/// Specialization for Dim == 3
///   Note that this is not a strided copy.  It should be used with dense
///   blocks only.  If padding is desired, it must be dealt with explicitly.
template <typename Order, typename T>
struct copy_block<3, Order, T>
{
  static inline void host_to_dev(Gpu_block_impl<3, T, Order> const& block)
  {
    assert(block.impl_stride(3, Order::impl_dim0) == 
      static_cast<stride_type>(block.size(3, Order::impl_dim1)) *
      static_cast<stride_type>(block.size(3, Order::impl_dim2)));
    assert(block.impl_stride(3, Order::impl_dim1) == 
      static_cast<stride_type>(block.size(3, Order::impl_dim2)));
    assert(block.impl_stride(3, Order::impl_dim2) == 1);

    T const* src = block.storage_.data();
    T* dest = block.device_storage_.data();
    cudaMemcpy(dest, src, block.size() * sizeof(T), cudaMemcpyHostToDevice);
    ASSERT_CUDA_OK();
  }
  static inline void dev_to_host(Gpu_block_impl<3, T, Order> const& block)
  {
    assert(block.impl_stride(3, Order::impl_dim0) == 
      static_cast<stride_type>(block.size(3, Order::impl_dim1)) *
      static_cast<stride_type>(block.size(3, Order::impl_dim2)));
    assert(block.impl_stride(3, Order::impl_dim1) == 
      static_cast<stride_type>(block.size(3, Order::impl_dim2)));
    assert(block.impl_stride(3, Order::impl_dim2) == 1);

    T const* src = block.device_storage_.data();
    T* dest = block.storage_.data();
    cudaMemcpy(dest, src, block.size() * sizeof(T), cudaMemcpyDeviceToHost);
    ASSERT_CUDA_OK();
  }
};

} // namespace cuda


/// Specialize block layout trait for Gpu_blocks.
template <dimension_type Dim,
          typename       T,
          typename       Order>
struct Block_layout<cuda::Gpu_block<Dim, T, Order> >
{
  static dimension_type const dim = Dim;

  typedef Direct_access_tag  access_type;
  typedef Order              order_type;
  typedef Stride_unit_dense  pack_type;
  typedef Cmplx_inter_fmt    complex_type;
                                        
  typedef Layout<Dim,
    typename Row_major<Dim>::type,
    Stride_unit_dense,
    Cmplx_inter_fmt>         layout_type;
};



template <dimension_type Dim,
          typename       T,
          typename       Order>
struct Is_modifiable_block<cuda::Gpu_block<Dim, T, Order> >
{
  static bool const value = true;
};


} // namespace impl
} // namespace vsip

#endif // VSIP_OPT_GPU_BLOCK_HPP
