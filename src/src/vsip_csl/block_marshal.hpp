/* Copyright (c) 2009 by CodeSourcery.  All rights reserved. */

#ifndef VSIP_CSL_BLOCK_MARSHAL_HPP
#define VSIP_CSL_BLOCK_MARSHAL_HPP

#include <vsip/dense.hpp>
#include <vsip_csl/dda.hpp>
#include <vsip/core/metaprogramming.hpp>

namespace vsip_csl
{
namespace impl
{
namespace block_marshal
{
enum { CHAR, FLOAT, DOUBLE, CFLOAT, CDOUBLE};
enum { SPLIT = vsip::impl::cmplx_split_fmt,
       INTERLEAVED = vsip::impl::cmplx_inter_fmt};

struct Descriptor
{
  vsip::impl::uint8_type value_type;
  vsip::impl::uint8_type dimensions;
  vsip::impl::uint8_type complex_type;
  vsip::impl::uint8_type block_type;
  vsip::impl::uint32_type size0;
  vsip::impl::uint32_type stride0;
  vsip::impl::uint32_type size1;
  vsip::impl::uint32_type stride1;
  vsip::impl::uint32_type size2;
  vsip::impl::uint32_type stride2;
};

template <typename Block,
	  bool Complex = vsip::impl::Is_complex<typename Block::value_type>::value>
class Marshal_base;

template <typename Block>
class Marshal_base<Block, false>
{
  typedef typename Block::value_type value_type;
  static vsip::dimension_type const dim = Block::dim;
public:
  static void rebind(Block &b,
		     void *data, void *, vsip::impl::uint8_type,
		     vsip::Domain<dim> const &domain)
  {
    b.rebind(static_cast<value_type *>(data), domain);
  }
};

template <typename Block>
class Marshal_base<Block, true>
{
  typedef typename Block::value_type value_type;
  typedef typename vsip::impl::Scalar_of<value_type> scalar_type;
  static vsip::dimension_type const dim = Block::dim;
public:
  static void rebind(Block &b,
		     void *real, void *imag, vsip::impl::uint8_type format,
		     vsip::Domain<dim> const &domain)
  {
    if (format == SPLIT)
      b.rebind(static_cast<scalar_type *>(real),
	       static_cast<scalar_type *>(imag),
               domain);
    else
      b.rebind(static_cast<value_type *>(real), domain);
  }
};

template <typename Block> 
class Marshal
{
public:
  static void marshal(Block const &, Descriptor &)
  { VSIP_IMPL_THROW(std::invalid_argument());}
  static bool can_unmarshal(Descriptor const &) { return false;}
  static void unmarshal(Descriptor const &, Block &)
  { VSIP_IMPL_THROW(std::invalid_argument());}
};

template <vsip::dimension_type D, typename T, typename O>
class Marshal<vsip::Dense<D, T, O, vsip::Local_map> >
  : public Marshal_base<vsip::Dense<D, T, O, vsip::Local_map> >
{
  typedef void *data_pointers[2];
public:
  typedef vsip::Dense<D, T, O, vsip::Local_map> block_type;

  static void marshal(block_type const &b,
		      data_pointers &data, Descriptor &d)
  {
    using namespace vsip::impl;
    if (Type_equal<T, char>::value) d.value_type = CHAR;
    else if (Type_equal<T, float>::value) d.value_type = FLOAT;
    else if (Type_equal<T, double>::value) d.value_type = DOUBLE;
    else if (Type_equal<T, complex<float> >::value) d.value_type = CFLOAT;
    else if (Type_equal<T, complex<double> >::value) d.value_type = CDOUBLE;
    d.dimensions = block_type::dim;
    if (Is_split_block<block_type>::value)
      d.complex_type = SPLIT;
    else
      d.complex_type = INTERLEAVED;
    // We know this is a Dense block with DDA, thus no copy.
    dda::Rt_ext_data<block_type> ext_data
      (b, vsip::impl::block_layout<block_type::dim>(b), dda::SYNC_IN);
    d.size0 = ext_data.size(0);
    d.stride0 = ext_data.stride(0);
    if (d.dimensions > 1)
    {
      d.size1 = ext_data.size(1);
      d.stride1 = ext_data.stride(1);
    }
    if (d.dimensions > 2)
    {
      d.size2 = ext_data.size(2);
      d.stride2 = ext_data.stride(2);
    }
    if (d.complex_type == SPLIT)
    {
      data[0] = static_cast<void *>(ext_data.data().as_split().first);
      data[1] = static_cast<void *>(ext_data.data().as_split().second);
    }
    else
    {
      data[0] = static_cast<void *>(ext_data.data().as_inter());
      data[1] = 0;
    }
  };

  static bool can_unmarshal(Descriptor const &d)
  {
    using namespace vsip::impl;
    if ((Type_equal<T, char>::value && d.value_type != CHAR) ||
	(Type_equal<T, float>::value && d.value_type != FLOAT) ||
	(Type_equal<T, double>::value && d.value_type != DOUBLE) ||
	(Type_equal<T, std::complex<float> >::value && d.value_type != CFLOAT) ||
	(Type_equal<T, std::complex<double> >::value && d.value_type != CDOUBLE))
      return false; // type mismatch
    else if (D != d.dimensions)
      return false; // dimension mismatch
    else if (vsip::impl::Is_complex<T>::value &&
	     (Type_equal<dense_complex_type, Cmplx_inter_fmt>::value &&
	      d.complex_type != INTERLEAVED))
      return false; // complex type mismatch

    // The minor dimension needs to be unit-stride.
    if (d.dimensions == 1 && d.stride0 != 1) return false;
    else if (d.dimensions == 2 && d.stride1 != 1) return false;
    else if (d.dimensions == 3 && d.stride2 != 1) return false;
    // Make sure strides match sizes.
    if (d.dimensions == 2 && d.stride0 != d.size1) return false;
    else if (d.dimensions == 3 && d.stride1 != d.size2 * d.size1) return false;
    return true;
  }
  static void unmarshal(void **data, Descriptor const &d, block_type &b)
  {
    using namespace vsip;
    if (!can_unmarshal(d))
      VSIP_IMPL_THROW(std::invalid_argument("Incompatible Block type"));
    Domain<block_type::dim> domain;
    switch (d.dimensions)
    {
      case 3:
	domain.impl_at(2) = Domain<1>(0, d.stride2, d.size2);
      case 2:
	domain.impl_at(1) = Domain<1>(0, d.stride1, d.size1);
      case 1:
	domain.impl_at(0) = Domain<1>(0, d.stride0, d.size0);
	break;
      default:
	VSIP_IMPL_THROW(std::invalid_argument("Invalid dimension"));
    }
    rebind(b, data[0], data[1], d.complex_type, domain);
  }
};

} // namespace vsip_csl::impl::block_marshal

struct Block_marshal
{
  template <typename Block>
  void marshal(Block const &b)
  { block_marshal::Marshal<Block>::marshal(b, data, descriptor);}

  template <typename Block>
  bool can_unmarshal()
  { return block_marshal::Marshal<Block>::can_unmarshal(descriptor);}

  template <typename Block>
  void unmarshal(Block &b)
  {
    block_marshal::Marshal<Block>::unmarshal(data, descriptor, b);
  }

  block_marshal::Descriptor descriptor;
  void *data[2];
};

} // namespace vsip_csl::impl
} // namespace vsip_csl

#endif
