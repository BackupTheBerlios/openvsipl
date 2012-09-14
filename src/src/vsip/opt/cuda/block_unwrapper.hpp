/* Copyright (c) 2009 by CodeSourcery.  All rights reserved. */

/** @file    vsip/opt/cuda/block_unwrapper.hpp
    @author  Don McCoy
    @date    2009-07-13
    @brief   VSIPL++ Library: Helper class for unwrapping blocks
*/

#ifndef VSIP_OPT_CUDA_BLOCK_UNWRAPPER_HPP
#define VSIP_OPT_CUDA_BLOCK_UNWRAPPER_HPP


/***********************************************************************
  Included Files
***********************************************************************/

#include <vsip/support.hpp>
#include <vsip/core/subblock.hpp>


/***********************************************************************
  Declarations
***********************************************************************/

namespace vsip
{

namespace impl
{

namespace cuda
{

/// Provides a uniform interface for obtaining the proper block from
/// a wrapper block.  Un-wrapped blocks return themselves.
template <typename Block>
struct Block_unwrapper
{
  typedef Block block_type;

  static 
  block_type const&
  underlying_block(Block const &block)  { return block; }
};
    
/// Specialization for the transpose wrapper class.
template <typename Block>
struct Block_unwrapper<Transposed_block<Block> >
{
  typedef Transposed_block<Block>  wrapped_block_type;
  typedef Block                    block_type;
  
  static
  block_type const&
  underlying_block(wrapped_block_type const& block) { return block.impl_block(); }
};
    
    

} // namespace vsip::impl::cuda
} // namespace vsip::impl
} // namespace vsip

#endif // VSIP_OPT_CUDA_BLOCK_UNWRAPPER_HPP
