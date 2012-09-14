/* Copyright (c) 2007 by CodeSourcery.  All rights reserved. */

/** @file    vsip/opt/cbe/ppu/eval_fastconv.hpp
    @author  Jules Bergmann
    @date    2007-03-05
    @brief   VSIPL++ Library: General evaluator for fast convolution

*/

#ifndef VSIP_OPT_CBE_PPU_EVAL_FASTCONV_HPP
#define VSIP_OPT_CBE_PPU_EVAL_FASTCONV_HPP

#include <vsip/opt/expr/assign_fwd.hpp>
#include <vsip/core/fft.hpp>
#include <vsip/opt/cbe/ppu/fastconv.hpp>

namespace vsip_csl
{
namespace dispatcher
{

template <typename LHS,
	  typename VecBlockT,
	  typename MatBlockT,
	  typename Backend1T,
	  typename Workspace1T,
	  typename Backend2T,
	  typename Workspace2T>
struct Evaluator<op::assign<2>, be::cbe_sdk,
  void(LHS &,
       expr::Unary<expr::op::fft<2, Backend1T, Workspace1T>::template Functor,
         expr::Vmmul<0, VecBlockT,
            expr::Unary<expr::op::fft<2, Backend2T, Workspace2T>::template Functor,
              MatBlockT> const> const> const &)>
{
  static char const* name() { return "Cbe_sdk_tag"; }

  typedef
  expr::Unary<expr::op::fft<2, Backend1T, Workspace1T>::template Functor,
    expr::Vmmul<0, VecBlockT,
      expr::Unary<expr::op::fft<2, Backend2T, Workspace2T>::template Functor,
	MatBlockT> const> const> RHS;

  typedef typename LHS::value_type dst_type;
  typedef typename RHS::value_type src_type;
  typedef typename impl::Block_layout<LHS>::complex_type complex_type;
  typedef impl::cbe::Fastconv<1, complex<float>, complex_type> fconv_type;

  static bool const ct_valid = 
    impl::Type_equal<complex<float>, typename VecBlockT::value_type>::value &&
    impl::Type_equal<complex<float>, typename MatBlockT::value_type>::value;

  static bool rt_valid(LHS &lhs, RHS const &rhs)
  {
    impl::Ext_data<VecBlockT> ext_kernel(rhs.arg().get_vblk());
    impl::Ext_data<MatBlockT> ext_in(rhs.arg().get_mblk().arg());
    impl::Ext_data<LHS> ext_out(lhs);

    return 
      fconv_type::rt_valid_size(lhs.size(2, 1)) &&
      ext_kernel.stride(0) == 1 &&
      ext_in.stride(1) == 1 &&
      ext_out.stride(1) == 1;
  }
  
  static void exec(LHS &lhs, RHS const &rhs)
  {
    length_type cols = lhs.size(2, 1);
    Matrix<complex<float> > tmp(1, cols);

    Vector<complex<float>, VecBlockT> w 
      (const_cast<VecBlockT&>(rhs.arg().get_vblk()));
    Matrix<complex<float>, MatBlockT> in 
      (const_cast<MatBlockT&>(rhs.arg().get_mblk().arg()));
    Matrix<complex<float>, LHS> out(lhs);

    fconv_type fconv(w, cols, false);

    fconv(in, out);
  }
};


template <typename LHS,
	  typename CoeffsMatBlockT,
	  typename MatBlockT,
	  typename Backend1T,
	  typename Workspace1T,
	  typename Backend2T,
	  typename Workspace2T>
struct Evaluator<op::assign<2>, be::cbe_sdk,
  void(LHS &,
       expr::Unary<expr::op::fft<2, Backend1T, Workspace1T>::template Functor,
         expr::Binary<expr::op::Mult, CoeffsMatBlockT,
           expr::Unary<expr::op::fft<2, Backend2T, Workspace2T>::template Functor, MatBlockT>
         const> const> const &)>
{
  static char const* name() { return "Cbe_sdk_tag"; }

  typedef
    expr::Unary<expr::op::fft<2, Backend1T, Workspace1T>::template Functor,
      expr::Binary<expr::op::Mult, CoeffsMatBlockT,
	expr::Unary<expr::op::fft<2, Backend2T, Workspace2T>::template Functor, MatBlockT>
      const> const> RHS;

  typedef typename LHS::value_type dst_type;
  typedef typename RHS::value_type src_type;
  typedef typename impl::Block_layout<LHS>::complex_type complex_type;
  typedef impl::cbe::Fastconv_base<2, complex<float>, complex_type> fconv_type;

  static bool const ct_valid = 
    impl::Type_equal<complex<float>, typename CoeffsMatBlockT::value_type>::value &&
    impl::Type_equal<complex<float>, typename MatBlockT::value_type>::value &&
    impl::Ext_data_cost<CoeffsMatBlockT>::value == 0 &&
    impl::Ext_data_cost<MatBlockT>::value == 0 &&
    impl::Ext_data_cost<LHS>::value == 0;

  static bool rt_valid(LHS &lhs, RHS const &rhs)
  {
    impl::Ext_data<CoeffsMatBlockT> ext_kernel(rhs.arg().arg1());
    impl::Ext_data<MatBlockT> ext_in(rhs.arg().arg2().arg());
    impl::Ext_data<LHS> ext_out(lhs);

    return 
      fconv_type::rt_valid_size(lhs.size(2, 1)) &&
      ext_kernel.stride(1) == 1 &&
      ext_in.stride(1) == 1 &&
      ext_out.stride(1) == 1;
  }
  
  static void exec(LHS &lhs, RHS const &rhs)
  {
    length_type cols = lhs.size(2, 1);
    Matrix<complex<float>, CoeffsMatBlockT> w 
      (const_cast<CoeffsMatBlockT&>(rhs.arg().arg1()));
    Matrix<complex<float>, MatBlockT> in 
      (const_cast<MatBlockT&>(rhs.arg().arg2().arg()));
    Matrix<complex<float>, LHS> out(lhs);

    fconv_type fconv(cols, false);

    fconv.convolve(in, w, out);
  }
};


template <typename LHS,
	  typename CoeffsMatBlockT,
	  typename MatBlockT,
	  typename Backend1T,
	  typename Workspace1T,
	  typename Backend2T,
	  typename Workspace2T>
struct Evaluator<op::assign<2>, be::cbe_sdk,
  void(LHS &,
       expr::Unary<expr::op::fft<2, Backend1T, Workspace1T>::template Functor,
         expr::Binary<expr::op::Mult,
           expr::Unary<expr::op::fft<2, Backend2T, Workspace2T>::template Functor,
             MatBlockT> const,
               CoeffsMatBlockT, true> const> const &)>
{
  static char const* name() { return "Cbe_sdk_tag";}

  typedef
    expr::Unary<expr::op::fft<2, Backend1T, Workspace1T>::template Functor,
      expr::Binary<expr::op::Mult,
	expr::Unary<expr::op::fft<2, Backend2T, Workspace2T>::template Functor,
	  MatBlockT> const,
	    CoeffsMatBlockT, true> const> RHS;

  typedef typename LHS::value_type dst_type;
  typedef typename RHS::value_type src_type;
  typedef typename impl::Block_layout<LHS>::complex_type complex_type;
  typedef impl::cbe::Fastconv_base<2, complex<float>, complex_type> fconv_type;

  static bool const ct_valid = 
    impl::Type_equal<complex<float>, typename CoeffsMatBlockT::value_type>::value &&
    impl::Type_equal<complex<float>, typename MatBlockT::value_type>::value &&
    impl::Ext_data_cost<CoeffsMatBlockT>::value == 0 &&
    impl::Ext_data_cost<MatBlockT>::value == 0 &&
    impl::Ext_data_cost<LHS>::value == 0;

  static bool rt_valid(LHS &lhs, RHS const &rhs)
  {
    impl::Ext_data<CoeffsMatBlockT> ext_kernel(rhs.arg().arg2());
    impl::Ext_data<MatBlockT> ext_in(rhs.arg().arg1().arg());
    impl::Ext_data<LHS> ext_out(lhs);

    return 
      fconv_type::rt_valid_size(lhs.size(2, 1)) &&
      ext_kernel.stride(1) == 1 &&
      ext_in.stride(1) == 1 &&
      ext_out.stride(1) == 1;
  }
  
  static void exec(LHS &lhs, RHS const &rhs)
  {
    length_type cols = lhs.size(2, 1);
    Matrix<complex<float>, CoeffsMatBlockT> w 
      (const_cast<CoeffsMatBlockT&>(rhs.arg().arg2()));
    Matrix<complex<float>, MatBlockT> in 
      (const_cast<MatBlockT&>(rhs.arg().arg1().arg()));
    Matrix<complex<float>, LHS> out(lhs);

    fconv_type fconv(cols, false);

    fconv.convolve(in, w, out);
  }
};

} // namespace vsip_csl::dispatcher
} // namespace vsip_csl

#endif // VSIP_OPT_CBE_PPU_EVAL_FASTCONV_HPP
