/* Copyright (c) 2009 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    python/view.cpp
    @author  Stefan Seefeld
    @date    2009-08-12
    @brief   VSIPL++ Library: Python bindings for view API.

*/
#include <boost/python.hpp>
#include <boost/python/slice.hpp>
#include <boost/noncopyable.hpp>
#include <vsip/initfin.hpp>
#include <vsip/support.hpp>
#include <vsip/vector.hpp>
#include <vsip/matrix.hpp>
#include "block.hpp"
#include <stdexcept>

namespace pyvsip
{

/// Construct a view from a numpy.ndarray
template <typename V>
std::auto_ptr<V> view(bpl::object o)
{
  Block<V::dim, typename V::value_type> block(o);
  return std::auto_ptr<V>(new V(block));
}

/// Provide access to view as a numpy.ndarray
template <typename V>
bpl::object array(V &v)
{
  return bpl::object(v.block().array());
}

inline vsip::length_type 
max_slice_length(int start, int step, int size)
{
  if (step > 0)
    return ceil((size - start) / float(step));
  else
    return ceil((start + 1) / (float)-step);
}

template <typename T>
vsip::Vector<T, Block<1, T> > 
get_slice(vsip::Vector<T, Block<1, T> > &v, bpl::slice s)
{
  int start = s.start() ? bpl::extract<int>(s.start())() : 0;
  if (start < 0) start += v.size();
  int step = s.step() ? bpl::extract<int>(s.step())() : 1;
  int length = max_slice_length(start, step, v.size());
  if (s.stop())
  {
    int stop = bpl::extract<int>(s.stop());
    if (stop < 0) stop += v.size();
    length = ceil(std::abs((stop - start)/float(step)));
  }
  vsip::Domain<1> domain(start, step * v.block().impl_stride(1, 0), length);
  Block<1, T> block(v.block(), start, domain);
  return vsip::Vector<T, Block<1, T> >(block);
}

template <typename T>
void set_slice(vsip::Vector<T, Block<1, T> > &v, bpl::slice s, T value)
{ 
  get_slice(v, s) = value;
}

template <typename T>
vsip::Vector<T, Block<1, T> > 
copy_vector(vsip::Vector<T, Block<1, T> > v)
{
  Block<1, T> block(v.block().copy());
  return vsip::Vector<T, Block<1, T> >(block);
}

template <typename T>
void fill_vector(vsip::Vector<T, Block<1, T> > &v, T value)
{ 
  v = value;
}

template <typename T>
void define_vector(char const *type_name)
{
  typedef vsip::Vector<T, Block<1, T> > vector_type;
  T (vector_type::*get)(vsip::index_type) const = &vector_type::get;
  void (vector_type::*put)(vsip::index_type, T) const =
    &vector_type::put;

  bpl::class_<vector_type> vector(type_name, bpl::init<vsip::length_type>());
  vector.def(bpl::init<vsip::length_type, T>());
  /// Conversion from array.
  vector.def("__init__", bpl::make_constructor(view<vector_type>));
  /// Conversion to array.
  vector.def("array", array<vector_type>);
  vector.def("__array__", array<vector_type>);
  vector.def("length", &vector_type::length);
  vector.def("__getitem__", get);
  vector.def("__getitem__", get_slice<T>);
  vector.def("__setitem__", put);
  vector.def("__setitem__", set_slice<T>);
  vector.def("copy", copy_vector<T>);
  vector.def("fill", fill_vector<T>);
}

template <typename T>
bpl::object create_vector(vsip::length_type size)
{ return bpl::object(vsip::Vector<T, Block<1, T> >(size));}

template <typename T>
bpl::object create_vector(bpl::object array)
{
  Block<1, T> block(array);
  return bpl::object(vsip::Vector<T, Block<1, T> >(block));
}

bpl::object create_vector_from_type(bpl::object type, vsip::length_type size)
{
  if (PyType_Check(type.ptr()))
  {
    if (type.ptr() == (PyObject*)&PyInt_Type)
      return create_vector<long>(size);
    else if (type.ptr() == (PyObject*)&PyFloat_Type)
      return create_vector<double>(size);
    else if (type.ptr() == (PyObject*)&PyComplex_Type)
      return create_vector<std::complex<double> >(size);
    else throw std::runtime_error("unsupported type");
  }
  else throw std::runtime_error("argument not a type");
}

bpl::object create_vector_from_array(bpl::object o)
{
  if (!PyArray_Check(o.ptr()))
    THROW(TypeError, "argument is not a numpy array");
  NPY_TYPES typenum = NPY_TYPES(PyArray_TYPE(o.ptr()));
  switch (typenum)
  {
    case NPY_INT: return create_vector<int>(o);
    case NPY_FLOAT: return create_vector<float>(o);
    case NPY_DOUBLE: return create_vector<double>(o);
    case NPY_CFLOAT: return create_vector<std::complex<float> >(o);
    case NPY_CDOUBLE: return create_vector<std::complex<double> >(o);
    default: throw std::runtime_error("wrong type_code");
  }
}

template <typename T>
vsip::Vector<T, Block<1, T> > 
get_row(vsip::Matrix<T, Block<2, T> > &m, int index)
{
  if (index < 0) index += m.size(0);
  vsip::Domain<1> domain(0, 1, m.size(1));
  Block<1, T> block(m.block(), index * m.size(1), domain);
  return vsip::Vector<T, Block<1, T> >(block);
}

template <typename T>
void set_row(vsip::Matrix<T, Block<2, T> > &m, int index, T value)
{
  get_row(m, index) = value;
}

template <typename T>
vsip::Matrix<T, Block<2, T> > 
get_rows(vsip::Matrix<T, Block<2, T> > &m, bpl::slice i)
{
  int start = i.start() ? bpl::extract<int>(i.start())() : 0;
  if (start < 0) start += m.size(0);
  int step = i.step() ? bpl::extract<int>(i.step())() : 1;
  vsip::length_type length = max_slice_length(start, step, m.size(0));
  if (i.stop())
  {
    int stop = bpl::extract<int>(i.stop());
    if (stop < 0) stop += m.size(0);
    length = ceil(std::abs((stop - start)/float(step)));
  }
  vsip::stride_type stride = m.block().impl_stride(2, 0);
  vsip::Domain<1> domain(start * stride, step * stride, length);
  Block<2, T> block(m.block(), start * stride, vsip::Domain<2>(domain, m.size(1)));
  return vsip::Matrix<T, Block<2, T> >(block);
}

template <typename T>
void set_rows(vsip::Matrix<T, Block<2, T> > &m, bpl::slice i, T value)
{
  get_rows(m, i) = value;
}

template <typename T>
vsip::Vector<T, Block<1, T> > 
get_row_slice(vsip::Matrix<T, Block<2, T> > &m, int i, bpl::slice j)
{
  if (i < 0) i += m.size(0);
  int start = j.start() ? bpl::extract<int>(j.start())() : 0;
  if (start < 0) start += m.size(1);
  int step = j.step() ? bpl::extract<int>(j.step())() : 1;
  vsip::length_type length = max_slice_length(start, step, m.size(1));
  if (j.stop())
  {
    int stop = bpl::extract<int>(j.stop());
    if (stop < 0) stop += m.size(1);
    length = ceil(std::abs((stop - start)/float(step)));
  }
  Block<2, T> &b = m.block();
  vsip::stride_type stride = b.impl_stride(2, 1);
  int offset = i * b.impl_stride(2, 0) + start * stride;
  vsip::Domain<1> domain(start * stride, step * stride, length);
  Block<1, T> block(m.block(), offset, domain);
  return vsip::Vector<T, Block<1, T> >(block);
}

template <typename T>
vsip::Vector<T, Block<1, T> > 
get_column_slice(vsip::Matrix<T, Block<2, T> > &m, bpl::slice i, int j)
{
  int start = i.start() ? bpl::extract<int>(i.start())() : 0;
  if (start < 0) start += m.size(0);
  int step = i.step() ? bpl::extract<int>(i.step())() : 1;
  vsip::length_type length = max_slice_length(start, step, m.size(0));
  if (i.stop())
  {
    int stop = bpl::extract<int>(i.stop());
    if (stop < 0) stop += m.size(0);
    length = ceil(std::abs((stop - start)/float(step)));
  }
  if (j < 0) j += m.size(1);
  Block<2, T> &b = m.block();
  vsip::stride_type stride = b.impl_stride(2, 0);
  int offset = start * stride + j * b.impl_stride(2, 1);
  vsip::Domain<1> domain(start * stride, step * stride, length);
  Block<1, T> block(m.block(), offset, domain);
  return vsip::Vector<T, Block<1, T> >(block);
}

template <typename T>
vsip::Matrix<T, Block<2, T> > 
get_sub_matrix(vsip::Matrix<T, Block<2, T> > &m, bpl::slice i, bpl::slice j)
{
  int start = i.start() ? bpl::extract<int>(i.start())() : 0;
  if (start < 0) start += m.size(0);
  int step = i.step() ? bpl::extract<int>(i.step())() : 1;
  int length = max_slice_length(start, step, m.size(0));
  if (i.stop())
  {
    int stop = bpl::extract<int>(i.stop());
    if (stop < 0) stop += m.size(0);
    length = ceil(std::abs((stop - start)/float(step)));
  }
  vsip::stride_type stride = m.block().impl_stride(2, 0);
  vsip::Domain<1> idomain(start * stride, step * stride, length);
  int offset = start * stride;
  start = j.start() ? bpl::extract<int>(j.start())() : 0;
  if (start < 0) start += m.size(1);
  step = j.step() ? bpl::extract<int>(j.step())() : 1;
  length = max_slice_length(start, step, m.size(1));
  if (j.stop())
  {
    int stop = bpl::extract<int>(j.stop());
    if (stop < 0) stop += m.size(1);
    length = ceil(std::abs((stop - start)/float(step)));
  }
  stride = m.block().impl_stride(2, 1);
  vsip::Domain<1> jdomain(start * stride, step * stride, length);
  offset += start * stride;
  Block<2, T> block(m.block(), offset, vsip::Domain<2>(idomain, jdomain));
  return vsip::Matrix<T, Block<2, T> >(block);
}

// Depending on the 'index' tuple, this function may return a single element or
// a subview.
template <typename T>
bpl::object get_item(vsip::Matrix<T, Block<2, T> > &m, bpl::tuple index)
{
  // Since the index isn't an int (which is caught by a different overload),
  // it must be a pair.
  if (bpl::len(index) != 2) throw std::runtime_error("invalid index size");
  // If the index pair contains two ints, we return an element.
  // Otherwise, if the index pair contains an int and a slice, or a slice and an int,
  // we return a vector (a subset of either a row or a column).
  // Otherwise, if the index pair contains two slices, we return a submatrix.
  bpl::extract<int> ei(index[0]);
  bpl::extract<bpl::slice> esi(index[0]);
  bpl::extract<int> ej(index[1]);
  bpl::extract<bpl::slice> esj(index[1]);
  if (ei.check() && ej.check())
  {
    int i = ei();
    if (i < 0) i += m.size(0);
    int j = ej();
    if (j < 0) j += m.size(1);
    return bpl::object(m.get(i, j));
  }
  else if (ei.check() && esj.check())
    return bpl::object(get_row_slice(m, ei(), esj()));
  else if (esi.check() && ej.check())
    return bpl::object(get_column_slice(m, esi(), ej()));
  else if (esi.check() && esj.check())
    return bpl::object(get_sub_matrix(m, esi(), esj()));
  else throw std::runtime_error("index argument must either be an integer or a slice.");
}

template <typename T>
void set_item(vsip::Matrix<T, Block<2, T> > &m, bpl::tuple index, T value)
{
  // This function is very similar to get_item() above.
  if (bpl::len(index) != 2) throw std::runtime_error("invalid index size");
  bpl::extract<int> ei(index[0]);
  bpl::extract<bpl::slice> esi(index[0]);
  bpl::extract<int> ej(index[1]);
  bpl::extract<bpl::slice> esj(index[1]);
  if (ei.check() && ej.check())
  {
    int i = ei();
    if (i < 0) i += m.size(0);
    int j = ej();
    if (j < 0) j += m.size(1);
    m.put(i, j, value);
  }
  else if (ei.check() && esj.check())
    get_row_slice(m, ei(), esj()) = value;
  else if (esi.check() && ej.check())
    get_column_slice(m, esi(), ej()) = value;
  else if (esi.check() && esj.check())
    get_sub_matrix(m, esi(), esj()) = value;
  else throw std::runtime_error("index argument must either be an integer or a slice.");
}

template <typename T>
vsip::Matrix<T, Block<2, T> > 
copy_matrix(vsip::Matrix<T, Block<2, T> > m)
{
  Block<2, T> block(m.block().copy());
  return vsip::Matrix<T, Block<2, T> >(block);
}

template <typename T>
void fill_matrix(vsip::Matrix<T, Block<2, T> > &m, T value)
{ 
  m = value;
}

template <typename T>
void define_matrix(char const *type_name)
{
  typedef vsip::Matrix<T, Block<2, T> > matrix_type;
  bpl::class_<matrix_type> matrix(type_name, bpl::init<vsip::length_type,
				                       vsip::length_type>());
  matrix.def(bpl::init<vsip::length_type, vsip::length_type, T>());
  // Conversion from array.
  matrix.def("__init__", bpl::make_constructor(view<matrix_type>));
  // Conversion to array.
  matrix.def("array", array<matrix_type>);
  matrix.def("__array__", array<matrix_type>);
  // matrix[i]
  matrix.def("__getitem__", get_row<T>);
  // matrix[i:j:k]
  matrix.def("__getitem__", get_rows<T>);
  // matrix[i, j] (where i and j are either ints or slices)
  matrix.def("__getitem__", get_item<T>);
  // matrix[i]
  matrix.def("__setitem__", set_row<T>);
  // matrix[i:j:k]
  matrix.def("__setitem__", set_rows<T>);
  // matrix[i, j] (where i and j are either ints or slices)
  matrix.def("__setitem__", set_item<T>);
  matrix.def("copy", copy_matrix<T>);
  matrix.def("fill", fill_matrix<T>);
}

template <typename T>
bpl::object create_matrix(vsip::length_type rows, vsip::length_type cols,
			  dimension_ordering o)
{
  Block<2, T> block(vsip::Domain<2>(rows, cols), 
		    typename Block<2, T>::map_type(), o);
  return bpl::object(vsip::Matrix<T, Block<2, T> >(block));
}

template <typename T>
bpl::object create_matrix(bpl::object array)
{
  Block<2, T> block(array);
  return bpl::object(vsip::Matrix<T, Block<2, T> >(block));
}

bpl::object create_matrix_from_type_and_order(bpl::object type,
					      vsip::length_type rows,
					      vsip::length_type cols,
					      dimension_ordering o)
{
  if (PyType_Check(type.ptr()))
  {
    if (type.ptr() == (PyObject*)&PyInt_Type)
      return create_matrix<long>(rows, cols, o);
    else if (type.ptr() == (PyObject*)&PyFloat_Type)
      return create_matrix<double>(rows, cols, o);
    else if (type.ptr() == (PyObject*)&PyComplex_Type)
      return create_matrix<std::complex<double> >(rows, cols, o);
    else throw std::runtime_error("unsupported type");
  }
  else throw std::runtime_error("argument not a type");
}

bpl::object create_matrix_from_type(bpl::object type,
				    vsip::length_type rows, vsip::length_type cols)
{
  return create_matrix_from_type_and_order(type, rows, cols, row);
}

bpl::object create_matrix_from_array(bpl::object o)
{
  if (!PyArray_Check(o.ptr()))
    THROW(TypeError, "argument is not a numpy array");
  NPY_TYPES typenum = NPY_TYPES(PyArray_TYPE(o.ptr()));
  switch (typenum)
  {
    case NPY_INT: return create_matrix<int>(o);
    case NPY_FLOAT: return create_matrix<float>(o);
    case NPY_DOUBLE: return create_matrix<double>(o);
    case NPY_CFLOAT: return create_matrix<std::complex<float> >(o);
    case NPY_CDOUBLE: return create_matrix<std::complex<double> >(o);
    default: throw std::runtime_error("wrong type_code");
  }
}

}

BOOST_PYTHON_MODULE(view)
{
  using namespace pyvsip;

  bpl::enum_<dimension_ordering> order("dimension_ordering");
  order.value("row", row);
  order.value("column", column);

  pyvsip::define_vector<int>("IVector");
  pyvsip::define_vector<float>("FVector");
  pyvsip::define_vector<double>("DVector");
  pyvsip::define_vector<std::complex<double> >("CVector");
  bpl::def("vector", pyvsip::create_vector_from_type);
  bpl::def("vector", pyvsip::create_vector_from_array);

  pyvsip::define_matrix<int>("IMatrix");
  pyvsip::define_matrix<float>("FMatrix");
  pyvsip::define_matrix<double>("DMatrix");
  pyvsip::define_matrix<std::complex<double> >("CMatrix");
  bpl::def("matrix", pyvsip::create_matrix_from_type);
  bpl::def("matrix", pyvsip::create_matrix_from_type_and_order);
  bpl::def("matrix", pyvsip::create_matrix_from_array);
}
