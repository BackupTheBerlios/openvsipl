/* Copyright (c) 2007 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    vsip/opt/cbe/ppu/alf.hpp
    @author  Stefan Seefeld
    @date    2007-01-22
    @brief   VSIPL++ Library: Wrappers and traits to bridge with IBMs ALF.
*/

#ifndef VSIP_OPT_CBE_PPU_ALF_HPP
#define VSIP_OPT_CBE_PPU_ALF_HPP

/***********************************************************************
  Included Files
***********************************************************************/

#include <cstring>
#include <vsip/core/config.hpp>

#if VSIP_IMPL_REF_IMPL
# error "vsip/opt files cannot be used as part of the reference impl."
#endif

#include <cml.h>
#include <vsip/core/metaprogramming.hpp>
#include <vsip/support.hpp>
#include <csl_alf.h>
#include <lwp_task.h>
#include <lwp_params.h>


/***********************************************************************
  Declarations
***********************************************************************/

namespace vsip
{
namespace impl
{
namespace cbe
{

class ALF;
class Task;

template <typename D> struct alf_data_type;
template <> struct alf_data_type<float> 
{
  static ALF_DATA_TYPE_T const value = ALF_DATA_FLOAT;
};
template <> struct alf_data_type<std::complex<float> > 
{
  static ALF_DATA_TYPE_T const value = ALF_DATA_FLOAT;
};
template <> struct alf_data_type<double> 
{
  static ALF_DATA_TYPE_T const value = ALF_DATA_DOUBLE;
};
template <> struct alf_data_type<std::complex<double> > 
{
  static ALF_DATA_TYPE_T const value = ALF_DATA_DOUBLE;
};

class Workblock
{
  friend class Task;

public:
  ~Workblock() {}

  template <typename P>
  void set_parameters(P const &p) 
  { 
    int status = alf_wb_parm_add(workblock_,
				 const_cast<P *>(&p),
				 sizeof(P), ALF_DATA_BYTE, 0);
    assert(status >= 0);
  }

  void enqueue()
  {
    int status = alf_wb_enqueue(workblock_);
    assert(status >= 0);
  }
private:
  Workblock() {}

  alf_wb_handle_t workblock_;
};



class Task
{
  friend class ALF;

public:
  Task()
    : image_(0), ssize_(0), psize_(0), isize_(0), osize_(0),
      iosize_(0), tsize_(0), task_(0) {}
  ~Task() { destroy(); }
  Task(char const* library,
       char const* image, length_type ssize, length_type psize,
       length_type isize, length_type osize, length_type iosize,
       length_type tsize, unsigned int spes);
  Task(char const* library, char const* image, unsigned int spes);
  void destroy();
  char const *image() const { return image_;}
  length_type ssize() const { return ssize_;}
  length_type psize() const { return psize_;}
  length_type isize() const { return isize_;}
  length_type osize() const { return osize_;}
  length_type tsize() const { return tsize_;}
  operator bool() const { return image_;}
  Workblock create_workblock()
  {
    Workblock block;
    int status = alf_wb_create(task_, ALF_WB_SINGLE, 1, &block.workblock_);
    if (status != 0)      
      VSIP_IMPL_THROW(std::runtime_error("Task::create_workblock(single) -- alf_wb_create failed."));
    return block;
  }
  Workblock create_workblock(int times)
  {
    Workblock block;
    int status = alf_wb_create(task_, ALF_WB_MULTI, times, &block.workblock_);
    if (status != 0)      
      VSIP_IMPL_THROW(std::runtime_error("Task::create_workblock(multi) -- alf_wb_create failed."));
    return block;
  }
  void sync()
  {
    unsigned int queue_size = 1;
    while(queue_size > 0)
    {
      int status = alf_task_query(task_, &queue_size, 0);
      assert(status >= 0);
      if (status == 0) break;
    }
  }

private:
  char const *image_;
  length_type ssize_;
  length_type psize_;
  length_type isize_;
  length_type osize_;
  length_type iosize_;
  length_type tsize_;
  alf_task_desc_handle_t desc_;
  alf_task_handle_t task_;
};

namespace lwp
{

class Workblock
{
  friend class Task;

public:
  template <typename P>
  void set_parameters(P const &p)
  { lwp_wb_set_parameters(&block_, (void*)&p, sizeof(P));}
  void enqueue()
  { lwp_wb_enqueue(&block_);}
private:
  Workblock(lwp_workblock const &wb) : block_(wb) {}

  lwp_workblock block_;
};

class Task
{
  friend class ALF;

public:
  Task(unsigned int spes,
       int buffer_size, int num_bufs,
       uintptr_t code_ea, int code_size,
       int cmd)
  {
    lwp_task_create(&task_, spes, buffer_size, num_bufs, code_ea, code_size, cmd);
  }
  ~Task() { destroy();}
  void destroy() { lwp_task_destroy(&task_);}
  Workblock create_workblock(int num_iter)
  {
    lwp_workblock block;
    lwp_task_wb_create(&task_, &block, num_iter);
    return Workblock(block);
  }

  void sync() { lwp_task_sync(&task_);}

private:
  lwp_task task_;
};

inline void
load_plugin(char *&code_ea, int &code_size,
	    char const *library_subdir, char const *kernel)
{ return lwp_load_plugin(&code_ea, &code_size, library_subdir, kernel);}

}


class ALF
{
  friend class Task_manager;
public:
  ALF(unsigned int num_accelerators)
    : num_accelerators_(0)
  {
    int   argc = 3;
    char* argv[3];
    char  number[256];
    sprintf(number, "%u", num_accelerators);
    argv[0] = "VSIPL++";
    argv[1] = "--cml-num-spes";
    argv[2] = number;
    cml_init_argv(&argc, argv);
    num_accelerators_ = csl_alf_num_spes();
  }
  ~ALF()
  {
    cml_fini();
  }
  void set_num_accelerators(unsigned int n)
  {
    // In ALF 3.0, this function can only be called once, in between
    // alf_init and first alf_task_create.
    assert(num_accelerators_ == 0);

    unsigned int num_spus = csl_alf_num_spes();
    if (num_spus > n || n == 0)
      n = num_spus;
    
    int status = alf_num_instances_set(csl_alf_handle(), n);
    assert(status > 0);
    num_accelerators_ = status;
  }
  unsigned int num_accelerators() const { return num_accelerators_;}

  Task create_task(const char* library,
		   const char* image,
		   int ssize,
                   length_type psize, length_type isize,
		   length_type osize, length_type iosize,
		   length_type tsize)
  {
    return Task(library, image, ssize, psize, isize, osize, iosize, tsize,
		num_accelerators());
  }

private:
  unsigned int num_accelerators_;
};

} // namespace vsip::impl::cbe
} // namespace vsip::impl
} // namespace vsip

#endif
