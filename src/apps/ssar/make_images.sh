#! /bin/sh

# Copyright (c) 2008 by CodeSourcery.  All rights reserved.
#
#  This file is available for license from CodeSourcery, Inc. under the terms
#  of a commercial license and under the GPL.  It is not part of the VSIPL++
#   reference implementation and is not available under the BSD license.
#
#   @file    make_images.sh
#   @author  Don McCoy
#   @date    2008-08-19
#   @brief   VSIPL++ implementation of SSCA #3: Kernel 1, Image Formation

# This script creates images from the input and output raw data files produced 
# during Kernel 1 processing.  It also creates intermediate images from the
# VSIPL++ views that are saved when VERBOSE is defined in kernel1.hpp (these
# are helpful in diagnosing problems and/or providing visual feedback as to
# what is occuring during each stage of processing).
#
# Requires 'rawtopgm' and 'viewtoraw' (the source for the latter is included).

# Parameters
#   dir    The directory where the image files are located (data1, data3, ...)
#   n      Input image rows
#   mc     Input image columns
#   m      Output image rows
#   nx     Output image columns

# Usage
#   ./make_images.sh DIR N MC M NX

dir=$1
n=$2
mc=$3
m=$4
nx=$5

echo "Converting to greyscale png..."
./viewtopng $dir/sar.view $dir/p00_sar.png $n $mc
if [ -f $dir/p62_s_filt.view ]; then 
    ./viewtopng -r $dir/p62_s_filt.view $dir/p62_s_filt.png $n $mc
fi
if [ -f $dir/p76_fs_ref.view ]; then 
    ./viewtopng -r $dir/p76_fs_ref.view $dir/p76_fs_ref.png $n $m
fi
if [ -f $dir/p77_fsm_half_fc.view ]; then 
    ./viewtopng $dir/p77_fsm_half_fc.view $dir/p77_fsm_half_fc.png $n $m
fi
if [ -f $dir/p77_fsm_row.view ]; then 
    ./viewtopng $dir/p77_fsm_row.view $dir/p77_fsm_row.png $n $m
fi
if [ -f $dir/p92_F.view ]; then 
    ./viewtopng $dir/p92_F.view $dir/p92_F.png $nx $m
fi
./viewtopng -s $dir/image.view $dir/p95_image.png $m $nx

