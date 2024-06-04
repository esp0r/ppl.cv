/**
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements. See the NOTICE file distributed with this
 * work for additional information regarding copyright ownership. The ASF
 * licenses this file to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance with the
 * License. You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 */

#include "ppl/cv/ocl/integral.h"

#include "ppl/common/ocl/pplopencl.h"
#include "utility/utility.hpp"

#include "kernels/integral.cl"

using namespace ppl::common;
using namespace ppl::common::ocl;

namespace ppl {
namespace cv {
namespace ocl {

#define BLOCK_X 128
#define BLOCK_Y 8

RetCode integralF32(const cl_mem src, int src_rows, int src_cols, int channels,
                 int src_stride, cl_mem dst, int dst_rows, int dst_cols,
                 int dst_stride, cl_command_queue queue) {
  PPL_ASSERT(src != nullptr);
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(src_rows >= 1 && src_cols >= 1);
  PPL_ASSERT(dst_rows >= 1 && dst_cols >= 1);
  PPL_ASSERT(dst_rows == src_rows + 1);
  PPL_ASSERT(dst_cols == src_cols + 1);
  PPL_ASSERT(channels == 1);
  PPL_ASSERT(src_stride >= src_cols * (int)sizeof(float));
  PPL_ASSERT(dst_stride >= dst_cols * (int)sizeof(float));

  FrameChain* frame_chain = getSharedFrameChain();
  frame_chain->setProjectName("cv");
  SET_PROGRAM_SOURCE(frame_chain, integral);

  size_t local_size[] = {kBlockDimX1, kBlockDimY1};
  size_t global_size[] = {(size_t)divideUp(src_cols, 2, 1), (size_t)src_rows};

  frame_chain->setCompileOptions("-D VERTICAL_F32");
  runOclKernel(frame_chain, "verticalF32Kernel", 2, global_size, local_size, src, src_rows, src_cols,
               src_stride, dst, dst_rows, dst_stride);

  size_t local_size_ = BLOCK_X;
  size_t global_size_  = src_rows * BLOCK_X;

  frame_chain->setCompileOptions("-D HORIZONTAL_F32");  
  runOclKernel(frame_chain, "horizontalF32Kernel", 1, &global_size_, &local_size_, src_rows, src_cols, dst,
                                                        dst_rows, dst_stride);

  return RC_SUCCESS;
}


template <>
RetCode Integral<float, float, 1>(cl_command_queue queue,
                                  int inHeight,
                                  int inWidth,
                                  int inWidthStride,
                                  const cl_mem inData,
                                  int outHeight,
                                  int outWidth,
                                  int outWidthStride,
                                  cl_mem outData) {
  inWidthStride  *= sizeof(float);
  outWidthStride *= sizeof(float);
  RetCode code = integralF32(inData, inHeight, inWidth, 1, inWidthStride, outData,
                          outHeight, outWidth, outWidthStride, queue);

  return code;
}

}  // ocl
}  // cv
}  // ppl
