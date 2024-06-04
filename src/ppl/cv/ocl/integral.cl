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
#define BLOCK_X 128
#define BLOCK_Y 8

#if defined(VERTICAL_F32) || defined(ALL_KERNELS)
__kernel
void verticalF32Kernel(global const float* src, int src_rows, int src_cols, int src_stride,
                    global float* dst, int dst_rows, int dst_stride) {
  local float data[BLOCK_Y][BLOCK_X * 2];
  local float block_sum[BLOCK_X * 2];

  int element_x = get_global_id(0);
  element_x = element_x << 1;
  int element_y = get_local_id(1);
  if (element_x >= src_cols || element_y >= src_rows) {
    return;
  }

  src_stride = src_stride >> 2;

  global float* input = (global float*)((global uchar*)src + mul24(element_y, src_stride));
  global float* output;
  dst_stride = dst_stride >> 2;
  output = (global float*)((global uchar*)dst + mul24(element_y + 1, dst_stride));

  if (get_local_id(1) == 0) {
    dst[element_x] = 0;
    dst[element_x + 1] = 0;
    if (element_x == src_cols - 1 || element_x == src_cols - 2) {
      dst[src_cols] = 0;
    }
  }

  int threadIdx_x = (get_local_id(0) << 1);
  if (get_local_id(1) == 0) {
    block_sum[threadIdx_x] = 0;
    block_sum[threadIdx_x + 1] = 0;
  }

  float element_sum0, element_sum1;
  while (element_y < src_rows) {
    data[get_local_id(1)][threadIdx_x]     = input[element_x];
    data[get_local_id(1)][threadIdx_x + 1] = input[element_x + 1];
    barrier(CLK_LOCAL_MEM_FENCE);

    element_sum0 = block_sum[threadIdx_x] + data[get_local_id(1)][threadIdx_x];
    element_sum1 = block_sum[threadIdx_x + 1] +
                   data[get_local_id(1)][threadIdx_x + 1];
    for (int i = 0; i < get_local_id(1); i++) {
      element_sum0 += data[i][threadIdx_x];
      element_sum1 += data[i][threadIdx_x + 1];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (element_x == 0 && src_rows != dst_rows) {
      output[0] = 0;
    }
    if (element_x < src_cols - 1) {
      output[element_x + 1] = element_sum0;
      output[element_x + 2] = element_sum1;
    }
    else {
      output[element_x + 1] = element_sum0;
    }

    if (get_local_id(1) == get_local_size(1) - 1) {
      block_sum[threadIdx_x]     = element_sum0;
      block_sum[threadIdx_x + 1] = element_sum1;
    }

    element_y += get_local_size(1);
    input  += get_local_size(1) * src_stride;
    output += get_local_size(1) * dst_stride;
  }
}
#endif

#if defined(HORIZONTAL_F32) || defined(ALL_KERNELS)
__kernel
void horizontalF32Kernel(int src_rows, int src_cols, global float* dst, int dst_rows,
                      int dst_stride) {
  local float data[BLOCK_X];
  local float block_sum;

  int threadIdx_x = get_local_id(0);
  int element_x = threadIdx_x;
  int element_y = get_global_id(0)/BLOCK_X;
  if (element_y >= src_rows) {
    return;
  }

  global float* output;
  output = (global float*)((global uchar*)dst + mul24(element_y + 1, dst_stride));

  if (element_x == 0) {
    block_sum = 0;
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  float element_sum;
  while (element_x < src_cols) {
    data[threadIdx_x] = output[element_x + 1];
    barrier(CLK_LOCAL_MEM_FENCE);

    element_sum = block_sum + data[threadIdx_x];
    for (int i = 0; i < threadIdx_x; i++) {
      element_sum += data[i];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (element_x == 0 && src_rows != dst_rows) {
      output[0] = 0;
    }
    output[element_x + 1] = element_sum;

    if (threadIdx_x == BLOCK_X - 1) {
      block_sum = element_sum;
    }
    element_x += BLOCK_X;
  }
}
#endif