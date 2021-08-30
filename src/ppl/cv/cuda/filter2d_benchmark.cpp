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

#include "ppl/cv/cuda/filter2d.h"

#include <time.h>
#include <sys/time.h>

#include "opencv2/opencv.hpp"
#include "benchmark/benchmark.h"

#include "ppl/cv/debug.h"
#include "infrastructure.hpp"

using namespace ppl::cv;
using namespace ppl::cv::cuda;
using namespace ppl::cv::debug;

template <typename Tsrc, typename Tdst, int channels, int ksize,
          BorderType border_type>
void BM_Filter2D_ppl_cuda(benchmark::State &state) {
  int width  = state.range(0);
  int height = state.range(1);
  cv::Mat src, kernel;
  src = createSourceImage(height, width,
                          CV_MAKETYPE(cv::DataType<Tsrc>::depth, channels));
  kernel = createSourceImage(1, ksize * ksize,
                             CV_MAKETYPE(cv::DataType<float>::depth, 1));
  cv::Mat dst(height, width,
              CV_MAKETYPE(cv::DataType<Tdst>::depth, channels));
  cv::cuda::GpuMat gpu_src(src);
  cv::cuda::GpuMat gpu_kernel(kernel);
  cv::cuda::GpuMat gpu_dst(dst);

  float delta = 0.f;

  int iterations = 1000;
  struct timeval start, end;

  // warm up the GPU
  for (int i = 0; i < iterations; i++) {
    Filter2D<Tsrc, channels>(0, gpu_src.rows, gpu_src.cols,
        gpu_src.step / sizeof(Tsrc), (Tsrc*)gpu_src.data, ksize,
        (float*)gpu_kernel.data, gpu_dst.step / sizeof(Tdst),
        (Tdst*)gpu_dst.data, delta, border_type);
  }
  cudaDeviceSynchronize();

  for (auto _ : state) {
    gettimeofday(&start, NULL);
    for (int i = 0; i < iterations; i++) {
      Filter2D<Tsrc, channels>(0, gpu_src.rows, gpu_src.cols,
          gpu_src.step / sizeof(Tsrc), (Tsrc*)gpu_src.data, ksize,
          (float*)gpu_kernel.data, gpu_dst.step / sizeof(Tdst),
          (Tdst*)gpu_dst.data, delta, border_type);
    }
    cudaDeviceSynchronize();
    gettimeofday(&end, NULL);
    int time = ((end.tv_sec * 1000000 + end.tv_usec) -
                (start.tv_sec * 1000000 + start.tv_usec)) / iterations;
    state.SetIterationTime(time * 1e-6);
  }
  state.SetItemsProcessed(state.iterations() * 1);
}

template <typename Tsrc, typename Tdst, int channels, int ksize,
          BorderType border_type>
void BM_Filter2D_opencv_cuda(benchmark::State &state) {
  int width  = state.range(0);
  int height = state.range(1);
  cv::Mat src, kernel;
  src = createSourceImage(height, width,
                          CV_MAKETYPE(cv::DataType<Tsrc>::depth, channels));
  kernel = createSourceImage(ksize, ksize,
                             CV_MAKETYPE(cv::DataType<float>::depth, 1));
  cv::Mat dst(height, width,
              CV_MAKETYPE(cv::DataType<Tdst>::depth, channels));
  cv::cuda::GpuMat gpu_src(src);
  cv::cuda::GpuMat gpu_kernel(kernel);
  cv::cuda::GpuMat gpu_dst(dst);

  cv::BorderTypes border = cv::BORDER_DEFAULT;
  if (border_type == BORDER_TYPE_REPLICATE) {
    border = cv::BORDER_REPLICATE;
  }
  else if (border_type == BORDER_TYPE_REFLECT) {
    border = cv::BORDER_REFLECT;
  }
  else if (border_type == BORDER_TYPE_REFLECT_101) {
    border = cv::BORDER_REFLECT_101;
  }
  else {
  }

  int iterations = 1000;
  struct timeval start, end;

  // warm up the GPU
  for (int i = 0; i < iterations; i++) {
    cv::Ptr<cv::cuda::Filter> linearfilter =
      cv::cuda::createLinearFilter(gpu_src.type(), gpu_dst.type(), kernel,
                                   cv::Point(-1, -1), border);
    linearfilter->apply(gpu_src, gpu_dst);
  }
  cudaDeviceSynchronize();

  for (auto _ : state) {
    gettimeofday(&start, NULL);
    for (int i = 0; i < iterations; i++) {
      cv::Ptr<cv::cuda::Filter> linearfilter =
        cv::cuda::createLinearFilter(gpu_src.type(), gpu_dst.type(), kernel,
                                    cv::Point(-1, -1), border);
      linearfilter->apply(gpu_src, gpu_dst);
    }
    cudaDeviceSynchronize();
    gettimeofday(&end, NULL);
    int time = ((end.tv_sec * 1000000 + end.tv_usec) -
                (start.tv_sec * 1000000 + start.tv_usec)) / iterations;
    state.SetIterationTime(time * 1e-6);
  }
  state.SetItemsProcessed(state.iterations() * 1);
}

template <typename Tsrc, typename Tdst, int channels, int ksize,
          BorderType border_type>
void BM_Filter2D_opencv_x86_cuda(benchmark::State &state) {
  int width  = state.range(0);
  int height = state.range(1);
  cv::Mat src, kernel;
  src = createSourceImage(height, width,
                          CV_MAKETYPE(cv::DataType<Tsrc>::depth, channels));
  kernel = createSourceImage(1, ksize * ksize,
                             CV_MAKETYPE(cv::DataType<float>::depth, 1));
  cv::Mat dst(height, width,
              CV_MAKETYPE(cv::DataType<Tdst>::depth, channels));

  cv::BorderTypes border = cv::BORDER_DEFAULT;
  if (border_type == BORDER_TYPE_REPLICATE) {
    border = cv::BORDER_REPLICATE;
  }
  else if (border_type == BORDER_TYPE_REFLECT) {
    border = cv::BORDER_REFLECT;
  }
  else if (border_type == BORDER_TYPE_REFLECT_101) {
    border = cv::BORDER_REFLECT_101;
  }
  else {
  }

  for (auto _ : state) {
    cv::filter2D(src, dst, dst.depth(), kernel, cv::Point(-1, -1), 0,
                 border);
  }
  state.SetItemsProcessed(state.iterations() * 1);
}

#define RUN_BENCHMARK0(src_type, dst_type, ksize, border_type, width, height)  \
BENCHMARK_TEMPLATE(BM_Filter2D_opencv_x86_cuda, src_type, dst_type, c1,        \
                   ksize, border_type)->Args({width, height});                 \
BENCHMARK_TEMPLATE(BM_Filter2D_ppl_cuda, src_type, dst_type, c1, ksize,        \
                   border_type)->Args({width, height})->UseManualTime()->      \
                   Iterations(10);                                             \
BENCHMARK_TEMPLATE(BM_Filter2D_opencv_x86_cuda, src_type, dst_type, c3,        \
                   ksize, border_type)->Args({width, height});                 \
BENCHMARK_TEMPLATE(BM_Filter2D_ppl_cuda, src_type, dst_type, c3, ksize,        \
                   border_type)->Args({width, height})->UseManualTime()->      \
                   Iterations(10);                                             \
BENCHMARK_TEMPLATE(BM_Filter2D_opencv_x86_cuda, src_type, dst_type, c4,        \
                   ksize, border_type)->Args({width, height});                 \
BENCHMARK_TEMPLATE(BM_Filter2D_ppl_cuda, src_type, dst_type, c4, ksize,        \
                   border_type)->Args({width, height})->UseManualTime()->      \
                   Iterations(10);

// RUN_BENCHMARK0(uchar, uchar, 5, BORDER_TYPE_REPLICATE, 640, 480)
// RUN_BENCHMARK0(uchar, uchar, 5, BORDER_TYPE_REFLECT, 640, 480)
// RUN_BENCHMARK0(uchar, uchar, 5, BORDER_TYPE_REFLECT_101, 640, 480)
// RUN_BENCHMARK0(uchar, uchar, 17, BORDER_TYPE_REPLICATE, 640, 480)
// RUN_BENCHMARK0(uchar, uchar, 17, BORDER_TYPE_REFLECT, 640, 480)
// RUN_BENCHMARK0(uchar, uchar, 17, BORDER_TYPE_REFLECT_101, 640, 480)
// RUN_BENCHMARK0(uchar, uchar, 25, BORDER_TYPE_REPLICATE, 640, 480)
// RUN_BENCHMARK0(uchar, uchar, 25, BORDER_TYPE_REFLECT, 640, 480)
// RUN_BENCHMARK0(uchar, uchar, 25, BORDER_TYPE_REFLECT_101, 640, 480)
// RUN_BENCHMARK0(uchar, uchar, 31, BORDER_TYPE_REPLICATE, 640, 480)
// RUN_BENCHMARK0(uchar, uchar, 31, BORDER_TYPE_REFLECT, 640, 480)
// RUN_BENCHMARK0(uchar, uchar, 31, BORDER_TYPE_REFLECT_101, 640, 480)

// RUN_BENCHMARK0(float, float, 5, BORDER_TYPE_REPLICATE, 640, 480)
// RUN_BENCHMARK0(float, float, 5, BORDER_TYPE_REFLECT, 640, 480)
// RUN_BENCHMARK0(float, float, 5, BORDER_TYPE_REFLECT_101, 640, 480)
// RUN_BENCHMARK0(float, float, 17, BORDER_TYPE_REPLICATE, 640, 480)
// RUN_BENCHMARK0(float, float, 17, BORDER_TYPE_REFLECT, 640, 480)
// RUN_BENCHMARK0(float, float, 17, BORDER_TYPE_REFLECT_101, 640, 480)
// RUN_BENCHMARK0(float, float, 25, BORDER_TYPE_REPLICATE, 640, 480)
// RUN_BENCHMARK0(float, float, 25, BORDER_TYPE_REFLECT, 640, 480)
// RUN_BENCHMARK0(float, float, 25, BORDER_TYPE_REFLECT_101, 640, 480)
// RUN_BENCHMARK0(float, float, 31, BORDER_TYPE_REPLICATE, 640, 480)
// RUN_BENCHMARK0(float, float, 31, BORDER_TYPE_REFLECT, 640, 480)
// RUN_BENCHMARK0(float, float, 31, BORDER_TYPE_REFLECT_101, 640, 480)

#define RUN_BENCHMARK1(src_type, dst_type, ksize, border_type, width, height)  \
BENCHMARK_TEMPLATE(BM_Filter2D_opencv_cuda, src_type, dst_type, c1,            \
                   ksize, border_type)->Args({width, height})->                \
                   UseManualTime()->Iterations(10);                            \
BENCHMARK_TEMPLATE(BM_Filter2D_ppl_cuda, src_type, dst_type, c1, ksize,        \
                   border_type)->Args({width, height})->UseManualTime()->      \
                   Iterations(10);                                             \
BENCHMARK_TEMPLATE(BM_Filter2D_opencv_cuda, src_type, dst_type, c4,            \
                   ksize, border_type)->Args({width, height})->                \
                   UseManualTime()->Iterations(10);                            \
BENCHMARK_TEMPLATE(BM_Filter2D_ppl_cuda, src_type, dst_type, c4, ksize,        \
                   border_type)->Args({width, height})->UseManualTime()->      \
                   Iterations(10);

// RUN_BENCHMARK1(uchar, uchar, 5, BORDER_TYPE_REPLICATE, 640, 480)
// RUN_BENCHMARK1(uchar, uchar, 5, BORDER_TYPE_REFLECT, 640, 480)
// RUN_BENCHMARK1(uchar, uchar, 5, BORDER_TYPE_REFLECT_101, 640, 480)
// RUN_BENCHMARK1(uchar, uchar, 17, BORDER_TYPE_REPLICATE, 640, 480)
// RUN_BENCHMARK1(uchar, uchar, 17, BORDER_TYPE_REFLECT, 640, 480)
// RUN_BENCHMARK1(uchar, uchar, 17, BORDER_TYPE_REFLECT_101, 640, 480)
// RUN_BENCHMARK1(uchar, uchar, 25, BORDER_TYPE_REPLICATE, 640, 480)
// RUN_BENCHMARK1(uchar, uchar, 25, BORDER_TYPE_REFLECT, 640, 480)
// RUN_BENCHMARK1(uchar, uchar, 25, BORDER_TYPE_REFLECT_101, 640, 480)
// RUN_BENCHMARK1(uchar, uchar, 31, BORDER_TYPE_REPLICATE, 640, 480)
// RUN_BENCHMARK1(uchar, uchar, 31, BORDER_TYPE_REFLECT, 640, 480)
// RUN_BENCHMARK1(uchar, uchar, 31, BORDER_TYPE_REFLECT_101, 640, 480)

// RUN_BENCHMARK1(float, float, 5, BORDER_TYPE_REPLICATE, 640, 480)
// RUN_BENCHMARK1(float, float, 5, BORDER_TYPE_REFLECT, 640, 480)
// RUN_BENCHMARK1(float, float, 5, BORDER_TYPE_REFLECT_101, 640, 480)
// RUN_BENCHMARK1(float, float, 17, BORDER_TYPE_REPLICATE, 640, 480)
// RUN_BENCHMARK1(float, float, 17, BORDER_TYPE_REFLECT, 640, 480)
// RUN_BENCHMARK1(float, float, 17, BORDER_TYPE_REFLECT_101, 640, 480)
// RUN_BENCHMARK1(float, float, 25, BORDER_TYPE_REPLICATE, 640, 480)
// RUN_BENCHMARK1(float, float, 25, BORDER_TYPE_REFLECT, 640, 480)
// RUN_BENCHMARK1(float, float, 25, BORDER_TYPE_REFLECT_101, 640, 480)
// RUN_BENCHMARK1(float, float, 31, BORDER_TYPE_REPLICATE, 640, 480)
// RUN_BENCHMARK1(float, float, 31, BORDER_TYPE_REFLECT, 640, 480)
// RUN_BENCHMARK1(float, float, 31, BORDER_TYPE_REFLECT_101, 640, 480)

#define RUN_OPENCV_TYPE_FUNCTIONS(src_type, dst_type, ksize, border_type)      \
BENCHMARK_TEMPLATE(BM_Filter2D_opencv_x86_cuda, src_type, dst_type, c1, ksize, \
                   border_type)->Args({640, 480});                             \
BENCHMARK_TEMPLATE(BM_Filter2D_opencv_x86_cuda, src_type, dst_type, c3, ksize, \
                   border_type)->Args({640, 480});                             \
BENCHMARK_TEMPLATE(BM_Filter2D_opencv_x86_cuda, src_type, dst_type, c4, ksize, \
                   border_type)->Args({640, 480});

#define RUN_PPL_CV_TYPE_FUNCTIONS(src_type, dst_type, ksize, border_type)      \
BENCHMARK_TEMPLATE(BM_Filter2D_ppl_cuda, src_type, dst_type, c1, ksize,        \
                   border_type)->Args({640, 480})->                            \
                   UseManualTime()->Iterations(10);                            \
BENCHMARK_TEMPLATE(BM_Filter2D_ppl_cuda, src_type, dst_type, c3, ksize,        \
                   border_type)->Args({640, 480})->                            \
                   UseManualTime()->Iterations(10);                            \
BENCHMARK_TEMPLATE(BM_Filter2D_ppl_cuda, src_type, dst_type, c4, ksize,        \
                   border_type)->Args({640, 480})->                            \
                   UseManualTime()->Iterations(10);

RUN_OPENCV_TYPE_FUNCTIONS(uchar, uchar, 5, BORDER_TYPE_REPLICATE)
RUN_OPENCV_TYPE_FUNCTIONS(uchar, uchar, 5, BORDER_TYPE_REFLECT)
RUN_OPENCV_TYPE_FUNCTIONS(uchar, uchar, 5, BORDER_TYPE_REFLECT_101)
RUN_OPENCV_TYPE_FUNCTIONS(float, float, 5, BORDER_TYPE_REPLICATE)
RUN_OPENCV_TYPE_FUNCTIONS(float, float, 5, BORDER_TYPE_REFLECT)
RUN_OPENCV_TYPE_FUNCTIONS(float, float, 5, BORDER_TYPE_REFLECT_101)

RUN_OPENCV_TYPE_FUNCTIONS(uchar, uchar, 17, BORDER_TYPE_REPLICATE)
RUN_OPENCV_TYPE_FUNCTIONS(uchar, uchar, 17, BORDER_TYPE_REFLECT)
RUN_OPENCV_TYPE_FUNCTIONS(uchar, uchar, 17, BORDER_TYPE_REFLECT_101)
RUN_OPENCV_TYPE_FUNCTIONS(float, float, 17, BORDER_TYPE_REPLICATE)
RUN_OPENCV_TYPE_FUNCTIONS(float, float, 17, BORDER_TYPE_REFLECT)
RUN_OPENCV_TYPE_FUNCTIONS(float, float, 17, BORDER_TYPE_REFLECT_101)

RUN_OPENCV_TYPE_FUNCTIONS(uchar, uchar, 25, BORDER_TYPE_REPLICATE)
RUN_OPENCV_TYPE_FUNCTIONS(uchar, uchar, 25, BORDER_TYPE_REFLECT)
RUN_OPENCV_TYPE_FUNCTIONS(uchar, uchar, 25, BORDER_TYPE_REFLECT_101)
RUN_OPENCV_TYPE_FUNCTIONS(float, float, 25, BORDER_TYPE_REPLICATE)
RUN_OPENCV_TYPE_FUNCTIONS(float, float, 25, BORDER_TYPE_REFLECT)
RUN_OPENCV_TYPE_FUNCTIONS(float, float, 25, BORDER_TYPE_REFLECT_101)

RUN_OPENCV_TYPE_FUNCTIONS(uchar, uchar, 31, BORDER_TYPE_REPLICATE)
RUN_OPENCV_TYPE_FUNCTIONS(uchar, uchar, 31, BORDER_TYPE_REFLECT)
RUN_OPENCV_TYPE_FUNCTIONS(uchar, uchar, 31, BORDER_TYPE_REFLECT_101)
RUN_OPENCV_TYPE_FUNCTIONS(float, float, 31, BORDER_TYPE_REPLICATE)
RUN_OPENCV_TYPE_FUNCTIONS(float, float, 31, BORDER_TYPE_REFLECT)
RUN_OPENCV_TYPE_FUNCTIONS(float, float, 31, BORDER_TYPE_REFLECT_101)

RUN_PPL_CV_TYPE_FUNCTIONS(uchar, uchar, 5, BORDER_TYPE_REPLICATE)
RUN_PPL_CV_TYPE_FUNCTIONS(uchar, uchar, 5, BORDER_TYPE_REFLECT)
RUN_PPL_CV_TYPE_FUNCTIONS(uchar, uchar, 5, BORDER_TYPE_REFLECT_101)
RUN_PPL_CV_TYPE_FUNCTIONS(float, float, 5, BORDER_TYPE_REPLICATE)
RUN_PPL_CV_TYPE_FUNCTIONS(float, float, 5, BORDER_TYPE_REFLECT)
RUN_PPL_CV_TYPE_FUNCTIONS(float, float, 5, BORDER_TYPE_REFLECT_101)

RUN_PPL_CV_TYPE_FUNCTIONS(uchar, uchar, 17, BORDER_TYPE_REPLICATE)
RUN_PPL_CV_TYPE_FUNCTIONS(uchar, uchar, 17, BORDER_TYPE_REFLECT)
RUN_PPL_CV_TYPE_FUNCTIONS(uchar, uchar, 17, BORDER_TYPE_REFLECT_101)
RUN_PPL_CV_TYPE_FUNCTIONS(float, float, 17, BORDER_TYPE_REPLICATE)
RUN_PPL_CV_TYPE_FUNCTIONS(float, float, 17, BORDER_TYPE_REFLECT)
RUN_PPL_CV_TYPE_FUNCTIONS(float, float, 17, BORDER_TYPE_REFLECT_101)

RUN_PPL_CV_TYPE_FUNCTIONS(uchar, uchar, 25, BORDER_TYPE_REPLICATE)
RUN_PPL_CV_TYPE_FUNCTIONS(uchar, uchar, 25, BORDER_TYPE_REFLECT)
RUN_PPL_CV_TYPE_FUNCTIONS(uchar, uchar, 25, BORDER_TYPE_REFLECT_101)
RUN_PPL_CV_TYPE_FUNCTIONS(float, float, 25, BORDER_TYPE_REPLICATE)
RUN_PPL_CV_TYPE_FUNCTIONS(float, float, 25, BORDER_TYPE_REFLECT)
RUN_PPL_CV_TYPE_FUNCTIONS(float, float, 25, BORDER_TYPE_REFLECT_101)

RUN_PPL_CV_TYPE_FUNCTIONS(uchar, uchar, 31, BORDER_TYPE_REPLICATE)
RUN_PPL_CV_TYPE_FUNCTIONS(uchar, uchar, 31, BORDER_TYPE_REFLECT)
RUN_PPL_CV_TYPE_FUNCTIONS(uchar, uchar, 31, BORDER_TYPE_REFLECT_101)
RUN_PPL_CV_TYPE_FUNCTIONS(float, float, 31, BORDER_TYPE_REPLICATE)
RUN_PPL_CV_TYPE_FUNCTIONS(float, float, 31, BORDER_TYPE_REFLECT)
RUN_PPL_CV_TYPE_FUNCTIONS(float, float, 31, BORDER_TYPE_REFLECT_101)