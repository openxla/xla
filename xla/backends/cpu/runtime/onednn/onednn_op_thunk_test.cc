/* Copyright 2025 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifdef INTEL_MKL

#include "xla/backends/cpu/runtime/onednn/onednn_op_thunk.h"

#include "gtest/gtest.h"
#include "xla/backends/cpu/runtime/thunk_testlib.h"
#include "xla/literal_util.h"
#include "xla/shape_util.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/platform/threadpool.h"

#define EIGEN_USE_THREADS
#include "unsupported/Eigen/CXX11/Tensor"

namespace xla::cpu {
namespace {
using dnnl::engine;
using dnnl::stream;

TEST(OneDnnOpThunkTest, SimpleOneDnnMatMulThunk) {
  // Set up a thread pool for parallel execution
  tsl::thread::ThreadPool threads(tsl::Env::Default(), "test", 8);
  Eigen::ThreadPoolDevice device(threads.AsEigenThreadPool(),
                                 threads.NumThreads());

  // Define shapes for lhs (2x3), rhs (3x2), and output (2x2)
  Shape lhs_shape = ShapeUtil::MakeShape(F32, {2, 3});
  Shape rhs_shape = ShapeUtil::MakeShape(F32, {3, 2});
  Shape out_shape = ShapeUtil::MakeShape(F32, {2, 2});

  // Prepare dummy data (row-major):
  // A = [ [1, 2, 3],
  //       [4, 5, 6] ]
  // B = [ [7,  8],
  //       [9, 10],
  //       [11,12] ]
  // C = A * B =
  //     [ [58, 64],
  //       [139,154] ]
  std::vector<float> input_a = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f};
  std::vector<float> input_b = {7.f, 8.f, 9.f, 10.f, 11.f, 12.f};
  std::vector<float> output(4, 0.f);

  // Create Literals from data
  Literal lhs_literal = LiteralUtil::CreateR2FromArray2D<float>(
      Array2D<float>({{1.f, 2.f, 3.f}, {4.f, 5.f, 6.f}}));
  Literal rhs_literal = LiteralUtil::CreateR2FromArray2D<float>(
      Array2D<float>({{7.f, 8.f}, {9.f, 10.f}, {11.f, 12.f}}));
  Literal out_literal = LiteralUtil::CreateR2FromArray2D<float>(
      Array2D<float>({{0.f, 0.f}, {0.f, 0.f}}));

  // Create buffer allocations and slices
  auto lhs_alloc = CreateBufferAllocation(0, lhs_literal);
  auto rhs_alloc = CreateBufferAllocation(1, rhs_literal);
  auto out_alloc = CreateBufferAllocation(2, out_literal);

  auto lhs_slice = CreateBufferAllocationSlice(lhs_alloc);
  auto rhs_slice = CreateBufferAllocationSlice(rhs_alloc);
  auto out_slice = CreateBufferAllocationSlice(out_alloc);

  BufferAllocations allocations =
      CreateBufferAllocations(lhs_literal, rhs_literal, out_literal);

  // Set up op_buffers
  OneDnnOpThunk::OpBuffers op_buffers;
  op_buffers.arguments_buffers = {lhs_slice, rhs_slice};
  op_buffers.arguments_shapes = {lhs_shape, rhs_shape};
  op_buffers.results_buffers = {out_slice};
  op_buffers.results_shapes = {out_shape};

  // Create thunk (matmul)
  TF_ASSERT_OK_AND_ASSIGN(
      auto thunk,
      OneDnnOpThunk::Create("__onednn$matmul", Thunk::Info(), op_buffers, {}));

  // Set up execute params
  Thunk::ExecuteParams params;
  params.buffer_allocations = &allocations;
  params.intra_op_threadpool = &device;

  // Execute the thunk
  auto exec_event = thunk->Execute(params);
  tsl::BlockUntilReady(exec_event);
  ASSERT_FALSE(exec_event.IsError()) << "OneDnnOpThunk execution failed";

  // Expected output literal
  Literal expected = LiteralUtil::CreateR2FromArray2D<float>(
      Array2D<float>({{58.f, 64.f}, {139.f, 154.f}}));

  // Load output and verify
  EXPECT_EQ(out_literal, expected);
}

// A very small 2D NHWC convolution test exercising the oneDNN convolution path
// through OneDnnOpThunk. We build a 1x1 kernel with weight=2 so the expected
// output is input * 2. The configuration mirrors the encoding used in
// oneDNN convolution runtime (spatial dims stored 1-based; window parameters
// stored with offsets that the runtime subtracts: stride values stored as
// (actual+1), pads as (actual+1), dilations as (actual+2)).
TEST(OneDnnOpThunkTest, SimpleOneDnnConvolutionThunk) {
  tsl::thread::ThreadPool threads(tsl::Env::Default(), "test", 4);
  Eigen::ThreadPoolDevice device(threads.AsEigenThreadPool(),
                                 threads.NumThreads());

  // Input: N=1,H=2,W=2,C=1  (NHWC)
  // Weights: KH=1, KW=1, IC=1, OC=1
  // Output: N=1,H=2,W=2,C=1
  Shape input_shape = ShapeUtil::MakeShape(F32, {1, 2, 2, 1});
  Shape weight_shape = ShapeUtil::MakeShape(F32, {1, 1, 1, 1});
  Shape output_shape = ShapeUtil::MakeShape(F32, {1, 2, 2, 1});

  // Input data
  Literal input_literal = LiteralUtil::CreateR4FromArray4D<float>(
      Array4D<float>(1, 2, 2, 1,
                     {/*h0w0*/ 1.f, /*h0w1*/ 2.f, /*h1w0*/ 3.f, /*h1w1*/ 4.f}));
  // Weight = 2
  Literal weight_literal = LiteralUtil::CreateR4FromArray4D<float>(
      Array4D<float>(1, 1, 1, 1, {2.f}));
  // Output buffer init zeros
  Literal output_literal = LiteralUtil::CreateR4FromArray4D<float>(
      Array4D<float>(1, 2, 2, 1, {0.f, 0.f, 0.f, 0.f}));

  // Expected = input * 2
  Literal expected_literal = LiteralUtil::CreateR4FromArray4D<float>(
      Array4D<float>(1, 2, 2, 1, {2.f, 4.f, 6.f, 8.f}));

  // Buffer allocations
  auto input_alloc = CreateBufferAllocation(0, input_literal);
  auto weight_alloc = CreateBufferAllocation(1, weight_literal);
  auto output_alloc = CreateBufferAllocation(2, output_literal);

  auto input_slice = CreateBufferAllocationSlice(input_alloc);
  auto weight_slice = CreateBufferAllocationSlice(weight_alloc);
  auto output_slice = CreateBufferAllocationSlice(output_alloc);

  BufferAllocations allocations =
      CreateBufferAllocations(input_literal, weight_literal, output_literal);

  // Build a minimal OneDnnConvolutionConfig proto.
  OneDnnConvolutionConfig conv_config;
  conv_config.set_dims(4);  // rank
  // Input layout NHWC: batch=0, feature=3, spatial (H,W) = (1,2) zero-based.
  auto* inp = conv_config.mutable_input();
  inp->set_dims(4);
  auto* inp_data = inp->mutable_data();
  inp_data->set_batch_dim(0);
  inp_data->set_feature_dim(3);
  // Spatial dims stored as one-based (so 1->2, 2->3).
  inp_data->add_spatial_dims(2);
  inp_data->add_spatial_dims(3);

  // Kernel layout assumed HWIO (H,W,In,Out):
  auto* ker = conv_config.mutable_kernel();
  ker->set_dims(4);
  auto* filter = ker->mutable_filter();
  filter->set_input_feature_dim(2);   // zero-based index of IC
  filter->set_output_feature_dim(3);  // zero-based index of OC
  // Spatial dims (H,W) one-based: (0->1,1->2) => 1,2
  filter->add_spatial_dims(1);
  filter->add_spatial_dims(2);

  // Output layout NHWC
  auto* out = conv_config.mutable_output();
  out->set_dims(4);
  auto* out_data = out->mutable_data();
  out_data->set_batch_dim(0);
  out_data->set_feature_dim(3);
  out_data->add_spatial_dims(2);
  out_data->add_spatial_dims(3);

  conv_config.set_feature_groups(1);

  // Window parameters: stride=1, pad=0, dilation=1 encoded with offsets.
  auto* win = conv_config.mutable_window();
  // Store (actual + 1) for strides so 2 -> (2 - 1 = 1 real stride).
  win->add_strides(2);
  win->add_strides(2);
  // Pads store (actual +1) so 1 -> 0 actual pad.
  win->add_pad_left(1);
  win->add_pad_left(1);
  win->add_pad_right(1);
  win->add_pad_right(1);
  // Dilations store (actual +2) so 3 -> 1 actual dilation.
  win->add_window_dilations(3);
  win->add_window_dilations(3);

  // Set up op buffers
  OneDnnOpThunk::OpBuffers op_buffers;
  op_buffers.arguments_buffers = {input_slice, weight_slice};
  op_buffers.arguments_shapes = {input_shape, weight_shape};
  op_buffers.results_buffers = {output_slice};
  op_buffers.results_shapes = {output_shape};

  // Wrap config in variant
  OneDnnOpThunk::OneDnnOpConfig config_variant = conv_config;

  TF_ASSERT_OK_AND_ASSIGN(
      auto thunk, OneDnnOpThunk::Create("__onednn$convolution", Thunk::Info(),
                                        op_buffers, config_variant));

  Thunk::ExecuteParams params;
  params.buffer_allocations = &allocations;
  params.intra_op_threadpool = &device;

  auto exec_event = thunk->Execute(params);
  tsl::BlockUntilReady(exec_event);
  ASSERT_FALSE(exec_event.IsError()) << "OneDnnOpThunk convolution execution failed";

  EXPECT_EQ(output_literal, expected_literal);
}

}  // namespace
}  // namespace xla::cpu

#endif  // INTEL_MKL
