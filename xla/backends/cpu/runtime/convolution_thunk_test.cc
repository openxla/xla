/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/backends/cpu/runtime/convolution_thunk.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "absl/base/casts.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/types/span.h"
#include "xla/backends/cpu/runtime/buffer_allocations.h"
#include "xla/backends/cpu/runtime/convolution_dims.h"
#include "xla/backends/cpu/runtime/convolution_lib.h"
#include "xla/backends/cpu/runtime/convolution_thunk_test_util.h"
#include "xla/backends/cpu/runtime/thunk.h"
#include "xla/layout_util.h"
#include "xla/service/buffer_assignment.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_address.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"
#include "xla/tsl/platform/threadpool.h"
#include "xla/xla_data.pb.h"

#define EIGEN_USE_THREADS
#include "Eigen/Core"
#include "unsupported/Eigen/CXX11/Tensor"

namespace xla::cpu {
namespace {

// NOTE: This file serves to verify the basic functionality of the convolution
// thunk. Comprehensive tests cases are common for all backends and are covered
// in xla/tests/convolution_test.cc file.

template <typename T>
class ConvolutionThunkTypedTest : public ::testing::Test {};

using CorrectTypes = ::testing::Types<float, Eigen::half>;
TYPED_TEST_SUITE(ConvolutionThunkTypedTest, CorrectTypes);

template <typename ElementType>
void SuccessfulConvolution(int convolution_rank) {
  ConvolutionThunkBuilder<ElementType> builder(
      ConvolutionDimensions{convolution_rank});
  TF_ASSERT_OK_AND_ASSIGN(auto thunk, builder.Build());
  BufferAllocations allocations = builder.GetAllocations();

  // Execute thunk and wait for completion.
  Thunk::ExecuteParams params;
  params.buffer_allocations = &allocations;

  auto execute_event = thunk->Execute(params);
  tsl::BlockUntilReady(execute_event);

  // Verify that the execution was successful.
  // NOTE: We don't check the correctness of the output here, just that it
  // executes without errors. Numerics is verified in generic convolution tests.
  ASSERT_FALSE(execute_event.IsError()) << execute_event.GetError();
}

TYPED_TEST(ConvolutionThunkTypedTest, SuccessfulConvolution1D) {
  SuccessfulConvolution<TypeParam>(/*convolution_rank=*/1);
}

TYPED_TEST(ConvolutionThunkTypedTest, SuccessfulConvolution2D) {
  SuccessfulConvolution<TypeParam>(/*convolution_rank=*/2);
}

TYPED_TEST(ConvolutionThunkTypedTest, SuccessfulConvolution3D) {
  SuccessfulConvolution<TypeParam>(/*convolution_rank=*/3);
}

TEST(ConvolutionThunkTest, CreationErrorOnUnsupportedType) {
  ConvolutionThunkBuilder<int> builder;

  auto status_or_thunk = builder.Build();
  EXPECT_EQ(status_or_thunk.status().code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(status_or_thunk.status().message(),
              ::testing::HasSubstr("Unsupported element type (S32)"));
}

TEST(ConvolutionThunkTest, CreationErrorOnTooHighConvolutionRank) {
  ConvolutionThunkBuilder<float> builder(
      ConvolutionDimensions(/*convolution_rank=*/4));

  auto status_or_thunk = builder.Build();
  EXPECT_EQ(status_or_thunk.status().code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(status_or_thunk.status().message(),
              ::testing::HasSubstr("Incorrect convolution rank (4)"));
}

TEST(ConvolutionThunkTest, CreationErrorOnTooLowConvolutionRank) {
  ConvolutionThunkBuilder<float> builder(
      ConvolutionDimensions(/*convolution_rank=*/0));

  auto status_or_thunk = builder.Build();
  EXPECT_EQ(status_or_thunk.status().code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(status_or_thunk.status().message(),
              ::testing::HasSubstr("Incorrect convolution rank (0)"));
}

TEST(ConvolutionThunkTest, CreationErrorOnMismatchedKernelBufferRank) {
  ConvolutionDimensions dims_2d(/*convolution_rank=*/2);
  auto input_dims = MakeInputDims(dims_2d);
  auto output_dims = MakeOutputDims(dims_2d);

  // Create kernel buffer with mismatched rank.
  ConvolutionDimensions dims_3d(/*convolution_rank=*/3);
  auto kernel_dims = MakeKernelDims(dims_3d);

  ConvolutionThunkBuilder<float> builder(input_dims, kernel_dims, output_dims);

  auto status_or_thunk = builder.Build();
  EXPECT_EQ(status_or_thunk.status().code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(status_or_thunk.status().message(),
              ::testing::HasSubstr("Buffer ranks mismatch. Input rank (4) vs "
                                   "kernel rank (5) vs output rank (4)"));
}

TEST(ConvolutionThunkTest, CreationErrorOnMismatchedOutputBufferRank) {
  ConvolutionDimensions dims_2d(/*convolution_rank=*/2);
  auto input_dims = MakeInputDims(dims_2d);
  auto kernel_dims = MakeKernelDims(dims_2d);

  // Create output buffer with mismatched rank.
  ConvolutionDimensions dims_3d(/*convolution_rank=*/3);
  auto output_dims = MakeOutputDims(dims_3d);

  ConvolutionThunkBuilder<float> builder(input_dims, kernel_dims, output_dims);
  auto status_or_thunk = builder.Build();

  EXPECT_EQ(status_or_thunk.status().code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(status_or_thunk.status().message(),
              ::testing::HasSubstr("Buffer ranks mismatch. Input rank (4) vs "
                                   "kernel rank (4) vs output rank (5)"));
}

TEST(ConvolutionThunkTest, CreationErrorOnBatchSizeMismatch) {
  ConvolutionDimensions dims;
  dims.batch_size = 1;
  auto input_dims = MakeInputDims(dims);
  auto kernel_dims = MakeKernelDims(dims);

  // Create output with mismatched batch size.
  dims.batch_size = 2;
  auto output_dims = MakeOutputDims(dims);

  ConvolutionThunkBuilder<float> builder(input_dims, kernel_dims, output_dims);
  auto status_or_thunk = builder.Build();

  EXPECT_EQ(status_or_thunk.status().code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(status_or_thunk.status().message(),
              ::testing::HasSubstr(
                  "Batch sizes mismatch. Input batch (1) vs output batch (2)"));
}

TEST(ConvolutionThunkTest, CreationErrorOnOutputChannelsMismatch) {
  ConvolutionDimensions dims;
  dims.output_channels = 3;
  auto input_dims = MakeInputDims(dims);
  auto kernel_dims = MakeKernelDims(dims);

  // Create output with output channels different than the kernel filters count.
  dims.output_channels = 4;
  auto output_dims = MakeOutputDims(dims);

  ConvolutionThunkBuilder<float> builder(input_dims, kernel_dims, output_dims);
  auto status_or_thunk = builder.Build();

  EXPECT_EQ(status_or_thunk.status().code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(
      status_or_thunk.status().message(),
      ::testing::HasSubstr("Output channels mismatch. Kernel filters count (3) "
                           "should be the same as output channels count (4)"));
}

Window MakeCustomWindow(absl::Span<const int64_t> sizes,
                        absl::Span<const int64_t> strides,
                        absl::Span<const std::pair<int64_t, int64_t>> padding,
                        absl::Span<const int64_t> base_dilation,
                        absl::Span<const int64_t> window_dilation,
                        bool reversal = false) {
  Window window;
  for (size_t i = 0; i < sizes.size(); ++i) {
    WindowDimension* d = window.add_dimensions();
    d->set_size(sizes[i]);
    d->set_stride(strides[i]);
    d->set_padding_low(padding[i].first);
    d->set_padding_high(padding[i].second);
    d->set_base_dilation(base_dilation[i]);
    d->set_window_dilation(window_dilation[i]);
    d->set_window_reversal(reversal);
  }
  return window;
}

template <typename T>
std::unique_ptr<T[]> MakeHeapBuffer(size_t count, T fill = T(0.0f)) {
  auto buf = std::make_unique<T[]>(count);
  std::fill_n(buf.get(), count, fill);
  return buf;
}

template <typename T>
absl::Status ExecuteRawConv(const Shape& in_shape, T* in_ptr,
                            const Shape& ker_shape, T* ker_ptr,
                            const Shape& out_shape, T* out_ptr,
                            const ConvolutionDimensionNumbers& dnums,
                            const Window& window,
                            int64_t feature_group_count = 1,
                            ConvolutionThunk::Options options = {true},
                            const Eigen::ThreadPoolDevice* device = nullptr) {
  const int64_t in_bytes = ShapeUtil::ByteSizeOf(in_shape);
  const int64_t ker_bytes = ShapeUtil::ByteSizeOf(ker_shape);
  const int64_t out_bytes = ShapeUtil::ByteSizeOf(out_shape);
  BufferAllocation in_alloc(0, in_bytes, 0);
  BufferAllocation ker_alloc(1, ker_bytes, 0);
  BufferAllocation out_alloc(2, out_bytes, 0);
  ABSL_ASSIGN_OR_RETURN(
      auto thunk,
      ConvolutionThunk::Create(
          {"conv"}, options, BufferAllocation::Slice(&in_alloc, 0, in_bytes),
          in_shape, BufferAllocation::Slice(&ker_alloc, 0, ker_bytes),
          ker_shape, BufferAllocation::Slice(&out_alloc, 0, out_bytes),
          out_shape, dnums, window, feature_group_count));
  const std::array<se::DeviceAddressBase, 3> bufs = {
      se::DeviceAddressBase(in_ptr, in_bytes),
      se::DeviceAddressBase(ker_ptr, ker_bytes),
      se::DeviceAddressBase(out_ptr, out_bytes)};
  BufferAllocations allocs(bufs);
  Thunk::ExecuteParams params;
  params.buffer_allocations = &allocs;
  params.intra_op_threadpool = device;
  auto event = thunk->Execute(params);
  tsl::BlockUntilReady(event);
  return event.IsError() ? event.GetError() : absl::OkStatus();
}

// Bit-exact HLO reference for 2D transposed convolution (stride=1, rhs_dil=1,
// group=1) accumulating in ascending (in_x, in_y) order matching Pack2DPatches.
template <typename T>
std::vector<T> ReferenceTransposedConv2D(const T* lhs, const T* rhs,
                                         int64_t batch, int64_t in_x,
                                         int64_t in_y, int64_t ic, int64_t kx,
                                         int64_t ky, int64_t oc, int64_t out_x,
                                         int64_t out_y, int64_t pad_x_lo,
                                         int64_t pad_y_lo, int64_t dil_x,
                                         int64_t dil_y) {
  std::vector<T> out(batch * out_x * out_y * oc, T(0.0f));
  for (int64_t b = 0; b < batch; ++b) {
    for (int64_t ix = 0; ix < out_x; ++ix) {
      for (int64_t iy = 0; iy < out_y; ++iy) {
        for (int64_t f = 0; f < oc; ++f) {
          T acc = T(0.0f);
          for (int64_t rkx = 0; rkx < kx; ++rkx) {
            const int64_t ux = ix - pad_x_lo + rkx;
            if (ux < 0 || ux % dil_x != 0 || ux / dil_x >= in_x) {
              continue;
            }
            const int64_t sx = ux / dil_x;
            for (int64_t rky = 0; rky < ky; ++rky) {
              const int64_t uy = iy - pad_y_lo + rky;
              if (uy < 0 || uy % dil_y != 0 || uy / dil_y >= in_y) {
                continue;
              }
              const int64_t sy = uy / dil_y;
              T prod = T(0.0f);
              for (int64_t c = 0; c < ic; ++c) {
                prod += lhs[((b * in_x + sx) * in_y + sy) * ic + c] *
                        rhs[((rkx * ky + rky) * ic + c) * oc + f];
              }
              acc += prod;
            }
          }
          out[((b * out_x + ix) * out_y + iy) * oc + f] = acc;
        }
      }
    }
  }
  return out;
}

class CountingThreadPool final : public Eigen::ThreadPoolInterface {
 public:
  explicit CountingThreadPool(int num_threads)
      : num_threads_(num_threads),
        pool_(num_threads > 0
                  ? std::make_unique<tsl::thread::ThreadPool>(
                        tsl::Env::Default(), "conv_test", num_threads)
                  : nullptr) {}

  void Schedule(std::function<void()> fn) override {
    schedules_.fetch_add(1, std::memory_order_relaxed);
    if (pool_ != nullptr) {
      pool_->Schedule(std::move(fn));
    } else {
      fn();
    }
  }
  int NumThreads() const override { return num_threads_; }
  int CurrentThreadId() const override {
    return pool_ != nullptr ? pool_->CurrentThreadId() : -1;
  }
  int schedules() const { return schedules_.load(std::memory_order_relaxed); }

 private:
  int num_threads_;
  std::unique_ptr<tsl::thread::ThreadPool> pool_;
  std::atomic<int> schedules_{0};
};

TEST(ConvolutionThunkTest, ZeroInputPaddedTransposedConvFillsPositiveZero) {
  // 2D: f32[1,0,2,1] * f32[2,2,1,1], pad={2_2,1_1}, lhs_dilate={2,1} ->
  // f32[1,3,3,1]. Recomputing patch count via truncating division previously
  // read 8 floats past a 0-element conv_matrix.
  auto ker2d =
      MakeHeapBuffer<float>(4, std::numeric_limits<float>::quiet_NaN());
  auto out2d = MakeHeapBuffer<float>(9, -7.0f);
  auto in_empty = MakeHeapBuffer<float>(0);
  ASSERT_TRUE(
      ExecuteRawConv<float>(
          ShapeUtil::MakeShape(F32, {1, 0, 2, 1}), in_empty.get(),
          ShapeUtil::MakeShape(F32, {2, 2, 1, 1}), ker2d.get(),
          ShapeUtil::MakeShape(F32, {1, 3, 3, 1}), out2d.get(),
          MakeConvolutionDimensionNumbers(2),
          MakeCustomWindow({2, 2}, {1, 1}, {{2, 2}, {1, 1}}, {2, 1}, {1, 1}))
          .ok());
  for (int i = 0; i < 9; ++i) {
    EXPECT_EQ(absl::bit_cast<uint32_t>(out2d[i]), uint32_t{0});
  }

  // 1D: f32[1,0,1] * f32[2,1,1], pad=2_2, lhs_dilate=2 -> f32[1,3,1], plus F16.
  auto ker1d = MakeHeapBuffer<float>(2, 5.0f);
  auto out1d = MakeHeapBuffer<float>(3, -3.0f);
  ASSERT_TRUE(
      ExecuteRawConv<float>(ShapeUtil::MakeShape(F32, {1, 0, 1}), nullptr,
                            ShapeUtil::MakeShape(F32, {2, 1, 1}), ker1d.get(),
                            ShapeUtil::MakeShape(F32, {1, 3, 1}), out1d.get(),
                            MakeConvolutionDimensionNumbers(1),
                            MakeCustomWindow({2}, {1}, {{2, 2}}, {2}, {1}))
          .ok());
  for (int i = 0; i < 3; ++i) {
    EXPECT_EQ(absl::bit_cast<uint32_t>(out1d[i]), uint32_t{0});
  }

  auto ker16 = MakeHeapBuffer<Eigen::half>(4, Eigen::half(3.0f));
  auto out16 = MakeHeapBuffer<Eigen::half>(9, Eigen::half(-1.0f));
  ASSERT_TRUE(
      ExecuteRawConv<Eigen::half>(
          ShapeUtil::MakeShape(F16, {1, 0, 2, 1}), nullptr,
          ShapeUtil::MakeShape(F16, {2, 2, 1, 1}), ker16.get(),
          ShapeUtil::MakeShape(F16, {1, 3, 3, 1}), out16.get(),
          MakeConvolutionDimensionNumbers(2),
          MakeCustomWindow({2, 2}, {1, 1}, {{2, 2}, {1, 1}}, {2, 1}, {1, 1}))
          .ok());
  for (int i = 0; i < 9; ++i) {
    EXPECT_EQ(absl::bit_cast<uint16_t>(out16[i]), uint16_t{0});
  }
}

TEST(ConvolutionThunkTest, AsymmetricTransposedConvBitExactAndIeeeSpecials) {
  // Asymmetric 2D transposed conv covering less-than-full and negative padding.
  constexpr int64_t kB = 2, kInX = 3, kInY = 2, kIC = 2, kKX = 3, kKY = 2,
                    kOC = 3;
  auto lhs = MakeHeapBuffer<float>(kB * kInX * kInY * kIC);
  auto rhs = MakeHeapBuffer<float>(kKX * kKY * kIC * kOC);
  for (int i = 0; i < kB * kInX * kInY * kIC; ++i) {
    lhs[i] = 0.125f * static_cast<float>((i * 7 + 3) % 19 - 9);
  }
  for (int i = 0; i < kKX * kKY * kIC * kOC; ++i) {
    rhs[i] = 0.25f * static_cast<float>((i * 5 + 1) % 17 - 8);
  }
  const std::pair<int64_t, int64_t> kPads[] = {{0, 0}, {1, 2}, {-1, 0}, {3, 1}};
  for (auto [px_lo, px_hi] : kPads) {
    for (auto [py_lo, py_hi] : std::array<std::pair<int64_t, int64_t>, 3>{
             {{0, 1}, {2, 0}, {-1, 1}}}) {
      const int64_t ox =
          std::max<int64_t>(0, (kInX - 1) * 3 + 1 + px_lo + px_hi - kKX + 1);
      const int64_t oy =
          std::max<int64_t>(0, (kInY - 1) * 2 + 1 + py_lo + py_hi - kKY + 1);
      auto out = MakeHeapBuffer<float>(kB * ox * oy * kOC, -99.0f);
      ASSERT_TRUE(ExecuteRawConv<float>(
                      ShapeUtil::MakeShape(F32, {kB, kInX, kInY, kIC}),
                      lhs.get(),
                      ShapeUtil::MakeShape(F32, {kKX, kKY, kIC, kOC}),
                      rhs.get(), ShapeUtil::MakeShape(F32, {kB, ox, oy, kOC}),
                      out.get(), MakeConvolutionDimensionNumbers(2),
                      MakeCustomWindow({kKX, kKY}, {1, 1},
                                       {{px_lo, px_hi}, {py_lo, py_hi}}, {3, 2},
                                       {1, 1}))
                      .ok());
      auto ref = ReferenceTransposedConv2D<float>(
          lhs.get(), rhs.get(), kB, kInX, kInY, kIC, kKX, kKY, kOC, ox, oy,
          px_lo, py_lo, 3, 2);
      for (size_t i = 0; i < ref.size(); ++i) {
        EXPECT_EQ(absl::bit_cast<uint32_t>(out[i]),
                  absl::bit_cast<uint32_t>(ref[i]));
      }
    }
  }

  // 1D transposed conv.
  auto lhs1 = MakeHeapBuffer<float>(3);
  auto rhs1 = MakeHeapBuffer<float>(3);
  lhs1[0] = 2.0f;
  lhs1[1] = -0.5f;
  lhs1[2] = 1.25f;
  rhs1[0] = 0.5f;
  rhs1[1] = -1.5f;
  rhs1[2] = 4.0f;
  auto out1 = MakeHeapBuffer<float>(5, -1.0f);
  ASSERT_TRUE(
      ExecuteRawConv<float>(ShapeUtil::MakeShape(F32, {1, 3, 1}), lhs1.get(),
                            ShapeUtil::MakeShape(F32, {3, 1, 1}), rhs1.get(),
                            ShapeUtil::MakeShape(F32, {1, 5, 1}), out1.get(),
                            MakeConvolutionDimensionNumbers(1),
                            MakeCustomWindow({3}, {1}, {{1, 1}}, {2}, {1}))
          .ok());
  auto ref1 = ReferenceTransposedConv2D<float>(lhs1.get(), rhs1.get(), 1, 1, 3,
                                               1, 1, 3, 1, 1, 5, 0, 1, 1, 2);
  for (size_t i = 0; i < ref1.size(); ++i) {
    EXPECT_EQ(absl::bit_cast<uint32_t>(out1[i]),
              absl::bit_cast<uint32_t>(ref1[i]));
  }

  // 3D transposed conv and IEEE 754 special values (-0.0, subnormal, Inf, NaN).
  auto lhs3 = MakeHeapBuffer<float>(2);
  lhs3[0] = 2.0f;
  lhs3[1] = -3.0f;
  auto rhs3 = MakeHeapBuffer<float>(2);
  rhs3[0] = 5.0f;
  rhs3[1] = 7.0f;
  auto out3 = MakeHeapBuffer<float>(2, 0.0f);
  ASSERT_TRUE(
      ExecuteRawConv<float>(
          ShapeUtil::MakeShape(F32, {1, 2, 1, 1, 1}), lhs3.get(),
          ShapeUtil::MakeShape(F32, {2, 1, 1, 1, 1}), rhs3.get(),
          ShapeUtil::MakeShape(F32, {1, 2, 1, 1, 1}), out3.get(),
          MakeConvolutionDimensionNumbers(3),
          MakeCustomWindow({2, 1, 1}, {1, 1, 1}, {{0, 0}, {0, 0}, {0, 0}},
                           {2, 1, 1}, {1, 1, 1}))
          .ok());
  EXPECT_EQ(out3[0], 10.0f);
  EXPECT_EQ(out3[1], -21.0f);

  auto spec_in = MakeHeapBuffer<float>(5);
  spec_in[0] = -0.0f;
  spec_in[1] = std::numeric_limits<float>::denorm_min();
  spec_in[2] = -std::numeric_limits<float>::denorm_min();
  spec_in[3] = -std::numeric_limits<float>::infinity();
  spec_in[4] = std::numeric_limits<float>::quiet_NaN();
  auto spec_k = MakeHeapBuffer<float>(1, 1.0f);
  auto spec_out = MakeHeapBuffer<float>(9, 1.0f);
  ASSERT_TRUE(
      ExecuteRawConv<float>(
          ShapeUtil::MakeShape(F32, {1, 5, 1, 1}), spec_in.get(),
          ShapeUtil::MakeShape(F32, {1, 1, 1, 1}), spec_k.get(),
          ShapeUtil::MakeShape(F32, {1, 9, 1, 1}), spec_out.get(),
          MakeConvolutionDimensionNumbers(2),
          MakeCustomWindow({1, 1}, {1, 1}, {{0, 0}, {0, 0}}, {2, 1}, {1, 1}))
          .ok());
  EXPECT_EQ(absl::bit_cast<uint32_t>(spec_out[0]), uint32_t{0});
  EXPECT_EQ(absl::bit_cast<uint32_t>(spec_out[2]),
            absl::bit_cast<uint32_t>(std::numeric_limits<float>::denorm_min()));
  EXPECT_EQ(
      absl::bit_cast<uint32_t>(spec_out[4]),
      absl::bit_cast<uint32_t>(-std::numeric_limits<float>::denorm_min()));
  EXPECT_TRUE(std::isinf(spec_out[6]) && spec_out[6] < 0.0f);
  EXPECT_TRUE(std::isnan(spec_out[8]));
}

TEST(ConvolutionThunkTest, TransposedConvExtremeInt64PaddingNoOverflow) {
  constexpr int64_t kMin = std::numeric_limits<int64_t>::min();
  constexpr int64_t kMax = std::numeric_limits<int64_t>::max();
  auto lhs = MakeHeapBuffer<float>(4, 3.0f);
  auto rhs = MakeHeapBuffer<float>(4, 5.0f);
  for (auto pad_x : std::array<std::pair<int64_t, int64_t>, 2>{
           {{kMin, kMax}, {kMax, kMin}}}) {
    auto out = MakeHeapBuffer<float>(1, -1.0f);
    ASSERT_TRUE(
        ExecuteRawConv<float>(
            ShapeUtil::MakeShape(F32, {1, 2, 2, 1}), lhs.get(),
            ShapeUtil::MakeShape(F32, {2, 2, 1, 1}), rhs.get(),
            ShapeUtil::MakeShape(F32, {1, 1, 1, 1}), out.get(),
            MakeConvolutionDimensionNumbers(2),
            MakeCustomWindow({2, 2}, {1, 1}, {pad_x, {0, 0}}, {2, 1}, {1, 1}))
            .ok());
    EXPECT_EQ(absl::bit_cast<uint32_t>(out[0]), uint32_t{0});
  }
}

TEST(ConvolutionThunkTest, CanUseCustomTransposedConvOverflowAndBoundary) {
  using internal::CanUseCustomTransposedConv;
  constexpr Eigen::Index kMax = std::numeric_limits<Eigen::Index>::max();

  // 32-bit int truncation: 65536 * 65536 -> 0 and 65537 * 65536 -> 65536.
  EXPECT_FALSE(CanUseCustomTransposedConv<float>(
      1, 65536, 65536, 1, 1, 1, 131071, 65536, 1, 1, 2, 1, 1, 1, 1));
  EXPECT_FALSE(CanUseCustomTransposedConv<Eigen::half>(
      1, 65537, 65536, 1, 1, 1, 131073, 65536, 1, 1, 2, 1, 1, 1, 1));
  // 64-bit size_t wrap: buffer_size = 2^62, buffer_size * 4 == 0 mod 2^64.
  EXPECT_FALSE(CanUseCustomTransposedConv<float>(int64_t{1} << 30, 1, 16, 1, 1,
                                                 int64_t{1} << 28, 1, 31, 1, 1,
                                                 1, 2, 1, 1, 1));
  // Exact 8 GiB boundary (F16: 2 * 4096 * 256 * 2048 = 2^32 elements = 8 GiB).
  EXPECT_TRUE(CanUseCustomTransposedConv<Eigen::half>(
      2, 1, 4096, 1, 256, 2048, 1, 8191, 1, 1, 1, 2, 1, 1, 1));
  EXPECT_FALSE(CanUseCustomTransposedConv<Eigen::half>(
      2, 1, 4097, 1, 256, 2048, 1, 8193, 1, 1, 1, 2, 1, 1, 1));
  EXPECT_FALSE(CanUseCustomTransposedConv<float>(2, 1, 4096, 1, 256, 2048, 1,
                                                 8191, 1, 1, 1, 2, 1, 1, 1));
  // Batch = 2^31 (fits 8 GiB F32 and 4 GiB F16) vs 2^31 + 1.
  EXPECT_TRUE(CanUseCustomTransposedConv<float>(int64_t{1} << 31, 1, 1, 1, 1, 1,
                                                1, 1, 1, 1, 2, 1, 1, 1, 1));
  EXPECT_FALSE(CanUseCustomTransposedConv<float>(
      (int64_t{1} << 31) + 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1));
  // 64-bit output size (fits) vs 4-factor output overflow and dilation
  // overflow.
  EXPECT_TRUE(CanUseCustomTransposedConv<float>(
      1, 100000, 1, 1, 1, 1, 9999900001LL, 1, 1, 1, 100000, 1, 1, 1, 1));
  EXPECT_FALSE(
      CanUseCustomTransposedConv<float>(1, 2, 2, 1, 1, 1, int64_t{1} << 40,
                                        int64_t{1} << 30, 1, 1, 2, 1, 1, 1, 1));
  EXPECT_FALSE(CanUseCustomTransposedConv<float>(1, 3, 1, 1, 1, 1, 1, 1, 1, 1,
                                                 kMax / 2 + 2, 1, 1, 1, 1));
  // Zero/negative dilation or dimensions.
  EXPECT_FALSE(CanUseCustomTransposedConv<float>(1, 2, 2, 1, 1, 1, 3, 2, 1, 1,
                                                 2, 0, 1, 1, 1));
  EXPECT_FALSE(CanUseCustomTransposedConv<float>(1, 0, 2, 2, 2, 1, 3, 3, 1, 1,
                                                 2, 1, 1, 1, 1));
}

TEST(ConvolutionThunkTest, ZeroSizedOutputAndZeroChannelsShortCircuit) {
  auto ker = MakeHeapBuffer<float>(8, std::numeric_limits<float>::quiet_NaN());
  // Zero-element output never dereferences output buffer.
  ASSERT_TRUE(
      ExecuteRawConv<float>(
          ShapeUtil::MakeShape(F32, {0, 2, 2, 1}), nullptr,
          ShapeUtil::MakeShape(F32, {2, 2, 1, 1}), ker.get(),
          ShapeUtil::MakeShape(F32, {0, 1, 1, 1}), nullptr,
          MakeConvolutionDimensionNumbers(2),
          MakeCustomWindow({2, 2}, {1, 1}, {{0, 0}, {0, 0}}, {1, 1}, {1, 1}))
          .ok());
  // Zero input channels with non-empty output (grouped 2D and 3D) writes +0.0.
  auto out2 = MakeHeapBuffer<float>(4, -5.0f);
  ASSERT_TRUE(
      ExecuteRawConv<float>(
          ShapeUtil::MakeShape(F32, {1, 2, 2, 0}), nullptr,
          ShapeUtil::MakeShape(F32, {2, 2, 0, 4}), nullptr,
          ShapeUtil::MakeShape(F32, {1, 1, 1, 4}), out2.get(),
          MakeConvolutionDimensionNumbers(2),
          MakeCustomWindow({2, 2}, {1, 1}, {{0, 0}, {0, 0}}, {1, 1}, {1, 1}),
          /*feature_group_count=*/2)
          .ok());
  for (int i = 0; i < 4; ++i) {
    EXPECT_EQ(absl::bit_cast<uint32_t>(out2[i]), uint32_t{0});
  }

  auto out3 = MakeHeapBuffer<float>(4, 9.0f);
  ASSERT_TRUE(
      ExecuteRawConv<float>(
          ShapeUtil::MakeShape(F32, {1, 0, 2, 2, 2}), nullptr,
          ShapeUtil::MakeShape(F32, {1, 2, 2, 1, 2}), ker.get(),
          ShapeUtil::MakeShape(F32, {1, 2, 1, 1, 2}), out3.get(),
          MakeConvolutionDimensionNumbers(3),
          MakeCustomWindow({1, 2, 2}, {1, 1, 1}, {{1, 1}, {0, 0}, {0, 0}},
                           {1, 1, 1}, {1, 1, 1}),
          /*feature_group_count=*/2)
          .ok());
  for (int i = 0; i < 4; ++i) {
    EXPECT_EQ(absl::bit_cast<uint32_t>(out3[i]), uint32_t{0});
  }
}

TEST(ConvolutionThunkTest, DegenerateThreadPoolDevicesFallBackSafely) {
  CountingThreadPool zero_pool(/*num_threads=*/0);
  Eigen::ThreadPoolDevice zero_pool_dev(&zero_pool, /*num_cores=*/0);
  Eigen::ThreadPoolDevice null_zero_dev(/*pool=*/nullptr, /*num_cores=*/0);
  Eigen::ThreadPoolDevice null_multi_dev(/*pool=*/nullptr, /*num_cores=*/4);

  auto in2 = MakeHeapBuffer<float>(2, 2.0f);
  auto ker2 = MakeHeapBuffer<float>(2, 3.0f);
  for (const Eigen::ThreadPoolDevice* dev :
       {&zero_pool_dev, &null_zero_dev, &null_multi_dev}) {
    // Grouped 2D, transposed 2D, ungrouped 2D, and 3D.
    auto out_g = MakeHeapBuffer<float>(2, 0.0f);
    ASSERT_TRUE(
        ExecuteRawConv<float>(
            ShapeUtil::MakeShape(F32, {1, 1, 1, 2}), in2.get(),
            ShapeUtil::MakeShape(F32, {1, 1, 1, 2}), ker2.get(),
            ShapeUtil::MakeShape(F32, {1, 1, 1, 2}), out_g.get(),
            MakeConvolutionDimensionNumbers(2),
            MakeCustomWindow({1, 1}, {1, 1}, {{0, 0}, {0, 0}}, {1, 1}, {1, 1}),
            /*feature_group_count=*/2, {true}, dev)
            .ok());
    EXPECT_EQ(out_g[0], 6.0f);
    EXPECT_EQ(out_g[1], 6.0f);

    auto out_t = MakeHeapBuffer<float>(3, 0.0f);
    ASSERT_TRUE(
        ExecuteRawConv<float>(
            ShapeUtil::MakeShape(F32, {1, 2, 1, 1}), in2.get(),
            ShapeUtil::MakeShape(F32, {1, 1, 1, 1}), ker2.get(),
            ShapeUtil::MakeShape(F32, {1, 3, 1, 1}), out_t.get(),
            MakeConvolutionDimensionNumbers(2),
            MakeCustomWindow({1, 1}, {1, 1}, {{0, 0}, {0, 0}}, {2, 1}, {1, 1}),
            /*feature_group_count=*/1, {true}, dev)
            .ok());
    EXPECT_EQ(out_t[0], 6.0f);
    EXPECT_EQ(out_t[1], 0.0f);
    EXPECT_EQ(out_t[2], 6.0f);

    auto out_u = MakeHeapBuffer<float>(1, 0.0f);
    ASSERT_TRUE(
        ExecuteRawConv<float>(
            ShapeUtil::MakeShape(F32, {1, 1, 1, 1}), in2.get(),
            ShapeUtil::MakeShape(F32, {1, 1, 1, 1}), ker2.get(),
            ShapeUtil::MakeShape(F32, {1, 1, 1, 1}), out_u.get(),
            MakeConvolutionDimensionNumbers(2),
            MakeCustomWindow({1, 1}, {1, 1}, {{0, 0}, {0, 0}}, {1, 1}, {1, 1}),
            /*feature_group_count=*/1, {true}, dev)
            .ok());
    EXPECT_EQ(out_u[0], 6.0f);

    auto out_3d = MakeHeapBuffer<float>(1, 0.0f);
    ASSERT_TRUE(
        ExecuteRawConv<float>(
            ShapeUtil::MakeShape(F32, {1, 1, 1, 1, 1}), in2.get(),
            ShapeUtil::MakeShape(F32, {1, 1, 1, 1, 1}), ker2.get(),
            ShapeUtil::MakeShape(F32, {1, 1, 1, 1, 1}), out_3d.get(),
            MakeConvolutionDimensionNumbers(3),
            MakeCustomWindow({1, 1, 1}, {1, 1, 1}, {{0, 0}, {0, 0}, {0, 0}},
                             {1, 1, 1}, {1, 1, 1}),
            /*feature_group_count=*/1, {true}, dev)
            .ok());
    EXPECT_EQ(out_3d[0], 6.0f);
  }
  EXPECT_EQ(zero_pool.schedules(), 0);
}

TEST(ConvolutionThunkTest, RejectsInvalidShapesTypesLayoutsGroupsAndWindows) {
  const Shape valid_in = ShapeUtil::MakeShape(F32, {1, 3, 3, 4});
  const Shape valid_ker = ShapeUtil::MakeShape(F32, {2, 2, 4, 2});
  const Shape valid_out = ShapeUtil::MakeShape(F32, {1, 2, 2, 2});
  const auto valid_dnums = MakeConvolutionDimensionNumbers(2);
  const Window valid_win =
      MakeCustomWindow({2, 2}, {1, 1}, {{0, 0}, {0, 0}}, {1, 1}, {1, 1});

  auto check_invalid = [](const Shape& in, const Shape& ker, const Shape& out,
                          const ConvolutionDimensionNumbers& dnums,
                          const Window& win, int64_t groups = 1) {
    ConvolutionSlices slices{{}, in, {}, ker, {}, out};
    auto status_or = GetConvolutionCanonicalDims(slices, dnums, win, groups);
    EXPECT_EQ(status_or.status().code(), absl::StatusCode::kInvalidArgument)
        << in.ToString(true) << " * " << ker.ToString(true) << " -> "
        << out.ToString(true);
  };

  // tfcompile AOT contract: empty slices with valid shapes succeed (incl.
  // unset window dim size == 0).
  ConvolutionSlices empty_slices{{}, valid_in, {}, valid_ker, {}, valid_out};
  EXPECT_TRUE(
      GetConvolutionCanonicalDims(empty_slices, valid_dnums, MakeWindow(2), 1)
          .ok());

  // Element type mismatches.
  check_invalid(valid_in, valid_ker, ShapeUtil::MakeShape(F16, {1, 2, 2, 2}),
                valid_dnums, valid_win);
  check_invalid(valid_in, ShapeUtil::MakeShape(F16, {2, 2, 4, 2}), valid_out,
                valid_dnums, valid_win);

  // Rank 0/1, non-array, dynamic, layoutless, tiled, and tail-padded shapes.
  check_invalid(ShapeUtil::MakeShape(F32, {}), valid_ker, valid_out,
                valid_dnums, valid_win);
  check_invalid(ShapeUtil::MakeShape(F32, {4}), valid_ker, valid_out,
                valid_dnums, valid_win);
  check_invalid(ShapeUtil::MakeTupleShape({valid_in}), valid_ker, valid_out,
                valid_dnums, valid_win);
  Shape dyn_in = valid_in;
  dyn_in.set_dynamic_dimension(0, true);
  check_invalid(dyn_in, valid_ker, valid_out, valid_dnums, valid_win);
  Shape no_layout = valid_in;
  LayoutUtil::ClearLayout(&no_layout);
  check_invalid(no_layout, valid_ker, valid_out, valid_dnums, valid_win);
  Shape tiled = valid_in;
  tiled.mutable_layout()->add_tiles()->add_dimensions(2);
  check_invalid(tiled, valid_ker, valid_out, valid_dnums, valid_win);
  Shape tail_padded = valid_in;
  tail_padded.mutable_layout()->set_tail_padding_alignment_in_elements(8);
  check_invalid(tail_padded, valid_ker, valid_out, valid_dnums, valid_win);

  // Out-of-bounds/duplicate dnums and non-canonical NCHW {3,2,1,0} layout.
  auto bad_dnums = valid_dnums;
  bad_dnums.set_input_batch_dimension(4);
  check_invalid(valid_in, valid_ker, valid_out, bad_dnums, valid_win);
  bad_dnums = valid_dnums;
  bad_dnums.set_input_feature_dimension(0);
  check_invalid(valid_in, valid_ker, valid_out, bad_dnums, valid_win);
  ConvolutionDimensionNumbers nchw_dnums = valid_dnums;
  nchw_dnums.set_input_feature_dimension(1);
  nchw_dnums.set_input_spatial_dimensions(0, 2);
  nchw_dnums.set_input_spatial_dimensions(1, 3);
  check_invalid(ShapeUtil::MakeShape(F32, {1, 4, 3, 3}), valid_ker, valid_out,
                nchw_dnums, valid_win);

  // Feature group count and channel divisibility/overflow.
  check_invalid(valid_in, valid_ker, valid_out, valid_dnums, valid_win, 0);
  check_invalid(valid_in, valid_ker, valid_out, valid_dnums, valid_win, -2);
  check_invalid(valid_in, valid_ker, valid_out, valid_dnums, valid_win, 2);
  check_invalid(valid_in, ShapeUtil::MakeShape(F32, {2, 2, 2, 3}),
                ShapeUtil::MakeShape(F32, {1, 2, 2, 3}), valid_dnums, valid_win,
                2);
  check_invalid(
      ShapeUtil::MakeShape(F32, {1, 1, 1, 8}),
      ShapeUtil::MakeShape(F32, {1, 1, int64_t{1} << 60, 0}),
      ShapeUtil::MakeShape(F32, {1, 1, 1, 0}), valid_dnums,
      MakeCustomWindow({1, 1}, {1, 1}, {{0, 0}, {0, 0}}, {1, 1}, {1, 1}), 8);

  // Window rank (including 1D conv with 2 window dims), kernel=0, size
  // mismatch, non-positive stride/dilation, reversal, dilated overflow, output
  // mismatch, and pre_contract_dims overflow.
  check_invalid(ShapeUtil::MakeShape(F32, {1, 3, 1}),
                ShapeUtil::MakeShape(F32, {2, 1, 1}),
                ShapeUtil::MakeShape(F32, {1, 2, 1}),
                MakeConvolutionDimensionNumbers(1), valid_win);
  check_invalid(valid_in, ShapeUtil::MakeShape(F32, {0, 2, 4, 2}), valid_out,
                valid_dnums, MakeWindow(2));
  check_invalid(
      valid_in, valid_ker, valid_out, valid_dnums,
      MakeCustomWindow({3, 2}, {1, 1}, {{0, 0}, {0, 0}}, {1, 1}, {1, 1}));
  check_invalid(
      valid_in, valid_ker, valid_out, valid_dnums,
      MakeCustomWindow({2, 2}, {0, 1}, {{0, 0}, {0, 0}}, {1, 1}, {1, 1}));
  check_invalid(
      valid_in, valid_ker, valid_out, valid_dnums,
      MakeCustomWindow({2, 2}, {1, 1}, {{0, 0}, {0, 0}}, {0, 1}, {1, 1}));
  check_invalid(
      valid_in, valid_ker, valid_out, valid_dnums,
      MakeCustomWindow({2, 2}, {1, 1}, {{0, 0}, {0, 0}}, {1, 1}, {0, 1}));
  check_invalid(valid_in, valid_ker, valid_out, valid_dnums,
                MakeCustomWindow({2, 2}, {1, 1}, {{0, 0}, {0, 0}}, {1, 1},
                                 {1, 1}, /*reversal=*/true));
  check_invalid(
      valid_in, valid_ker, valid_out, valid_dnums,
      MakeCustomWindow({2, 2}, {1, 1}, {{0, 0}, {0, 0}},
                       {std::numeric_limits<int64_t>::max(), 1}, {1, 1}));
  check_invalid(valid_in, valid_ker, ShapeUtil::MakeShape(F32, {1, 3, 2, 2}),
                valid_dnums, valid_win);

  // Overflowing spatial dimensions with 0 output channels.
  constexpr int64_t kBig = int64_t{1} << 32;
  Shape big_zero =
      ShapeUtil::MakeShapeWithDenseLayout(F32, {1, 0, 1, 1}, {1, 3, 2, 0});
  big_zero.set_dimensions(2, kBig);
  big_zero.set_dimensions(3, kBig);
  ConvolutionDimensionNumbers nchw_all;
  nchw_all.set_input_batch_dimension(0);
  nchw_all.set_input_feature_dimension(1);
  nchw_all.add_input_spatial_dimensions(2);
  nchw_all.add_input_spatial_dimensions(3);
  nchw_all.set_kernel_output_feature_dimension(0);
  nchw_all.set_kernel_input_feature_dimension(1);
  nchw_all.add_kernel_spatial_dimensions(2);
  nchw_all.add_kernel_spatial_dimensions(3);
  nchw_all.set_output_batch_dimension(0);
  nchw_all.set_output_feature_dimension(1);
  nchw_all.add_output_spatial_dimensions(2);
  nchw_all.add_output_spatial_dimensions(3);
  check_invalid(
      big_zero,
      ShapeUtil::MakeShapeWithDenseLayout(F32, {0, 0, 1, 1}, {0, 1, 3, 2}),
      big_zero, nchw_all,
      MakeCustomWindow({1, 1}, {1, 1}, {{0, 0}, {0, 0}}, {1, 1}, {1, 1}));
}

TEST(ConvolutionThunkTest, AcceptsPhysicallyCanonicalNchwAndOihwLayouts) {
  auto in = MakeHeapBuffer<float>(2 * 3 * 4 * 4);
  auto ker = MakeHeapBuffer<float>(2 * 3 * 2 * 4);
  for (int i = 0; i < 2 * 3 * 4 * 4; ++i) {
    in[i] = 0.25f * static_cast<float>((i % 11) - 5);
  }
  for (int i = 0; i < 2 * 3 * 2 * 4; ++i) {
    ker[i] = 0.5f * static_cast<float>((i % 7) - 3);
  }
  auto out_nhwc = MakeHeapBuffer<float>(2 * 2 * 2 * 4, 0.0f);
  auto out_nchw = MakeHeapBuffer<float>(2 * 2 * 2 * 4, -1.0f);
  const Window win =
      MakeCustomWindow({2, 3}, {1, 1}, {{0, 0}, {0, 0}}, {1, 1}, {1, 1});
  ASSERT_TRUE(ExecuteRawConv<float>(
                  ShapeUtil::MakeShape(F32, {2, 3, 4, 4}), in.get(),
                  ShapeUtil::MakeShape(F32, {2, 3, 2, 4}), ker.get(),
                  ShapeUtil::MakeShape(F32, {2, 2, 2, 4}), out_nhwc.get(),
                  MakeConvolutionDimensionNumbers(2), win, 2)
                  .ok());

  ConvolutionDimensionNumbers dnums;
  dnums.set_input_batch_dimension(0);
  dnums.set_input_feature_dimension(1);
  dnums.add_input_spatial_dimensions(2);
  dnums.add_input_spatial_dimensions(3);
  dnums.set_kernel_output_feature_dimension(0);
  dnums.set_kernel_input_feature_dimension(1);
  dnums.add_kernel_spatial_dimensions(2);
  dnums.add_kernel_spatial_dimensions(3);
  dnums.set_output_batch_dimension(0);
  dnums.set_output_feature_dimension(1);
  dnums.add_output_spatial_dimensions(2);
  dnums.add_output_spatial_dimensions(3);
  ASSERT_TRUE(
      ExecuteRawConv<float>(
          ShapeUtil::MakeShapeWithDenseLayout(F32, {2, 4, 3, 4}, {1, 3, 2, 0}),
          in.get(),
          ShapeUtil::MakeShapeWithDenseLayout(F32, {4, 2, 2, 3}, {0, 1, 3, 2}),
          ker.get(),
          ShapeUtil::MakeShapeWithDenseLayout(F32, {2, 4, 2, 2}, {1, 3, 2, 0}),
          out_nchw.get(), dnums, win, 2)
          .ok());
  for (int i = 0; i < 2 * 2 * 2 * 4; ++i) {
    EXPECT_EQ(absl::bit_cast<uint32_t>(out_nchw[i]),
              absl::bit_cast<uint32_t>(out_nhwc[i]));
  }
}

TEST(ConvolutionThunkTest, ExecuteValidatesBuffersAlignmentAndAliasing) {
  const Shape in_shape = ShapeUtil::MakeShape(F32, {1, 2, 1, 1});
  const Shape ker_shape = ShapeUtil::MakeShape(F32, {1, 1, 1, 1});
  const Shape out_shape = ShapeUtil::MakeShape(F32, {1, 3, 1, 1});
  const auto dnums = MakeConvolutionDimensionNumbers(2);
  const Window win =
      MakeCustomWindow({1, 1}, {1, 1}, {{0, 0}, {0, 0}}, {2, 1}, {1, 1});

  BufferAllocation alloc0(0, 64, 0);
  BufferAllocation alloc1(1, 64, 0);
  BufferAllocation alloc2(2, 64, 0);
  TF_ASSERT_OK_AND_ASSIGN(
      auto thunk,
      ConvolutionThunk::Create(
          {"conv"}, {false}, BufferAllocation::Slice(&alloc0, 0, 8), in_shape,
          BufferAllocation::Slice(&alloc1, 0, 4), ker_shape,
          BufferAllocation::Slice(&alloc2, 0, 12), out_shape, dnums, win, 1));

  // Null buffer_allocations.
  Thunk::ExecuteParams null_params;
  auto ev = thunk->Execute(null_params);
  tsl::BlockUntilReady(ev);
  EXPECT_TRUE(ev.IsError());

  // Null non-empty buffer and undersized backing DeviceAddressBase.
  alignas(64) std::byte storage[128] = {};
  auto run_with = [&](se::DeviceAddressBase a, se::DeviceAddressBase b,
                      se::DeviceAddressBase c) {
    const std::array<se::DeviceAddressBase, 3> bufs = {a, b, c};
    BufferAllocations allocs(bufs);
    Thunk::ExecuteParams p;
    p.buffer_allocations = &allocs;
    auto e = thunk->Execute(p);
    tsl::BlockUntilReady(e);
    return e.IsError() ? e.GetError().code() : absl::StatusCode::kOk;
  };
  std::byte* base = storage;
  EXPECT_EQ(run_with(se::DeviceAddressBase(nullptr, 8),
                     se::DeviceAddressBase(base + 16, 4),
                     se::DeviceAddressBase(base + 32, 12)),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(run_with(se::DeviceAddressBase(base, 4),
                     se::DeviceAddressBase(base + 16, 4),
                     se::DeviceAddressBase(base + 32, 12)),
            absl::StatusCode::kInvalidArgument);

  // Undersized slice.
  TF_ASSERT_OK_AND_ASSIGN(
      auto short_thunk,
      ConvolutionThunk::Create(
          {"conv"}, {false}, BufferAllocation::Slice(&alloc0, 0, 8), in_shape,
          BufferAllocation::Slice(&alloc1, 0, 4), ker_shape,
          BufferAllocation::Slice(&alloc2, 0, 8), out_shape, dnums, win, 1));
  {
    const std::array<se::DeviceAddressBase, 3> bufs = {
        se::DeviceAddressBase(base, 64), se::DeviceAddressBase(base + 16, 64),
        se::DeviceAddressBase(base + 32, 64)};
    BufferAllocations allocs(bufs);
    Thunk::ExecuteParams p;
    p.buffer_allocations = &allocs;
    auto e = short_thunk->Execute(p);
    tsl::BlockUntilReady(e);
    EXPECT_TRUE(e.IsError());
  }

  // Odd-byte / 2-byte misaligned F32 rejected.
  EXPECT_EQ(run_with(se::DeviceAddressBase(base + 1, 8),
                     se::DeviceAddressBase(base + 16, 4),
                     se::DeviceAddressBase(base + 32, 12)),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(run_with(se::DeviceAddressBase(base + 2, 8),
                     se::DeviceAddressBase(base + 16, 4),
                     se::DeviceAddressBase(base + 32, 12)),
            absl::StatusCode::kInvalidArgument);

  // Overlapping output with input or kernel (exact and partial) rejected.
  EXPECT_EQ(run_with(se::DeviceAddressBase(base, 8),
                     se::DeviceAddressBase(base + 16, 4),
                     se::DeviceAddressBase(base, 12)),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(run_with(se::DeviceAddressBase(base, 8),
                     se::DeviceAddressBase(base + 16, 4),
                     se::DeviceAddressBase(base + 4, 12)),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(run_with(se::DeviceAddressBase(base, 8),
                     se::DeviceAddressBase(base + 16, 4),
                     se::DeviceAddressBase(base + 8, 12)),
            absl::StatusCode::kInvalidArgument);

  // Sub-16B-aligned (4B/12B offset) and adjacent slices succeed.
  const float in_vals[2] = {2.5f, -1.5f};
  float ker_val = 4.0f;
  std::memcpy(base + 4, in_vals, sizeof(in_vals));
  std::memcpy(base + 12, &ker_val, sizeof(ker_val));
  EXPECT_EQ(run_with(se::DeviceAddressBase(base + 4, 8),
                     se::DeviceAddressBase(base + 12, 4),
                     se::DeviceAddressBase(base + 16, 12)),
            absl::StatusCode::kOk);
  float out_vals[3] = {};
  std::memcpy(out_vals, base + 16, sizeof(out_vals));
  EXPECT_EQ(out_vals[0], 10.0f);
  EXPECT_EQ(out_vals[1], 0.0f);
  EXPECT_EQ(out_vals[2], -6.0f);

  // Input == kernel read-only self-convolution and 2B-offset F16 succeed.
  const Shape s1 = ShapeUtil::MakeShape(F32, {1, 1, 1, 1});
  float self_out = 0.0f;
  ASSERT_TRUE(
      ExecuteRawConv<float>(
          s1, &ker_val, s1, &ker_val, s1, &self_out, dnums,
          MakeCustomWindow({1, 1}, {1, 1}, {{0, 0}, {0, 0}}, {1, 1}, {1, 1}))
          .ok());
  EXPECT_EQ(self_out, 16.0f);

  const Shape h1 = ShapeUtil::MakeShape(F16, {1, 1, 1, 1});
  const Eigen::half hv(3.0f);
  std::memcpy(base + 2, &hv, sizeof(hv));
  std::memcpy(base + 6, &hv, sizeof(hv));
  ASSERT_TRUE(
      ExecuteRawConv<Eigen::half>(
          h1, reinterpret_cast<Eigen::half*>(base + 2), h1,
          reinterpret_cast<Eigen::half*>(base + 6), h1,
          reinterpret_cast<Eigen::half*>(base + 10), dnums,
          MakeCustomWindow({1, 1}, {1, 1}, {{0, 0}, {0, 0}}, {1, 1}, {1, 1}))
          .ok());
  Eigen::half hout(0.0f);
  std::memcpy(&hout, base + 10, sizeof(hout));
  EXPECT_EQ(static_cast<float>(hout), 9.0f);
}

TEST(ConvolutionThunkTest, HonorsMultiThreadedOption) {
  auto in = MakeHeapBuffer<float>(4, 2.0f);
  auto ker = MakeHeapBuffer<float>(4, 3.0f);
  const Window win2 =
      MakeCustomWindow({1, 1}, {1, 1}, {{0, 0}, {0, 0}}, {1, 1}, {1, 1});

  CountingThreadPool single_pool(/*num_threads=*/4);
  Eigen::ThreadPoolDevice single_dev(&single_pool, /*num_cores=*/4);
  auto out_single = MakeHeapBuffer<float>(4, 0.0f);
  ASSERT_TRUE(ExecuteRawConv<float>(
                  ShapeUtil::MakeShape(F32, {1, 1, 1, 4}), in.get(),
                  ShapeUtil::MakeShape(F32, {1, 1, 1, 4}), ker.get(),
                  ShapeUtil::MakeShape(F32, {1, 1, 1, 4}), out_single.get(),
                  MakeConvolutionDimensionNumbers(2), win2,
                  /*feature_group_count=*/4,
                  /*options=*/{false}, &single_dev)
                  .ok());
  auto out3_single = MakeHeapBuffer<float>(4, 0.0f);
  ASSERT_TRUE(
      ExecuteRawConv<float>(
          ShapeUtil::MakeShape(F32, {1, 1, 1, 1, 4}), in.get(),
          ShapeUtil::MakeShape(F32, {1, 1, 1, 1, 4}), ker.get(),
          ShapeUtil::MakeShape(F32, {1, 1, 1, 1, 4}), out3_single.get(),
          MakeConvolutionDimensionNumbers(3),
          MakeCustomWindow({1, 1, 1}, {1, 1, 1}, {{0, 0}, {0, 0}, {0, 0}},
                           {1, 1, 1}, {1, 1, 1}),
          /*feature_group_count=*/4, /*options=*/{false}, &single_dev)
          .ok());
  EXPECT_EQ(single_pool.schedules(), 0);

  CountingThreadPool multi_pool(/*num_threads=*/4);
  Eigen::ThreadPoolDevice multi_dev(&multi_pool, /*num_cores=*/4);
  auto out_multi = MakeHeapBuffer<float>(4, 0.0f);
  ASSERT_TRUE(
      ExecuteRawConv<float>(ShapeUtil::MakeShape(F32, {1, 1, 1, 4}), in.get(),
                            ShapeUtil::MakeShape(F32, {1, 1, 1, 4}), ker.get(),
                            ShapeUtil::MakeShape(F32, {1, 1, 1, 4}),
                            out_multi.get(), MakeConvolutionDimensionNumbers(2),
                            win2, /*feature_group_count=*/4,
                            /*options=*/{true}, &multi_dev)
          .ok());
  EXPECT_GT(multi_pool.schedules(), 0);
  for (int i = 0; i < 4; ++i) {
    EXPECT_EQ(out_single[i], 6.0f);
    EXPECT_EQ(out3_single[i], 6.0f);
    EXPECT_EQ(out_multi[i], 6.0f);
  }
}

}  // namespace
}  // namespace xla::cpu
