/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "xla/literal_util.h"
#include "xla/tests/literal_test_util.h"
#include "xla/tools/multihost_hlo_runner/hlo_runner.h"
#include "tsl/lib/core/status_test_util.h"

namespace xla {

TEST(MultiHostHloRunnerTest, TestAllReduce) {
  constexpr absl::string_view all_reduce_test_hlo = R"(
    HloModule hlo_runner_test_0.1
    primitive_computation_add__2.3 {
      parameter.4 = f32[] parameter(0), parameter_replication={false}
      parameter.5 = f32[] parameter(1), parameter_replication={false}
      ROOT add.6 = f32[] add(parameter.4, parameter.5)
    }
    ENTRY hlo_runner_test_0.1 {
      constant.2 = pred[] constant(false)
      parameter.1 = f32[100]{0} parameter(0), parameter_replication={false}
      all-reduce.7 = f32[100]{0} all-reduce(parameter.1), replica_groups={{0,1,2,3}}, to_apply=primitive_computation_add__2.3
      tuple.8 = (f32[100]{0}) tuple(all-reduce.7)
      get-tuple-element.9 = f32[100]{0} get-tuple-element(tuple.8), index=0
      ROOT tuple.10 = (f32[100]{0}) tuple(get-tuple-element.9)
  }
  )";
  MultiHostHloRunner::Options options;
  options.num_replicas = 4;
  options.num_repeats = 5;
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<MultiHostHloRunner> hlo_runner,
                          MultiHostHloRunner::CreateMultiHostHloRunner(
                              options, MultiHostHloRunner::DeviceType::kGpu));
  auto run_status = hlo_runner->ParseAndRun(all_reduce_test_hlo);
  TF_EXPECT_OK(run_status.status());
}

TEST(MultiHostHloRunnerTest, TestSpmd) {
  constexpr absl::string_view spmd_test_hlo = R"(
    HloModule hlo_runner_test_0.1

    ENTRY hlo_runner_test_0.1 {
      %lhs = f32[32,32,64,128] parameter(0), sharding={devices=[1,4,1,1]0,1,2,3}
      %rhs = f32[32,1024,64,128] parameter(1), sharding={devices=[1,4,1,1]0,1,2,3}
      ROOT %conv = f32[32,32,1024,1] convolution(%lhs, %rhs),
        dim_labels=0bf1_0oi1->0bf1, window={size=32x128 stride=31x1 lhs_dilate=32x1}, sharding={devices=[1,4,1,1]0,1,2,3}
  }
  )";

  MultiHostHloRunner::Options options;
  options.num_partitions = 4;
  options.spmd_mode = MultiHostHloRunner::SpmdMode::kUseSpmdPartitioning;
  options.num_repeats = 2;
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<MultiHostHloRunner> hlo_runner,
                          MultiHostHloRunner::CreateMultiHostHloRunner(
                              options, MultiHostHloRunner::DeviceType::kGpu));
  auto run_status = hlo_runner->ParseAndRun(spmd_test_hlo);
  TF_EXPECT_OK(run_status.status());
}

TEST(MultiHostHloRunnerTest, TestCollectivePermute) {
  constexpr absl::string_view collective_permute_test_hlo = R"(
    HloModule collective_permute_test

    ENTRY after_optimizations_test {
      %parameter.1 = bf16[8]{0} parameter(0), sharding={replicated}
      ROOT %collective-permute.1 = bf16[8]{0} collective-permute(bf16[8]{0} parameter.1), source_target_pairs={{0,1},{1,2},{2,3}}, channel_id=1, sharding={replicated}
    }
  )";

  MultiHostHloRunner::Options options;
  options.num_partitions = 4;
  options.spmd_mode = MultiHostHloRunner::SpmdMode::kUseSpmdPartitioning;
  options.num_repeats = 3;
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<MultiHostHloRunner> hlo_runner,
                          MultiHostHloRunner::CreateMultiHostHloRunner(
                              options, MultiHostHloRunner::DeviceType::kGpu));
  TF_EXPECT_OK(hlo_runner->ParseAndRun(collective_permute_test_hlo).status());
}

TEST(MultiHostHloRunnerTest, TestSpmdAfterOptimizations) {
  constexpr absl::string_view spmd_after_optimizations_test_hlo = R"(
    HloModule spmd_after_optimizations_test, is_scheduled=true
    ENTRY %after_optimizations_test_spmd (param: bf16[8]) -> bf16[8] {
      %param = bf16[8]{0} parameter(0), sharding={replicated}
      %collective-permute = bf16[8]{0} collective-permute(bf16[8]{0} %param), channel_id=1, source_target_pairs={{0,1},{1,2},{2,3}}, backend_config="{\"flag_configs\":[],\"barrier_config\":{\"barrier_type\":\"CUSTOM\",\"id\":\"0\"},\"scoped_memory_configs\":[]}"
      ROOT %copy.1 = bf16[8]{0} copy(bf16[8]{0} %collective-permute), backend_config="{\"flag_configs\":[],\"window_config\":{\"kernel_window_bounds\":[],\"output_window_bounds\":[\"1\"],\"input_window_bounds\":[],\"estimated_cycles\":\"587\",\"iteration_bounds\":[\"1\"]},\"scoped_memory_configs\":[]}"
    }
  )";

  MultiHostHloRunner::Options options;
  options.num_partitions = 4;
  options.hlo_passes_mode =
      MultiHostHloRunner::HloPassesMode::kRunXLABackendOnly;
  options.spmd_mode = MultiHostHloRunner::SpmdMode::kUseSpmdPartitioning;
  options.spmd_partitioned_mode =
      MultiHostHloRunner::SpmdPartitionedMode::kIsSpmdPartitionedModule;
  options.num_repeats = 3;
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<MultiHostHloRunner> hlo_runner,
                          MultiHostHloRunner::CreateMultiHostHloRunner(
                              options, MultiHostHloRunner::DeviceType::kGpu));
  TF_EXPECT_OK(
      hlo_runner->ParseAndRun(spmd_after_optimizations_test_hlo).status());
}

TEST(MultiHostHloRunnerTest, TestResidualBlock) {
#if defined(THREAD_SANITIZER)
  GTEST_SKIP();
#endif
  constexpr absl::string_view residual_block_hlo = R"(
HloModule cluster_residual_block

%max_BF16.48 (lhs.49: bf16[], rhs.50: bf16[]) -> bf16[] {
  %lhs.49 = bf16[] parameter(0)
  %rhs.50 = bf16[] parameter(1)
  ROOT %maximum.51 = bf16[] maximum(bf16[] %lhs.49, bf16[] %rhs.50)
}

%resnet34_batch_normalization_sufficient_statistics_var_ss-reduction.132 (x.133: f32[], y.134: f32[]) -> f32[] {
  %x.133 = f32[] parameter(0)
  %y.134 = f32[] parameter(1)
  ROOT %add.135 = f32[] add(f32[] %x.133, f32[] %y.134)
}

%resnet34_batch_normalization_sufficient_statistics_mean_ss-reduction.136 (x.137: f32[], y.138: f32[]) -> f32[] {
  %x.137 = f32[] parameter(0)
  %y.138 = f32[] parameter(1)
  ROOT %add.139 = f32[] add(f32[] %x.137, f32[] %y.138)
}

%final_sum (x.1: bf16[], y.1: bf16[]) -> bf16[] {
  %x.1 = bf16[] parameter(0)
  %y.1 = bf16[] parameter(1)
  ROOT %add.1 = bf16[] add(bf16[] %x.1, bf16[] %y.1)
}

ENTRY xla_computation_residual_block {
  %param.1 = bf16[4,153,153,12]{3,2,1,0} parameter(0), sharding={devices=[1,4,1,1]0,1,2,3}
  %param.2 = bf16[4,4,12,64]{3,2,1,0} parameter(1)
  %param.3 = bf16[3,3,64,64]{3,2,1,0} parameter(2)
  %param.4 = bf16[3,3,64,64]{3,2,1,0} parameter(3)
  %param.5 = bf16[3,3,64,64]{3,2,1,0} parameter(4)
  %param.6 = bf16[3,3,64,128]{3,2,1,0} parameter(5)
  %param.7 = bf16[3,3,128,128]{3,2,1,0} parameter(6)
  %param.8 = bf16[1,1,64,128]{3,2,1,0} parameter(7)
  %param.9 = bf16[3,3,128,256]{3,2,1,0} parameter(8)
  %param.10 = bf16[3,3,256,256]{3,2,1,0} parameter(9)
  %param.11 = bf16[1,1,128,256]{3,2,1,0} parameter(10)
  %param.12 = f32[64]{0} parameter(11)
  %param.13 = f32[64]{0} parameter(12)

  %convolution.547 = bf16[4,150,150,64]{3,2,1,0} convolution(bf16[4,153,153,12]{3,2,1,0} %param.1, bf16[4,4,12,64]{3,2,1,0} %param.2), window={size=4x4}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="resnet34/conv2d/Conv2D"}
  
  %constant.548 = bf16[] constant(-inf), metadata={op_type="MaxPool" op_name="resnet34/max_pooling2d/MaxPool"}
  
  %reduce-window.549 = bf16[4,75,75,64]{3,2,1,0} reduce-window(bf16[4,150,150,64]{3,2,1,0} %convolution.547, bf16[] %constant.548), window={size=1x3x3x1 stride=1x2x2x1 pad=0_0x0_1x0_1x0_0}, to_apply=%max_BF16.48, metadata={op_type="MaxPool" op_name="resnet34/max_pooling2d/MaxPool"}
  
  %convolution.576 = bf16[4,75,75,64]{3,2,1,0} convolution(bf16[4,75,75,64]{3,2,1,0} %reduce-window.549, bf16[3,3,64,64]{3,2,1,0} %param.4), window={size=3x3 pad=1_1x1_1}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="resnet34/conv2d_1/Conv2D"}

  %convert.1118 = f32[4,75,75,64]{3,2,1,0} convert(bf16[4,75,75,64]{3,2,1,0} %convolution.576), metadata={op_type="Cast" op_name="resnet34/batch_normalization/Cast"}

  %convert.1125 = f32[4,75,75,64]{3,2,1,0} convert(f32[4,75,75,64]{3,2,1,0} %convert.1118), metadata={op_type="Sum" op_name="resnet34/batch_normalization/sufficient_statistics/mean_ss"}
  
  %constant.1126 = f32[] constant(0), metadata={op_type="Sum" op_name="resnet34/batch_normalization/sufficient_statistics/mean_ss"}
  
  %convert.1127 = f32[] convert(f32[] %constant.1126), metadata={op_type="Sum" op_name="resnet34/batch_normalization/sufficient_statistics/mean_ss"}
  
  %reduce.1128 = f32[64]{0} reduce(f32[4,75,75,64]{3,2,1,0} %convert.1125, f32[] %convert.1127), dimensions={0,1,2}, to_apply=%resnet34_batch_normalization_sufficient_statistics_mean_ss-reduction.136, metadata={op_type="Sum" op_name="resnet34/batch_normalization/sufficient_statistics/mean_ss"}
  
  %convert.1129 = f32[64]{0} convert(f32[64]{0} %reduce.1128), metadata={op_type="Sum" op_name="resnet34/batch_normalization/sufficient_statistics/mean_ss"}
  
  %multiply.1119 = f32[4,75,75,64]{3,2,1,0} multiply(f32[4,75,75,64]{3,2,1,0} %convert.1118, f32[4,75,75,64]{3,2,1,0} %convert.1118), metadata={op_type="Square" op_name="resnet34/batch_normalization/sufficient_statistics/Square"}
  
  %convert.1120 = f32[4,75,75,64]{3,2,1,0} convert(f32[4,75,75,64]{3,2,1,0} %multiply.1119), metadata={op_type="Sum" op_name="resnet34/batch_normalization/sufficient_statistics/var_ss"}
  
  %constant.1121 = f32[] constant(0), metadata={op_type="Sum" op_name="resnet34/batch_normalization/sufficient_statistics/var_ss"}
  
  %convert.1122 = f32[] convert(f32[] %constant.1121), metadata={op_type="Sum" op_name="resnet34/batch_normalization/sufficient_statistics/var_ss"}
  
  %reduce.1123 = f32[64]{0} reduce(f32[4,75,75,64]{3,2,1,0} %convert.1120, f32[] %convert.1122), dimensions={0,1,2}, to_apply=%resnet34_batch_normalization_sufficient_statistics_var_ss-reduction.132, metadata={op_type="Sum" op_name="resnet34/batch_normalization/sufficient_statistics/var_ss"}
  
  %convert.1124 = f32[64]{0} convert(f32[64]{0} %reduce.1123), metadata={op_type="Sum" op_name="resnet34/batch_normalization/sufficient_statistics/var_ss"}
  
  %concatenate.1130 = f32[128]{0} concatenate(f32[64]{0} %convert.1129, f32[64]{0} %convert.1124), dimensions={0}, metadata={op_type="ConcatV2" op_name="resnet34/batch_normalization/concat"}
  
  %slice.1136 = f32[64]{0} slice(f32[128]{0} %concatenate.1130), slice={[64:128]}, metadata={op_type="Slice" op_name="resnet34/batch_normalization/Slice_1"}
  
  %constant.1137 = f32[] constant(4.44444449e-05), metadata={op_type="Mul" op_name="resnet34/batch_normalization/normalize/Mul"}
  
  %broadcast.1138 = f32[64]{0} broadcast(f32[] %constant.1137), dimensions={}, metadata={op_type="Mul" op_name="resnet34/batch_normalization/normalize/Mul"}

  %multiply.1139 = f32[64]{0} multiply(f32[64]{0} %slice.1136, f32[64]{0} %broadcast.1138), metadata={op_type="Mul" op_name="resnet34/batch_normalization/normalize/Mul"}

  %slice.1131 = f32[64]{0} slice(f32[128]{0} %concatenate.1130), slice={[0:64]}, metadata={op_type="Slice" op_name="resnet34/batch_normalization/Slice"}

  %constant.1132 = f32[] constant(4.44444449e-05), metadata={op_type="Mul" op_name="resnet34/batch_normalization/normalize/mean"}

  %broadcast.1133 = f32[64]{0} broadcast(f32[] %constant.1132), dimensions={}, metadata={op_type="Mul" op_name="resnet34/batch_normalization/normalize/mean"}

  %multiply.1134 = f32[64]{0} multiply(f32[64]{0} %slice.1131, f32[64]{0} %broadcast.1133), metadata={op_type="Mul" op_name="resnet34/batch_normalization/normalize/mean"}

  %multiply.1135 = f32[64]{0} multiply(f32[64]{0} %multiply.1134, f32[64]{0} %multiply.1134), metadata={op_type="Square" op_name="resnet34/batch_normalization/normalize/Square"}

  %subtract.1140 = f32[64]{0} subtract(f32[64]{0} %multiply.1139, f32[64]{0} %multiply.1135), metadata={op_type="Sub" op_name="resnet34/batch_normalization/normalize/variance"}

  %constant.1141 = f32[] constant(1e-05), metadata={op_type="AddV2" op_name="resnet34/batch_normalization/batchnorm/add"}

  %broadcast.1142 = f32[64]{0} broadcast(f32[] %constant.1141), dimensions={}, metadata={op_type="AddV2" op_name="resnet34/batch_normalization/batchnorm/add"}
  		  
  %add.1143 = f32[64]{0} add(f32[64]{0} %subtract.1140, f32[64]{0} %broadcast.1142), metadata={op_type="AddV2" op_name="resnet34/batch_normalization/batchnorm/add"}
  
  %rsqrt.1144 = f32[64]{0} rsqrt(f32[64]{0} %add.1143), metadata={op_type="Rsqrt" op_name="resnet34/batch_normalization/batchnorm/Rsqrt"}


  %multiply.1156 = f32[64]{0} multiply(f32[64]{0} %rsqrt.1144, f32[64]{0} %param.12), metadata={op_type="Mul" op_name="resnet34/batch_normalization/batchnorm/mul"}
  
  %broadcast.1157 = f32[4,75,75,64]{3,2,1,0} broadcast(f32[64]{0} %multiply.1156), dimensions={3}, metadata={op_type="Mul" op_name="resnet34/batch_normalization/batchnorm/mul_1"}

%multiply.1158 = f32[4,75,75,64]{3,2,1,0} multiply(f32[4,75,75,64]{3,2,1,0} %convert.1118, f32[4,75,75,64]{3,2,1,0} %broadcast.1157), metadata={op_type="Mul" op_name="resnet34/batch_normalization/batchnorm/mul_1"}

  %multiply.1159 = f32[64]{0} multiply(f32[64]{0} %multiply.1134, f32[64]{0} %multiply.1156), metadata={op_type="Mul" op_name="resnet34/batch_normalization/batchnorm/mul_2"}
  
  %subtract.1160 = f32[64]{0} subtract(f32[64]{0} %param.13, f32[64]{0} %multiply.1159), metadata={op_type="Sub" op_name="resnet34/batch_normalization/batchnorm/sub"}
  
  %broadcast.1161 = f32[4,75,75,64]{3,2,1,0} broadcast(f32[64]{0} %subtract.1160), dimensions={3}, metadata={op_type="AddV2" op_name="resnet34/batch_normalization/batchnorm/add_1"}
  
  %add.1162 = f32[4,75,75,64]{3,2,1,0} add(f32[4,75,75,64]{3,2,1,0} %multiply.1158, f32[4,75,75,64]{3,2,1,0} %broadcast.1161), metadata={op_type="AddV2" op_name="resnet34/batch_normalization/batchnorm/add_1"}
  
  %convert.1163 = bf16[4,75,75,64]{3,2,1,0} convert(f32[4,75,75,64]{3,2,1,0} %add.1162), metadata={op_type="Cast" op_name="resnet34/batch_normalization/Cast_1"}

  %constant.1164 = bf16[] constant(0), metadata={op_type="Relu" op_name="resnet34/Relu"}

  %broadcast.1165 = bf16[4,75,75,64]{3,2,1,0} broadcast(bf16[] %constant.1164), dimensions={}, metadata={op_type="Relu" op_name="resnet34/Relu"}
  
  %maximum.1166 = bf16[4,75,75,64]{3,2,1,0} maximum(bf16[4,75,75,64]{3,2,1,0} %broadcast.1165, bf16[4,75,75,64]{3,2,1,0} %convert.1163), metadata={op_type="Relu" op_name="resnet34/Relu"}
  
  %convolution.603 = bf16[4,75,75,64]{3,2,1,0} convolution(bf16[4,75,75,64]{3,2,1,0} %maximum.1166, bf16[3,3,64,64]{3,2,1,0} %param.5), window={size=3x3 pad=1_1x1_1}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="resnet34/conv2d_2/Conv2D"}

  %constant.605 = bf16[] constant(0), metadata={op_type="Relu" op_name="resnet34/Relu"}

  %broadcast.606 = bf16[4,75,75,64]{3,2,1,0} broadcast(bf16[] %constant.605), dimensions={}, metadata={op_type="Relu" op_name="resnet34/Relu"}

  %add.604 = bf16[4,75,75,64]{3,2,1,0} add(bf16[4,75,75,64]{3,2,1,0} %convolution.603, bf16[4,75,75,64]{3,2,1,0} %reduce-window.549), metadata={op_type="AddV2" op_name="resnet34/add"}

  %maximum.607 = bf16[4,75,75,64]{3,2,1,0} maximum(bf16[4,75,75,64]{3,2,1,0} %broadcast.606, bf16[4,75,75,64]{3,2,1,0} %add.604), metadata={op_type="Relu" op_name="resnet34/Relu"}
  
  %constant.637 = bf16[] constant(0), metadata={op_type="Pad" op_name="resnet34/Pad_3"}
  
  %pad.638 = bf16[4,77,77,64]{3,2,1,0} pad(bf16[4,75,75,64]{3,2,1,0} %maximum.607, bf16[] %constant.637), padding=0_0x1_1x1_1x0_0, metadata={op_type="Pad" op_name="resnet34/Pad_3"}

  %convolution.665 = bf16[4,38,38,128]{3,2,1,0} convolution(bf16[4,77,77,64]{3,2,1,0} %pad.638, bf16[3,3,64,128]{3,2,1,0} %param.6), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="resnet34/conv2d_4/Conv2D"}

  %convolution.692 = bf16[4,38,38,128]{3,2,1,0} convolution(bf16[4,38,38,128]{3,2,1,0} %convolution.665, bf16[3,3,128,128]{3,2,1,0} %param.7), window={size=3x3 pad=1_1x1_1}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="resnet34/conv2d_5/Conv2D"}

  %constant.608 = bf16[] constant(0), metadata={op_type="Pad" op_name="resnet34/Pad_2"}

  %pad.609 = bf16[4,75,75,64]{3,2,1,0} pad(bf16[4,75,75,64]{3,2,1,0} %maximum.607, bf16[] %constant.608), padding=0_0x0_0x0_0x0_0, metadata={op_type="Pad" op_name="resnet34/Pad_2"}
  
  %convolution.636 = bf16[4,38,38,128]{3,2,1,0} convolution(bf16[4,75,75,64]{3,2,1,0} %pad.609, bf16[1,1,64,128]{3,2,1,0} %param.8), window={size=1x1 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="resnet34/conv2d_3/Conv2D"}
  
  %add.693 = bf16[4,38,38,128]{3,2,1,0} add(bf16[4,38,38,128]{3,2,1,0} %convolution.692, bf16[4,38,38,128]{3,2,1,0} %convolution.636), metadata={op_type="AddV2" op_name="resnet34/add_1"}
  
  %constant.694 = bf16[] constant(0), metadata={op_type="Relu" op_name="resnet34/Relu_1"}

  %broadcast.695 = bf16[4,38,38,128]{3,2,1,0} broadcast(bf16[] %constant.694), dimensions={}, metadata={op_type="Relu" op_name="resnet34/Relu_1"}
  
  %maximum.696 = bf16[4,38,38,128]{3,2,1,0} maximum(bf16[4,38,38,128]{3,2,1,0} %broadcast.695, bf16[4,38,38,128]{3,2,1,0} %add.693), metadata={op_type="Relu" op_name="resnet34/Relu_1"}
  
  %convolution.750 = bf16[4,38,38,256]{3,2,1,0} convolution(bf16[4,38,38,128]{3,2,1,0} %maximum.696, bf16[3,3,128,256]{3,2,1,0} %param.9), window={size=3x3 pad=1_1x1_1}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="resnet34/conv2d_7/Conv2D"}

  %convolution.777 = bf16[4,38,38,256]{3,2,1,0} convolution(bf16[4,38,38,256]{3,2,1,0} %convolution.750, bf16[3,3,256,256]{3,2,1,0} %param.10), window={size=3x3 pad=1_1x1_1}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="resnet34/conv2d_8/Conv2D"}
  
  %convolution.749 = bf16[4,38,38,256]{3,2,1,0} convolution(bf16[4,38,38,128]{3,2,1,0} %maximum.696, bf16[1,1,128,256]{3,2,1,0} %param.11), window={size=1x1}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="resnet34/conv2d_6/Conv2D"}
  
  %add.778 = bf16[4,38,38,256]{3,2,1,0} add(bf16[4,38,38,256]{3,2,1,0} %convolution.777, bf16[4,38,38,256]{3,2,1,0} %convolution.749), metadata={op_type="AddV2" op_name="resnet34/add_2"}

  %constant.2000 = bf16[] constant(0), metadata={op_type="Constant", op_name="FinalSum"}

  ROOT %reduce.2002 = bf16[4] reduce(bf16[4,38,38,256]{3,2,1,0} %add.778, bf16[] %constant.2000), dimensions={1,2,3}, to_apply=%final_sum
}
)";
  MultiHostHloRunner::Options options;
  options.num_partitions = 4;
  options.spmd_mode = MultiHostHloRunner::SpmdMode::kUseSpmdPartitioning;
  options.num_repeats = 1;

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<MultiHostHloRunner> hlo_runner,
                          MultiHostHloRunner::CreateMultiHostHloRunner(
                              options, MultiHostHloRunner::DeviceType::kGpu));
  TF_EXPECT_OK(hlo_runner->ParseAndRun(residual_block_hlo).status());
}

TEST(MultiHostHloRunnerTest, ReplicaCorrectness) {
  constexpr absl::string_view all_reduce_test_hlo = R"(
    HloModule hlo_runner_test_0.1
    primitive_computation_add__2.3 {
      parameter.4 = f32[] parameter(0), parameter_replication={false}
      parameter.5 = f32[] parameter(1), parameter_replication={false}
      ROOT add.6 = f32[] add(parameter.4, parameter.5)
    }
    ENTRY hlo_runner_test_0.1 {
      constant.2 = pred[] constant(false)
      parameter.1 = f32[100]{0} parameter(0), parameter_replication={false}
      ROOT all-reduce.7 = f32[100]{0} all-reduce(parameter.1), replica_groups={{0,1,2,3,4,5,6,7}}, to_apply=primitive_computation_add__2.3
  }
  )";
  MultiHostHloRunner::Options options;
  options.num_replicas = 8;
  options.spmd_mode = MultiHostHloRunner::SpmdMode::kNotUseSpmdPartitioning;
  options.num_repeats = 1;
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<MultiHostHloRunner> hlo_runner,
                          MultiHostHloRunner::CreateMultiHostHloRunner(
                              options, MultiHostHloRunner::DeviceType::kGpu));
  MultiHostHloRunner::PerDeviceLiteralVecType arguments;
  for (int i = 0; i < 8; ++i) {
    std::vector<float> vals(100, i);
    Literal val_literal = LiteralUtil::CreateR1<float>(vals);
    arguments[i].push_back(std::move(val_literal));
  }
  TF_ASSERT_OK_AND_ASSIGN(
      auto run_results,
      hlo_runner->ParseAndRun(all_reduce_test_hlo, arguments));
  std::vector<float> expected_results(100, 28);
  Literal expected_literal = LiteralUtil::CreateR1<float>(expected_results);
  for (int i = 0; i < 8; ++i) {
    auto near_or_equal = LiteralTestUtil::NearOrEqual(
        expected_literal, run_results.at(i)[0], ErrorSpec{1e-4, 1e-4});
    EXPECT_TRUE(near_or_equal);
  }
}

TEST(MultiHostHloRunnerTest, TestLargeArgument) {
#if defined(ADDRESS_SANITIZER) || defined(MEMORY_SANITIZER) || \
    defined(THREAD_SANITIZER)
  GTEST_SKIP();
#endif
  constexpr absl::string_view test_hlo = R"(
    HloModule hlo_runner_test_0.1
    primitive_computation_add__2.3 {
      parameter.4 = f32[] parameter(0), parameter_replication={false}
      parameter.5 = f32[] parameter(1), parameter_replication={false}
      ROOT add.6 = f32[] add(parameter.4, parameter.5)
    }
    ENTRY hlo_runner_test_0.1 {
      constant.2 = pred[] constant(false)
      parameter.1 = f32[524288,8,128] parameter(0), parameter_replication={false}
      all-reduce.7 = f32[524288,8,128] all-reduce(parameter.1), replica_groups={{0,1,2,3,4,5,6,7}}, to_apply=primitive_computation_add__2.3
      tuple.8 = (f32[524288,8,128]) tuple(all-reduce.7)
      get-tuple-element.9 = f32[524288,8,128] get-tuple-element(tuple.8), index=0
      ROOT tuple.10 = (f32[524288,8,128]) tuple(get-tuple-element.9)
  }
  )";
  MultiHostHloRunner::Options options;
  options.num_replicas = 8;
  options.module_argument_mode =
      MultiHostHloRunner::ModuleArgumentMode::kUseSharedRandomInputs;
  options.module_output_mode =
      MultiHostHloRunner::ModuleOutputMode::kNotReturnOutputs;
  options.num_repeats = 1;
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<MultiHostHloRunner> hlo_runner,
                          MultiHostHloRunner::CreateMultiHostHloRunner(
                              options, MultiHostHloRunner::DeviceType::kGpu));
  auto run_status = hlo_runner->ParseAndRun(test_hlo);
  TF_EXPECT_OK(run_status.status());
}

}  // namespace xla

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
