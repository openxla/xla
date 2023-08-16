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

#include <string>

#include <gtest/gtest.h>
#include "xla/pjrt/compile_options.pb.h"
#include "xla/xla.pb.h"
#include "tsl/lib/strings/proto_serialization.h"

namespace {

using xla::CompileOptionsProto;

CompileOptionsProto BuildCompileOptions() {
  CompileOptionsProto co_proto;
  xla::ExecutableBuildOptionsProto* ebo_proto =
      co_proto.mutable_executable_build_options();
  xla::DebugOptions* d_proto = ebo_proto->mutable_debug_options();

  d_proto->set_xla_backend_optimization_level(3);
  d_proto->set_xla_eliminate_hlo_implicit_broadcast(true);
  d_proto->set_xla_cpu_multi_thread_eigen(true);
  d_proto->set_xla_gpu_cuda_data_dir("./cuda_sdk_lib");
  d_proto->set_xla_llvm_enable_alias_scope_metadata(true);
  d_proto->set_xla_llvm_enable_noalias_metadata(true);
  d_proto->set_xla_llvm_enable_invariant_load_metadata(true);
  d_proto->set_xla_force_host_platform_device_count(1);
  d_proto->set_xla_cpu_fast_math_honor_nans(true);
  d_proto->set_xla_cpu_fast_math_honor_infs(true);
  d_proto->set_xla_allow_excess_precision(true);
  d_proto->set_xla_gpu_autotune_level(4);
  d_proto->set_xla_cpu_fast_math_honor_division(true);
  d_proto->set_xla_cpu_fast_math_honor_functions(true);
  d_proto->set_xla_dump_max_hlo_modules(-1);
  d_proto->set_xla_multiheap_size_constraint_per_heap(-1);
  d_proto->set_xla_detailed_logging_and_dumping(true);
  d_proto->set_xla_gpu_enable_async_all_reduce(true);
  d_proto->set_xla_gpu_strict_conv_algorithm_picker(true);
  d_proto->set_xla_gpu_all_reduce_combine_threshold_bytes(31457280);
  d_proto->set_xla_gpu_enable_cudnn_frontend(true);
  d_proto->set_xla_gpu_nccl_termination_timeout_seconds(-1);
  d_proto->set_xla_gpu_enable_shared_constants(true);
  d_proto->set_xla_gpu_redzone_scratch_max_megabytes(4096);
  d_proto->set_xla_gpu_simplify_all_fp_conversions(true);
  d_proto->set_xla_gpu_enable_xla_runtime_executable(true);
  // d_proto->set_xla_gpu_shape_checks(RUNTIME);
  d_proto->set_xla_gpu_normalize_layouts(true);
  d_proto->set_xla_cpu_enable_mlir_tiling_and_fusion(true);
  d_proto->set_xla_dump_enable_mlir_pretty_form(true);
  d_proto->set_xla_gpu_enable_triton_gemm(true);
  d_proto->set_xla_gpu_enable_cudnn_int8x32_convolution_reordering(true);
  d_proto->set_xla_cpu_enable_experimental_deallocation(true);
  d_proto->set_xla_cpu_enable_mlir_fusion_outlining(true);
  d_proto->set_xla_gpu_graph_level(1);
  d_proto->set_xla_cpu_matmul_tiling_m_dim(8);
  d_proto->set_xla_cpu_matmul_tiling_n_dim(8);
  d_proto->set_xla_cpu_matmul_tiling_k_dim(8);
  d_proto->set_xla_gpu_graph_num_runs_to_instantiate(-1);
  d_proto->set_xla_gpu_collective_inflation_factor(1);
  d_proto->set_xla_gpu_graph_min_graph_size(5);
  d_proto->set_xla_gpu_enable_reassociation_for_converted_ar(true);
  d_proto->set_xla_gpu_all_gather_combine_threshold_bytes(31457280);
  d_proto->set_xla_gpu_reduce_scatter_combine_threshold_bytes(31457280);
  d_proto->set_xla_gpu_enable_highest_priority_async_stream(true);
  d_proto->set_xla_gpu_enable_triton_softmax_fusion(true);
  d_proto->set_xla_gpu_auto_spmd_partitioning_memory_budget_ratio(1.1);
  d_proto->set_xla_gpu_redzone_padding_bytes(8388608);
  d_proto->set_xla_gpu_triton_fusion_level(2);
  d_proto->set_xla_gpu_graph_eviction_timeout_seconds(60);
  d_proto->set_xla_gpu_enable_gpu2_hal(true);
  d_proto->set_xla_gpu_copy_insertion_use_region_analysis(true);
  d_proto->set_xla_gpu_collective_permute_decomposer_threshold(
      9223372036854775807);

  ebo_proto->set_device_ordinal(-1);
  ebo_proto->set_num_replicas(2);
  ebo_proto->set_num_partitions(3);
  ebo_proto->set_use_spmd_partitioning(true);
  ebo_proto->add_allow_spmd_sharding_propagation_to_output(false);

  co_proto.set_profile_version(-1);
  auto* env_overrides = co_proto.mutable_env_option_overrides();
  (*env_overrides)["1"].set_string_field("1");
  (*env_overrides)["2"].set_string_field("2");

  return co_proto;
}

TEST(SerializationTest, DISABLED_Serialize) {
  auto co_proto = BuildCompileOptions();
  std::string serialized_proto;
  EXPECT_TRUE(tsl::SerializeToStringDeterministic(co_proto, &serialized_proto));
  std::string serialized_proto_2;
  EXPECT_TRUE(
      tsl::SerializeToStringDeterministic(co_proto, &serialized_proto_2));
  EXPECT_EQ(serialized_proto, serialized_proto_2);
}

}  // namespace
