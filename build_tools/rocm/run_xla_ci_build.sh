#!/usr/bin/env bash
# Copyright 2025 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ==============================================================================

SCRIPT_DIR=$(realpath $(dirname $0))
TAG_FILTERS=$($SCRIPT_DIR/rocm_tag_filters.sh),-skip_rocprofiler_sdk,-oss_excluded,-oss_serial
BAZEL_DISK_CACHE_DIR="/tf/disk_cache/rocm-jaxlib"
mkdir -p ${BAZEL_DISK_CACHE_DIR}
mkdir -p /tf/pkg

for arg in "$@"; do
    if [[ "$arg" == "--config=asan" ]]; then
        TAG_FILTERS="${TAG_FILTERS},-noasan"
    fi
    if [[ "$arg" == "--config=tsan" ]]; then
        TAG_FILTERS="${TAG_FILTERS},-notsan"
    fi
    if [[ "$arg" == "--config=ci_multi_gpu" ]]; then
        TAG_FILTERS="${TAG_FILTERS},multi_gpu"
    fi
    if [[ "$arg" == "--config=ci_single_gpu" ]]; then
        TAG_FILTERS="${TAG_FILTERS},gpu,-multi_gpu"
    fi
done

set -x

bazel --bazelrc="$SCRIPT_DIR/rocm_xla.bazelrc" test \
    --disk_cache=${BAZEL_DISK_CACHE_DIR} \
    --config=rocm_rbe \
    --disk_cache=${BAZEL_DISK_CACHE_DIR} \
    --build_tag_filters=$TAG_FILTERS \
    --test_tag_filters=$TAG_FILTERS \
    --test_timeout=920,2400,7200,9600 \
    --profile=/tf/pkg/profile.json.gz \
    --keep_going \
    --test_env=TF_TESTS_PER_GPU=1 \
    --action_env=XLA_FLAGS="--xla_gpu_enable_llvm_module_compilation_parallelism=true --xla_gpu_force_compilation_parallelism=16" \
    --test_output=errors \
    --local_test_jobs=4 \
    "$@" \
    -//xla/tests:collective_pipeline_parallelism_test \
    -//xla/backends/gpu/codegen/emitters/tests:reduce_row/mof_scalar_variadic.hlo.test \
    -//xla/backends/gpu/codegen/emitters/tests:reduce_row/side_output_broadcast.hlo.test \
    -//xla/backends/gpu/codegen/triton:dot_algorithms_test_amdgpu_any \
    -//xla/backends/gpu/codegen/triton:fusion_emitter_device_legacy_port_test_amdgpu_any \
    -//xla/backends/gpu/codegen/triton:fusion_emitter_device_test_amdgpu_any \
    -//xla/backends/gpu/codegen/triton:fusion_emitter_int4_device_test_amdgpu_any \
    -//xla/backends/gpu/codegen/triton:fusion_emitter_parametrized_test_amdgpu_any \
    -//xla/backends/gpu/codegen/triton:support_legacy_test_amdgpu_any \
    -//xla/backends/gpu/codegen/triton:support_test \
    -//xla/backends/gpu/runtime:command_buffer_conversion_pass_test_amdgpu_any \
    -//xla/backends/gpu/runtime:kernel_thunk_test_amdgpu_any \
    -//xla/backends/gpu/runtime:topk_test_amdgpu_any \
    -//xla/codegen/emitters/tests:loop/broadcast_constant_block_dim_limit.hlo.test \
    -//xla/hlo/builder/lib:self_adjoint_eig_test_amdgpu_any \
    -//xla/hlo/builder/lib:svd_test_amdgpu_any \
    -//xla/hlo/builder/lib:svd_test_amdgpu_any_notfrt \
    -//xla/pjrt/c:pjrt_c_api_gpu_test_amdgpu_any \
    -//xla/service/gpu:determinism_test_amdgpu_any \
    -//xla/service/gpu:dot_algorithm_support_test_amdgpu_any \
    -//xla/service/gpu/tests:command_buffer_test_amdgpu_any \
    -//xla/service/gpu/tests:command_buffer_test_amdgpu_any_notfrt \
    -//xla/service/gpu/tests:gpu_kernel_tiling_test_amdgpu_any \
    -//xla/service/gpu/tests:gpu_triton_custom_call_test_amdgpu_any \
    -//xla/service/gpu/tests:sorting_test_amdgpu_any \
    -//xla/service/gpu/transforms:cublas_gemm_rewriter_test_amdgpu_any \
    -//xla/service/gpu/transforms:layout_assignment_a100.hlo.test \
    -//xla/service/gpu/transforms:layout_assignment_h100.hlo.test \
    -//xla/service/gpu/transforms:layout_assignment_v100.hlo.test \
    -//xla/service/gpu/transforms:triton_fusion_numerics_verifier_test_amdgpu_any \
    -//xla/service/gpu/transforms:triton_fusion_numerics_verifier_test_amdgpu_any_notfrt \
    -//xla/tests:convolution_test_amdgpu_any \
    -//xla/tests:convolution_test_amdgpu_any_notfrt \
    -//xla/tests:multioutput_fusion_test_amdgpu_any \
    -//xla/tests:sample_file_test_amdgpu_any \
    -//xla/tests:scatter_test_amdgpu_any \
    -//xla/tests:scatter_test_amdgpu_any_notfrt \
    -//xla/tools/hlo_opt:tests/gpu_hlo_llvm.hlo.test \
    -//xla/backends/gpu/codegen/triton:dot_algorithms_legacy_test_amdgpu_any \
    -//xla/tests:cholesky_test_amdgpu_any \
    -//xla/service/gpu/tests:sorting.hlo.test \
    -//xla/service/gpu/llvm_gpu_backend:amdgpu_bitcode_link_test \
    -//xla/tests:triangular_solve_test_amdgpu_any \
    -//xla/tests:batch_norm_training_test_amdgpu_any_notfrt \
    -//xla/backends/gpu/codegen/triton:fusion_emitter_parametrized_legacy_test_amdgpu_any \
    -//xla/hlo/builder/lib:self_adjoint_eig_test_amdgpu_any_notfrt \
    -//xla/tests:convert_test_amdgpu_any \
    -//xla/pjrt/gpu/tfrt:tfrt_gpu_buffer_test \
    -//xla/service/gpu/tests:gpu_cub_sort_test_amdgpu_any \
    -//xla/service/gpu:auto_sharding_gpu_compiler_test_amdgpu_any \
    -//xla/backends/gpu/codegen/triton:fusion_emitter_large_test_amdgpu_any \
    -//xla/backends/gpu/codegen:dynamic_slice_fusion_test_amdgpu_any \
    -//xla/tests:nccl_group_execution_test_amdgpu_any \
    -//xla/tools/multihost_hlo_runner:functional_hlo_runner_test_amdgpu_any \
    -//xla/tests:collective_ops_e2e_test_amdgpu_any

result=$?

# clean up nccl- files
rm -rf /dev/shm/nccl-*

# clean up bazel disk_cache
bazel shutdown \
    --disk_cache=${BAZEL_DISK_CACHE_DIR} \
    --experimental_disk_cache_gc_max_size=100G

exit $result
