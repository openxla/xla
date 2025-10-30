#!/usr/bin/env bash
# Copyright 2024 The TensorFlow Authors. All Rights Reserved.
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

# This is a rocm specific script housed under `build_tools/rocm`
# It runs following distributed tests which require more >= 4 gpus and these tests
# are skipped currently in the CI due to tag selection. These tests are tagged either as manual or with oss
# ```
# //xla/tests:collective_ops_e2e_test_gpu_amd_any
# //xla/tests:collective_ops_test_gpu_amd_any
# //xla/tests:replicated_io_feed_test_gpu_amd_any
# //xla/tools/multihost_hlo_runner:functional_hlo_runner_test_gpu_amd_any
# //xla/pjrt/distributed:topology_util_test
# //xla/pjrt/distributed:client_server_test
# ```
# Also these tests do not use `--run_under=//build_tools/ci:parallel_gpu_execute` with bazel which
# locks down individual gpus thus making multi gpu tests impossible to run

set -e
set -x

N_BUILD_JOBS=$(grep -c ^processor /proc/cpuinfo)
# If rocm-smi exists locally (it should) use it to find
# out how many GPUs we have to test with.
rocm-smi -i
STATUS=$?
if [ $STATUS -ne 0 ]; then TF_GPU_COUNT=1; else
   TF_GPU_COUNT=$(rocm-smi -i|grep 'Device ID' |grep 'GPU' |wc -l)
fi
if [[ $TF_GPU_COUNT -lt 4 ]]; then
    echo "Found only ${TF_GPU_COUNT} gpus, multi-gpu tests need atleast 4 gpus."
    exit
fi

amdgpuname=(`rocminfo | grep gfx | head -n 1`)
AMD_GPU_GFX_ID=${amdgpuname[1]}

export PYTHON_BIN_PATH=`which python3`
export TF_NEED_ROCM=1
export ROCM_PATH="/opt/rocm"

BAZEL_DISK_CACHE_SIZE=100G
BAZEL_DISK_CACHE_DIR="/tf/disk_cache/rocm-jaxlib-v0.7.1"
mkdir -p ${BAZEL_DISK_CACHE_DIR}
if [ ! -d /tf/pkg ]; then
	mkdir -p /tf/pkg
fi

EXCLUDED_TESTS=(
  CollectiveOpsTestE2E.MemcpyP2pLargeMessage
  RaggedAllToAllTest/RaggedAllToAllTest.RaggedAllToAll_8GPUs_2ReplicasPerGroups/sync_decomposer
  RaggedAllToAllTest/RaggedAllToAllTest.RaggedAllToAll_8GPUs_2ReplicasPerGroups/async_decomposer
  # //xla/backends/gpu/codegen/triton:fusion_emitter_parametrized_legacy_test_amdgpu_any
  ElementwiseTestSuiteF32/BinaryElementwiseTest.ElementwiseFusionExecutesCorrectly/f32_atan2
  # //xla/tests:collective_ops_e2e_test_amdgpu_any
  CollectiveOpsTestE2EPipelinedNonPipelined.CollectivePipelinerBackward
  CollectiveOpsTestE2EPipelinedNonPipelined.CollectivePipelinerBackwardStartFromOne
  # //xla/tools/multihost_hlo_runner:functional_hlo_runner_test
  FunctionalHloRunnerTest.Sharded2DevicesHloUnoptimizedSnapshot
  FunctionalHloRunnerTest.ShardedComputationUnderStreamCapture
)

SCRIPT_DIR=$(realpath $(dirname $0))
TAG_FILTERS="$($SCRIPT_DIR/rocm_tag_filters.sh)"

RBE_OPTIONS=()
SANITIZER_ARGS=()
if [[ $1 == "asan" ]]; then
    SANITIZER_ARGS+=("--run_under=//build_tools/rocm:sanitizer_wrapper")
    SANITIZER_ARGS+=("--config=asan")
    TAG_FILTERS="$TAG_FILTERS,-noasan"
    shift
elif [[ $1 == "tsan" ]]; then
    SANITIZER_ARGS+=("--run_under=//build_tools/rocm:sanitizer_wrapper")
    SANITIZER_ARGS+=("--config=tsan")
    TAG_FILTERS="$TAG_FILTERS,-notsan"
    # excluded from tsan
    EXCLUDED_TESTS+=(
        CollectiveOpsTest*
        Fp8CollectiveOpsTest.AllGather_8BitFloat
        Fp8CollectiveOpsTest.CollectivePermute_8BitFloat
        Fp8CollectiveOpsTest.AllToAll_8BitFloat
        AsyncCollectiveOps*
        AllReduceTest*
        RaggedAllToAllTest*
        AsyncCollectiveOps*
        AsyncMemcpyCollectiveOps*
        RaggedAllToAllTest*
    )

    #  tsan tests appear to be flaky in rbe due to the heavy load
    #  force them to run locally
    RBE_OPTIONS+=(
         --strategy=TestRunner=local
    )
    shift
fi

bazel --bazelrc=build_tools/rocm/rocm_xla.bazelrc test \
    --config=rocm_ci \
    --config=xla_mgpu \
    --profile=/tf/pkg/profile.json.gz \
    --disk_cache=${BAZEL_DISK_CACHE_DIR} \
    --experimental_disk_cache_gc_max_size=${BAZEL_DISK_CACHE_SIZE} \
    --experimental_guard_against_concurrent_changes \
    --build_tag_filters=$TAG_FILTERS \
    --test_tag_filters=$TAG_FILTERS \
    --test_timeout=920,2400,7200,9600 \
    --test_sharding_strategy=disabled \
    --test_output=errors \
    --flaky_test_attempts=3 \
    --keep_going \
    --run_under=//build_tools/rocm:exclusive_local_wrapper \
    --repo_env=TF_ROCM_AMDGPU_TARGETS=gfx908,gfx90a,gfx942,gfx1100 \
    --action_env=XLA_FLAGS=--xla_gpu_force_compilation_parallelism=16 \
    --action_env=XLA_FLAGS=--xla_gpu_enable_llvm_module_compilation_parallelism=true \
    --action_env=NCCL_MAX_NCHANNELS=1 \
    --test_filter=-$(IFS=: ; echo "${EXCLUDED_TESTS[*]}") \
    "${SANITIZER_ARGS[@]}" \
    "$@" \
    "${RBE_OPTIONS[@]}"

# clean up bazel disk_cache
bazel shutdown \
  --disk_cache=${BAZEL_DISK_CACHE_DIR} \
  --experimental_disk_cache_gc_max_size=${BAZEL_DISK_CACHE_SIZE}
