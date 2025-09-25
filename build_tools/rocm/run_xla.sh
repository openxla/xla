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

# This script runs XLA unit tests on ROCm platform by selecting tests that are
# tagged with requires-gpu-amd

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
TF_TESTS_PER_GPU=1
N_TEST_JOBS=$(expr ${TF_GPU_COUNT} \* ${TF_TESTS_PER_GPU})
amdgpuname=(`rocminfo | grep gfx | head -n 1`)
AMD_GPU_GFX_ID=${amdgpuname[1]}
echo ""
echo "Bazel will use ${N_BUILD_JOBS} concurrent build job(s) and ${N_TEST_JOBS} concurrent test job(s) for gpu ${AMD_GPU_GFX_ID}."
echo ""

export PYTHON_BIN_PATH=`which python3`
export TF_NEED_ROCM=1
export ROCM_PATH="/opt/rocm"

GPU_NAME=(`rocminfo | grep -m 1 gfx`)
GPU_NAME=${GPU_NAME[1]}

EXCLUDED_TESTS=(
# //xla/pjrt/c:pjrt_c_api_gpu_test_gpu_amd_any
PjrtCAPIGpuExtensionTest.TritonCompile
# //xla/backends/gpu/codegen/triton:fusion_emitter_device_test_gpu_amd_any
TritonEmitterTest.CheckRocmWarpSize
TritonEmitterTest.ConvertF16ToF8E5M2Exhaustive
TritonEmitterTest.FP8ToFP8EndToEnd
TritonEmitterTest.FusionWithOutputContainingMoreThanInt32MaxElementsExecutesCorrectly
BasicDotAlgorithmEmitterTestSuite/BasicDotAlgorithmEmitterTest.BasicAlgorithmIsEmittedCorrectly/ALG_DOT_F64_F64_F64
# //xla/backends/gpu/codegen/triton:fusion_emitter_device_legacy_test_gpu_amd_any
TritonGemmTest.BroadcastOfVectorConstantIsFused
TritonGemmTest.FailIfTooMuchShmem
TritonGemmTest.SplitAndTransposeLhsExecutesCorrectly
# //xla/backends/gpu/codegen/triton:fusion_emitter_int4_device_test_gpu_amd_any
TritonTest.NonstandardLayoutWithManyNonContractingDims
TritonTest.NonstandardLayoutWithManyNonContractingDimsReversedLayout
# //xla/hlo/builder/lib:self_adjoint_eig_test_gpu_amd_any marked as flaky but randomly red after 3 attempts
RandomEighTestInstantiation/RandomEighTest.Random/*
)

BAZEL_DISK_CACHE_SIZE=100G
BAZEL_DISK_CACHE_DIR="/tf/disk_cache/rocm-jaxlib-v0.6.0"
mkdir -p ${BAZEL_DISK_CACHE_DIR}

SCRIPT_DIR=$(realpath $(dirname $0))
TAG_FILTERS=$($SCRIPT_DIR/rocm_tag_filters.sh),gpu,-multigpu,-multi_gpu_h100,requires-gpu-amd,-skip_rocprofiler_sdk

SANITIZER_ARGS=()
if [[ $1 == "asan" ]]; then
    SANITIZER_ARGS+=("--test_env=ASAN_OPTIONS=suppressions=${SCRIPT_DIR}/asan_ignore_list.txt:use_sigaltstack=0")
    SANITIZER_ARGS+=("--test_env=LSAN_OPTIONS=suppressions=${SCRIPT_DIR}/lsan_ignore_list.txt:use_sigaltstack=0")
    SANITIZER_ARGS+=("--config=asan")
    TAG_FILTERS=$TAG_FILTERS,-noasan
    shift
elif [[ $1 == "tsan" ]]; then
    SANITIZER_ARGS+=("--test_env=TSAN_OPTIONS=suppressions=${SCRIPT_DIR}/tsan_ignore_list.txt::history_size=7:ignore_noninstrumented_modules=1")
    SANITIZER_ARGS+=("--config=tsan")
    TAG_FILTERS=$TAG_FILTERS,-notsan
    shift
fi

bazel --bazelrc=build_tools/rocm/rocm_xla.bazelrc test \
    --config=rocm_ci \
    --config=xla_sgpu \
    --disk_cache=${BAZEL_DISK_CACHE_DIR} \
    --profile=/tf/pkg/profile.json.gz \
    --experimental_disk_cache_gc_max_size=${BAZEL_DISK_CACHE_SIZE} \
    --experimental_guard_against_concurrent_changes \
    --build_tag_filters=$TAG_FILTERS \
    --test_tag_filters=$TAG_FILTERS \
    --test_timeout=920,2400,7200,9600 \
    --test_sharding_strategy=disabled \
    --test_output=errors \
    --flaky_test_attempts=3 \
    --keep_going \
    --local_test_jobs=${N_TEST_JOBS} \
    --test_env=TF_TESTS_PER_GPU=$TF_TESTS_PER_GPU \
    --test_env=TF_GPU_COUNT=$TF_GPU_COUNT \
    --action_env=TF_ROCM_AMDGPU_TARGETS=${GPU_NAME} \
    --action_env=XLA_FLAGS="--xla_gpu_enable_llvm_module_compilation_parallelism=true --xla_gpu_force_compilation_parallelism=16" \
    --run_under=//build_tools/ci:parallel_gpu_execute \
    --test_env=MIOPEN_FIND_ENFORCE=5 \
    --test_env=MIOPEN_FIND_MODE=1 \
    --test_filter=-$(IFS=: ; echo "${EXCLUDED_TESTS[*]}") \
    "${SANITIZER_ARGS[@]}" \
    "$@"

# clean up bazel disk_cache
bazel shutdown \
  --disk_cache=${BAZEL_DISK_CACHE_DIR} \
  --experimental_disk_cache_gc_max_size=${BAZEL_DISK_CACHE_SIZE}
