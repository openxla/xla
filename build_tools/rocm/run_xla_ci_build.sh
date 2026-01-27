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

set -x

SCRIPT_DIR=$(realpath $(dirname $0))
TAG_FILTERS=$($SCRIPT_DIR/rocm_tag_filters.sh),-skip_rocprofiler_sdk,-no_oss,-oss_excluded,-oss_serial

mkdir -p /tf/pkg
BEP_JSON="/tmp/bep.json"

for arg in "$@"; do
    if [[ "$arg" == "--config=ci_multi_gpu" ]]; then
        TAG_FILTERS="${TAG_FILTERS},multi_gpu"
    fi
    if [[ "$arg" == "--config=ci_single_gpu" ]]; then
        TAG_FILTERS="${TAG_FILTERS},gpu,-multi_gpu"
    fi
done

SCRIPT_DIR=$(dirname $0)

set +e
bazel --bazelrc="$SCRIPT_DIR/rocm_xla_ci.bazelrc" test \
    --build_tag_filters=$TAG_FILTERS \
    --test_tag_filters=$TAG_FILTERS \
    --build_event_json_file="$BEP_JSON" \
    --profile=/tf/pkg/profile.json.gz \
    --keep_going \
    --test_env=TF_TESTS_PER_GPU=1 \
    --action_env=XLA_FLAGS="--xla_gpu_enable_llvm_module_compilation_parallelism=true --xla_gpu_force_compilation_parallelism=16" \
    --test_output=errors \
    --run_under=//build_tools/rocm:parallel_gpu_execute \
    "$@"

BAZEL_EXIT_CODE=$?
set -e

# Classify and exit: 0 = success, 78 = infra failure, other = build/test failure
if [[ $BAZEL_EXIT_CODE -eq 0 ]]; then
    exit 0
elif [[ -f "$BEP_JSON" ]] && jq -e 'select(.aborted.reason) | .aborted.reason | test("REMOTE_FAILURE|OUT_OF_MEMORY|INTERNAL|LOADING_FAILURE|NO_ANALYZE|NO_BUILD")' "$BEP_JSON" >/dev/null 2>&1; then
    echo "::warning::Infrastructure failure detected (exit code: $BAZEL_EXIT_CODE)"
    exit 78
else
    echo "::error::Build/test failure (exit code: $BAZEL_EXIT_CODE)"
    exit $BAZEL_EXIT_CODE
fi
