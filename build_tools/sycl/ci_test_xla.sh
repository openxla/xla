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
# ==============================================================================

# This script builds and executes tests. It can be run only on a system that
# has an Intel GPU with the appropriate driver installed.
# TEST_TARGETS="//xla/stream_executor/sycl/..." OPTIONAL_BAZEL_TEST_OPTIONS="--output_user_root=/tmp/bazel_tests --jobs=16 --cxxopt=-DTENSORFLOW_USE_SYCL" build_tools/sycl/ci_test_xla.sh

set -euo pipefail

jobs=256
no_of_gpus=$(ls /dev/dri/ | fgrep render | wc -l)
if [[ "${no_of_gpus}" -eq 0 ]]; then
  echo "unknown number of gpus."
  exit 1
fi

local_test_jobs=$((no_of_gpus * 8))
if [[ ${jobs} -lt 1 || ${jobs} -ge 512 ]]; then
  echo "jobs invalid setting ${jobs}"
  exit 1
fi
if [[ ${local_test_jobs} -lt 1 || ${local_test_jobs} -ge 64 ]]; then
  echo "local_test_jobs invalid setting ${local_test_jobs}"
  exit 1
fi

TEST_TARGETS="${TEST_TARGETS:-//xla/stream_executor/...}"
TEST_TO_SKIP="${TEST_TO_SKIP:-}"
TEST_TO_ADD="${TEST_TO_ADD:-}"
OPTIONAL_BAZEL_TEST_OPTIONS="${OPTIONAL_BAZEL_TEST_OPTIONS:-}"

DEFAULT_PARALLEL_FLAGS=""
if [[ ! " ${OPTIONAL_BAZEL_TEST_OPTIONS} " =~ [[:space:]]--jobs= ]]; then
  DEFAULT_PARALLEL_FLAGS+=" --jobs=${jobs}"
fi
if [[ ! " ${OPTIONAL_BAZEL_TEST_OPTIONS} " =~ [[:space:]]--local_test_jobs= ]]; then
  DEFAULT_PARALLEL_FLAGS+=" --local_test_jobs=${local_test_jobs}"
fi

DEFAULT_ROOT=""
if [[ " ${OPTIONAL_BAZEL_TEST_OPTIONS} " =~ [[:space:]](--output_user_root=[^[:space:]]+) ]]; then
  DEFAULT_ROOT="${BASH_REMATCH[1]}"
  OPTIONAL_BAZEL_TEST_OPTIONS="${OPTIONAL_BAZEL_TEST_OPTIONS/${BASH_REMATCH[1]}/}"
fi

TEST_PATTERNS="${TEST_TARGETS}"

if [[ -n "${TEST_TO_ADD}" ]]; then
  for target in ${TEST_TO_ADD}; do
    TEST_PATTERNS+=" ${target}"
  done
fi

if [[ -n "${TEST_TO_SKIP}" ]]; then
  for target in ${TEST_TO_SKIP}; do
    TEST_PATTERNS+=" -${target}"
  done
fi

echo "TEST_TARGETS=${TEST_TARGETS}"
echo "TEST_TO_SKIP=${TEST_TO_SKIP}"
echo "TEST_TO_ADD=${TEST_TO_ADD}"
echo "TEST_PATTERNS=${TEST_PATTERNS}"
echo "OPTIONAL_BAZEL_TEST_OPTIONS=${OPTIONAL_BAZEL_TEST_OPTIONS}"
echo "DEFAULT_PARALLEL_FLAGS=${DEFAULT_PARALLEL_FLAGS}"
echo "DEFAULT_ROOT=${DEFAULT_ROOT}"

bazel ${DEFAULT_ROOT:-} test \
  --config=sycl_hermetic --verbose_failures -c opt \
  ${DEFAULT_PARALLEL_FLAGS} \
  --test_timeout=60,300,900,3600 --flaky_test_attempts=2 --keep_going --test_keep_going \
  --build_tag_filters=gpu,oneapi-only,requires-gpu-intel,-requires-gpu-amd,-requires-gpu-nvidia,-no_oss,-cuda-only,-rocm-only,-no-oneapi \
  --test_tag_filters=gpu,oneapi-only,requires-gpu-intel,-requires-gpu-amd,-requires-gpu-nvidia,-no_oss,-cuda-only,-rocm-only,-no-oneapi \
  ${OPTIONAL_BAZEL_TEST_OPTIONS} \
  -- \
  ${TEST_PATTERNS}
