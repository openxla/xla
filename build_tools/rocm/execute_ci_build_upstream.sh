#!/usr/bin/env bash
# Copyright 2026 The OpenXLA Authors. All Rights Reserved.
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
# ============================================================================

set -ex

SCRIPT_DIR=$(realpath "$(dirname "$0")")

EXCLUDED_TESTS=(
    "*ParametersUsedByCollectiveMosaicShouldBeCopiedToCollectiveMemory"
    "SortingTest*"
    "*IotaR1Test*"
    "HostMemoryAllocateTest.Numa"
    "CubSort*"
)

TAG_FILTERS=$("${SCRIPT_DIR}/rocm_tag_filters.sh")

for arg in "$@"; do
    if [[ "$arg" == "--config=ci_multi_gpu" ]]; then
        TAG_FILTERS="${TAG_FILTERS},multi_gpu,-no_oss"
    fi
    if [[ "$arg" == "--config=ci_single_gpu" ]]; then
        TAG_FILTERS="${TAG_FILTERS},gpu,-multi_gpu,-no_oss"
    fi
done

"${SCRIPT_DIR}/run_xla_ci_build.sh" \
    "$@" \
    --build_tag_filters="$TAG_FILTERS" \
    --test_tag_filters="$TAG_FILTERS" \
    --execution_log_compact_file=execution_log.binpb.zst \
    --spawn_strategy=local \
    --repo_env=REMOTE_GPU_TESTING=1 \
    --repo_env=TF_ROCM_AMDGPU_TARGETS=gfx90a,gfx942 \
    --remote_download_outputs=minimal \
    --grpc_keepalive_time=30s \
    --test_sharding_strategy=disabled \
    --test_verbose_timeout_warnings \
    --curses=no \
    --color=yes \
    --test_filter=-$(
        IFS=:
        echo "${EXCLUDED_TESTS[*]}"
    ) \
    -- //xla/...
