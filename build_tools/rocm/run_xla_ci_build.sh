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

set -e
set -x

SCRIPT_DIR=$(realpath $(dirname $0))
TAG_FILTERS=$($SCRIPT_DIR/rocm_tag_filters.sh)

mkdir -p /tf/pkg

for arg in "$@"; do
    if [[ "$arg" == "--config=ci_multi_gpu" ]]; then
        TAG_FILTERS="${TAG_FILTERS},multi_gpu"
    fi
    if [[ "$arg" == "--config=ci_single_gpu" ]]; then
        TAG_FILTERS="${TAG_FILTERS},requires-gpu-rocm,requires-gpu-amd,-multi_gpu"
    fi
    if [[ "$arg" == "--config=ci_rocm_cpu" ]]; then
        TAG_FILTERS="${TAG_FILTERS},gpu,-requires-gpu-rocm,-requires-gpu-amd"
    fi
    if [[ "$arg" == "--config=no_flaky" ]]; then
        TAG_FILTERS="${TAG_FILTERS},-flaky"
    fi
done

SCRIPT_DIR=$(dirname $0)
BAZEL_ARGS=()
if [[ -n "${XLA_ROCM_TARGET_PATTERN_FILE:-}" && -f "${XLA_ROCM_TARGET_PATTERN_FILE}" ]]; then
    TAG_FILTERS="${TAG_FILTERS},-manual"
    for arg in "$@"; do
        if [[ "$arg" == "--" ]]; then
            break
        fi
        BAZEL_ARGS+=("$arg")
    done
    BAZEL_ARGS+=(
        "--skip_incompatible_explicit_targets"
        "--build_tests_only"
        "--target_pattern_file=${XLA_ROCM_TARGET_PATTERN_FILE}"
    )
else
    BAZEL_ARGS=("$@")
fi

# Hang probe. In the single GPU run, runs only the tests that hung, several
# times each, under the hang_probe target, which attaches rocgdb to a test that
# is still running after HANG_PROBE_AFTER_S seconds and prints all thread
# backtraces. Results are not taken from the cache.
if [[ "${XLA_CI_HANG_PROBE:-1}" == "1" && " $* " == *" --config=ci_single_gpu "* ]]; then
    read -r -a HANG_PROBE_TARGETS <<< "${XLA_CI_HANG_PROBE_TARGETS:-//xla/tests:convolution_gpu_alternative_layout_test_amdgpu_any //xla/tests:convolution_autotune_disabled_test_amdgpu_any //xla/hlo/builder/lib:svd_test_amdgpu_any}"
    BAZEL_ARGS=()
    for arg in "$@"; do
        if [[ "$arg" == "--" ]]; then
            break
        fi
        BAZEL_ARGS+=("$arg")
    done
    BAZEL_ARGS+=(
        "--run_under=//build_tools/rocm:hang_probe"
        "--test_env=HANG_PROBE_AFTER_S=${XLA_CI_HANG_PROBE_AFTER_S:-300}"
        "--runs_per_test=${XLA_CI_HANG_PROBE_RUNS:-10}"
        "--nocache_test_results"
        "--keep_going"
        "--remote_download_regex=.*/testlogs/.*/test\.log"
        "--"
        "${HANG_PROBE_TARGETS[@]}"
    )
    HANG_PROBE_MARKER="$(mktemp "${RUNNER_TEMP:-/tmp}/hangprobe_start.XXXXXX")"
fi

# Summarizes the RUNTIME lines that hang_probe.sh wrote to the test logs of
# this run, so passing tests report how the C++ runtime was linked too.
print_hang_probe_summary() {
    local logs
    logs="$(find -L bazel-testlogs -name test.log -newer "$HANG_PROBE_MARKER" 2>/dev/null)"
    echo "================ HANGPROBE runtime summary ================"
    echo "test logs: $(grep -c . <<< "$logs")"
    [[ -z "$logs" ]] && return
    echo "-- verdicts:"
    xargs -r grep -h "HANGPROBE RUNTIME VERDICT" <<< "$logs" | sort | uniq -c
    echo "-- distinct check results:"
    xargs -r grep -hE "HANGPROBE RUNTIME (lib |guard_copies|bindings_cxa|guard_providers|not_found|captured)" <<< "$logs" |
        sed 's/^HANGPROBE RUNTIME //' | sort | uniq -c | sort -rn | head -40
    echo "-- hangs: $(xargs -r grep -l "HANGPROBE HANG test still running" <<< "$logs" | wc -l)"
    echo "==========================================================="
}

set +e
bazel --bazelrc="$SCRIPT_DIR/rocm_xla_ci.bazelrc" test \
    --build_tag_filters=$TAG_FILTERS \
    --test_tag_filters=$TAG_FILTERS \
    --nokeep_going \
    --test_env=TF_TESTS_PER_GPU=1 \
    --action_env=XLA_FLAGS="--xla_gpu_force_compilation_parallelism=16 --xla_gpu_blas_max_algorithms=8" \
    --test_output=errors \
    --sandbox_add_mount_pair=/dev/null:/etc/ld.so.cache \
    --curses=no \
    --color=yes \
    "${BAZEL_ARGS[@]}"
BAZEL_RC=$?
set -e

if [[ -n "${HANG_PROBE_MARKER:-}" ]]; then
    { set +x; } 2>/dev/null
    print_hang_probe_summary
    rm -f "$HANG_PROBE_MARKER"
    set -x
fi

if [[ $BAZEL_RC -eq 4 && -n "${XLA_ROCM_TARGET_PATTERN_FILE:-}" ]]; then
    echo "Bazel returned exit code 4 (no matching test targets in ${XLA_ROCM_TARGET_PATTERN_FILE}); treating as success."
    exit 0
fi
exit $BAZEL_RC
