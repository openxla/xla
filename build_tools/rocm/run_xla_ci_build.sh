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

# Memory probe. Wraps every test in the mem_probe target, which logs the
# container memory before and after each test and its peak, and downloads all
# test.log files so the summary below can include passing tests. Test results
# are not taken from the cache, so every run measures the tests again. Configs
# that set their own run_under come later in BAZEL_ARGS and take precedence.
MEM_PROBE_ARGS=()
MEM_PROBE_MARKER=""
if [[ "${XLA_CI_MEM_PROBE:-1}" == "1" ]]; then
    MEM_PROBE_ARGS=(
        "--run_under=//build_tools/rocm:mem_probe"
        "--remote_download_regex=.*/testlogs/.*/test\.log"
        "--nocache_test_results"
    )
    MEM_PROBE_MARKER="$(mktemp "${RUNNER_TEMP:-/tmp}/memprobe_start.XXXXXX")"
fi

# Prints one row per test action of this run from the MEMPROBE lines in the
# test logs, sorted by container memory growth, then one row per container in
# start order to show whether memory left behind by earlier actions builds up.
print_mem_probe_summary() {
    local testlogs="bazel-testlogs"
    local rows
    if [[ ! -d "$testlogs" ]]; then
        echo "MEMPROBE summary skipped, no $testlogs directory"
        return
    fi
    rows="$(mem_probe_rows "$testlogs")"
    echo "===================== MEMPROBE summary (MiB) ====================="
    echo "oom_kill counts kills of any kind, oom_lim only the container hitting"
    echo "its own limit. A kill with oom_lim unchanged and low worker_min means"
    echo "the whole worker ran out of memory."
    printf "%-8s %8s %9s %9s %9s %10s %9s %12s %6s  %-20s %-8s %-14s %-12s %10s %7s  %s\n" \
        exit test_peak before growth peak limit oom_kill container_age procs \
        start_utc boot_id container gpu worker_min oom_lim action
    printf "%s\n" "$rows" | sort -k4,4nr
    echo "============ MEMPROBE per container, in start order (MiB) ============"
    echo "before is the container memory when an action starts. A rising trend"
    echo "means earlier actions in that container left memory behind."
    printf "%-14s %-8s %-12s %7s %-20s %-20s %8s %8s %8s %7s %10s %6s  %s\n" \
        container boot_id gpu actions first_start last_start before_1st \
        before_last before_max trend max_growth kills flag
    printf "%s\n" "$rows" | awk '$12 != "-"' | sort -k12,12 -k10,10 | awk '
        function num(v) { return v ~ /^[0-9]+$/ }
        function flush(   flag, trend) {
            if (ctr == "") return
            trend = (num(b1) && num(bl)) ? bl - b1 : "-"
            flag = (trend != "-" && trend > 256) || bmax > 512 ? "ACCUMULATES?" : ""
            printf "%-14s %-8s %-12s %7d %-20s %-20s %8s %8s %8d %7s %10d %6d  %s\n",
                ctr, boot, gpu, n, s1, sl, b1, bl, bmax, trend, gmax, kills, flag
        }
        $12 != ctr {
            flush()
            ctr = $12; boot = $11; gpu = $13; n = 0; s1 = $10; b1 = $3
            bmax = 0; gmax = 0; kills = 0
        }
        {
            n++; sl = $10; bl = $3
            if (num($3) && $3 > bmax) bmax = $3
            if (num($4) && $4 > gmax) gmax = $4
            if ($1 == "KILLED?" || $1 == "137") kills++
        }
        END { flush() }' | sort -k10,10nr
    echo "=================================================================="
}

# Turns the MEMPROBE lines of every test.log of this run into one row each.
mem_probe_rows() {
    find -L "$1" -name test.log -newer "$MEM_PROBE_MARKER" -print0 2>/dev/null |
        xargs -0 -r awk '
            function val(s,   kv) { split(s, kv, "="); return kv[2] }
            function flush() {
                if (name == "") return
                printf "%-8s %8s %9s %9s %9s %10s %9s %12s %6s  %-20s %-8s %-14s %-12s %10s %7s  %s\n",
                    ex, rss, before, growth, peak, limit, oom, age, procs,
                    start, boot, ctr, gpu, wmin, olim, name
            }
            FNR == 1 {
                flush()
                name = FILENAME
                sub(/^bazel-testlogs\//, "", name)
                sub(/\/test\.log$/, "", name)
                ex = "KILLED?"; rss = "-"; before = "-"; growth = "-"; peak = "-"
                limit = "-"; oom = "-"; age = "-"; procs = "-"
                start = "-"; boot = "-"; ctr = "-"; gpu = "-"; wmin = "-"; olim = "-"
            }
            /^MEMPROBE BEFORE worker / {
                for (i = 4; i <= NF; i++) if ($i ~ /^avail_mib=/) wmin = val($i)
            }
            /^MEMPROBE BEFORE id / {
                for (i = 4; i <= NF; i++) {
                    if ($i ~ /^start_utc=/) start = val($i)
                    if ($i ~ /^boot_id=/) boot = substr(val($i), 1, 8)
                    if ($i ~ /^container=/) ctr = substr(val($i), 1, 14)
                    if ($i ~ /^gpus=/) { gpu = val($i); sub(/\/.*/, "", gpu) }
                }
            }
            /^MEMPROBE BEFORE host=/ {
                for (i = 3; i <= NF; i++) {
                    if ($i ~ /^container_age_s=/) age = val($i)
                    if ($i ~ /^procs=/) procs = val($i)
                }
            }
            /^MEMPROBE BEFORE memory_max=/ {
                for (i = 3; i <= NF; i++) {
                    if ($i ~ /^memory_max=/) {
                        limit = val($i)
                        if (limit ~ /^[0-9]+$/) limit = int(limit / 1048576)
                    }
                    if ($i ~ /^current_mib=/) before = val($i)
                }
            }
            /^MEMPROBE LIVE / {
                for (i = 3; i <= NF; i++) {
                    if ($i ~ /^growth_mib=/) growth = val($i)
                    if ($i ~ /^worker_avail_mib=/ && (wmin == "-" || val($i) + 0 < wmin + 0))
                        wmin = val($i)
                }
            }
            /^MEMPROBE ACTION / {
                for (i = 3; i <= NF; i++) {
                    if ($i ~ /^exit=/) ex = val($i)
                    if ($i ~ /^test_peak_rss_mib=/) rss = val($i)
                    if ($i ~ /^container_before_mib=/) before = val($i)
                    if ($i ~ /^container_growth_mib=/) growth = val($i)
                    if ($i ~ /^container_peak_during_action_mib=/) peak = val($i)
                    if ($i ~ /^oom_kill_before=/) ob = val($i)
                    if ($i ~ /^oom_kill_after=/) oom = ob "->" val($i)
                    if ($i ~ /^oom_before=/) lb = val($i)
                    if ($i ~ /^oom_after=/) olim = lb "->" val($i)
                    if ($i ~ /^worker_avail_min_mib=/) wmin = val($i)
                }
            }
            END { flush() }'
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
    "${MEM_PROBE_ARGS[@]}" \
    "${BAZEL_ARGS[@]}"
BAZEL_RC=$?
set -e

if [[ -n "$MEM_PROBE_MARKER" ]]; then
    { set +x; } 2>/dev/null
    print_mem_probe_summary
    rm -f "$MEM_PROBE_MARKER"
    set -x
fi

if [[ $BAZEL_RC -eq 4 && -n "${XLA_ROCM_TARGET_PATTERN_FILE:-}" ]]; then
    echo "Bazel returned exit code 4 (no matching test targets in ${XLA_ROCM_TARGET_PATTERN_FILE}); treating as success."
    exit 0
fi
exit $BAZEL_RC
