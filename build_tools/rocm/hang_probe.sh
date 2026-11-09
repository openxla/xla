#!/usr/bin/env bash
# Copyright 2026 The OpenXLA Authors.
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
#
# Hang probe for test actions, used as bazel test run_under.
#
# Logs the CPU resources the action gets and checks how the C++ runtime is
# linked into the test (see runtime_check), then runs the test. If the test is
# still running after HANG_PROBE_AFTER_S seconds, dumps the state of every
# thread from /proc, attaches rocgdb to print all thread backtraces and the
# loaded libraries, and kills the test with exit code 124. Every probe line
# starts with HANGPROBE and goes to stderr, so it ends up in test.log.
#
# Containers usually run with Yama ptrace_scope=1, which only lets an ancestor
# attach. The test therefore runs in a subshell that execs rocgdb, so rocgdb
# becomes the test's parent.

set -u

AFTER_S="${HANG_PROBE_AFTER_S:-300}"

log() { echo "HANGPROBE $*" >&2; }

cg_read() { [[ -r "$1" ]] && tr '\n' ' ' < "$1" 2>/dev/null; }

log_env() {
  log "ENV host=$(hostname 2>/dev/null) kernel=$(uname -r) nproc=$(nproc 2>/dev/null)" \
      "online=$(getconf _NPROCESSORS_ONLN 2>/dev/null)" \
      "cpus_allowed=$(awk '/^Cpus_allowed_list:/ { print $2 }' /proc/self/status)" \
      "loadavg=\"$(cut -d' ' -f1-3 /proc/loadavg)\""
  log "ENV cpu.max=\"$(cg_read /sys/fs/cgroup/cpu.max)\"" \
      "cfs_quota=$(cg_read /sys/fs/cgroup/cpu/cpu.cfs_quota_us)" \
      "cfs_period=$(cg_read /sys/fs/cgroup/cpu/cpu.cfs_period_us)" \
      "cpuset=$(cg_read /sys/fs/cgroup/cpuset.cpus.effective)" \
      "memory.max=$(cg_read /sys/fs/cgroup/memory.max)"
  log "ENV ptrace_scope=$(cat /proc/sys/kernel/yama/ptrace_scope 2>/dev/null || echo na)" \
      "shm=\"$(df -h /dev/shm 2>/dev/null | awk 'NR == 2 { print $2 }')\"" \
      "target=${TEST_TARGET:-na} shard=${TEST_SHARD_INDEX:-na}/${TEST_TOTAL_SHARDS:-na}" \
      "xla_flags=\"${XLA_FLAGS:-}\""
}

find_gdb() {
  local g
  for pattern in "*/bin/rocgdb-pynone" "*/bin/rocgdb"; do
    g="$(find -L "${TEST_SRCDIR:-.}" -path "$pattern" -type f 2>/dev/null | head -n 1)"
    [[ -n "$g" ]] && { echo "$g"; return; }
  done
  command -v gdb 2>/dev/null
}

# Thread names, states and kernel wait channels. Needs no ptrace, so it is
# printed even if rocgdb cannot attach.
dump_proc() {
  local pid="$1" t
  log "PROC pid=$pid threads=$(ls /proc/$pid/task 2>/dev/null | wc -l)"
  for t in /proc/"$pid"/task/*; do
    log "PROC tid=${t##*/} comm=$(cat "$t/comm" 2>/dev/null)" \
        "state=$(awk '{ print $3 }' "$t/stat" 2>/dev/null)" \
        "wchan=$(cat "$t/wchan" 2>/dev/null)"
  done
}

# Checks how the C++ runtime is linked into the test process:
#   * where libc++, libc++abi, libunwind, libatomic and libstdc++ resolve from;
#   * how many Bazel-built objects carry their own __cxa_guard_acquire. More
#     than one means several libc++abi copies share the guard variables of
#     inline statics, which can deadlock static initialization;
#   * which object the __cxa_* references of the ROCm libraries (built against
#     libstdc++) bind to. Binding to libc++abi.so.1 means the unversioned libc++
#     runtime captured libstdc++ users.
# Prints RUNTIME lines and sets RUNTIME_RC to 125 if a check failed.
RUNTIME_RC=0
runtime_check() {
  local bin="$1" tmp trace not_found objects copies=0 path bindings
  [[ "${HANG_PROBE_RUNTIME_CHECK:-1}" == "1" ]] || return 0
  [[ "$bin" == */* ]] || bin="./$bin"
  tmp="$(mktemp -d "${TEST_TMPDIR:-/tmp}/runtimecheck.XXXXXX")"

  trace="$(LD_TRACE_LOADED_OBJECTS=1 "$bin" 2>&1)"
  not_found="$(grep -c 'not found' <<< "$trace")"
  objects="$(awk '$2 == "=>" && $3 ~ /^\// { print $3 }' <<< "$trace")"
  log "RUNTIME objects=$(wc -l <<< "$objects") not_found=$not_found"
  grep 'not found' <<< "$trace" | head -n 5 | while read -r line; do
    log "RUNTIME not_found $line"
  done
  for lib in libc++.so.1 libc++abi.so.1 libunwind.so.1 libatomic.so.1 libstdc++.so.6; do
    log "RUNTIME lib $lib -> $(awk -v l="$lib" '$1 == l { print $3; exit }' <<< "$trace")"
  done

  if command -v nm > /dev/null; then
    while read -r path; do
      [[ -z "$path" ]] && continue
      case "$path" in
        *rocm_dist*|*libstdc++*|/lib/*|/lib64/*|/usr/*) continue ;;
      esac
      if nm "$path" 2> /dev/null | grep -qE ' [tTW] __cxa_guard_acquire$' ||
         nm -D --defined-only "$path" 2> /dev/null | grep -q ' __cxa_guard_acquire$'; then
        copies=$((copies + 1))
        (( copies <= 3 )) && log "RUNTIME guard_copy $path"
      fi
    done <<< "$(printf '%s\n%s\n' "$bin" "$objects")"
    log "RUNTIME guard_copies=$copies"
  else
    copies=-1
    log "RUNTIME guard_copies=unknown (no nm)"
  fi

  # Bind every symbol at load time, so the references of the ROCm libraries
  # show up without running any GPU code.
  LD_BIND_NOW=1 LD_DEBUG=bindings LD_DEBUG_OUTPUT="$tmp/bind" \
      timeout 300 "$bin" --gtest_list_tests > /dev/null 2>&1
  bindings="$(cat "$tmp"/bind.* 2> /dev/null |
      sed -nE "s/.*binding file ([^ ]+) \[[0-9]+\] to ([^ ]+) \[[0-9]+\]: normal symbol \`(__cxa_[a-z_]+)'.*/\1 \2 \3/p")"
  log "RUNTIME bindings_cxa=$(grep -c . <<< "$bindings")" \
      "rocm_to_libstdcxx=$(awk '$1 ~ /rocm_dist/ && $2 ~ /libstdc\+\+/' <<< "$bindings" | grep -c .)" \
      "rocm_to_libcxxabi=$(awk '$1 ~ /rocm_dist/ && $2 ~ /libc\+\+abi/' <<< "$bindings" | grep -c .)"
  awk '$1 ~ /rocm_dist/ && $2 ~ /libc\+\+abi/ { print $1, $3 }' <<< "$bindings" |
      sort -u | head -n 5 | while read -r from sym; do
    log "RUNTIME captured ${from##*/} $sym -> libc++abi"
  done
  log "RUNTIME guard_providers=$(awk '$1 !~ /rocm_dist/ && $3 == "__cxa_guard_acquire" { n = $2; sub(/.*\//, "", n); print n }' <<< "$bindings" | sort -u | tr '\n' ' ')"

  local reasons=""
  (( not_found > 0 )) && reasons+=" not_found"
  (( copies > 1 )) && reasons+=" guard_copies=$copies"
  awk '$1 ~ /rocm_dist/ && $2 ~ /libc\+\+abi/ { found = 1 } END { exit !found }' <<< "$bindings" &&
      reasons+=" rocm_bound_to_libcxxabi"
  if [[ -n "$reasons" ]]; then
    log "RUNTIME VERDICT FAIL$reasons"
    RUNTIME_RC=125
  else
    log "RUNTIME VERDICT OK"
  fi
  rm -rf "$tmp"
}

if (( $# == 0 )); then
  log "usage hang_probe.sh TEST_BINARY ARGS"
  exit 2
fi

log_env
runtime_check "$1"
GDB="$(find_gdb)"
log "GDB ${GDB:-none}"
PIDFILE="$(mktemp "${TEST_TMPDIR:-/tmp}/hangprobe.XXXXXX")"
START="$(date +%s)"

(
  "$@" &
  test_pid=$!
  echo "$test_pid" > "$PIDFILE"
  while kill -0 "$test_pid" 2>/dev/null; do
    if (( $(date +%s) - START >= AFTER_S )); then
      log "HANG test still running after ${AFTER_S}s, pid=$test_pid"
      dump_proc "$test_pid"
      if [[ -z "$GDB" ]]; then
        log "HANG no debugger found, killing the test"
        kill -KILL "$test_pid"
        exit 124
      fi
      log "STACKS BEGIN"
      exec "$GDB" -batch -nx -iex "set debuginfod enabled off" -p "$test_pid" \
          -ex "set pagination off" -ex "set width 0" -ex "set confirm off" \
          -ex "set print thread-events off" \
          -ex "info threads" -ex "thread apply all bt 80" \
          -ex "info sharedlibrary" -ex "kill" -ex "quit 124" >&2
    fi
    sleep 1
  done
  wait "$test_pid"
) &
RUNNER=$!
trap 'kill -TERM "$(cat "$PIDFILE" 2>/dev/null)" 2>/dev/null' TERM INT

wait "$RUNNER"
RC=$?
# If rocgdb failed to attach, it exits without killing the test.
TEST_PID="$(cat "$PIDFILE" 2>/dev/null)"
if [[ -n "$TEST_PID" ]] && kill -0 "$TEST_PID" 2>/dev/null; then
  log "STACKS END rocgdb rc=$RC, test still alive, killing it"
  kill -KILL "$TEST_PID" 2>/dev/null
  RC=124
elif (( $(date +%s) - START >= AFTER_S )); then
  log "STACKS END rc=$RC"
fi
rm -f "$PIDFILE"
# A failed runtime check fails an otherwise passing test, so it shows up.
if (( RC == 0 && RUNTIME_RC != 0 )); then
  RC="$RUNTIME_RC"
fi
log "EXIT rc=$RC wall_s=$(( $(date +%s) - START ))"
exit "$RC"
