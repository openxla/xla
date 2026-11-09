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
# Logs the CPU resources the action gets, then runs the test. If the test is
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

if (( $# == 0 )); then
  log "usage hang_probe.sh TEST_BINARY ARGS"
  exit 2
fi

log_env
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
log "EXIT rc=$RC wall_s=$(( $(date +%s) - START ))"
exit "$RC"
