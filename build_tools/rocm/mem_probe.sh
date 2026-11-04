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
# Memory probe for test actions, used as bazel test run_under.
#
# Prints the memory state of the container the test runs in before and after
# the test, plus the peak memory of the test and of the container while the
# test ran. RBE workers reuse containers across actions, so the BEFORE snapshot
# shows anything earlier actions left behind. Every line starts with MEMPROBE
# and goes to stderr, so it ends up in test.log. The LIVE lines are printed
# while the test runs, so they survive even if the action is OOM killed.

set -u

INTERVAL="${MEMPROBE_INTERVAL:-0.5}"
STEP_MIB="${MEMPROBE_STEP_MIB:-1024}"
CLK_TCK="$(getconf CLK_TCK 2>/dev/null || echo 100)"

log() { echo "MEMPROBE $*" >&2; }
mib() { [[ "$1" =~ ^[0-9]+$ ]] && echo $(( $1 / 1048576 )) || echo "${1:-na}"; }

CG=/sys/fs/cgroup
if [[ -r "$CG/memory.current" ]]; then
  CGV=2
elif [[ -r "$CG/memory/memory.usage_in_bytes" ]]; then
  CGV=1
  CG="$CG/memory"
else
  CGV=0
fi

# Reads a cgroup file by its v2 name, falling back to the v1 name.
cg_file() {
  local f
  if (( CGV == 2 )); then f="$CG/$1"; else f="$CG/$2"; fi
  [[ -r "$f" ]] && cat "$f" 2>/dev/null
}
cg_current() { cg_file memory.current memory.usage_in_bytes; }

# Seconds since a process started, from /proc/PID/stat field 22.
proc_age() {
  local stat rest uptime
  stat="$(cat "/proc/$1/stat" 2>/dev/null)" || return
  rest="${stat##*) }"
  read -r -a f <<< "$rest"
  read -r uptime _ < /proc/uptime
  awk -v up="$uptime" -v st="${f[19]}" -v hz="$CLK_TCK" \
      'BEGIN { printf "%d", up - st / hz }'
}

# Largest processes in the container, skipping the probe and its helpers.
top_procs() {
  local d pid comm rss
  for d in /proc/[0-9]*; do
    pid="${d#/proc/}"
    [[ "$pid" == "$$" || "$pid" == "$BASHPID" ]] && continue
    comm="$(cat "$d/comm" 2>/dev/null)" || continue
    case "$comm" in
      mem_probe.sh|sleep|awk|sort|head|cat|du|df|getconf|date) continue ;;
    esac
    rss="$(awk '/^VmRSS:/ { print $2 }' "$d/status" 2>/dev/null)"
    echo "${rss:-0} $pid $comm"
  done | sort -rn | head -n 8
}

# GPUs from the KFD topology as PCI address and unique id. The host boot_id is
# shared by all containers on one machine, so together they identify the
# worker even though hostname is only the container id.
gpu_ids() {
  local p loc dom uid out=""
  for p in /sys/class/kfd/kfd/topology/nodes/*/properties; do
    [[ -r "$p" ]] || continue
    awk '$1 == "simd_count" && $2 > 0 { found = 1 } END { exit !found }' "$p" || continue
    loc="$(awk '$1 == "location_id" { print $2 }' "$p")"
    dom="$(awk '$1 == "domain" { print $2 }' "$p")"
    uid="$(awk '$1 == "unique_id" { print $2 }' "$p")"
    out+="$(printf "%04x:%02x:%02x.%x" "${dom:-0}" $(( ${loc:-0} >> 8 )) \
            $(( (${loc:-0} >> 3) & 31 )) $(( ${loc:-0} & 7 )))/uid=$(printf "%x" "${uid:-0}"),"
  done
  echo "${out%,}"
}

identity() {
  local boot
  boot="$(cat /proc/sys/kernel/random/boot_id 2>/dev/null)"
  log "$1 id start_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) boot_id=${boot:-na}" \
      "container=$(hostname 2>/dev/null) gpus=$(gpu_ids)" \
      "visible=${ROCR_VISIBLE_DEVICES:-}/${HIP_VISIBLE_DEVICES:-}" \
      "target=${TEST_TARGET:-na} shard=${TEST_SHARD_INDEX:-na}/${TEST_TOTAL_SHARDS:-na}"
}

# Worker memory in KiB. Containers see the host /proc/meminfo, so this is the
# whole machine, including actions of other jobs running on it.
host_mem() { awk -v k="$1:" '$1 == k { print $2 }' /proc/meminfo 2>/dev/null; }

# Memory pressure of the whole machine over the last 10 seconds.
host_psi() {
  awk '{ split($2, a, "="); printf "psi_%s_avg10=%s ", $1, a[2] }' \
      /proc/pressure/memory 2>/dev/null
}

snapshot() {
  local label="$1" stat events nprocs line rss pid comm used
  nprocs="$(ls -d /proc/[0-9]* 2>/dev/null | wc -l)"
  log "$label host=$(hostname 2>/dev/null) nproc=$(nproc 2>/dev/null)" \
      "kernel=$(uname -r) cgroup=v$CGV container_age_s=$(proc_age 1)" \
      "procs=$nprocs"
  log "$label memory_max=$(cg_file memory.max memory.limit_in_bytes)" \
      "current_mib=$(mib "$(cg_current)")" \
      "peak_mib=$(mib "$(cg_file memory.peak memory.max_usage_in_bytes)")"
  stat="$(cg_file memory.stat memory.stat | awk '
      $1 ~ /^(anon|file|shmem|kernel|kernel_stack|pagetables|sock|slab|rss|cache|mapped_file)$/ {
        printf "%s_mib=%d ", $1, $2 / 1048576 }')"
  log "$label stat $stat"
  log "$label worker total_mib=$(( $(host_mem MemTotal) / 1024 ))" \
      "avail_mib=$(( $(host_mem MemAvailable) / 1024 )) $(host_psi)"
  events="$(cg_file memory.events memory.oom_control | tr '\n' ' ')"
  log "$label events $events"
  used="$(df -k /dev/shm /tmp 2>/dev/null | awk 'NR > 1 { printf "%s=%dMiB(%s) ", $6, $3 / 1024, $1 }')"
  log "$label shm_kib=$(du -sk /dev/shm 2>/dev/null | cut -f1) fs $used"
  while read -r rss pid comm; do
    [[ -z "$pid" ]] && continue
    log "$label proc rss_mib=$(( rss / 1024 )) pid=$pid age_s=$(proc_age "$pid") comm=$comm"
  done < <(top_procs)
}

# Counters from memory.events. oom counts times this container hit its own
# limit. oom_kill counts its processes killed by any OOM killer, including a
# machine wide one when the worker runs out of memory.
oom_count() {
  cg_file memory.events memory.oom_control | awk -v k="$1" '$1 == k { print $2 }'
}

if (( $# == 0 )); then
  log "usage mem_probe.sh TEST_BINARY ARGS"
  exit 2
fi

identity BEFORE
snapshot BEFORE
OOM_BEFORE="$(oom_count oom)"
OOM_KILL_BEFORE="$(oom_count oom_kill)"
BASE="$(cg_current)"
BASE="${BASE:-0}"
HOST_AVAIL_BEFORE="$(host_mem MemAvailable)"
HOST_AVAIL_BEFORE="${HOST_AVAIL_BEFORE:-0}"
STATE="$(mktemp "${TEST_TMPDIR:-/tmp}/memprobe.XXXXXX")"
START="$(date +%s.%N)"

"$@" &
CHILD=$!
trap 'kill -TERM "$CHILD" 2>/dev/null' TERM INT

# Samples the container memory, the test's high water mark and the worker's
# available memory. hwm and avail are in KiB. A LIVE line is printed when the
# container grows by another STEP_MIB or the worker loses another 4 GiB.
(
  peak="$BASE"
  hwm_kib=0
  printed="$BASE"
  avail_min="$HOST_AVAIL_BEFORE"
  avail_printed="$HOST_AVAIL_BEFORE"
  while kill -0 "$CHILD" 2>/dev/null; do
    cur="$(cg_current)"
    cur="${cur:-0}"
    h="$(awk '/^VmHWM:/ { print $2 }' "/proc/$CHILD/status" 2>/dev/null)"
    (( ${h:-0} > hwm_kib )) && hwm_kib="$h"
    avail="$(host_mem MemAvailable)"
    avail="${avail:-0}"
    (( avail < avail_min )) && avail_min="$avail"
    (( cur > peak )) && peak="$cur"
    if (( (cur - printed) / 1048576 >= STEP_MIB || \
          (avail_printed - avail) / 1048576 >= 4 )); then
      printed="$cur"
      avail_printed="$avail"
      log "LIVE t_s=$(awk -v s="$START" -v n="$(date +%s.%N)" 'BEGIN { printf "%.1f", n - s }')" \
          "container_current_mib=$(mib "$cur")" \
          "growth_mib=$(( (cur - BASE) / 1048576 )) test_hwm_mib=$(( hwm_kib / 1024 ))" \
          "worker_avail_mib=$(( avail / 1024 )) $(host_psi)"
    fi
    echo "$peak $hwm_kib $avail_min" > "$STATE"
    sleep "$INTERVAL"
  done
) &
SAMPLER=$!

wait "$CHILD"
RC=$?
kill "$SAMPLER" 2>/dev/null
wait "$SAMPLER" 2>/dev/null
read -r PEAK HWM_KIB AVAIL_MIN < "$STATE" 2>/dev/null ||
  { PEAK="$BASE"; HWM_KIB=0; AVAIL_MIN="$HOST_AVAIL_BEFORE"; }
rm -f "$STATE"
OOM_AFTER="$(oom_count oom)"
OOM_KILL_AFTER="$(oom_count oom_kill)"

log "ACTION exit=$RC" \
    "wall_s=$(awk -v s="$START" -v n="$(date +%s.%N)" 'BEGIN { printf "%.1f", n - s }')" \
    "test_peak_rss_mib=$(( ${HWM_KIB:-0} / 1024 ))" \
    "container_before_mib=$(mib "$BASE")" \
    "container_peak_during_action_mib=$(mib "$PEAK")" \
    "container_growth_mib=$(( (${PEAK:-0} - BASE) / 1048576 ))" \
    "worker_avail_before_mib=$(( HOST_AVAIL_BEFORE / 1024 ))" \
    "worker_avail_min_mib=$(( ${AVAIL_MIN:-0} / 1024 ))" \
    "oom_before=${OOM_BEFORE:-na} oom_after=${OOM_AFTER:-na}" \
    "oom_kill_before=${OOM_KILL_BEFORE:-na} oom_kill_after=${OOM_KILL_AFTER:-na}" \
    "target=${TEST_TARGET:-na} shard=${TEST_SHARD_INDEX:-na}"
snapshot AFTER
exit "$RC"
