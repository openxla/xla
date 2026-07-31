#!/usr/bin/env bash
# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
#
#
# A script to run GPU tests in parallel, isolating each test to the GPU(s) it
# was allocated.
#
# ROCm exposes two independent visibility layers:
#   * ROCR_VISIBLE_DEVICES selects the physical GPUs at the ROCr/HSA level.
#   * HIP_VISIBLE_DEVICES indexes into the ROCr-visible set (so it should be a
#     plain 0..N-1 list, NOT a physical device id).
# We therefore allocate physical GPUs to a test via ROCR_VISIBLE_DEVICES -- only
# ever drawing from the pool the environment already handed us (e.g. from a
# container orchestrator such as Kubernetes) -- and expose them re-indexed to
# 0..N-1 via HIP_VISIBLE_DEVICES.
#
# Multi-GPU tests (multi-process / distributed tests, identified by
# num_processes * gpus_per_process > 1) reserve a *block* of GPUs rather than a
# single one. The previous behavior of pinning every test to one GPU via
# HIP_VISIBLE_DEVICES=$i made such tests impossible to run.
#
# Required environment variables:
#     TF_GPU_COUNT = Number of GPUs available.

set -u

ROCMINFO=$(find -L "${TEST_SRCDIR:-.}" -name "rocminfo" -path "*/bin/rocminfo" | head -n 1)
TF_GPU_COUNT=$($ROCMINFO | grep "Name: *gfx*" | wc -l)
TF_TESTS_PER_GPU=${TF_TESTS_PER_GPU:-8}

# There are certain tests in xla that do not require any gpu in order to be executed
# here we allow executing these tests on a machine without gpu support.
# if there are no GPUs on that system e.g rbe default pool then execute the test without lock
if [[ $TF_GPU_COUNT == 0 ]];then
    echo "Execute with no GPU support"
    exec "$@"
fi

# This function is used below in rlocation to check that a path is absolute
function is_absolute {
  [[ "$1" = /* ]] || [[ "$1" =~ ^[a-zA-Z]:[/\\].* ]]
}

export TF_PER_DEVICE_MEMORY_LIMIT_MB=${TF_PER_DEVICE_MEMORY_LIMIT_MB:-4096}

# *******************************************************************
#         This section of the script is needed to
#         make things work on windows under msys.
# *******************************************************************
RUNFILES_MANIFEST_FILE="${TEST_SRCDIR}/MANIFEST"
function rlocation() {
  if is_absolute "$1" ; then
    # If the file path is already fully specified, simply return it.
    echo "$1"
  elif [[ -e "$TEST_SRCDIR/$1" ]]; then
    # If the file exists in the $TEST_SRCDIR then just use it.
    echo "$TEST_SRCDIR/$1"
  elif [[ -e "$RUNFILES_MANIFEST_FILE" ]]; then
    # If a runfiles manifest file exists then use it.
    echo "$(grep "^$1 " "$RUNFILES_MANIFEST_FILE" | sed 's/[^ ]* //')"
  fi
}

TEST_BINARY="$(rlocation $TEST_WORKSPACE/${1#./})"
shift
# *******************************************************************

# Determine how many GPUs this test needs. Multi-process/distributed tests pass
# --num_processes and --gpus_per_process; everything else needs a single GPU.
NUM_PROCESSES=1
GPUS_PER_PROCESS=1
for arg in "$@"; do
  case "$arg" in
    --num_processes=*)    NUM_PROCESSES="${arg#*=}" ;;
    --gpus_per_process=*) GPUS_PER_PROCESS="${arg#*=}" ;;
  esac
done
[[ "$NUM_PROCESSES" =~ ^[0-9]+$ ]] || NUM_PROCESSES=1
[[ "$GPUS_PER_PROCESS" =~ ^[0-9]+$ ]] || GPUS_PER_PROCESS=1
NEEDED_GPUS=$((NUM_PROCESSES * GPUS_PER_PROCESS))
(( NEEDED_GPUS < 1 )) && NEEDED_GPUS=1

# Physical GPU pool: honor any externally provided ROCR_VISIBLE_DEVICES
# allocation (e.g. from the container orchestrator); otherwise use all GPUs.
if [[ -n "${ROCR_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -r -a GPU_POOL <<< "$ROCR_VISIBLE_DEVICES"
else
  GPU_POOL=($(seq 0 $((TF_GPU_COUNT - 1))))
fi
POOL_SIZE=${#GPU_POOL[@]}

if (( POOL_SIZE < NEEDED_GPUS )); then
  echo "Test needs ${NEEDED_GPUS} GPU(s) but only ${POOL_SIZE} are allocated" \
       "(ROCR_VISIBLE_DEVICES=${ROCR_VISIBLE_DEVICES:-unset}); exiting with failure..."
  exit 1
fi

mkdir -p /var/lock
# Reserve a contiguous block of NEEDED_GPUS GPUs from the pool. Prefer spreading
# tests across GPUs (iterate the oversubscription round `j` outermost) before
# doubling up, matching the original single-GPU behavior for NEEDED_GPUS=1.
for j in `seq 0 $((TF_TESTS_PER_GPU - 1))`; do
  for (( start=0; start + NEEDED_GPUS <= POOL_SIZE; start++ )); do
    lock_fds=()
    acquired=1
    for (( k=0; k < NEEDED_GPUS; k++ )); do
      gpu=${GPU_POOL[$((start + k))]}
      exec {lock_fd}>/var/lock/gpulock${gpu}_${j} || exit 1
      if flock -n "$lock_fd"; then
        lock_fds+=("$lock_fd")
      else
        eval "exec ${lock_fd}>&-"  # close the fd we opened but failed to lock
        acquired=0
        break
      fi
    done
    if (( acquired == 1 )); then
      block=("${GPU_POOL[@]:start:NEEDED_GPUS}")
      (
        # This export only works within the brackets, so it is isolated to one
        # single command.
        # ROCr restricts the process to the allocated physical GPUs; HIP then
        # sees them re-indexed to 0..NEEDED_GPUS-1. Do not set
        # CUDA_VISIBLE_DEVICES: on ROCm HIP would treat it as an additional
        # (physical-id) filter and fight the ROCr allocation.
        export ROCR_VISIBLE_DEVICES=$(IFS=,; echo "${block[*]}")
        export HIP_VISIBLE_DEVICES=$(seq -s, 0 $((NEEDED_GPUS - 1)))
        unset CUDA_VISIBLE_DEVICES
        echo "Running test $TEST_BINARY $* on GPU(s) $ROCR_VISIBLE_DEVICES"
        "$TEST_BINARY" $@
      )
      return_code=$?
      # Releasing the locks (closing the fds) is deferred to process exit, but
      # do it explicitly for clarity.
      for fd in "${lock_fds[@]}"; do eval "exec ${fd}>&-"; done
      exit $return_code
    else
      # Release any partial locks before trying the next starting offset.
      for fd in "${lock_fds[@]}"; do eval "exec ${fd}>&-"; done
    fi
  done
done

echo "Cannot find ${NEEDED_GPUS} free GPU(s) to run the test $* on, exiting with failure..."
exit 1
