#!/usr/bin/env bash
set -euo pipefail

# Running inside GH Actions job container already
  wget -qO - https://repositories.intel.com/gpu/intel-graphics.key | \
    gpg --yes --dearmor --output /usr/share/keyrings/intel-graphics.gpg

  if ! command -v lsb_release >/dev/null 2>&1; then
    apt-get update && apt-get install -y lsb-release
  fi

  echo "deb [arch=amd64 signed-by=/usr/share/keyrings/intel-graphics.gpg] https://repositories.intel.com/gpu/ubuntu $(lsb_release -cs) unified" | \
   tee /etc/apt/sources.list.d/intel-gpu-$(lsb_release -cs).list

  apt-get update && \
    apt-get install -y --no-install-recommends \
      intel-opencl-icd intel-ocloc libze1 libze-dev xpu-smi && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

  bash build_tools/sycl/ci_test_xla.sh
