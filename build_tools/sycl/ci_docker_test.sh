#!/usr/bin/env bash
set -euo pipefail

if [[ "${GITHUB_ACTIONS:-}" == "true" ]]; then
  # Running inside GH Actions job container already
  wget -qO - https://repositories.intel.com/gpu/intel-graphics.key | \
    gpg --yes --dearmor --output /usr/share/keyrings/intel-graphics.gpg

  if ! command -v lsb_release >/dev/null 2>&1; then
    apt-get update && apt-get install -y lsb-release
  fi

  echo "deb [arch=amd64 signed-by=/usr/share/keyrings/intel-graphics.gpg] https://repositories.intel.com/gpu/ubuntu $(lsb_release -cs) unified" | \
    sudo tee /etc/apt/sources.list.d/intel-gpu-$(lsb_release -cs).list

  apt-get update && \
    apt-get install -y --no-install-recommends \
      intel-opencl-icd intel-ocloc libze1 libze-dev xpu-smi && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

  bash build_tools/sycl/ci_test_xla.sh
  exit 0
fi

# ----- LOCAL RUN -----
# Import base Docker image from Google
docker pull us-docker.pkg.dev/ml-oss-artifacts-published/ml-public-container/ml-build:latest
docker rm -f oneapi_xla >/dev/null 2>&1 || true

docker run --rm -i --name=oneapi_xla \
  -e https_proxy -e http_proxy -e HTTPS_PROXY -e HTTP_PROXY -e no_proxy -e NO_PROXY \
  --device=/dev/dri \
  -v "$PWD:/xla" \
  us-docker.pkg.dev/ml-oss-artifacts-published/ml-public-container/ml-build:latest \
  /bin/bash <<'EOF'

# Step 1: Add Intel GPU APT Signing Key
wget -qO - https://repositories.intel.com/gpu/intel-graphics.key | \
  gpg --yes --dearmor --output /usr/share/keyrings/intel-graphics.gpg

# Step 2: Add Intel GPU APT Repository
if ! command -v lsb_release >/dev/null 2>&1; then
  apt-get update && apt-get install -y lsb-release
fi

echo "deb [arch=amd64 signed-by=/usr/share/keyrings/intel-graphics.gpg] https://repositories.intel.com/gpu/ubuntu $(lsb_release -cs) unified" | \
  sudo tee /etc/apt/sources.list.d/intel-gpu-$(lsb_release -cs).list

# Step 3: Install Intel GPU Runtime Packages
apt-get update && \
  apt-get install -y --no-install-recommends \
    intel-opencl-icd intel-ocloc libze1 libze-dev xpu-smi && \
  apt-get clean && rm -rf /var/lib/apt/lists/*

cd /xla
bash build_tools/sycl/ci_test_xla.sh

EOF
