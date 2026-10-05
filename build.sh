#!/bin/bash

SCRIPT_DIR=$(dirname $0)

${SCRIPT_DIR}/build_tools/rocm/run_xla_ci_build.sh \
    --config=rocm_ci_hermetic \
    --config=rocm_clang_hermetic \
    --config=rocm_rbe \
    --config=ci_single_gpu \
    --@rules_ml_toolchain//common:enable_xla_test_global_symbol_version=True \
    --repo_env=REMOTE_GPU_TESTING=1 \
    --action_env=RBE_DOCKER_IMAGE="ghcr.io/rocm/jax-build-ubu24.rocmless@sha256:68854ed30b800d3e8c390f6e651721e2bead58830da1c59e1b828354eb8f077f" \
    --repo_env=ROCM_DISTRO_URL="https://stable.repo.amd.com/rocm/core/tarball/therock-dist-linux-multiarch-10.0.0.tar.gz" \
    --repo_env=ROCM_DISTRO_HASH="1c5e807875d26a2470ecc7323daa5b5b9009208a55c3290ac255a909cde15fc6" \
    --repo_env=TF_ROCM_AMDGPU_TARGETS="gfx950" \
    --repo_env=TF_ROCM_RBE_SINGLE_GPU_POOL=linux_x64_gpu_do_gfx950 \
    --repo_env=REMOTE_GPU_TESTING=1 \
    --strategy=TestRunner=remote \
    --nocache_test_results \
    --local_test_jobs=1 \
    --flaky_test_attempts=1 \
    --jobs=140 \
    --test_timeout=920,2400,7200,9600 \
    -- \
    //xla/...
