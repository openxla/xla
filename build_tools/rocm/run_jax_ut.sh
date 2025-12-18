#!/bin/bash

set -e

JAX_DIR=$1
XLA_DIR=$2

pushd $JAX_DIR

python build/build.py build \
    --wheels=jaxlib \
    --configure_only \
    --python_version=3.12 \
    --local_xla_path=${XLA_DIR} \
    --python_version=3.12 \
    --clang_path=/lib/llvm-18/bin/clang-18 \
    --rocm_version=7 \
    --rocm_amdgpu_targets=gfx942,gfx90a \
    --verbose

bazel test \
    --config=rocm \
    --build_tag_filters=cpu,gpu,-tpu,-config-cuda-only \
    --test_tag_filters=cpu,gpu,-tpu,-config-cuda-only \
    --action_env=TF_ROCM_AMDGPU_TARGETS=gfx908,gfx90a,gfx942 \
    --//jax:build_jaxlib=true \
    --action_env=TF_ROCM_AMDGPU_TARGETS="gfx906,gfx908,gfx90a,gfx942,gfx950,gfx1030,gfx1100,gfx1101,gfx1200,gfx1201" \
    --test_verbose_timeout_warnings \
    --test_output=errors \
    //tests:core_test_gpu \
    //tests:linalg_test_gpu \
    //tests:ffi_test_gpu \
    --test_filter=CoreTest \
    --test_filter=JaxprTypeChecks \
    --test_filter=DynamicShapesTest \
    --test_filter=testMatmul

popd
