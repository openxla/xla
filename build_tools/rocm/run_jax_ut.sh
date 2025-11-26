#!/bin/bash

set -e

JAX_DIR=$1
XLA_DIR=$2

pushd $JAX_DIR

python build/build.py build \
    --wheels=jax-rocm-plugin \
    --configure_only \
    --local_xla_path=${XLA_DIR} \
    --python_version=3.12

# TODO: run the tests when they are green
bazel build \
    --config=rocm \
    --build_tag_filters=cpu,gpu,-tpu,-config-cuda-only \
    --test_tag_filters=cpu,gpu,-tpu,-config-cuda-only \
    --action_env=TF_ROCM_AMDGPU_TARGETS=gfx908,gfx90a,gfx942 \
    --//jax:build_jaxlib=true \
    "//tests/..."

popd
