#!/bin/bash -e

printf -v LILITH_EXTRA_ARGS "%s " \
  "--path $TEST_SRCDIR/google3/third_party/tensorflow/compiler/xla/mlir/backends/cpu" \
  "--path $TEST_SRCDIR/google3/third_party/tensorflow/compiler/xla/mlir/backends/gpu" \
  "--path $TEST_SRCDIR/google3/third_party/tensorflow/compiler/xla/mlir/runtime" \
  "--path $TEST_SRCDIR/google3/third_party/tensorflow/compiler/xla/mlir_hlo/tosa" \
  "--path $TEST_SRCDIR/google3/third_party/tensorflow/compiler/xla/service/mlir_gpu" \

export LILITH_EXTRA_ARGS

exec "$TEST_SRCDIR/google3/third_party/llvm/llvm-project/mlir/run_lit.sh" \
  "$@"
