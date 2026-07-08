#!/bin/bash
set -e

source "$(dirname "${BASH_SOURCE[0]}")/shell_common.sh"

REAL_BIN="$PWD/external/%LLVM_REPO_NAME%/bin/clang-tidy"
if [ ! -f "$REAL_BIN" ]; then
  echoerr "Failed to locate clang-tidy binary at: $REAL_BIN"
  exit 1
fi
echo "Using clang-tidy at: " $REAL_BIN
REAL_LIB_DIR="$(dirname "$REAL_BIN")/../lib"
export LD_LIBRARY_PATH="${REAL_LIB_DIR}:${LD_LIBRARY_PATH}"
# Intentional unquoted $@ expansion.
# Word-splitting is required here to resolve composite tokens passed by Bazel
# (e.g., "-include file.h" -> "-include" "file.h").
exec "$REAL_BIN" $@
