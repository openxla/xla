#!/bin/bash

# Copyright 2026 The OpenXLA Authors. All Rights Reserved.
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
set -xe

BUILD_WORKSPACE_DIRECTORY=${BUILD_WORKSPACE_DIRECTORY:-$(pwd)}
cd "$BUILD_WORKSPACE_DIRECTORY"
BAZEL_CMD=${BAZEL_CMD:-bazelisk}

if ! git rev-parse --is-inside-work-tree > /dev/null 2>&1; then
  echo "Error: This script must be run inside a Git repository."
  exit 1
fi
# Aways exit with 0 if no C++ files are changed.
CHANGED_FILES=$(git diff --name-only $(git merge-base origin/main HEAD) | grep -E '\.(cc|h)$' || true)
if [ -z "$CHANGED_FILES" ]; then
  echo "No C++ files changed."
  exit 0
fi
PACKAGES=$(echo "$CHANGED_FILES" | while read -r file; do
    echo "//$(dirname "$file"):all"
done | sort -u | tr '\n' ' ')
TARGETS=$($BAZEL_CMD query --config=clang-tidy "kind('cc_(library|binary|test)', rdeps(set($PACKAGES), set($CHANGED_FILES), 1))")
if [ -z "$TARGETS" ]; then
  echo "No relevant targets found for changed files."
  exit 0
fi
$BAZEL_CMD build --config=clang-tidy --keep_going $TARGETS
