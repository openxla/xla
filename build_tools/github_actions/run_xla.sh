#!/bin/bash

# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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

BUILD_BINARY=$1

echo "Running XLA's run_hlo_module at ${BUILD_BINARY}"
echo "---------"

# Run run_hlo_module
num_iterations=5
run_start_time="$(date +%s)"
echo "run_hlo_module execution start time: ${run_start_time}"
# TODO(b/277240370): use `run_hlo_module`'s timing utils instead of `date`.
${BUILD_BINARY} -- \
    --input_format=hlo \
    --platform=CPU \  # TODO(b/277243133): use GPU to run benchmarks
    --iterations=$num_iterations \
    --reference_platform= \
    xla/tools/data/benchmarking/mobilenet_v2.hlo
run_end_time="$(date +%s)"
runtime="$((run_end_time - run_start_time))"
echo "Run time for ${num_iterations} iterations is ${runtime} seconds."
