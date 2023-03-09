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

echo "XLA build script is running..."

start_time="$(date +%s)"
echo "Start Time: ${start_time}"
bazel build -c opt --nocheck_visibility --keep_going xla/tools:run_hlo_module
end_time="$(date +%s)"
echo "End Time: ${end_time}"
runtime="$((end_time - start_time))"
echo "Run time is ${runtime} seconds."
