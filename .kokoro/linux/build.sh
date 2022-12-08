#!/bin/bash
# Copyright 2022 Google LLC All Rights Reserved.
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

# -e: abort script if one command fails
# -u: error if undefined variable used
# -o pipefail: entire command fails if pipe fails. watch out for yes | ...
# -o history: record shell history
set -euo pipefail -o history

# Pull the container (in case it was updated since the instance started) and
# store its SHA in the Sponge log.
docker pull "$DOCKER_IMAGE"
echo "TF_INFO_DOCKER_IMAGE,$DOCKER_IMAGE" >> "$KOKORO_ARTIFACTS_DIR/custom_sponge_config.csv"
echo "TF_INFO_DOCKER_SHA,$(docker pull "$DOCKER_IMAGE" | sed -n '/Digest:/s/Digest: //g p')" >> "$KOKORO_ARTIFACTS_DIR/custom_sponge_config.csv"

# Start a container in the background
docker run --name xla -w /tf/xla -itd --rm \
    -v "$KOKORO_ARTIFACTS_DIR/github/xla:/tf/xla" \
    "$DOCKER_IMAGE" \
    bash

# Build XLA
docker exec xla bazel build \
    --output_filter="" \
    --nocheck_visibility \
    --keep_going \
    -- //xla/...

# Stop container
docker stop xla
