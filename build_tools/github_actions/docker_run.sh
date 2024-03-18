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

# Runs docker configured for usage with GitHub Actions, translating GitHub
# Actions environment variables into generic ones and then invoking the generic
# docker_run script.

# Drawn from https://github.com/openxla/iree/blob/0c0c34f7c8d5a920942f888db4521f64737d598c/build_tools/github_actions/docker_run.sh

set -euo pipefail

export DOCKER_HOST_WORKDIR="${GITHUB_WORKSPACE}"
export DOCKER_HOST_TMPDIR="${RUNNER_TEMP}"

"${GITHUB_WORKSPACE}/build_tools/docker/docker_run.sh" "$@"
