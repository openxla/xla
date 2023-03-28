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
"""Adds assignees to PRs based on contents."""
import os

import github_api


if __name__ == "__main__":
  pr_number = int(os.getenv("PR_NUMBER"))
  api = github_api.GitHubAPI(os.getenv("GH_TOKEN"))
  # TODO(ddunleavy): make this more sophisticated
  assignees = ["ddunleavy", "tpopp", "xla-rotation"]
  api.add_issue_assignees("openxla/xla", pr_number, assignees)
