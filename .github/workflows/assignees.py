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
"""Adds assignees to issues and PRs."""
import os

import github_api


def add_assignees(api: github_api.GitHubAPI, number: int) -> None:
  """Adds assignees to issues and PRs.

  Arguments:
    api: API instance to use for API calls
    number: the number of the issue/PR
  """
  issue_info = api.get_issue("openxla/xla", number)
  author = api.get_user(issue_info["user"]["login"])
  domain = author["email"].split("@")[1] if "email" in author else None

  assignees = ["xla-rotation"]  # add rotation to every issue/pr

  if domain == "nvidia.com":
    assignees.extend(["cheshire", "gcforster", "reedwm", "chsigg"])

  # TODO(ddunleavy): AMD, others?
  api.add_issue_assignees(assignees)


def maybe_run_ci(api: github_api.GitHubAPI, number: int) -> None:
  """Runs CI based on user email.

  TODO(ddunleavy): see if we can just run on all PRs without checking email.

  Arguments:
    api: API instance to use for API calls
    number: the number of the issue/PR
  """
  issue_info = api.get_issue("openxla/xla", number)
  if "pull_request" not in issue_info:
    # Github calls both PRs and Issues issues, so we need to check if we have a
    # PR before adding the label that triggers CI.
    return
  author = api.get_user(issue_info["user"]["login"])
  _, domain = author["email"].split("@")

  partner_domains = [
      "amd.com",
      "apple.com",
      "arm.com",
      "intel.com",
      "linaro.org",
      "google.com",
      "nvidia.com",
  ]

  if domain in partner_domains:
    api.add_issue_labels("openxla/xla", number, ["kokoro:force-run"])


def main():
  api = github_api.GitHubAPI(os.getenv("GH_TOKEN"))
  number = int(os.getenv("ISSUE_NUMBER"))
  add_assignees(api, number)
  maybe_run_ci(api, number)


if __name__ == "__main__":
  main()
