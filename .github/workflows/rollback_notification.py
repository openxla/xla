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
"""Notifies PR authors of rollbacks on committed PRs."""
import itertools
import json
import logging
import os
import re
import subprocess
from typing import Any, Generator, Sequence


def call_gh_api(
    endpoint: str, *, http_method: str = "GET", **kwargs
) -> dict[Any, Any]:
  """Calls the GitHub API via the command line using `gh api`.

  Arguments:
    endpoint: the endpoint to send the request to
    http_method: the http_method to use (e.g. GET)
    **kwargs: the fields that will be used in the request. For example, if given
      `state='"open"'`, `-f state='open'` will be given as arguments

  Returns:
    A dict representing the json response if successful.
  """

  fields = itertools.chain(*[("-f", f"{k}={v}") for k, v in kwargs.items()])
  logging.info("Fields passed to `gh api`: %s", fields)
  proc = subprocess.run(
      ["gh", "api", "--method", http_method, endpoint, *fields],
      stdout=subprocess.PIPE,
      check=True,
      text=True,
  )

  return json.loads(proc.stdout)


def get_reverted_commit_hashes(message: str) -> list[str]:
  """Searches a commit message for `reverts <commit hash>` and returns the found SHAs.

  Arguments:
    message: the commit message to search

  Returns:
    A list of SHAs as strings.
  """
  regex = re.compile(r"reverts ([0-9a-f]{5,40})", flags=re.IGNORECASE)
  commit_hashes = regex.findall(message)
  if commit_hashes:
    logging.info(
        "Found commit hashes reverted in this commit: %s", commit_hashes
    )
  return commit_hashes


def get_associated_prs(
    commit_hashes: Sequence[str],
) -> Generator[tuple[str, str], None, None]:
  """Finds PRs associated with commits.

  Arguments:
    commit_hashes: A sequence of SHAs which may have PRs associated with them

  Yields:
    Associated pairs of (PR number, SHA), both as strings
  """
  regex = re.compile(r"PR #(\d+)")
  for commit_hash in commit_hashes:
    response = call_gh_api(f"repos/openxla/xla/commits/{commit_hash}")
    message = response["commit"]["message"]
    if maybe_match := regex.match(message):
      pr_number = maybe_match.group(1)
      logging.info(
          "Found PR #%s associated with commit_hash %s", pr_number, commit_hash
      )
      yield commit_hash, pr_number


def write_pr_comment_and_reopen(commit_hash: str, pr_number: int) -> None:
  """Writes a comment on the PR notifying that the PR has been reverted.

  Arguments:
    commit_hash: the SHA of the commit that reverted the PR
    pr_number: the number of the PR that we want to comment on
  """

  comment_body = f"This PR was rolled back in {commit_hash}!"

  # write PR comment
  call_gh_api(
      f"/repos/openxla/xla/issues/{pr_number}/comments",
      http_method="POST",
      body=comment_body,
  )

  # reopen PR
  call_gh_api(
      f"/repos/openxla/xla/issues/{pr_number}",
      http_method="POST",
      body='"open"',  # API fails with 422 without quotes
  )


def main():
  head_commit_message = os.getenv("HEAD_COMMIT_MESSAGE")
  if head_commit_message is None:
    raise EnvironmentError("Environment variable HEAD_COMMIT_MESSAGE not set!")

  commit_hashes = get_reverted_commit_hashes(head_commit_message)

  for commit_hash, pr_number in get_associated_prs(commit_hashes):
    write_pr_comment_and_reopen(commit_hash, pr_number)


if __name__ == "__main__":
  main()
