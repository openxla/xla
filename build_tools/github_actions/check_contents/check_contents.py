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
"""Command line tool for checking for regexes in diffs.

Filters `git diff` by path, then checks to make sure no lines matching a regex
have been added in the diff.
"""

import argparse
import dataclasses
import itertools
import logging  # Intended to run on vanilla Github Actions runner
import re
import subprocess
import sys
from typing import Iterable, Generator, Optional, Sequence, TypeVar


_T = TypeVar("_T")


@dataclasses.dataclass
class FileDiff:
  """Represents a diff of a file.

  Attributes:
    path: Path of the file being diffed.
    added_lines: List of tuples with (line_number, text)
  """

  path: str
  added_lines: list[tuple[int, str]]


@dataclasses.dataclass
class RegexLocation:
  """Path and line where a prohibited regex was found.

  Attributes:
    path: Path of the file which has the prohibited regex.
    line_number: The number of the offending line.
    line_contents: The text of the offending line.
    matched_text: The exact string matched by the regex.
  """

  path: str
  line_number: int
  line_contents: str
  matched_text: str


def get_git_diff_stdout() -> str:
  """Run git diff with appropriate arguments and capture stdout as a str."""
  proc = subprocess.run(
      ["git", "diff", "origin/main", "HEAD"],
      capture_output=True,
      check=True,
      text=True,
  )
  return proc.stdout


def batch(
    iterable: Iterable[_T], n: int
) -> Generator[tuple[_T, ...], None, None]:
  """Splits an iterable into chunks of size n.

  TODO(ddunleavy): once python 3.12 is available, use itertools.batch.

  Arguments:
    iterable: the iterable to batch.
    n: the number of elements in each batch.

  Yields:
    A tuple of length n of the type that the iterable produces.
  """
  iterator = iter(iterable)
  while True:
    try:
      # Unnecessary list here, but a generator won't raise StopIteration,
      # instead it will raise RuntimeError: "generator raises StopIteration".
      # I'd rather have a list comprehension in place of a generator expression
      # than catch RuntimeError and have to inspect the payload to verify it's
      # the one I want to be catching.
      yield tuple([next(iterator) for _ in range(n)])
    except StopIteration:
      return


def parse_diff(diff: str) -> list[FileDiff]:
  """Parses the otuput of git diff into structured FileDiff objects.

  Arguments:
    diff: The raw output of git diff.

  Returns:
    A list of FileDiffs which contain the added lines for each file in the diff.
  """
  diff_pattern = r"diff --git a/.* b/(.*)\n"  # capture filename
  chunk_header_pattern = r"@@ -\d+,\d+ \+(\d+),\d+ @@\n"  # capture line number

  # ignore initial empty match
  raw_per_file_diffs = re.split(diff_pattern, diff)[1:]

  file_diffs = []
  for path, raw_chunks in batch(raw_per_file_diffs, 2):
    chunks = re.split(chunk_header_pattern, raw_chunks, re.MULTILINE)

    # ignore extraneous diff metadata
    for _, starting_line_no, diff in batch(chunks, 3):
      starting_line_no = int(starting_line_no)
      lines = diff.split("\n")

      added_lines = [
          (line_no, line[1:])
          for line_no, line in zip(itertools.count(starting_line_no), lines)
          if line.startswith("+")
      ]
      file_diffs.append(FileDiff(path=path, added_lines=added_lines))

  return file_diffs


def filter_diffs_by_path(
    diffs: Iterable[FileDiff],
    *,
    path_regexes: list[str],
    path_regex_exclusions: list[str],
) -> list[FileDiff]:
  """Filters files according to path_regexes.

  If a file matches both a path_regex and a path_regex_exclusion, then
  it will be filtered out.

  Arguments:
    diffs: A sequence of FileDiff objects representing the diffs of each file in
      the change.
    path_regexes: A list of regexes. Paths matching these will pass through the
      filter. By default, every path is matched.
    path_regex_exclusions: A list of regexes. Paths that match both a path_regex
      and a path_regex_exclusion won't pass through the filter.

  Returns:
    A list of FileDiffs whose paths match a path_regex and don't match
      any path_regex_exclusions.
  """

  if not path_regexes:
    path_regexes = [".*"]  # by default match everything

  path_regexes = [re.compile(regex) for regex in path_regexes]

  def should_include(path: str) -> bool:
    return any(regex.search(path) for regex in path_regexes)

  path_regex_exclusions = [re.compile(regex) for regex in path_regex_exclusions]

  def should_exclude(path: str) -> bool:
    return any(regex.search(path) for regex in path_regex_exclusions)

  return [
      diff
      for diff in diffs
      if should_include(diff.path) and not should_exclude(diff.path)
  ]


def check_diffs(
    diffs: Iterable[FileDiff],
    *,
    prohibited_regex: str,
    suppression_regex: Optional[str] = None,  # TODO(ddunleavy): CI not on 3.10
) -> list[RegexLocation]:
  """Checks FileDiffs for prohibited regexes.

  Arguments:
    diffs: A sequence of FileDiff objects representing the diffs of each file in
      the change.
    prohibited_regex: The regex that isn't allowed in the diff.
    suppression_regex: A regex used as an escape hatch to allow the prohibited
      regex in the diff. If this is found on the same line as prohibited_regex,
      there is no error.

  Returns:
    A list of RegexLocations where the prohibited_regex is found.
  """

  prohibited_regex = re.compile(prohibited_regex)
  if suppression_regex is not None:
    suppression_regex = re.compile(suppression_regex)

  def should_not_suppress(line) -> bool:
    if suppression_regex:
      return not suppression_regex.search(line)
    return True

  regex_locations = []
  for diff in diffs:
    for line_no, line in diff.added_lines:
      if should_not_suppress(line):
        regex_locations.extend(
            [
                RegexLocation(diff.path, line_no, line, regex_match.group())
                for regex_match in prohibited_regex.finditer(line)
            ]
        )

  return regex_locations


def main(argv: Sequence[str]):
  parser = argparse.ArgumentParser(
      description="Check `git diff` for prohibited regexes."
  )
  parser.add_argument("--path_regex", nargs="*", default=[])
  parser.add_argument("--path_regex_exclusion", nargs="*", default=[])
  parser.add_argument("--prohibited_regex", required=True)
  parser.add_argument("--suppression_regex")
  parser.add_argument("--failure_message", required=True)

  # We don't want to include path/to/check_contents.py as an argument
  args = parser.parse_args(argv[1:])

  file_diffs = filter_diffs_by_path(
      parse_diff(get_git_diff_stdout()),
      path_regexes=args.path_regex,
      path_regex_exclusions=args.path_regex_exclusion,
  )

  regex_locations = check_diffs(
      file_diffs,
      prohibited_regex=args.prohibited_regex,
      suppression_regex=args.suppression_regex,
  )

  if regex_locations:
    for loc in regex_locations:
      logging.error(
          "Found `%s` in %s:%s",
          args.prohibited_regex,
          loc.path,
          loc.line_number,
      )
      logging.error(
          "Matched `%s` in line `%s`", loc.matched_text, loc.line_contents
      )
      logging.error("Failure message: %s", args.failure_message)
    sys.exit(1)
  else:
    logging.info(
        "Prohibited regex `%s` not found in diff!", args.prohibited_regex
    )
    sys.exit(0)


if __name__ == "__main__":
  main(sys.argv)
