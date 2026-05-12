# Copyright 2026 The OpenXLA Authors.
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
# ==============================================================================

"""Filters Clang-Tidy errors based on modified lines in a Git patch.

This script reads a unified diff (patch file) to determine which lines of which
files have been modified. It then parses the Bazel Build Event Protocol (BEP)
JSON file to find the paths of all generated Clang-Tidy YAML reports. For each
report, it checks if any reported errors fall within the modified line ranges
and prints the matching errors.
"""

from __future__ import annotations

import argparse
import bisect
import collections
from collections.abc import Callable, Sequence
import dataclasses
import json
import logging
import pathlib
import sys
from typing import IO, TypedDict

from build_tools.lint import diff_parser


@dataclasses.dataclass(frozen=True)
class AppConfig:
  """Configuration for the ClangTidyDiffFilter application.

  Attributes:
    patch: Path to the patch file containing the diff.
    repo_root: Absolute path to the repository root.
    bep_file: Path to the Bazel Build Event Protocol JSON file.
    warnings_as_errors: If True, treat Clang-Tidy warnings as errors.
  """

  patch: str
  repo_root: str
  bep_file: str
  warnings_as_errors: bool


@dataclasses.dataclass(frozen=True)
class Diagnostic:
  """Represents a single Clang-Tidy diagnostic.

  Attributes:
    file_path: The path to the file where the diagnostic was found, relative to
      the repo root.
    line_num: The 1-based line number of the diagnostic.
    col_num: The 1-based column number of the diagnostic.
    level: The severity level of the diagnostic (e.g., "warning", "error").
    name: The name of the Clang-Tidy check (e.g., "clang-diagnostic-error").
    message: The diagnostic message.
    yaml_file: The path to the .clang-tidy.yaml file where this diagnostic was
      reported.
  """

  file_path: str
  line_num: int
  col_num: int
  level: str
  name: str
  message: str
  yaml_file: str


# Can't use TypedDict with classes because linting will complain about fields
# starting with capital letters.
ClangTidyDiagnostic = TypedDict(
    "ClangTidyDiagnostic",
    {
        "DiagnosticName": str,
        "Message": str,
        "FilePath": str,
        "FileOffset": int,
        "Level": str,
    },
    total=False,
)


ClangTidyReport = TypedDict(
    "ClangTidyReport",
    {
        "MainSourceFile": str,
        "Diagnostics": list[ClangTidyDiagnostic],
    },
    total=False,
)


def _logger() -> logging.Logger:
  """Returns the logger for this module."""
  return logging.getLogger(__name__)


def _set_log_level(log_level: str) -> None:
  """Sets the log level for the application."""
  logging.basicConfig(
      format=(
          "[%(asctime)s] [%(levelname)s][%(filename)s:%(funcName)s:%(lineno)d]"
          " %(message)s"
      ),
      datefmt="%H:%M:%S",
  )

  def _set_level(level: int) -> None:
    _logger().setLevel(level)

  match log_level:
    case "DEBUG":
      _set_level(logging.DEBUG)
    case "INFO":
      _set_level(logging.INFO)
    case "WARNING":
      _set_level(logging.WARNING)
    case "ERROR":
      _set_level(logging.ERROR)
    case _:
      raise ValueError(f"Unsupported log level: {log_level}")


def parse_diff(diff_path: str) -> dict[str, set[int]]:
  """Parses a unified diff file using diff_parser and returns a dictionary mapping filenames to a set of modified line numbers."""
  with open(diff_path, "r") as f:
    diff_str = f.read()
  hunks = diff_parser.parse_hunks(diff_str)
  file_to_lines: dict[str, set[int]] = collections.defaultdict(set)
  for hunk in hunks:
    for line_no, _ in hunk.added_lines():
      file_to_lines[hunk.file].add(line_no)
  return file_to_lines


def get_line_offsets(file_path: str) -> tuple[int, ...]:
  """Returns a list of byte offsets for the start of each line in the file."""
  offsets = [0]
  with open(file_path, "rb") as f:
    while f.readline():
      offsets.append(f.tell())
  return tuple(offsets)


def offset_to_line(offsets: Sequence[int], offset: int) -> int:
  """Converts a byte offset to a 1-based line number using binary search."""
  if not offsets:
    return -1
  # bisect_right returns the index where the offset would be inserted after
  # existing entries. Since offsets contains start of lines, bisect_right - 1
  # gives the line index (0-based).
  return bisect.bisect_right(offsets, offset)


def normalize_path(path: str, repo_root: str) -> str:
  """Normalizes a path to be relative to the repo root.

  This is not foolproof for all possible path formats,
  but it handles common cases seen locally and in CI.

  Args:
    path: The path to normalize.
    repo_root: The absolute path to the repository root.

  Returns:
      The normalized path as a string.
  """
  if not path:
    return ""
  p = pathlib.Path(path)
  # Handle bazel execroot paths
  if "execroot" in p.parts:
    idx = p.parts.index("execroot")
    if idx + 2 < len(p.parts):
      return pathlib.Path(*p.parts[idx + 2 :]).as_posix()
  # Handle local absolute paths under repo_root (CI Runner paths)
  # This handles /__w/xla/xla by removing the prefix.
  if p.is_absolute() and p.is_relative_to(repo_root):
    return p.relative_to(repo_root).as_posix()
  # Handle remote execution paths
  # p is like "/b/f/w/xla/..."
  # NB: We don't quite know the top level directory to look for in the remote
  # path, but since all CPP sources live mostly under "xla/" we use it as the
  # anchor. We also include "third_party" as an anchor since some files may
  # be under that directory.
  _top_level_pkgs = ("xla", "third_party")
  for pkg in _top_level_pkgs:
    if pkg in p.parts:
      parts = list(p.parts)
      idx = parts.index(pkg)  # Find FIRST occurrence in remote path
      return pathlib.Path(*parts[idx:]).as_posix()
  return path


def parse_bep(bep_path: str, repo_root: str) -> list[str]:
  """Parses a Bazel BEP JSON file and returns a list of paths to .clang-tidy.yaml files.

  Args:
    bep_path: Path to the Bazel BEP JSON file.
    repo_root: Absolute path to the repository root.

  Returns:
    A list of paths to .clang-tidy.yaml files.

  Raises:
    ValueError: If a file entry in the BEP is missing 'name' or 'pathPrefix'
      fields.
  """
  yaml_files: list[str] = []
  with open(bep_path, "r") as f:
    for line in f:
      try:
        event = json.loads(line)
        if "namedSetOfFiles" not in event:
          continue
        files = event["namedSetOfFiles"].get("files", [])
        for file_info in files:
          name = file_info.get("name")
          prefix = file_info.get("pathPrefix")
          if name is None:
            raise ValueError("File entry in BEP is missing 'name' field.")
          if not name.endswith(".clang-tidy.yaml"):
            continue
          if prefix is None:
            raise ValueError("File entry in BEP is missing 'pathPrefix' field.")
          path = (
              pathlib.Path(repo_root) / pathlib.Path(*prefix) / name
          ).as_posix()
          yaml_files.append(path)
      except json.JSONDecodeError:
        _logger().warning(
            "Skipping invalid JSON line in BEP file: %s", line.strip()
        )
        continue

  return yaml_files


def parse_clang_tidy_yaml(yaml_path: str) -> ClangTidyReport:
  """A simple, specialized parser for clang-tidy YAML reports to avoid PyYAML dependency."""

  def extract(s: str) -> str:
    _, value = s.split(":", 1)
    return value.strip().strip("'\"")

  result: ClangTidyReport = {"Diagnostics": []}

  current_diag: ClangTidyDiagnostic | None = None
  in_diag_message = False

  with open(yaml_path, "r") as f:
    for line in f:
      stripped = line.strip()
      if stripped.startswith("MainSourceFile:"):
        result["MainSourceFile"] = extract(stripped)
      elif stripped.startswith("- DiagnosticName:"):
        current_diag = {"DiagnosticName": extract(stripped)}
        result["Diagnostics"].append(current_diag)
        in_diag_message = False
      elif stripped.startswith("DiagnosticMessage:"):
        in_diag_message = True
      elif in_diag_message and stripped.startswith("Message:"):
        if current_diag is not None:
          current_diag["Message"] = extract(stripped)
      elif in_diag_message and stripped.startswith("FilePath:"):
        if current_diag is not None:
          current_diag["FilePath"] = extract(stripped)
      elif in_diag_message and stripped.startswith("FileOffset:"):
        if current_diag is not None:
          current_diag["FileOffset"] = int(extract(stripped))
      elif stripped.startswith("Level:"):
        if current_diag is not None:
          current_diag["Level"] = extract(stripped)
      elif stripped.startswith("Replacements:"):
        in_diag_message = False

  return result


RESET = "\033[0m"
BOLD = "\033[1m"
RED = "\033[31m"
YELLOW = "\033[33m"
GREEN = "\033[32m"
CYAN = "\033[36m"


def ansiformat(text: str, color: str = "", bold: bool = False) -> str:
  """Returns text wrapped in ANSI color codes."""
  bold_str = BOLD if bold else ""
  return f"{bold_str}{color}{text}{RESET}"


def print_diagnostic(
    diag: Diagnostic,
    repo_root: str,
    *,
    warnings_as_errors: bool = False,
    stream: IO[str] = sys.stderr,
) -> None:
  """Prints a diagnostic message with code snippet and color."""

  # Clang-tidy format: file:line:col: level: message [name]
  def get_level_str() -> str:
    level_str = diag.level.lower() if diag.level else "warning"
    if warnings_as_errors and level_str == "warning":
      return "error"
    return level_str

  level_str = get_level_str()
  diag_line = " ".join((
      ansiformat(
          ":".join((diag.file_path, str(diag.line_num), str(diag.col_num))),
          bold=True,
      ),
      ansiformat(
          f"{level_str}:",
          color=RED if level_str == "error" else YELLOW,
          bold=True,
      ),
      ansiformat(diag.message, bold=True),
      ansiformat(f"[{diag.name}]", color=CYAN, bold=True),
  ))
  stream.write(f"{diag_line}\n")
  abs_path = pathlib.Path(repo_root) / diag.file_path
  try:
    with open(abs_path, "r") as f:
      lines = f.readlines()
  except FileNotFoundError:
    _logger().warning(
        "Could not read file %s to print diagnostic snippet.", abs_path
    )
    return

  context_lines = 5
  start = max(0, diag.line_num - context_lines - 1)
  end = min(len(lines), diag.line_num + context_lines)
  for linenum, line in enumerate(lines[start:end], start=start + 1):
    line_content = line.rstrip("\n")
    prefix = f"{linenum:5d} | "
    if linenum == diag.line_num:
      stream.write(f"{ansiformat(prefix, bold=True)}{line_content}\n")
      # Print caret
      spaces = "".join(
          "\t" if ch == "\t" else " " for ch in line_content[: diag.col_num - 1]
      )
      stream.write(
          f"{' ' * len(prefix)}{spaces}{ansiformat('^', color=GREEN, bold=True)}\n"
      )
    else:
      stream.write(f"{prefix}{line_content}\n")


class ClangTidyDiffFilter:
  """Filters Clang-Tidy diagnostics based on a diff."""

  def __init__(
      self,
      config: AppConfig,
      offset_provider: Callable[[str], Sequence[int]] = get_line_offsets,
  ):
    """Initializes the ClangTidyDiffFilter.

    Args:
      config: An AppConfig object containing the application configuration.
      offset_provider: A callable that takes a file path and returns a list of
        byte offsets for the start of each line in the file. Defaults to
        `get_line_offsets`.
    """
    self.diff_ranges = parse_diff(config.patch)
    self.yaml_files = parse_bep(config.bep_file, config.repo_root)
    self.repo_root = config.repo_root
    self.warnings_as_errors = config.warnings_as_errors
    self.offset_provider = offset_provider
    self.file_offsets_cache: dict[str, Sequence[int]] = {}
    self.seen_files: set[str] = set()

  def process_file(self, yaml_file: str) -> Sequence[Diagnostic]:
    """Processes a single Clang-Tidy YAML report file."""
    matched_diagnostics: list[Diagnostic] = []
    data = parse_clang_tidy_yaml(yaml_file)
    _logger().debug(
        "Processing clang-tidy report file: %s with content:\n%s",
        yaml_file,
        data,
    )

    if not data:
      return []

    main_source = data.get("MainSourceFile")
    if main_source:
      norm_main_source = normalize_path(main_source, self.repo_root)
      self.seen_files.add(norm_main_source)
    report_source = normalize_path(
        yaml_file.removesuffix(".clang-tidy.yaml"), self.repo_root
    )
    if report_source in self.diff_ranges:
      self.seen_files.add(report_source)
    if "Diagnostics" not in data:
      return []

    for diag in data["Diagnostics"]:
      msg: ClangTidyDiagnostic = diag.get("DiagnosticMessage", {})
      file_path = msg.get("FilePath") or diag.get("FilePath")
      offset = msg.get("FileOffset") or diag.get("FileOffset")

      if not file_path or offset is None:
        continue

      norm_path = normalize_path(file_path, self.repo_root)

      if norm_path not in self.diff_ranges:
        _logger().info(
            "Skipping diagnostic for file %s as it is not in the diff.",
            norm_path,
        )
        continue

      abs_path = pathlib.Path(self.repo_root) / norm_path
      if norm_path not in self.file_offsets_cache:
        self.file_offsets_cache[norm_path] = self.offset_provider(
            abs_path.as_posix()
        )

      offsets = self.file_offsets_cache[norm_path]
      if not offsets:
        _logger().warning(
            "Could not read file %s to calculate line number.",
            abs_path.as_posix(),
        )
        continue

      line_num = offset_to_line(offsets, offset)

      line_start_offset = offsets[line_num - 1]
      col_num = offset - line_start_offset + 1

      lines = self.diff_ranges[norm_path]
      if line_num in lines:
        matched_diagnostics.append(
            Diagnostic(
                file_path=norm_path,
                line_num=line_num,
                col_num=col_num,
                level=diag.get("Level") or "",
                name=diag.get("DiagnosticName") or "",
                message=msg.get("Message") or diag.get("Message") or "",
                yaml_file=yaml_file,
            )
        )
    return matched_diagnostics

  def report_missing(self) -> None:
    """Reports any touched files that were not processed using the logger."""
    touched_files = set(self.diff_ranges.keys())
    missing_files = [
        f for f in touched_files - self.seen_files if f.endswith((".h", ".cc"))
    ]
    if missing_files:
      _logger().warning(
          "No Clang-Tidy reports were processed for the following modified"
          " files:"
      )
      for f in sorted(missing_files):
        _logger().warning("  - %s", f)

  def run(self) -> bool:
    """Runs the Clang-Tidy diff filter.

    Returns:
      True if the check was successful (no errors found), False if errors were
      found or running the check failed.
    """
    if not self.diff_ranges or not self.yaml_files:
      _logger().error("No YAML files provided or found in BEP.")
      return False

    found_diagnostics = False
    for y in self.yaml_files:
      diagnostics = self.process_file(y)
      if diagnostics:
        found_diagnostics = True
      for d in diagnostics:
        print_diagnostic(
            d, self.repo_root, warnings_as_errors=self.warnings_as_errors
        )

    self.report_missing()
    return not found_diagnostics


def main() -> None:
  """Main entry point for the Clang-Tidy diff filter."""

  parser = argparse.ArgumentParser(
      description="Filter Clang-Tidy errors by Git diff."
  )
  parser.add_argument("--patch", required=True, help="Path to the patch file.")
  parser.add_argument(
      "--repo-root", required=True, help="Absolute path to the repo root."
  )
  parser.add_argument(
      "--bep-file",
      required=True,
      help="Path to Bazel Build Event Protocol JSON file.",
  )
  parser.add_argument(
      "--warnings-as-errors",
      default="true",
      choices=["true", "false"],
      help="Treat warnings as errors.",
  )
  parser.add_argument(
      "--log-level",
      default="INFO",
      choices=["DEBUG", "INFO", "WARNING", "ERROR"],
      help="Set the log level.",
  )

  args = parser.parse_args()
  _set_log_level(args.log_level)

  config = AppConfig(
      patch=args.patch,
      repo_root=args.repo_root,
      bep_file=args.bep_file,
      warnings_as_errors=args.warnings_as_errors == "true",
  )

  filterer = ClangTidyDiffFilter(config)
  if not filterer.run():
    sys.exit(1)


if __name__ == "__main__":
  main()
