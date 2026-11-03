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
# ============================================================================
"""Git diff inspection and file-based fast-skip detection for XLA and JAX CI.

Provides generic git diff utilities operating on a caller-supplied directory
and evaluates whether a CI build can be skipped based on the modified files.
"""

import argparse
from collections.abc import Sequence
import dataclasses
import logging
import os
import re
import subprocess
import sys
import time
from typing import List, Optional, Tuple

# Global configuration and external repository definition files whose
# modification invalidates incremental bazel-diff target hashes (which run with
# --excludeExternalTargets and fine-grained hash only @tsl) and necessitates a
# full test suite run.
GLOBAL_BAZEL_CONFIG_PATTERNS: Tuple[str, ...] = (
    r"(^|/)MODULE\.bazel(\.lock)?$",
    r"(^|/)REPO\.bazel$",
    r"(^|/)WORKSPACE(\.bzlmod)?$",
    r"\.bazelrc$",
    r"\.bazelversion$",
    # Top-level workspace initialization macros and pip lockfiles.
    r"^workspace[0-9]+\.bzl$",
    r"^requirements_lock_.*\.txt$",
    # External repository definitions, Bzlmod module extensions, and repo rules.
    r"(^|/)workspace\.bzl$",
    r"^third_party/(repo|quilt_repo)\.bzl$",
    r"^third_party/extensions/.*\.bzl$",
    r"^third_party/(?!tsl/tsl/).*extension\.bzl$",
    (
        r"^third_party/(?!tsl/tsl/).*"
        r"(configure|init_repositor|redist|download|repo|setup).*\.bzl$"
    ),
    # External repository overlay BUILD files, patches, and templates (excluding
    # first-party @tsl and pywrap/python rules tracked by bazel-diff).
    r"(^|/)[^/]+\.BUILD(\.bazel)?$",
    r"^third_party/(?!tsl/tsl/).*\.(patch|diff|tpl)$",
    r"^third_party/(?!tsl/tsl/).*(/patches/|/series$)",
    (
        r"^third_party/(?!tsl/tsl/|py/(numpy|rules_pywrap)/|rules_python/)"
        r".*/BUILD(\.bazel|\.system)?$"
    ),
    r"^third_party/py/python_wheel\.bzl$",
    r"^tools/toolchains/",
)

# Strict allowlist pattern for Git revision arguments (hexadecimal commit SHAs
# or alphanumeric ref names not starting with '-') to prevent flag injection.
SAFE_GIT_REF_PATTERN = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_./~^-]*$")

# Regex matching files that only contain documentation or repository metadata.
DOCS_OR_METADATA_PATTERN = re.compile(
    r"(\.(md|rst|png|jpg|jpeg|svg|webp|gif)$"
    r"|(^|/)OWNERS$"
    r"|(^|/)LICENSE(\.(txt|md))?$"
    r"|(^|/)\.clang-(format|tidy)$"
    r"|(^|/)\.clangd$"
    r"|(^|/)\.gitignore$)"
)

# Regex matching BUILD or .bzl files that could alter build graphs.
BUILD_OR_BZL_PATTERN = re.compile(r"((^|/)BUILD(\.bazel)?$|\.bzl$)")

# Regex matching BUILD, .bzl, or .proto files that could alter build graphs.
BUILD_OR_PROTO_PATTERN = re.compile(r"((^|/)BUILD(\.bazel)?$|\.bzl$|\.proto$)")

# Regex matching XLA unit test, benchmark, and HLO snapshot files.
XLA_TEST_OR_BENCHMARK_PATTERN = re.compile(
    r"((_test|_tests|_test_gpu|_test_cpu|_benchmark)\.(cc|h|cu\.cc|py)$"
    r"|\.hlo$)"
)

# Safe characters allowed in Bazel target labels passed to `bazel query`.
SAFE_BAZEL_LABEL_PATTERN = re.compile(
    r"^@[A-Za-z0-9_.-]+//[A-Za-z0-9_./+-]*:[A-Za-z0-9_./+-]+$"
)

# Root JAX targets whose transitive @xla dependencies are part of CPU jaxlib.
JAX_CPU_ROOT_TARGETS: Tuple[str, ...] = (
    "//jaxlib/tools:jaxlib_sources",
    "//jaxlib/tools:build_utils",
)

# Additional JAX root targets for CUDA GPU builds.
JAX_CUDA_ROOT_TARGETS: Tuple[str, ...] = (
    "//jaxlib/cuda:cuda_gpu_support",
    "//jaxlib/cuda:cuda_plugin_extension",
    "//jaxlib/mosaic/gpu:mosaic_gpu",
    "@xla//xla/pjrt/c:pjrt_c_api_gpu",
    "@xla//xla/service:gpu_plugin",
    "@xla//xla/stream_executor:cuda_platform",
)

# Additional JAX root targets for ROCm GPU builds.
JAX_ROCM_ROOT_TARGETS: Tuple[str, ...] = (
    "//jaxlib/rocm:rocm_gpu_support",
    "//jaxlib/rocm:rocm_plugin_extension",
    "@xla//xla/pjrt/c:pjrt_c_api_gpu",
    "@xla//xla/pjrt/c:pjrt_c_api_gpu_version_script.lds",
    "@xla//xla/service:gpu_plugin",
    "@xla//xla/stream_executor:rocm_platform",
)


@dataclasses.dataclass(frozen=True)
class SkipDecision:
  """Result of evaluating whether a CI build can be skipped.

  Attributes:
    should_skip: Whether the CI build can be safely skipped.
    reason: Human-readable explanation for the decision.
    changed_files_count: Number of git changed files between base and head.
    elapsed_seconds: Wall-clock duration of the skip evaluation in seconds.
  """

  should_skip: bool
  reason: str = ""
  changed_files_count: int = 0
  elapsed_seconds: float = 0.0


def validate_git_ref(ref: str) -> str:
  """Validates that ref is a safe Git revision string and cannot inject flags.

  Args:
    ref: Git commit SHA or ref name to validate.

  Returns:
    The validated ref string.

  Raises:
    ValueError: If ref is empty, starts with '-', or contains disallowed chars.
  """
  if not ref or ref.startswith("-") or not SAFE_GIT_REF_PATTERN.fullmatch(ref):
    raise ValueError(f"Invalid or unsafe git revision parameter: {ref!r}")
  return ref


def matches_any_pattern(path: str, patterns: Sequence[str]) -> bool:
  """Returns True if path matches any regex pattern in patterns."""
  normalized = path.replace("\\", "/")
  return any(re.search(pattern, normalized) for pattern in patterns)


def is_global_config_changed(changed_files: Sequence[str]) -> bool:
  """Returns True if any changed file matches global bazel config patterns."""
  return any(
      matches_any_pattern(file_path, GLOBAL_BAZEL_CONFIG_PATTERNS)
      for file_path in changed_files
  )


def is_docs_or_metadata_only(changed_files: Sequence[str]) -> bool:
  """Returns True if all changed files are docs or non-build metadata."""
  if not changed_files:
    return False
  return all(
      bool(DOCS_OR_METADATA_PATTERN.search(file_path))
      for file_path in changed_files
  )


def run_git_best_effort(
    args: Sequence[str], cwd: str = "."
) -> Optional[subprocess.CompletedProcess[str]]:
  """Runs a git command best-effort, returning None on OS/subprocess error."""
  try:
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )
  except (OSError, subprocess.SubprocessError):
    return None


def get_merge_base(
    base_sha: str, head_sha: str = "HEAD", cwd: str = "."
) -> Optional[str]:
  """Computes git merge-base between base_sha and head_sha.

  Args:
    base_sha: Base git commit SHA.
    head_sha: Head git commit SHA or ref.
    cwd: Directory where the git command should be executed.

  Returns:
    The merge base commit SHA as a string, or None if not found.
  """
  try:
    safe_base = validate_git_ref(base_sha)
    safe_head = validate_git_ref(head_sha)
  except ValueError:
    return None
  result = run_git_best_effort(
      ["merge-base", "--", safe_base, safe_head], cwd=cwd
  )
  if result and result.returncode == 0 and result.stdout.strip():
    return result.stdout.strip()
  return None


def get_diff_base(base_sha: str, head_sha: str = "HEAD", cwd: str = ".") -> str:
  """Returns the base commit to diff against head_sha.

  Uses git merge-base when available so that changes that landed on the base
  branch after the feature branch diverged are not falsely attributed to the PR.

  Args:
    base_sha: Base git commit SHA.
    head_sha: Head git commit SHA or ref.
    cwd: Directory where the git command should be executed.

  Returns:
    The merge-base SHA if available, otherwise validated base_sha.
  """
  safe_base = validate_git_ref(base_sha)
  safe_head = validate_git_ref(head_sha)
  merge_base = get_merge_base(safe_base, safe_head, cwd=cwd)
  return validate_git_ref(merge_base) if merge_base else safe_base


def ensure_base_fetched(
    base_sha: str, head_sha: str = "HEAD", cwd: str = "."
) -> None:
  """Best-effort fetches base_sha/head_sha and unshallows if merge-base is missing."""
  try:
    safe_base = validate_git_ref(base_sha)
    safe_head = validate_git_ref(head_sha)
  except ValueError:
    return
  if get_merge_base(safe_base, safe_head, cwd=cwd):
    return
  fetch_refs = [safe_base]
  if safe_head != "HEAD" and safe_head != safe_base:
    fetch_refs.append(safe_head)
  run_git_best_effort(
      ["fetch", "--no-tags", "--depth=50", "--", "origin", *fetch_refs],
      cwd=cwd,
  )
  if not get_merge_base(safe_base, safe_head, cwd=cwd):
    run_git_best_effort(["fetch", "--unshallow", "origin"], cwd=cwd)


def get_changed_files(
    base_sha: str, head_sha: str = "HEAD", cwd: str = "."
) -> List[str]:
  """Returns list of changed filepaths between base_sha and head_sha.

  Args:
    base_sha: Base git commit SHA.
    head_sha: Head git commit SHA or ref.
    cwd: Directory where the git command should be executed.

  Returns:
    List of relative paths of changed files.
  """
  diff_base = get_diff_base(base_sha, head_sha, cwd=cwd)
  safe_head = validate_git_ref(head_sha)
  command = [
      "git",
      "diff",
      "--name-only",
      "--end-of-options",
      diff_base,
      safe_head,
      "--",
  ]
  result = subprocess.run(
      command, cwd=cwd, capture_output=True, text=True, check=True
  )
  return [
      line.strip().replace("\\", "/")
      for line in result.stdout.splitlines()
      if line.strip()
      and not line.strip().replace("\\", "/").startswith("build_tools/ci/")
  ]


def is_ci_or_global_config_file(file_path: str) -> bool:
  """Returns True if file_path is a global Bazel config or CI workflow/tool."""
  normalized = file_path.replace("\\", "/")
  if matches_any_pattern(normalized, GLOBAL_BAZEL_CONFIG_PATTERNS):
    return True
  if normalized.startswith((".github/", "build_tools/")) and not (
      DOCS_OR_METADATA_PATTERN.search(normalized)
  ):
    return True
  return False


def is_build_or_ci_core_file(file_path: str) -> bool:
  """Returns True if file_path is a BUILD, .bzl, global config, or CI file."""
  normalized = file_path.replace("\\", "/")
  return is_ci_or_global_config_file(normalized) or bool(
      BUILD_OR_BZL_PATTERN.search(normalized)
  )


def is_build_or_proto_or_ci_core_file(file_path: str) -> bool:
  """Returns True if file_path requires a full run without Bazel graph query."""
  normalized = file_path.replace("\\", "/")
  return is_build_or_ci_core_file(normalized) or bool(
      BUILD_OR_PROTO_PATTERN.search(normalized)
  )


def is_unimpacted_for_all_jax(file_path: str) -> bool:
  """Returns True if file_path is a test-only or non-code file for JAX CI."""
  normalized = file_path.replace("\\", "/")
  if is_build_or_proto_or_ci_core_file(normalized):
    return False
  if DOCS_OR_METADATA_PATTERN.search(normalized):
    return True
  if (
      normalized.startswith("xla/")
      and not normalized.startswith("xla/mosaic/")
      and XLA_TEST_OR_BENCHMARK_PATTERN.search(normalized)
  ):
    return True
  return False


def normalize_external_bazel_label(label: str) -> Optional[str]:
  """Normalizes an @xla or @tsl Bazel label (including Bzlmod canonical forms).

  Canonicalizes repository names (`@@xla+//...`, `@@xla~override//...`,
  `@@+tsl_extension+tsl//...`) to `@xla//...` or `@tsl//...`, maps
  `@xla//third_party/tsl/...` to `@tsl//...`, and normalizes `:BUILD.bazel` to
  `:BUILD` so build file labels match regardless of extension.

  Args:
    label: Raw Bazel label string.

  Returns:
    Normalized `@xla//pkg:target` or `@tsl//pkg:target` string, or None if the
    label does not belong to `@xla` or `@tsl`.
  """
  stripped = label.strip()
  if not stripped.startswith("@") or "//" not in stripped:
    return None
  repo_part, pkg_and_target = stripped.split("//", 1)
  if ":" not in pkg_and_target:
    return None

  if "tsl" in repo_part:
    repo = "@tsl"
  elif "xla" in repo_part:
    if pkg_and_target.startswith("third_party/tsl/"):
      repo = "@tsl"
      pkg_and_target = pkg_and_target[len("third_party/tsl/") :]
    elif pkg_and_target.startswith("third_party/tsl:"):
      repo = "@tsl"
      pkg_and_target = pkg_and_target[len("third_party/tsl") :]
    else:
      repo = "@xla"
  else:
    return None

  pkg, target_name = pkg_and_target.split(":", 1)
  if target_name == "BUILD.bazel":
    target_name = "BUILD"
  normalized = f"{repo}//{pkg}:{target_name}"
  if not SAFE_BAZEL_LABEL_PATTERN.fullmatch(normalized):
    return None
  return normalized


def resolve_bazel_file_label(file_path: str, xla_dir: str) -> Optional[str]:
  """Maps a relative file path in xla_dir to its external Bazel target label.

  Walks up ancestor directories of `file_path` inside `xla_dir` to find the
  enclosing Bazel package (`BUILD.bazel` or `BUILD`) and returns the label as
  referenced from the JAX workspace (`@xla//pkg:target` or `@tsl//pkg:target`).
  When `file_path` was deleted and no longer exists on disk, maps it to its
  enclosing package's `:BUILD` target so reachability is checked against
  `buildfiles(deps(...))`.

  Args:
    file_path: Relative path of the modified file within `xla_dir`.
    xla_dir: Path to the XLA repository root on disk.

  Returns:
    The external Bazel label string, or None if the file cannot be safely mapped
    to a package in `@xla` or `@tsl`.
  """
  normalized = file_path.replace("\\", "/").lstrip("/")
  if not normalized:
    return None
  full_path = os.path.join(xla_dir, normalized)
  file_exists = os.path.isfile(full_path)

  curr_dir = os.path.dirname(normalized)
  pkg: Optional[str] = None
  target_name: Optional[str] = None
  while True:
    abs_dir = os.path.join(xla_dir, curr_dir) if curr_dir else xla_dir
    if os.path.isfile(os.path.join(abs_dir, "BUILD.bazel")) or os.path.isfile(
        os.path.join(abs_dir, "BUILD")
    ):
      pkg = curr_dir
      if file_exists:
        target_name = (
            normalized[len(curr_dir) + 1 :] if curr_dir else normalized
        )
      else:
        target_name = "BUILD"
      break
    if not curr_dir:
      return None
    parent = os.path.dirname(curr_dir)
    if parent == curr_dir:
      return None
    curr_dir = parent

  if not pkg or not target_name:
    return None

  if pkg == "third_party/tsl" or pkg.startswith("third_party/tsl/"):
    tsl_pkg = pkg[len("third_party/tsl") :].lstrip("/")
    raw_label = f"@tsl//{tsl_pkg}:{target_name}"
  else:
    raw_label = f"@xla//{pkg}:{target_name}"

  return normalize_external_bazel_label(raw_label)


def get_jax_query_roots(build_name: str) -> Tuple[str, ...]:
  """Returns the JAX root targets to query for the given JAX build name."""
  upper_name = build_name.upper()
  if "ROCM" in upper_name or "AMD" in upper_name or "INSTINCT" in upper_name:
    return JAX_CPU_ROOT_TARGETS + JAX_ROCM_ROOT_TARGETS
  if "GPU" in upper_name or "CUDA" in upper_name:
    return JAX_CPU_ROOT_TARGETS + JAX_CUDA_ROOT_TARGETS
  if "CPU" in upper_name:
    return JAX_CPU_ROOT_TARGETS
  return ()


def are_files_reachable_in_jax(
    file_labels: Sequence[str],
    jax_roots: Sequence[str],
    jax_dir: str,
    *,
    startup_options: Sequence[str] = (),
    query_options: Sequence[str] = (),
) -> Optional[bool]:
  """Checks via `bazel query` whether any label in file_labels is in deps(jax_roots).

  Queries `deps(jax_roots) + buildfiles(deps(jax_roots))` filtered to `@xla` and
  `@tsl` targets and intersects the normalized labels in Python. This avoids
  passing `set(file_labels)` to Bazel, which would fail with exit code 7
  (`NoSuchTargetException`) when a modified file is conditionally excluded from
  the current platform's BUILD evaluation (e.g. `if_cuda_is_configured` or
  `if_rocm_is_configured` during a CPU build).

  Args:
    file_labels: External Bazel file labels (e.g. `@xla//pkg:file.cc`).
    jax_roots: Root JAX targets for the current backend.
    jax_dir: Path to the JAX repository root where `bazel query` is executed.
    startup_options: Bazel startup flags placed before `query`.
    query_options: Additional `bazel query` flags (such as `--config`,
      `--repo_env`, `--override_repository`, and `--override_module`).

  Returns:
    True if at least one file label is reachable from `jax_roots`, False if
    `bazel query` succeeds and the intersection is empty, or None if the query
    cannot be executed or fails.
  """
  if not file_labels or not jax_roots:
    return None
  normalized_targets = set()
  for label in file_labels:
    norm = normalize_external_bazel_label(label)
    if norm is None:
      return None
    normalized_targets.add(norm)

  roots_expr = " + ".join(jax_roots)
  query_expr = (
      r'filter("^@+[^/]*(xla|tsl)[^/]*//", '
      f"deps({roots_expr}) + buildfiles(deps({roots_expr})))"
  )
  command = [
      "bazel",
      *startup_options,
      "query",
      "--noshow_progress",
      *query_options,
      query_expr,
  ]
  try:
    result = subprocess.run(
        command,
        cwd=jax_dir,
        capture_output=True,
        text=True,
        check=False,
        timeout=600,
    )
  except (OSError, subprocess.SubprocessError) as e:
    logging.warning("Failed to run bazel query in %s: %s", jax_dir, e)
    return None

  if result.returncode != 0:
    logging.warning(
        "bazel query failed with exit code %d: %s",
        result.returncode,
        result.stderr.strip(),
    )
    return None

  for line in result.stdout.splitlines():
    norm = normalize_external_bazel_label(line)
    if norm in normalized_targets:
      return True
  return False


def evaluate_skip(
    base_sha: str,
    head_sha: str = "HEAD",
    cwd: str = ".",
    *,
    is_jax_build: bool = False,
    jax_dir: Optional[str] = None,
    build_name: str = "",
    startup_options: Sequence[str] = (),
    query_options: Sequence[str] = (),
) -> SkipDecision:
  """Evaluates whether a CI build can be skipped based on git changed files.

  Guaranteed fail-open: any git, OS, or validation failure returns
  should_skip=False.

  Args:
    base_sha: Base git commit SHA to diff against.
    head_sha: Head git commit SHA or ref.
    cwd: Path to the XLA git repository root to inspect.
    is_jax_build: Whether the build under evaluation is a JAX CI job.
    jax_dir: Optional path to the JAX workspace root. When provided for a JAX
      build, uses `bazel query` to check whether modified XLA files are in the
      transitive dependency graph of the JAX build's targets.
    build_name: Name or BuildType string of the build under evaluation.
    startup_options: Optional Bazel startup flags for `bazel query` in
      `jax_dir`.
    query_options: Optional `bazel query` flags (`--config`, `--repo_env`,
      `--override_repository`, `--override_module`) for `jax_dir`.

  Returns:
    A SkipDecision indicating whether the build can be skipped and why.
  """
  start_time = time.time()
  cwd = os.path.abspath(cwd)

  try:
    base_sha = validate_git_ref(base_sha)
    head_sha = validate_git_ref(head_sha or "HEAD")
  except ValueError as e:
    return SkipDecision(
        should_skip=False,
        reason=f"Invalid git revision parameter: {e}",
        elapsed_seconds=time.time() - start_time,
    )

  ensure_base_fetched(base_sha, head_sha, cwd=cwd)

  try:
    changed_files = get_changed_files(base_sha, head_sha, cwd=cwd)
  except (OSError, subprocess.SubprocessError, ValueError) as e:
    return SkipDecision(
        should_skip=False,
        reason=f"Failed to get git changed files: {e}",
        elapsed_seconds=time.time() - start_time,
    )

  changed_count = len(changed_files)
  if not changed_files:
    return SkipDecision(
        should_skip=False,
        reason="No changed files detected",
        changed_files_count=0,
        elapsed_seconds=time.time() - start_time,
    )

  if is_docs_or_metadata_only(changed_files):
    return SkipDecision(
        should_skip=True,
        reason="Only documentation or repository metadata modified",
        changed_files_count=changed_count,
        elapsed_seconds=time.time() - start_time,
    )

  if not is_jax_build:
    return SkipDecision(
        should_skip=False,
        reason="Non-documentation changes require target analysis",
        changed_files_count=changed_count,
        elapsed_seconds=time.time() - start_time,
    )

  if any(is_ci_or_global_config_file(f) for f in changed_files):
    return SkipDecision(
        should_skip=False,
        reason="Build configuration, proto, or CI workflow modified",
        changed_files_count=changed_count,
        elapsed_seconds=time.time() - start_time,
    )

  if all(is_unimpacted_for_all_jax(f) for f in changed_files):
    return SkipDecision(
        should_skip=True,
        reason="Modified files do not affect JAX builds",
        changed_files_count=changed_count,
        elapsed_seconds=time.time() - start_time,
    )

  if jax_dir:
    jax_roots = get_jax_query_roots(build_name)
    if jax_roots:
      candidate_files = [
          f for f in changed_files if not is_unimpacted_for_all_jax(f)
      ]
      file_labels: List[str] = []
      all_mapped = True
      for file_path in candidate_files:
        label = resolve_bazel_file_label(file_path, xla_dir=cwd)
        if label is None:
          all_mapped = False
          break
        file_labels.append(label)

      if all_mapped and file_labels:
        reachable = are_files_reachable_in_jax(
            file_labels,
            jax_roots,
            os.path.abspath(jax_dir),
            startup_options=startup_options,
            query_options=query_options,
        )
        if reachable is not None:
          if not reachable:
            return SkipDecision(
                should_skip=True,
                reason=(
                    "Modified files are not in the Bazel dependency graph of"
                    " JAX build targets"
                ),
                changed_files_count=changed_count,
                elapsed_seconds=time.time() - start_time,
            )
          return SkipDecision(
              should_skip=False,
              reason=(
                  "Modified files are in the Bazel dependency graph of JAX"
                  " build targets"
              ),
              changed_files_count=changed_count,
              elapsed_seconds=time.time() - start_time,
          )

  if any(is_build_or_proto_or_ci_core_file(f) for f in changed_files):
    return SkipDecision(
        should_skip=False,
        reason="Build configuration, proto, or CI workflow modified",
        changed_files_count=changed_count,
        elapsed_seconds=time.time() - start_time,
    )

  return SkipDecision(
      should_skip=False,
      reason="Changed files may impact JAX build targets",
      changed_files_count=changed_count,
      elapsed_seconds=time.time() - start_time,
  )


def report_skip(decision: SkipDecision, build_name: str) -> None:
  """Emits skip decision summary to stdout and appends to $GITHUB_STEP_SUMMARY."""
  status = "SKIP" if decision.should_skip else "RUN"
  summary_line = (
      f"change-detector Build='{build_name}': "
      f"Decision={status}, "
      f"ChangedFiles={decision.changed_files_count}, "
      f"Reason='{decision.reason}', "
      f"Duration={decision.elapsed_seconds:.2f}s"
  )
  separator = "=" * 55
  sys.stdout.write(f"\n{separator}\n")
  sys.stdout.write(f"{summary_line}\n")
  sys.stdout.write(f"{separator}\n\n")
  sys.stdout.flush()

  step_summary_file = os.environ.get("GITHUB_STEP_SUMMARY")
  if step_summary_file:
    try:
      with open(step_summary_file, "a", encoding="utf-8") as f:
        f.write("### Change Detector Analysis\n")
        f.write(
            "| Parameter | Value |\n"
            "|---|---|\n"
            f"| **Build** | `{build_name}` |\n"
            f"| **Decision** | **`{status}`** |\n"
            f"| **Changed files** | {decision.changed_files_count} |\n"
            f"| **Reason** | {decision.reason} |\n"
            f"| **Duration** | {decision.elapsed_seconds:.2f}s |\n\n"
        )
    except OSError as e:
      logging.warning("Failed to write to GITHUB_STEP_SUMMARY: %s", e)


def _write_github_output(github_output: str, should_skip: bool) -> None:
  """Writes should_skip output key to $GITHUB_OUTPUT if configured."""
  if not github_output:
    return
  try:
    with open(github_output, "a", encoding="utf-8") as f:
      f.write(f"should_skip={'true' if should_skip else 'false'}\n")
  except OSError as e:
    logging.warning(
        "Failed to write to GITHUB_OUTPUT (%s): %s", github_output, e
    )


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
  """Parses CLI arguments for evaluating fast-skip decisions in CI."""
  parser = argparse.ArgumentParser(
      description="Evaluate whether a CI build can be skipped from git diffs."
  )
  parser.add_argument(
      "--base_sha",
      default=os.environ.get("XLA_CI_BAZEL_DIFF_BASE_SHA", ""),
      help="Base git revision to compare against.",
  )
  parser.add_argument(
      "--head_sha",
      default=os.environ.get("XLA_CI_BAZEL_DIFF_HEAD_SHA", "HEAD") or "HEAD",
      help="Head git revision to compare.",
  )
  parser.add_argument(
      "--cwd",
      default=".",
      help="Path to the git repository root.",
  )
  parser.add_argument(
      "--build_name",
      default="ci_build",
      help="Build name used in step summary reports.",
  )
  parser.add_argument(
      "--is_jax_build",
      action="store_true",
      help="Whether the build under evaluation is a JAX CI job.",
  )
  parser.add_argument(
      "--jax_dir",
      default="",
      help="Optional path to the JAX workspace root for Bazel query analysis.",
  )
  parser.add_argument(
      "--github_output",
      default=os.environ.get("GITHUB_OUTPUT", ""),
      help="Path to GitHub Actions $GITHUB_OUTPUT file.",
  )
  return parser.parse_args(argv)


def run_cli(args: argparse.Namespace) -> int:
  """Executes fast-skip evaluation from parsed CLI flags.

  Args:
    args: Parsed command-line arguments from `_parse_args`.

  Returns:
    Process exit status (0 on completion or fail-open fallback).
  """
  disabled = os.environ.get("XLA_CI_DISABLE_BAZEL_DIFF", "false").lower() in (
      "true",
      "1",
      "yes",
  )
  base_sha = args.base_sha.strip()
  if disabled or not base_sha:
    _write_github_output(args.github_output, should_skip=False)
    return 0

  decision = evaluate_skip(
      base_sha,
      args.head_sha.strip() or "HEAD",
      cwd=args.cwd,
      is_jax_build=args.is_jax_build,
      jax_dir=args.jax_dir.strip() or None,
      build_name=args.build_name,
  )
  report_skip(decision, args.build_name)
  _write_github_output(args.github_output, should_skip=decision.should_skip)
  return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
  """CLI entry point for evaluating fast-skip decisions in GitHub Actions."""
  logging.basicConfig(level=logging.INFO)
  return run_cli(_parse_args(argv))


if __name__ == "__main__":
  sys.exit(main())
