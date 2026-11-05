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
"""Tests for change_detector module."""

import os
import subprocess
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized

from build_tools.ci import change_detector


class ChangeDetectorTest(parameterized.TestCase):

  @parameterized.parameters(
      (["MODULE.bazel"], True),
      (["MODULE.bazel.lock"], True),
      (["REPO.bazel"], True),
      (["WORKSPACE"], True),
      (["WORKSPACE.bzlmod"], True),
      ([".bazelrc"], True),
      ([".bazelversion"], True),
      (["workspace0.bzl"], True),
      (["workspace1.bzl"], True),
      (["workspace2.bzl"], True),
      (["workspace3.bzl"], True),
      (["workspace4.bzl"], True),
      (["requirements_lock_3_11.txt"], True),
      (["requirements_lock_3_12.txt"], True),
      (["models.BUILD"], True),
      (["arm_compiler.BUILD"], True),
      (["tools/def_file_filter/def_file_filter.BUILD"], True),
      (["third_party/repo.bzl"], True),
      (["third_party/quilt_repo.bzl"], True),
      (["third_party/extensions/third_party.bzl"], True),
      (["third_party/extensions/llvm.bzl"], True),
      (["third_party/cccl/cccl_extension.bzl"], True),
      (["third_party/libdrm/extension.bzl"], True),
      (["third_party/llvm/setup.bzl"], True),
      (["third_party/gpus/cuda_configure.bzl"], True),
      (["third_party/gpus/rocm_configure.bzl"], True),
      (
          ["third_party/gpus/cuda/hermetic/cuda_redist_init_repositories.bzl"],
          True,
      ),
      (["third_party/py/python_init_repositories.bzl"], True),
      (["third_party/mkl_dnn/workspace.bzl"], True),
      (["third_party/mkl_dnn/mkldnn_v1_async.BUILD"], True),
      (["third_party/mkl_dnn/onednn_acl_aarch64_build_fixes.patch"], True),
      (["third_party/mkl_dnn/BUILD"], True),
      (["third_party/triton/intel_xpu/workspace.bzl"], True),
      (["third_party/triton/common/series"], True),
      (["third_party/cuda_tile/patches/series"], True),
      (["third_party/xz/xz.BUILD.bazel"], True),
      (["third_party/xz/config/BUILD.bazel"], True),
      (["third_party/xz/patches/remove_have_config.diff"], True),
      (["third_party/sqlite/BUILD"], True),
      (["third_party/hwloc/BUILD.system"], True),
      (["third_party/boringssl.BUILD"], True),
      (["third_party/boringssl.patch"], True),
      (["third_party/gpus/cuda/hermetic/cuda_Automatic.BUILD.tpl"], True),
      (["xla/flags.bzl"], False),
      (["third_party/tsl/tsl/platform/default/rules.bzl"], False),
      (["third_party/tsl/REPO.bazel"], True),
      (["third_party/tsl/tsl/BUILD"], False),
      (["third_party/tsl/tsl/platform/denormal.cc"], False),
      (["third_party/py/rules_pywrap/pywrap.default.bzl"], False),
      (["third_party/py/rules_pywrap/BUILD"], False),
      (["third_party/py/numpy/BUILD"], False),
      (["third_party/py/numpy/tf_numpy_dummy.py"], False),
      (["third_party/rules_python/python/defs.bzl"], False),
      (["third_party/rules_python/python/BUILD"], False),
      (["third_party/compute_library/build_defs.bzl"], False),
      (["xla/service/cpu/cpu_compiler.cc"], False),
      (["docs/overview.md", "xla/BUILD"], False),
  )
  def test_is_global_config_changed(self, changed_files, expected):
    self.assertEqual(
        change_detector.is_global_config_changed(changed_files), expected
    )

  @parameterized.parameters(
      (["README.md", "docs/index.md"], True),
      (["OWNERS", "LICENSE", "LICENSE.txt"], True),
      ([".gitignore", ".clang-format", ".clang-tidy", ".clangd"], True),
      (["logo.png", "diagram.svg"], True),
      (["README.md", "xla/service/hlo_parser.cc"], False),
      (["LICENSE.py"], False),
      (["LICENSE.sh"], False),
      ([".clang_hook.sh"], False),
      ([".gitignore.py"], False),
      (["docs/script.py"], False),
      ([], False),
  )
  def test_is_docs_or_metadata_only(self, changed_files, expected):
    self.assertEqual(
        change_detector.is_docs_or_metadata_only(changed_files), expected
    )

  @parameterized.parameters(
      ("26672b856b96dfb0a893f160ee61525526366663",),
      ("HEAD",),
      ("HEAD~1",),
      ("origin/main",),
      ("base_sha",),
  )
  def test_validate_git_ref_valid(self, ref):
    self.assertEqual(change_detector.validate_git_ref(ref), ref)

  @parameterized.parameters(
      ("",),
      ("-f",),
      ("--upload-pack=evil",),
      ("--output=/tmp/pwn",),
      ("HEAD; rm -rf /",),
      ("ref with spaces",),
  )
  def test_validate_git_ref_invalid(self, ref):
    with self.assertRaises(ValueError):
      change_detector.validate_git_ref(ref)

  @mock.patch.object(
      change_detector.subprocess, "run", autospec=True, spec_set=True
  )
  def test_get_merge_base_success(self, mock_run):
    mock_run.return_value = subprocess.CompletedProcess(
        args=["git", "merge-base"], returncode=0, stdout="base_commit\n"
    )
    self.assertEqual(
        change_detector.get_merge_base("base", "head"), "base_commit"
    )
    mock_run.assert_called_once_with(
        ["git", "merge-base", "--", "base", "head"],
        cwd=".",
        capture_output=True,
        text=True,
        check=False,
    )

  @mock.patch.object(
      change_detector.subprocess, "run", autospec=True, spec_set=True
  )
  def test_get_merge_base_failure(self, mock_run):
    mock_run.return_value = subprocess.CompletedProcess(
        args=["git", "merge-base"], returncode=128, stdout=""
    )
    self.assertIsNone(change_detector.get_merge_base("base", "head"))

  @mock.patch.object(
      change_detector.subprocess, "run", autospec=True, spec_set=True
  )
  def test_get_merge_base_rejects_flag_injection(self, mock_run):
    self.assertIsNone(
        change_detector.get_merge_base("--upload-pack=evil", "HEAD")
    )
    mock_run.assert_not_called()

  @mock.patch.object(
      change_detector, "get_merge_base", autospec=True, spec_set=True
  )
  def test_get_diff_base_with_merge_base(self, mock_mb):
    mock_mb.return_value = "merge_base_sha"
    self.assertEqual(
        change_detector.get_diff_base("base_sha", "head_sha"), "merge_base_sha"
    )

  @mock.patch.object(
      change_detector, "get_merge_base", autospec=True, spec_set=True
  )
  def test_get_diff_base_fallback(self, mock_mb):
    mock_mb.return_value = None
    self.assertEqual(
        change_detector.get_diff_base("base_sha", "head_sha"), "base_sha"
    )

  @mock.patch.object(
      change_detector, "run_git_best_effort", autospec=True, spec_set=True
  )
  @mock.patch.object(
      change_detector, "get_merge_base", autospec=True, spec_set=True
  )
  def test_ensure_base_fetched_already_present(self, mock_mb, mock_git):
    mock_mb.return_value = "merge_base_sha"
    change_detector.ensure_base_fetched("base_sha", "head_sha", cwd="/repo")
    mock_mb.assert_called_once_with("base_sha", "head_sha", cwd="/repo")
    mock_git.assert_not_called()

  @mock.patch.object(
      change_detector, "run_git_best_effort", autospec=True, spec_set=True
  )
  @mock.patch.object(
      change_detector, "get_merge_base", autospec=True, spec_set=True
  )
  def test_ensure_base_fetched_shallow_fetch_succeeds(self, mock_mb, mock_git):
    mock_mb.side_effect = [None, "merge_base_sha"]
    mock_git.side_effect = [
        subprocess.CompletedProcess(
            args=["git", "cat-file", "commit", "head_sha"],
            returncode=0,
            stdout="tree 0\nparent other_sha\n\nmsg\n",
        ),
        subprocess.CompletedProcess(args=["git", "fetch"], returncode=0),
    ]
    change_detector.ensure_base_fetched("base_sha", "head_sha", cwd="/repo")
    self.assertEqual(
        mock_git.call_args_list,
        [
            mock.call(["cat-file", "commit", "head_sha"], cwd="/repo"),
            mock.call(
                [
                    "fetch",
                    "--no-tags",
                    "--depth=50",
                    "--",
                    "origin",
                    "base_sha",
                    "head_sha",
                ],
                cwd="/repo",
            ),
        ],
    )

  @mock.patch.object(
      change_detector, "run_git_best_effort", autospec=True, spec_set=True
  )
  @mock.patch.object(
      change_detector, "get_merge_base", autospec=True, spec_set=True
  )
  def test_ensure_base_fetched_falls_back_to_unshallow(self, mock_mb, mock_git):
    mock_mb.side_effect = [None, None]
    mock_git.side_effect = [
        subprocess.CompletedProcess(
            args=["git", "cat-file", "commit", "HEAD"],
            returncode=0,
            stdout="tree 0\nparent other_sha\n\nmsg\n",
        ),
        subprocess.CompletedProcess(args=["git", "fetch"], returncode=0),
        subprocess.CompletedProcess(args=["git", "fetch"], returncode=0),
    ]
    change_detector.ensure_base_fetched("base_sha", "HEAD", cwd="/repo")
    self.assertEqual(
        mock_git.call_args_list,
        [
            mock.call(["cat-file", "commit", "HEAD"], cwd="/repo"),
            mock.call(
                [
                    "fetch",
                    "--no-tags",
                    "--depth=50",
                    "--",
                    "origin",
                    "base_sha",
                ],
                cwd="/repo",
            ),
            mock.call(["fetch", "--unshallow", "origin"], cwd="/repo"),
        ],
    )

  @mock.patch.object(
      change_detector, "run_git_best_effort", autospec=True, spec_set=True
  )
  @mock.patch.object(
      change_detector, "get_merge_base", autospec=True, spec_set=True
  )
  def test_ensure_base_fetched_direct_parent_fetches_depth_one(
      self, mock_mb, mock_git
  ):
    mock_mb.return_value = None
    mock_git.side_effect = [
        subprocess.CompletedProcess(
            args=["git", "cat-file", "commit", "merge_sha"],
            returncode=0,
            stdout="tree 0\nparent base_sha\nparent pr_sha\n\nMerge\n",
        ),
        subprocess.CompletedProcess(args=["git", "cat-file"], returncode=1),
        subprocess.CompletedProcess(args=["git", "fetch"], returncode=0),
        subprocess.CompletedProcess(args=["git", "cat-file"], returncode=0),
    ]
    change_detector.ensure_base_fetched("base_sha", "merge_sha", cwd="/repo")
    self.assertEqual(
        mock_git.call_args_list,
        [
            mock.call(["cat-file", "commit", "merge_sha"], cwd="/repo"),
            mock.call(["cat-file", "-e", "base_sha^{commit}"], cwd="/repo"),
            mock.call(
                [
                    "fetch",
                    "--no-tags",
                    "--depth=1",
                    "--",
                    "origin",
                    "base_sha",
                ],
                cwd="/repo",
            ),
            mock.call(["cat-file", "-e", "base_sha^{commit}"], cwd="/repo"),
        ],
    )

  @mock.patch.object(
      change_detector, "run_git_best_effort", autospec=True, spec_set=True
  )
  @mock.patch.object(
      change_detector, "get_merge_base", autospec=True, spec_set=True
  )
  def test_ensure_base_fetched_direct_parent_falls_back_to_unshallow(
      self, mock_mb, mock_git
  ):
    mock_mb.return_value = None
    mock_git.side_effect = [
        subprocess.CompletedProcess(
            args=["git", "cat-file", "commit", "merge_sha"],
            returncode=0,
            stdout="tree 0\nparent base_sha\nparent pr_sha\n\nMerge\n",
        ),
        subprocess.CompletedProcess(args=["git", "cat-file"], returncode=1),
        subprocess.CompletedProcess(args=["git", "fetch"], returncode=128),
        subprocess.CompletedProcess(args=["git", "cat-file"], returncode=1),
        subprocess.CompletedProcess(args=["git", "fetch"], returncode=0),
    ]
    change_detector.ensure_base_fetched("base_sha", "merge_sha", cwd="/repo")
    self.assertEqual(
        mock_git.call_args_list,
        [
            mock.call(["cat-file", "commit", "merge_sha"], cwd="/repo"),
            mock.call(["cat-file", "-e", "base_sha^{commit}"], cwd="/repo"),
            mock.call(
                [
                    "fetch",
                    "--no-tags",
                    "--depth=1",
                    "--",
                    "origin",
                    "base_sha",
                ],
                cwd="/repo",
            ),
            mock.call(["cat-file", "-e", "base_sha^{commit}"], cwd="/repo"),
            mock.call(["fetch", "--unshallow", "origin"], cwd="/repo"),
        ],
    )

  @parameterized.named_parameters(
      dict(
          testcase_name="docs_only_xla_build",
          changed_files=["README.md", "docs/architecture.md"],
          is_jax_build=False,
          expected_skip=True,
      ),
      dict(
          testcase_name="docs_only_jax_build",
          changed_files=["README.md", "OWNERS"],
          is_jax_build=True,
          expected_skip=True,
      ),
      dict(
          testcase_name="non_docs_xla_build_defers_to_bazel_diff",
          changed_files=["xla/service/gpu/fusion_merger_test.cc"],
          is_jax_build=False,
          expected_skip=False,
      ),
      dict(
          testcase_name="xla_tests_and_benchmarks_skip_jax",
          changed_files=[
              "xla/service/gpu/fusion_merger_test.cc",
              "xla/hlo/ir/hlo_instruction_benchmark.cc",
              "xla/service/gpu/tests/gemm_rewrite.hlo",
          ],
          is_jax_build=True,
          expected_skip=True,
      ),
      dict(
          testcase_name="mosaic_test_does_not_skip_jax",
          changed_files=["xla/mosaic/gpu/mosaic_gpu_test.py"],
          is_jax_build=True,
          expected_skip=False,
      ),
      dict(
          testcase_name="workflow_and_build_tools_do_not_skip_jax",
          changed_files=[
              ".github/workflows/benchmark_presubmit.yml",
              "build_tools/rocm/run_xla.sh",
              "build_tools/lint/diff_parser.py",
          ],
          is_jax_build=True,
          expected_skip=False,
      ),
      dict(
          testcase_name="ci_yml_does_not_skip_jax",
          changed_files=[".github/workflows/ci.yml"],
          is_jax_build=True,
          expected_skip=False,
      ),
      dict(
          testcase_name="build_tools_ci_does_not_skip_jax",
          changed_files=["build_tools/ci/build.py"],
          is_jax_build=True,
          expected_skip=False,
      ),
      dict(
          testcase_name="gpu_sources_do_not_skip_jax",
          changed_files=["xla/service/gpu/nvptx_compiler.cc"],
          is_jax_build=True,
          expected_skip=False,
      ),
      dict(
          testcase_name="build_file_does_not_skip_jax",
          changed_files=[
              "xla/service/gpu/fusion_merger_test.cc",
              "xla/service/gpu/BUILD",
          ],
          is_jax_build=True,
          expected_skip=False,
      ),
      dict(
          testcase_name="proto_file_does_not_skip_jax",
          changed_files=["xla/service/gpu/model/hlo_op_profile.proto"],
          is_jax_build=True,
          expected_skip=False,
      ),
      dict(
          testcase_name="core_hlo_change_does_not_skip_jax",
          changed_files=["xla/hlo/ir/hlo_instruction.cc"],
          is_jax_build=True,
          expected_skip=False,
      ),
  )
  @mock.patch.object(
      change_detector, "ensure_base_fetched", autospec=True, spec_set=True
  )
  @mock.patch.object(
      change_detector, "get_changed_files", autospec=True, spec_set=True
  )
  def test_evaluate_skip(
      self,
      mock_changed,
      mock_fetch,
      changed_files,
      is_jax_build,
      expected_skip,
  ):
    mock_changed.return_value = changed_files
    decision = change_detector.evaluate_skip(
        "base_sha",
        "head_sha",
        cwd="/workspace/openxla/xla",
        is_jax_build=is_jax_build,
    )
    self.assertEqual(decision.should_skip, expected_skip)
    self.assertLen(changed_files, decision.changed_files_count)
    mock_fetch.assert_called_once_with(
        "base_sha", "head_sha", cwd="/workspace/openxla/xla"
    )

  @mock.patch.object(
      change_detector, "ensure_base_fetched", autospec=True, spec_set=True
  )
  @mock.patch.object(
      change_detector, "get_changed_files", autospec=True, spec_set=True
  )
  def test_evaluate_skip_git_fail_open(self, mock_changed, mock_fetch):
    mock_changed.side_effect = OSError("git failure")
    decision = change_detector.evaluate_skip(
        "base_sha", "head_sha", is_jax_build=True
    )
    self.assertFalse(decision.should_skip)
    self.assertIn("Failed to get git changed files", decision.reason)
    mock_fetch.assert_called_once()

  def test_report_skip(self):
    summary_file = self.create_tempfile("step_summary.md")
    with mock.patch.dict(
        os.environ, {"GITHUB_STEP_SUMMARY": summary_file.full_path}
    ):
      decision = change_detector.SkipDecision(
          should_skip=True,
          reason="Modified files do not affect JAX builds",
          changed_files_count=3,
          elapsed_seconds=0.04,
      )
      change_detector.report_skip(
          decision, "BuildType.JAX_LINUX_X86_CPU_BZLMOD_GITHUB_ACTIONS"
      )

    content = summary_file.read_text()
    self.assertIn("Change Detector Analysis", content)
    self.assertIn("JAX_LINUX_X86_CPU_BZLMOD_GITHUB_ACTIONS", content)
    self.assertIn("SKIP", content)
    self.assertIn("Modified files do not affect JAX builds", content)

  @mock.patch.object(
      change_detector, "evaluate_skip", autospec=True, spec_set=True
  )
  def test_main_writes_github_output(self, mock_eval):
    output_file = self.create_tempfile("github_output.txt")
    mock_eval.return_value = change_detector.SkipDecision(
        should_skip=True,
        reason="Modified files do not affect JAX builds",
        changed_files_count=1,
        elapsed_seconds=0.01,
    )
    rc = change_detector.main([
        "--base_sha=base_sha",
        "--head_sha=head_sha",
        "--is_jax_build",
        "--build_name=JAX ROCm",
        f"--github_output={output_file.full_path}",
    ])
    self.assertEqual(rc, 0)
    self.assertEqual(output_file.read_text(), "should_skip=true\n")

  def test_main_disabled_via_env(self):
    output_file = self.create_tempfile("github_output.txt")
    with mock.patch.dict(os.environ, {"XLA_CI_DISABLE_BAZEL_DIFF": "true"}):
      rc = change_detector.main([
          "--base_sha=base_sha",
          f"--github_output={output_file.full_path}",
      ])
    self.assertEqual(rc, 0)
    self.assertEqual(output_file.read_text(), "should_skip=false\n")


class _FakeGitRepo:
  """In-memory fake Git repository backing subprocess.run in tests."""

  def __init__(
      self,
      *,
      origin_commits: dict[str, tuple[tuple[str, ...], dict[str, str]]],
      checked_out_sha: str,
      shallow: bool = False,
  ):
    self.origin_commits = origin_commits
    self.head = checked_out_sha
    self.unshallow_called = False
    if shallow:
      self.local_commits = {checked_out_sha: origin_commits[checked_out_sha]}
      self.shallow_commits = {checked_out_sha}
    else:
      self.local_commits = dict(origin_commits)
      self.shallow_commits = set()

  def _ancestors(self, start: str) -> dict[str, int]:
    if start not in self.local_commits:
      return {}
    distances = {start: 0}
    queue = [start]
    while queue:
      curr = queue.pop(0)
      if curr in self.shallow_commits:
        continue
      parents, _ = self.local_commits[curr]
      for parent in parents:
        if parent in self.local_commits and parent not in distances:
          distances[parent] = distances[curr] + 1
          queue.append(parent)
    return distances

  def run(
      self,
      cmd: list[str],
      *,
      cwd: str = ".",
      capture_output: bool = False,
      text: bool = False,
      check: bool = False,
  ) -> subprocess.CompletedProcess[str]:
    del cwd, capture_output, text
    assert cmd[0] == "git", f"Unexpected command: {cmd}"
    args = cmd[1:]

    def _finish(
        returncode: int, stdout: str = "", stderr: str = ""
    ) -> subprocess.CompletedProcess[str]:
      if check and returncode != 0:
        raise subprocess.CalledProcessError(returncode, cmd, stdout, stderr)
      return subprocess.CompletedProcess(cmd, returncode, stdout, stderr)

    if args == ["rev-parse", "HEAD"]:
      return _finish(0, f"{self.head}\n")

    if len(args) == 3 and args[:2] == ["rev-parse", "--verify"]:
      target = args[2].removesuffix("^{commit}")
      resolved = self.head if target == "HEAD" else target
      if resolved in self.local_commits:
        return _finish(0, f"{resolved}\n")
      return _finish(128, stderr="unknown revision\n")

    if len(args) == 3 and args[:2] == ["cat-file", "-e"]:
      target = args[2].removesuffix("^{commit}")
      resolved = self.head if target == "HEAD" else target
      return _finish(0 if resolved in self.local_commits else 1)

    if len(args) == 3 and args[:2] == ["cat-file", "commit"]:
      resolved = self.head if args[2] == "HEAD" else args[2]
      if resolved not in self.local_commits:
        return _finish(128, stderr="bad object\n")
      parents, _ = self.local_commits[resolved]
      header_lines = ["tree " + "0" * 40]
      header_lines.extend(f"parent {p}" for p in parents)
      header_lines.append("author Test <test@example.com> 1700000000 +0000")
      header_lines.append("")
      header_lines.append("commit message with parent fake_line")
      return _finish(0, "\n".join(header_lines) + "\n")

    if len(args) == 4 and args[:2] == ["merge-base", "--"]:
      a_ref = self.head if args[2] == "HEAD" else args[2]
      b_ref = self.head if args[3] == "HEAD" else args[3]
      if a_ref not in self.local_commits or b_ref not in self.local_commits:
        return _finish(128)
      anc_a = self._ancestors(a_ref)
      anc_b = self._ancestors(b_ref)
      common = set(anc_a) & set(anc_b)
      if not common:
        return _finish(1)
      best = min(common, key=lambda c: anc_a[c] + anc_b[c])
      return _finish(0, f"{best}\n")

    if args[:2] == ["fetch", "--unshallow"]:
      self.unshallow_called = True
      self.local_commits = dict(self.origin_commits)
      self.shallow_commits.clear()
      return _finish(0)

    if args and args[0] == "fetch":
      sep = args.index("--")
      refs = args[sep + 2 :]
      is_depth_one = "--depth=1" in args
      for ref in refs:
        if ref not in self.origin_commits:
          return _finish(128, stderr=f"not our ref {ref}\n")
        self.local_commits[ref] = self.origin_commits[ref]
        if is_depth_one:
          self.shallow_commits.add(ref)
      return _finish(0)

    if args[:3] == ["diff", "--name-only", "--end-of-options"]:
      a_ref = self.head if args[3] == "HEAD" else args[3]
      b_ref = self.head if args[4] == "HEAD" else args[4]
      if a_ref not in self.local_commits or b_ref not in self.local_commits:
        return _finish(128, stderr="bad revision\n")
      _, files_a = self.local_commits[a_ref]
      _, files_b = self.local_commits[b_ref]
      changed = sorted(
          path
          for path in set(files_a) | set(files_b)
          if files_a.get(path) != files_b.get(path)
      )
      return _finish(0, "".join(f"{p}\n" for p in changed))

    raise AssertionError(f"Unhandled git args: {args}")


class ChangeDetectorPrMergeCommitTest(absltest.TestCase):

  def test_shallow_pr_merge_checkout_ignores_already_merged_changes(self):
    m0_sha = "0" * 40
    pr_sha = "1" * 40
    main_sha = "2" * 40
    merge_sha = "3" * 40
    repo = _FakeGitRepo(
        origin_commits={
            m0_sha: ((), {"xla/foo.cc": "v1\n", "README.md": "v1\n"}),
            pr_sha: (
                (m0_sha,),
                {"xla/foo.cc": "v2\n", "README.md": "v2\n"},
            ),
            main_sha: (
                (m0_sha,),
                {
                    "xla/foo.cc": "v2\n",
                    "xla/bar.cc": "main_only\n",
                    "README.md": "v1\n",
                },
            ),
            merge_sha: (
                (main_sha, pr_sha),
                {
                    "xla/foo.cc": "v2\n",
                    "xla/bar.cc": "main_only\n",
                    "README.md": "v2\n",
                },
            ),
        },
        checked_out_sha=merge_sha,
        shallow=True,
    )
    self.enter_context(
        mock.patch.object(
            change_detector.subprocess,
            "run",
            autospec=True,
            spec_set=True,
            side_effect=repo.run,
        )
    )

    decision = change_detector.evaluate_skip(
        main_sha, pr_sha, cwd="/repo", is_jax_build=False
    )

    self.assertTrue(decision.should_skip)
    self.assertEqual(decision.changed_files_count, 1)
    self.assertFalse(repo.unshallow_called)

  def test_resolve_pr_merge_refs_uses_merge_commit_first_parent(self):
    stale_base_sha = "0" * 40
    pr_sha = "1" * 40
    main_sha = "2" * 40
    merge_sha = "3" * 40
    other_sha = "4" * 40
    repo = _FakeGitRepo(
        origin_commits={
            stale_base_sha: ((), {}),
            pr_sha: ((stale_base_sha,), {}),
            main_sha: ((stale_base_sha,), {}),
            merge_sha: ((main_sha, pr_sha), {}),
            other_sha: ((main_sha,), {}),
        },
        checked_out_sha=merge_sha,
        shallow=True,
    )
    self.enter_context(
        mock.patch.object(
            change_detector.subprocess,
            "run",
            autospec=True,
            spec_set=True,
            side_effect=repo.run,
        )
    )

    # Resolves to (main_sha, merge_sha) even when base_sha was an older commit.
    self.assertEqual(
        change_detector.resolve_pr_merge_refs(
            stale_base_sha, pr_sha.upper(), cwd="/repo"
        ),
        (main_sha, merge_sha),
    )
    # Non-matching PR head SHA is left untouched.
    self.assertEqual(
        change_detector.resolve_pr_merge_refs(
            stale_base_sha, other_sha, cwd="/repo"
        ),
        (stale_base_sha, other_sha),
    )
    # Explicit HEAD or checked-out commit equal to head_sha is left untouched.
    self.assertEqual(
        change_detector.resolve_pr_merge_refs(
            stale_base_sha, "HEAD", cwd="/repo"
        ),
        (stale_base_sha, "HEAD"),
    )
    self.assertEqual(
        change_detector.resolve_pr_merge_refs(
            stale_base_sha, merge_sha, cwd="/repo"
        ),
        (stale_base_sha, merge_sha),
    )


if __name__ == "__main__":
  absltest.main()
