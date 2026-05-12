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

import io
import json
import pathlib
import tempfile
import textwrap

from absl.testing import absltest
from absl.testing import parameterized

from build_tools.ci import clang_tidy_diff


class TestClangTidyDiff(parameterized.TestCase):

  @parameterized.parameters(
      (0, 1),
      (5, 1),
      (10, 2),
      (15, 2),
      (20, 3),
      (25, 3),
      (30, 4),
      (35, 4),
  )
  def test_offset_to_line(self, offset, expected_line):
    offsets = [0, 10, 20, 30]
    self.assertEqual(
        clang_tidy_diff.offset_to_line(offsets, offset), expected_line
    )

  def test_offset_to_line_empty(self):
    self.assertEqual(clang_tidy_diff.offset_to_line([], 10), -1)

  def test_normalize_path_relative(self):
    self.assertEqual(
        clang_tidy_diff.normalize_path("foo/bar.cc", "/root"), "foo/bar.cc"
    )

  def test_normalize_path_absolute_in_repo(self):
    self.assertEqual(
        clang_tidy_diff.normalize_path("/root/foo/bar.cc", "/root"),
        "foo/bar.cc",
    )

  def test_normalize_path_absolute_outside_repo(self):
    self.assertEqual(
        clang_tidy_diff.normalize_path("/other/foo/bar.cc", "/root"),
        "/other/foo/bar.cc",
    )

  def test_normalize_path_execroot(self):
    path = "/usr/local/google/home/user/.cache/bazel/_bazel_user/a708d4fc59660ccd295a76cce84d113c/execroot/xla/xla/stream_executor/cuda/cuda_status.h"
    self.assertEqual(
        clang_tidy_diff.normalize_path(path, "/root"),
        "xla/stream_executor/cuda/cuda_status.h",
    )

  def test_normalize_path_remote_worker(self):
    path = "/b/f/w/xla/backends/gpu/codegen/triton/transforms/lowering_utils.h"
    # repo_root is /__w/xla/xla, so workspace name is 'xla'
    self.assertEqual(
        clang_tidy_diff.normalize_path(path, "/__w/xla/xla"),
        "xla/backends/gpu/codegen/triton/transforms/lowering_utils.h",
    )

  def test_normalize_path_local_ci_runner(self):
    path = "/__w/xla/xla/xla/backends/gpu/codegen/triton/transforms/lowering_utils.h"
    self.assertEqual(
        clang_tidy_diff.normalize_path(path, "/__w/xla/xla"),
        "xla/backends/gpu/codegen/triton/transforms/lowering_utils.h",
    )

  def test_normalize_path_remote_worker_third_party(self):
    path = "/b/f/w/third_party/gpus/cuda/include/cuda.h"
    self.assertEqual(
        clang_tidy_diff.normalize_path(path, "/__w/xla/xla"),
        "third_party/gpus/cuda/include/cuda.h",
    )

  def test_parse_diff(self):
    tmpdir = self.create_tempdir()
    diff_path = pathlib.Path(tmpdir) / "test.diff"
    with open(diff_path, "w") as f:

      f.write(textwrap.dedent("""\
                  diff --git a/file1.cc b/file1.cc
                  index 123456..789012 100644
                  --- a/file1.cc
                  +++ b/file1.cc
                  @@ -1,2 +1,3 @@
                   line1
                  +line2
                   line3
                  """))
    ranges = clang_tidy_diff.parse_diff(str(diff_path))
    self.assertEqual(ranges, {"file1.cc": {2}})

  def test_parse_bep(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      bep_path = pathlib.Path(tmpdir) / "test.bep"
      with open(bep_path, "w") as f:
        f.write(
            '{"namedSetOfFiles": {"files": [{"name": "file1.clang-tidy.yaml",'
            ' "pathPrefix": ["bazel-out", "k8-opt", "bin"]}]}}\n'
        )
      yaml_files = clang_tidy_diff.parse_bep(str(bep_path), "/root")
      self.assertEqual(
          yaml_files, ["/root/bazel-out/k8-opt/bin/file1.clang-tidy.yaml"]
      )

  def test_parse_clang_tidy_yaml(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      yaml_path = pathlib.Path(tmpdir) / "test.yaml"
      with open(yaml_path, "w") as f:

        f.write(textwrap.dedent("""\
                    ---
                    MainSourceFile:  '/root/file1.cc'
                    Diagnostics:
                      - DiagnosticName:  misc-unused
                        DiagnosticMessage:
                          Message:         'unused variable'
                          FilePath:        '/root/file1.cc'
                          FileOffset:      15
                    ...
                    """))

      data = clang_tidy_diff.parse_clang_tidy_yaml(str(yaml_path))

      self.assertEqual(data.get("MainSourceFile"), "/root/file1.cc")
      diagnostics = data.get("Diagnostics", [])
      with self.subTest("Diagnostics"):
        self.assertLen(diagnostics, 1)
        self.assertEqual(diagnostics[0].get("DiagnosticName"), "misc-unused")
        self.assertEqual(diagnostics[0].get("Message"), "unused variable")
        self.assertEqual(diagnostics[0].get("FilePath"), "/root/file1.cc")
        self.assertEqual(diagnostics[0].get("FileOffset"), 15)

  def test_process_file(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      yaml_path = pathlib.Path(tmpdir) / "test.clang-tidy.yaml"
      with open(yaml_path, "w") as f:
        f.write(textwrap.dedent(f"""\
                    ---
                    MainSourceFile:  '{tmpdir}/file1.cc'
                    Diagnostics:
                      - DiagnosticName:  misc-unused
                        DiagnosticMessage:
                          Message:         'unused variable'
                          FilePath:        '{tmpdir}/file1.cc'
                          FileOffset:      15
                        Level:           Error
                    ...
                    """))

      diff_path = pathlib.Path(tmpdir) / "test.diff"
      with open(diff_path, "w") as f:

        f.write(textwrap.dedent("""\
                    diff --git a/file1.cc b/file1.cc
                    index 123456..789012 100644
                    --- a/file1.cc
                    +++ b/file1.cc
                    @@ -1,2 +1,3 @@
                     line1
                    +line2
                     line3
                    """))

      bep_path = pathlib.Path(tmpdir) / "test.bep"
      with open(bep_path, "w") as f:
        f.write(
            json.dumps({
                "namedSetOfFiles": {
                    "files": [{
                        "name": "test.clang-tidy.yaml",
                        "pathPrefix": [],
                    }]
                }
            })
            + "\n"
        )

      config = clang_tidy_diff.AppConfig(
          patch=str(diff_path),
          repo_root=str(tmpdir),
          bep_file=str(bep_path),
          warnings_as_errors=True,
      )

      def mock_offset_provider(_: str) -> list[int]:
        return [0, 10, 20, 30]

      filterer = clang_tidy_diff.ClangTidyDiffFilter(
          config, offset_provider=mock_offset_provider
      )
      diagnostics: list[clang_tidy_diff.Diagnostic] = filterer.process_file(
          str(yaml_path)
      )

      with self.subTest("Diagnostics"):
        self.assertLen(diagnostics, 1)
        self.assertEqual(diagnostics[0].file_path, "file1.cc")
        self.assertEqual(diagnostics[0].line_num, 2)
        self.assertEqual(
            diagnostics[0].col_num, 6
        )  # Offset 15 - Line 2 start 10 + 1 = 6
        self.assertEqual(diagnostics[0].level, "Error")
        self.assertEqual(diagnostics[0].name, "misc-unused")
        self.assertEqual(diagnostics[0].message, "unused variable")

  def test_process_file_no_substring_false_positives(self):
    """Tests that we don't get false positives from diff file paths being substrings of other file paths."""
    tmpdir = self.create_tempdir()
    yaml_path = pathlib.Path(tmpdir) / "xla/long_util.cc.clang-tidy.yaml"
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with open(yaml_path, "w") as f:
      f.write(textwrap.dedent(f"""\
                  ---
                  MainSourceFile:  '{tmpdir}/xla/long_util.cc'
                  Diagnostics: []
                  """))
    # util and long_util are both in the diff but only long_util has a report.
    diff_path = pathlib.Path(tmpdir) / "test.diff"
    with open(diff_path, "w") as f:
      f.write(textwrap.dedent("""\
                  diff --git a/util.cc b/util.cc
                  index 123456..789012 100644
                  --- a/util.cc
                  +++ b/util.cc
                  @@ -1,1 +1,2 @@
                   line1
                  +line2
                  diff --git a/xla/long_util.cc b/xla/long_util.cc
                  index 123456..789012 100644
                  --- a/xla/long_util.cc
                  +++ b/xla/long_util.cc
                  @@ -1,1 +1,2 @@
                   line1
                  +line2
                  """))
    bep_path = pathlib.Path(tmpdir) / "test.bep"
    with open(bep_path, "w") as f:
      f.write(
          json.dumps({
              "namedSetOfFiles": {
                  "files": [{
                      "name": "xla/long_util.cc.clang-tidy.yaml",
                      "pathPrefix": [],
                  }]
              }
          })
          + "\n"
      )
    config = clang_tidy_diff.AppConfig(
        patch=str(diff_path),
        repo_root=str(tmpdir),
        bep_file=str(bep_path),
        warnings_as_errors=True,
    )
    filterer = clang_tidy_diff.ClangTidyDiffFilter(config)
    _ = filterer.process_file(str(yaml_path))
    self.assertIn("xla/long_util.cc", filterer.seen_files)
    self.assertNotIn("util.cc", filterer.seen_files)

  def test_run(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      yaml_path = pathlib.Path(tmpdir) / "test.clang-tidy.yaml"
      with open(yaml_path, "w") as f:
        f.write(textwrap.dedent(f"""\
                    ---
                    MainSourceFile:  '{tmpdir}/file1.cc'
                    Diagnostics:
                      - DiagnosticName:  misc-unused
                        DiagnosticMessage:
                          Message:         'unused variable'
                          FilePath:        '{tmpdir}/file1.cc'
                          FileOffset:      15
                        Level:           Error
                    ...
                    """))

      diff_path = pathlib.Path(tmpdir) / "test.diff"
      with open(diff_path, "w") as f:

        f.write(textwrap.dedent("""\
                    diff --git a/file1.cc b/file1.cc
                    index 123456..789012 100644
                    --- a/file1.cc
                    +++ b/file1.cc
                    @@ -1,2 +1,3 @@
                     line1
                    +line2
                     line3
                    """))

      bep_path = pathlib.Path(tmpdir) / "test.bep"
      with open(bep_path, "w") as f:
        f.write(
            json.dumps({
                "namedSetOfFiles": {
                    "files": [{
                        "name": "test.clang-tidy.yaml",
                        "pathPrefix": [],
                    }]
                }
            })
            + "\n"
        )

      config = clang_tidy_diff.AppConfig(
          patch=str(diff_path),
          repo_root=str(tmpdir),
          bep_file=str(bep_path),
          warnings_as_errors=True,
      )

      def mock_offset_provider(_: str) -> list[int]:
        return [0, 10, 20, 30]

      filterer = clang_tidy_diff.ClangTidyDiffFilter(
          config, offset_provider=mock_offset_provider
      )

      self.assertFalse(filterer.run())

  def test_print_diagnostic_sanity(self):
    diag = clang_tidy_diff.Diagnostic(
        file_path="file1.cc",
        line_num=2,
        col_num=3,
        level="warning",
        name="misc-unused",
        message="unused variable",
        yaml_file="test.clang-tidy.yaml",
    )
    with tempfile.TemporaryDirectory() as tmpdir:
      src_file = pathlib.Path(tmpdir) / "file1.cc"
      with open(src_file, "w") as f:
        f.write("line1\nline2\nline3\n")
      captured_stderr = io.StringIO()
      # Run it with warnings_as_errors=True to test that path too
      clang_tidy_diff.print_diagnostic(
          diag,
          repo_root=tmpdir,
          warnings_as_errors=True,
          stream=captured_stderr,
      )
      output = captured_stderr.getvalue()
      with self.subTest("diagnostic_string_sanity"):
        self.assertIn("file1.cc:2:3", output)
        self.assertIn("error:", output)
        self.assertIn("unused variable", output)
        self.assertIn("[misc-unused]", output)
        self.assertIn("  2 |", output)  # Snippet line
        self.assertIn("line2", output)
        self.assertIn("^", output)  # Caret


if __name__ == "__main__":
  absltest.main()
