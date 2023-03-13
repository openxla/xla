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
import textwrap

from absl.testing import absltest
from xla.build_tools.github_actions.check_contents import check_contents


CLEAN_DIFF = textwrap.dedent(
    """\
    diff --git a/src/important.cc b/src/important.cc
    new file mode 100644
    index 0000000..9c68461
    --- /dev/null
    +++ b/src/important.cc
    @@ -0,0 +1,3 @@
    +// Here we care if we find prohibited regexes.
    +std::unique_ptr<int> add(int a, int b) {
    +  return std::make_unique<int>(a + b);
    +}
"""
)

SUPPRESSED_DIFF = textwrap.dedent(
    """\
    diff --git a/src/dir/bad.cc b/src/dir/bad.cc
    new file mode 100644
    index 0000000..2508a76
    --- /dev/null
    +++ b/src/dir/bad.cc
    @@ -0,0 +1,7 @@
    +// This code is bad!
    +
    +using Make_Unique = std::make_unique; // OK
    +
    +std::unique_ptr<int> add(int a, int b) {
    +  return Make_Unique<int>(a + b); // OK. Fixed now!
    +}
    diff --git a/src/important.cc b/src/important.cc
    new file mode 100644
    index 0000000..dc06702
    --- /dev/null
    +++ b/src/important.cc
    @@ -0,0 +1,5 @@
    +// Here we care if we find prohibited regexes.
    +
    +std::unique_ptr<int> add(int a, int b) {
    +  return std::make_unique<int>(a + b);
    +}
"""
)


class ParseDiffTest(absltest.TestCase):

  def test_parse_clean_diff(self):
    file_diffs = check_contents.parse_diff(CLEAN_DIFF)
    self.assertLen(file_diffs, 1)

    important_cc_diff = file_diffs[0]
    self.assertEqual(important_cc_diff.path, "src/important.cc")
    expected_lines = [
        (1, "// Here we care if we find prohibited regexes."),
        (2, "std::unique_ptr<int> add(int a, int b) {"),
        (3, "  return std::make_unique<int>(a + b);"),
        (4, "}"),
    ]

    self.assertEqual(important_cc_diff.added_lines, expected_lines)

  def test_parse_suppressed_diff(self):
    file_diffs = check_contents.parse_diff(SUPPRESSED_DIFF)
    self.assertLen(file_diffs, 2)

    bad_cc_diff = file_diffs[0]
    self.assertEqual(bad_cc_diff.path, "src/dir/bad.cc")

    expected_lines = [
        (1, "// This code is bad!"),
        (2, ""),
        (3, "using Make_Unique = std::make_unique; // OK"),
        (4, ""),
        (5, "std::unique_ptr<int> add(int a, int b) {"),
        (6, "  return Make_Unique<int>(a + b); // OK. Fixed now!"),
        (7, "}"),
    ]

    self.assertEqual(bad_cc_diff.added_lines, expected_lines)


class CheckDiffsTest(absltest.TestCase):

  def test_check_good_diff(self):
    file_diffs = check_contents.parse_diff(CLEAN_DIFF)

    locs = check_contents.check_diffs(
        file_diffs, prohibited_regex="Make_Unique", suppression_regex="OK"
    )
    self.assertEmpty(locs, 0)

  def test_check_suppressed_diff_without_suppressions(self):
    file_diffs = check_contents.parse_diff(SUPPRESSED_DIFF)

    locs = check_contents.check_diffs(
        file_diffs, prohibited_regex="Make_Unique"
    )

    expected_locs = [
        check_contents.RegexLocation(
            path="src/dir/bad.cc",
            line_number=3,
            line_contents="using Make_Unique = std::make_unique; // OK",
            matched_text="Make_Unique",
        ),
        check_contents.RegexLocation(
            path="src/dir/bad.cc",
            line_number=6,
            line_contents="  return Make_Unique<int>(a + b); // OK. Fixed now!",
            matched_text="Make_Unique",
        ),
    ]

    self.assertEqual(locs, expected_locs)

  def test_check_suppressed_diff_with_path_regexes(self):
    file_diffs = check_contents.parse_diff(SUPPRESSED_DIFF)
    filtered_diffs = check_contents.filter_diffs_by_path(
        file_diffs,
        path_regexes=["src/important\\..*"],
        path_regex_exclusions=[],
    )

    self.assertLen(filtered_diffs, 1)

    locs = check_contents.check_diffs(
        filtered_diffs, prohibited_regex="Make_Unique"
    )

    self.assertEmpty(locs)

  def test_check_suppressed_diff_with_exclusions(self):
    file_diffs = check_contents.parse_diff(SUPPRESSED_DIFF)
    filtered_diffs = check_contents.filter_diffs_by_path(
        file_diffs,
        path_regexes=[],
        path_regex_exclusions=["src/dir/.*"],
    )

    self.assertLen(filtered_diffs, 1)

    locs = check_contents.check_diffs(
        filtered_diffs, prohibited_regex="Make_Unique"
    )

    self.assertEmpty(locs)

  def test_check_suppressed_diff_with_suppression(self):
    file_diffs = check_contents.parse_diff(SUPPRESSED_DIFF)

    filtered_diffs = check_contents.filter_diffs_by_path(
        file_diffs, path_regexes=[], path_regex_exclusions=[]
    )

    # filtering without path_regex(_exclusions) is a noop
    self.assertEqual(file_diffs, filtered_diffs)

    locs = check_contents.check_diffs(
        file_diffs, prohibited_regex="Make_Unique", suppression_regex="OK"
    )

    self.assertEmpty(locs)


if __name__ == "__main__":
  absltest.main()
