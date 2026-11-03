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
"""Library to compute the hash of a target.

Computes a deterministic hash of a binary for version sealing.
"""

# This file defines macros, not rules. See go/for-macros-only.

load("//xla:xla.default.bzl", "xla_cc_binary")
load("//xla/tools:source_hash_rules.bzl", _source_hash_rule = "source_hash_rule")

visibility("public")

def source_hash(name, target, out):
    """Generates a text file containing a hash of the executable target.

    Args:
      name: string, Name of the rule.
      target: label, cc_library to hash. Must be alwayslink.
      out: string, Name of the file to write the hash to.
    """
    cc_name = name + "_source_hash_internal.cc"
    native.genrule(
        name = "_%s_write_source_hash_internal" % cc_name,
        outs = [cc_name],
        cmd = "echo 'int main() { return 0; }' > \"$@\"",
    )

    binary_name = "_%s_source_hash_internal" % name
    xla_cc_binary(
        name = binary_name,
        srcs = [cc_name],
        deps = [target],
        stamp = 0,
    )

    _source_hash_rule(
        name = name,
        target = binary_name,
        original_target = target,
        out = out,
    )
