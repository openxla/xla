# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Bazel workspace for DUCC (CPU FFTs)."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def repo():
    http_archive(
        name = "ducc",
        strip_prefix = "ducc-e33fb4bdca38be02b869d96d6329ad5a2c3a335c",
        sha256 = "21371a0a7d2895d7813da7f2eb0ed097014d4a327b4b7712aeb17cf92634a202",
        urls = [
            "https://github.com/mreineck/ducc/archive/e33fb4bdca38be02b869d96d6329ad5a2c3a335c.tar.gz",
        ],
        build_file = "@//third_party/ducc:BUILD",
        patches = [
            "//third_party/ducc:threading.patch",
        ]
    )