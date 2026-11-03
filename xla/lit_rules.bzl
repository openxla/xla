# Copyright 2026 The OpenXLA Authors. All Rights Reserved.
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

"""Helper Starlark rules for LIT tests."""

load("@bazel_skylib//lib:paths.bzl", "paths")
load("@rules_cc//cc/common:cc_info.bzl", "CcInfo")

visibility("private")

def _tools_on_path_impl(ctx):
    runfiles = ctx.runfiles()

    # For Bazel 4.x support. Drop when Bazel 4.x is no longer supported
    to_merge = [d[DefaultInfo].default_runfiles for d in ctx.attr.srcs]
    if hasattr(runfiles, "merge_all"):
        runfiles = runfiles.merge_all(to_merge)
    else:
        for m in to_merge:
            runfiles = runfiles.merge(m)

    runfiles_symlinks = {}

    for src in ctx.attr.srcs:
        exe = src[DefaultInfo].files_to_run.executable
        if not exe:
            fail("All targets used as tools by lit tests must have exactly one" +
                 " executable, but {} has none".format(src))
        bin_path = paths.join(ctx.attr.bin_dir, exe.basename)
        if bin_path in runfiles_symlinks:
            fail("All tools used by lit tests must have unique basenames, as" +
                 " they are added to the path." +
                 " {} and {} conflict".format(runfiles_symlinks[bin_path], exe))
        runfiles_symlinks[bin_path] = exe

    # The loop below symlinks the libraries that are used by the tools.
    for dep in ctx.attr.deps:
        linker_inputs = dep[CcInfo].linking_context.linker_inputs.to_list()
        for linker_input in linker_inputs:
            if len(linker_input.libraries) == 0:
                continue
            lib = linker_input.libraries[0].dynamic_library
            if not lib:
                continue
            lib_path = paths.join(ctx.attr.lib_dir, lib.basename)
            if lib_path in runfiles_symlinks:
                if runfiles_symlinks[lib_path] == lib:
                    continue
                fail("All libs used by lit tests must have unique basenames, as" +
                     " they are added to the path." +
                     " {} and {} conflict".format(runfiles_symlinks[lib_path], lib))
            runfiles_symlinks[lib_path] = lib

    return [
        DefaultInfo(runfiles = ctx.runfiles(
            symlinks = runfiles_symlinks,
        ).merge(runfiles)),
    ]

tools_on_path = rule(
    _tools_on_path_impl,
    attrs = {
        "srcs": attr.label_list(allow_files = True, mandatory = True),
        "bin_dir": attr.string(mandatory = True),
        "lib_dir": attr.string(mandatory = True),
        "deps": attr.label_list(),
    },
    doc = "Symlinks srcs into a single lit_bin directory. All basenames must be unique.",
)
