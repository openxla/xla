# Copyright 2026 The OpenXLA Authors.
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

"""Build rule for normalizing an optional CuTeDSL runtime shared library."""

def _normalize_runtime_dso_impl(ctx):
    dso_available = ctx.attr.src.label != ctx.attr.unavailable.label
    if not ctx.attr.runtime_available:
        if ctx.attr.shared or dso_available:
            fail(
                "CuTeDSL runtime linkage is disabled, but shared-runtime " +
                "settings were provided",
            )
        return [DefaultInfo()]

    if ctx.attr.shared != dso_available:
        if ctx.attr.shared:
            fail("shared CuTeDSL runtime linkage requires cutedsl_runtime_dso")
        fail("cutedsl_runtime_dso requires cutedsl_runtime_is_shared=true")

    if not ctx.attr.shared:
        return [DefaultInfo()]

    files = ctx.attr.src[DefaultInfo].files.to_list()
    if len(files) != 1:
        fail(
            "{} must provide exactly one file; got {}".format(
                ctx.attr.src.label,
                len(files),
            ),
        )

    output = ctx.actions.declare_file("libcute_dsl_runtime.so")
    ctx.actions.symlink(output = output, target_file = files[0])
    return [DefaultInfo(
        files = depset([output]),
        runfiles = ctx.runfiles(files = [output]),
    )]

normalize_runtime_dso = rule(
    implementation = _normalize_runtime_dso_impl,
    attrs = {
        "runtime_available": attr.bool(mandatory = True),
        "shared": attr.bool(mandatory = True),
        "src": attr.label(allow_files = True, mandatory = True),
        "unavailable": attr.label(mandatory = True),
    },
)
