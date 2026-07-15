""" Macros to unpack a wheel and use its content as a py_library. """

load("@rules_python//python:defs.bzl", "py_library")

def _unpacked_wheel_impl(ctx):
    output_dir = ctx.actions.declare_directory(ctx.label.name)
    wheel = ctx.file.wheel
    args = ctx.actions.args()
    args.add("--wheel=%s" % wheel.path)
    args.add("--output_dir=%s" % output_dir.path)
    srcs = [wheel]

    # Collect runfiles from wheel_deps for propagation
    runfiles_depsets = []
    for d in ctx.attr.wheel_deps:
        # Collect default_runfiles and pass them to --wheel_files (for copying into wheel)
        if hasattr(d[DefaultInfo], 'default_runfiles'):
            default_files = d[DefaultInfo].default_runfiles.files
            for f in default_files.to_list():
                srcs.append(f)
                args.add("--wheel_files=%s" % (f.path))

        # Collect data_runfiles for propagation to tests (not copied into wheel)
        if hasattr(d[DefaultInfo], 'data_runfiles'):
            runfiles_depsets.append(d[DefaultInfo].data_runfiles)

    for z in ctx.files.zip_deps:
        srcs.append(z)
        args.add("--zip_files=%s" % (z.path))
    args.set_param_file_format("flag_per_line")
    args.use_param_file("@%s", use_always = False)
    ctx.actions.run(
        arguments = [args],
        inputs = srcs,
        outputs = [output_dir],
        executable = ctx.executable.unpack_wheel_and_unzip_archive_files,
        mnemonic = "UnpackWheelAndUnzipArchiveFiles",
    )

    return [
        DefaultInfo(
            files = depset([output_dir]),
            runfiles = ctx.runfiles(files = [output_dir]).merge_all(runfiles_depsets),
        ),
    ]

_unpacked_wheel = rule(
    implementation = _unpacked_wheel_impl,
    attrs = {
        "wheel": attr.label(mandatory = True, allow_single_file = True),
        "unpack_wheel_and_unzip_archive_files": attr.label(
            default = Label("//third_party/py:unpack_wheel_and_unzip_archive_files"),
            executable = True,
            cfg = "exec",
        ),
        "wheel_deps": attr.label_list(allow_files = True),
        "zip_deps": attr.label_list(allow_files = True),
    },
)

def py_import(
        name,
        wheel,
        deps = [],
        wheel_deps = [],
        zip_deps = [],
        testonly = False):
    unpacked_wheel_name = name + "_unpacked_wheel"
    _unpacked_wheel(
        name = unpacked_wheel_name,
        wheel = wheel,
        wheel_deps = wheel_deps,
        zip_deps = zip_deps,
        testonly = testonly,
    )
    py_library(
        name = name,
        data = [":" + unpacked_wheel_name],
        imports = [unpacked_wheel_name],
        deps = deps,
        visibility = ["//visibility:public"],
        testonly = testonly,
    )

"""Unpacks the wheel and uses its content as a py_library.
Args:
  wheel: wheel file to unpack.
  deps: dependencies of the py_library.
  wheel_deps: additional wheels to unpack. These wheels will be unpacked in the
              same folder as the wheel.
  zip_deps: additional zip files to unpack. These files will be extracted
              in the same folder as the wheel.
"""  # buildifier: disable=no-effect
