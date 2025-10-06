load("@rules_cc//cc:defs.bzl", "cc_toolchain", "cc_toolchain_suite")
#load(":cc_toolchain_config.bzl", "cc_toolchain_config")

package(default_visibility = ["//visibility:public"])

platform(
    name = "linux_x64",
    constraint_values = [
        "@platforms//os:linux",
        "@platforms//cpu:x86_64",
        "@bazel_tools//tools/cpp:clang",
    ],
    exec_properties = {
        "container-image": "docker://rocm/tensorflow-build:2.19-jammy-python3.9-rocm6.4.0@sha256:7b42dafbe5401fd84cf951a63f16dd44a84c64a6ba2eeb7675fd0919f03b2bdf",
        "OSFamily": "Linux",
    },
)
