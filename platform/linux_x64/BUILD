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
        "container-image": "docker://rocm/tensorflow-build:latest-jammy-python3.10-rocm6.4.0@sha256:9c2fbc861570735fc86dbf39dd3cefef6a13fcf6ae08358271af52f1aebd4248",
        "OSFamily": "Linux",
    },
)
