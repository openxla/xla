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
        "container-image": "rocm/tensorflow-build@sha256:7cd444ac48657fee2f5087fbda7766266704d3f8fb2299f681952ae4eabed060",
        "OSFamily": "Linux",
        "Pool": "linux_x64_large",
    },
)

platform(
    name = "linux_x64_gpu",
    constraint_values = [
        "@platforms//os:linux",
        "@platforms//cpu:x86_64",
        "@bazel_tools//tools/cpp:clang",
    ],
    exec_properties = {
        "container-image": "rocm/tensorflow-build@sha256:7cd444ac48657fee2f5087fbda7766266704d3f8fb2299f681952ae4eabed060",
        "OSFamily": "Linux",
        "Pool": "amd_gpu",
    },
)
