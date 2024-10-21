load("@tsl//tsl/platform:rules_cc.bzl", "cc_library")
load("//xla/tsl:tsl.default.bzl", "tsl_pybind_extension")

# copybara:uncomment package(default_applicable_licenses = ["//tensorflow:license"])

cc_library(
    name = "profile_data_lib",
    srcs = ["profile_data.cc"],
    hdrs = ["profile_data.h"],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    features = ["-use_header_modules"],
    visibility = [
        "//perftools/accelerators/xprof/api/python:__pkg__",
        "//xla/python:__pkg__",
    ],
    deps = [
        "@nanobind",
        "@tsl//tsl/platform:env",
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:protobuf",
        "@tsl//tsl/platform:status",
        "@tsl//tsl/profiler/protobuf:xplane_proto_cc",
    ],
    alwayslink = 1,
)

tsl_pybind_extension(
    name = "profile_data",
    srcs = ["py_profile_data.cc"],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    pytype_srcs = ["profile_data.pyi"],
    visibility = ["//visibility:public"],
    deps = [
        ":profile_data_lib",
        "@nanobind",
    ],
)
