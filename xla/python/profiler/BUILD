load("@tsl//tsl/platform:rules_cc.bzl", "cc_library")
load("//xla:strict.default.bzl", "py_strict_test")
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
        "@tsl//tsl/platform:protobuf",
    ],
)

py_strict_test(
    name = "profile_data_test",
    srcs = ["profile_data_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    tags = [
        "no_oss",  # TODO(phawkins): This test passes, but requires --config=monolithic.
        "optonly",
    ],
    deps = [
        ":profile_data",
        # copybara:uncomment "//pyglib:gfile",
        "@absl_py//absl/testing:absltest",
    ],
)
