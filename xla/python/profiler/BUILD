load("@tsl//tsl/platform:rules_cc.bzl", "cc_library")
load("//xla/tsl:tsl.default.bzl", "tsl_pybind_extension")

# copybara:uncomment package(default_applicable_licenses = ["//tensorflow:license"])

cc_library(
    name = "xplane_visitor_lib",
    srcs = ["xplane_visitor.cc"],
    hdrs = ["xplane_visitor.h"],
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
    name = "xplane_visitor",
    srcs = ["py_xplane_visitor.cc"],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    pytype_srcs = ["xplane_visitor.pyi"],
    visibility = ["//visibility:public"],
    deps = [
        ":xplane_visitor_lib",
        "@nanobind",
    ],
)
