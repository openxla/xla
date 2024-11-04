load("@local_config_rocm//rocm:build_defs.bzl", "if_rocm")
load(
    "@tsl//tsl/platform:build_config.bzl",
    "tf_additional_device_tracer_srcs",
)
load("@tsl//tsl/profiler/builds:build_config.bzl", "tf_profiler_copts")
load("//xla/tests:build_defs.bzl", "xla_test")
load(
    "//xla/tsl:tsl.bzl",
    "internal_visibility",
    "tsl_copts",
    "tsl_gpu_library",
)

cc_library(
    name = "common_headers",
    hdrs = [
        "call_stack.hpp",
        "defines.hpp",
        "filesystem.hpp",
        "name_info.hpp",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//xla/stream_executor/rocm:roctracer_wrapper"]
)