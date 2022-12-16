load("@tsl//tsl:tsl.default.bzl", "get_compatible_with_cloud")
load("@tsl//tsl/profiler/builds:build_config.bzl", "tf_profiler_copts")
load("@tsl//tsl/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//third_party/tensorflow:license"],
    default_visibility = ["//xla/python/profiler:__subpackages__"],
    licenses = ["notice"],
)

cc_library(
    name = "python_hooks",
    srcs = ["python_hooks.cc"],
    hdrs = ["python_hooks.h"],
    compatible_with = get_compatible_with_cloud(),
    copts = tf_profiler_copts() + ["-fexceptions"],
    features = ["-use_header_modules"],  # Incompatible with -fexceptions.
    visibility = [
        "//third_party/tensorflow/core/profiler:__subpackages__",
        "//third_party/tensorflow/python/profiler/internal:__subpackages__",
        "//xla/backends/profiler:__subpackages__",
    ],
    deps = [
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@pybind11",
        "@tsl//tsl/platform:env",
        "@tsl//tsl/platform:macros",
        "@tsl//tsl/platform:path",
        "@tsl//tsl/platform:types",
        "@tsl//tsl/profiler/protobuf:xplane_proto_cc",
        "@tsl//tsl/profiler/utils:time_utils",
        "@tsl//tsl/profiler/utils:xplane_builder",
        "@tsl//tsl/profiler/utils:xplane_schema",
        "@tsl//tsl/profiler/utils:xplane_utils",
    ],
    alwayslink = True,
)

cc_library(
    name = "traceme_wrapper",
    hdrs = ["traceme_wrapper.h"],
    copts = tf_profiler_copts(),
    visibility = [
        "//third_party/tensorflow/python/profiler/internal:__pkg__",
        "//xla/python:__pkg__",
    ],
    deps = [
        "@com_google_absl//absl/strings",
        "@pybind11",
        "@tsl//tsl/platform:macros",
        "@tsl//tsl/platform:types",
        "@tsl//tsl/profiler/lib:traceme_for_pybind",
    ],
)
