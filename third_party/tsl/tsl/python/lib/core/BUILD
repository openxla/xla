# Description:
# Implementation of custom numpy floats.

package(
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

filegroup(
    name = "numpy_hdr",
    srcs = ["numpy.h"],
)

filegroup(
    name = "basic_hdrs",
    srcs = [
        "numpy.h",
    ],
)

cc_library(
    name = "bfloat16_hdr",
    hdrs = [
        "bfloat16.h",
    ],
    deps = [
        "@local_config_python//:python_headers",  # build_cleaner: keep; DNR: b/35864863
    ],
)

cc_library(
    name = "bfloat16_lib",
    srcs = [
        "bfloat16.cc",
        "float8_e4m3b11.cc",
    ],
    hdrs = [
        "bfloat16.h",
        "float8_e4m3b11.h",
    ],
    deps = [
        ":custom_float",
        "//tsl/platform:logging",
        "//tsl/platform:types",
        "//tsl/python/lib/core:numpy",
        "@com_google_absl//absl/strings",
        "@local_config_python//:python_headers",  # build_cleaner: keep; DNR: b/35864863
        "@xla//third_party/eigen3",
    ],
)

cc_library(
    name = "custom_float",
    hdrs = [
        "custom_float.h",
    ],
    deps = [
        "//tsl/platform:logging",
        "//tsl/python/lib/core:numpy",
        "@com_google_absl//absl/strings",
        "@local_config_python//:python_headers",  # build_cleaner: keep; DNR: b/35864863
        "@xla//third_party/eigen3",
    ],
)

cc_library(
    name = "float8_hdr",
    hdrs = [
        "float8.h",
    ],
    deps = [
        "@local_config_python//:python_headers",  # build_cleaner: keep; DNR: b/35864863
    ],
)

cc_library(
    name = "float8_lib",
    srcs = [
        "float8.cc",
    ],
    hdrs = [
        "float8.h",
    ],
    deps = [
        ":custom_float",
        "//tsl/platform:logging",
        "//tsl/platform:types",
        "//tsl/python/lib/core:numpy",
        "@com_google_absl//absl/strings",
        "@local_config_python//:python_headers",  # build_cleaner: keep; DNR: b/35864863
        "@xla//third_party/eigen3",
    ],
)

cc_library(
    name = "custom_casts_lib",
    srcs = ["custom_casts.cc"],
    hdrs = ["custom_casts.h"],
    deps = [
        ":bfloat16_lib",
        ":float8_lib",
        "//tsl/platform:types",
        "//tsl/python/lib/core:numpy",
        "@local_config_python//:python_headers",  # build_cleaner: keep; DNR: b/35864863
    ],
)

cc_library(
    name = "custom_casts_hdr",
    hdrs = [
        "custom_casts.h",
    ],
)

cc_library(
    name = "numpy",
    srcs = ["numpy.cc"],
    hdrs = ["numpy.h"],
    deps = [
        "@local_config_python//:numpy_headers",
        "@local_config_python//:python_headers",
    ],
)

# Directory-level target.
cc_library(
    name = "core",
    deps = [
        ":bfloat16_lib",
        ":custom_casts_lib",
        ":float8_lib",
        ":numpy",
    ],
)
