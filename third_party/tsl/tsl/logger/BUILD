load("//third_party/tensorflow/compiler/xla:pytype.default.bzl", "pytype_strict_library")

package(
    default_visibility = ["//visibility:public"],
)

pytype_strict_library(name = "error_log_noop")

pytype_strict_library(
    name = "error_log_client",
    srcs = ["error_log_client.py"],
)

pytype_strict_library(
    name = "get_error_log_client",
    srcs = ["get_error_log_client.py"],
    deps = [
        ":error_log_client",
        "//third_party/tensorflow/google/tsl/logger:error_log_client_google",
    ],
)
