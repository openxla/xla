# Description:
#   Python API for XLA.
load("//xla/tests:build_defs.bzl", "generate_backend_suites")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

generate_backend_suites()

py_library(
    name = "types",
    srcs = ["types.py"],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        "//third_party/py/numpy",
        "//xla:xla_data_proto_py",
        "@tsl//tsl/python/lib/core:pywrap_bfloat16",
    ],
)

py_library(
    name = "xla_shape",
    srcs = ["xla_shape.py"],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        ":types",
        "//xla:xla_data_proto_py",
    ],
)

py_library(
    name = "xla_literal",
    srcs = ["xla_literal.py"],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        ":types",
        ":xla_shape",
        "//xla:xla_data_proto_py",
    ],
)

py_test(
    name = "xla_shape_test",
    srcs = ["xla_shape_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    tags = [
        "no_pip",
        "optonly",
    ],
    deps = [
        ":xla_shape",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:absltest",
    ],
)

py_test(
    name = "xla_literal_test",
    srcs = ["xla_literal_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    tags = [
        "no_pip",
        "optonly",
    ],
    deps = [
        ":xla_literal",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:absltest",
    ],
)
