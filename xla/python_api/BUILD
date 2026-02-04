load("@rules_python//python:py_library.bzl", "py_library")
load("//xla:py_strict.bzl", "py_strict_test")

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
    srcs = ["types_.py"],
    strict_deps = True,
    visibility = ["//visibility:public"],
    deps = [
        "//third_party/py/numpy",
        "//xla:xla_data_proto_py",
        "@ml_dtypes_py//ml_dtypes",
    ],
)

py_library(
    name = "xla_shape",
    srcs = ["xla_shape.py"],
    strict_deps = True,
    visibility = ["//visibility:public"],
    deps = [
        ":types",
        "//third_party/py/numpy",
        "//xla:xla_data_proto_py",
    ],
)

py_library(
    name = "xla_literal",
    srcs = ["xla_literal.py"],
    strict_deps = True,
    visibility = ["//visibility:public"],
    deps = [
        ":types",
        ":xla_shape",
        "//third_party/py/numpy",
        "//xla:xla_data_proto_py",
    ],
)

py_strict_test(
    name = "xla_shape_test",
    srcs = ["xla_shape_test.py"],
    tags = [
        "optonly",
    ],
    deps = [
        ":xla_shape",
        "//third_party/py/numpy",
        "//xla:xla_data_proto_py",
        "@absl_py//absl/testing:absltest",
    ],
)

py_strict_test(
    name = "xla_literal_test",
    srcs = ["xla_literal_test.py"],
    tags = [
        "optonly",
    ],
    deps = [
        ":xla_literal",
        "//third_party/py/numpy",
        "//xla:xla_data_proto_py",
        "@absl_py//absl/testing:absltest",
    ],
)
