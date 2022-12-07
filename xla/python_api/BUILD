# Description:
#   Python API for XLA.

package(
    # copybara:uncomment default_applicable_licenses = ["//third_party/tensorflow:license"],
    licenses = ["notice"],
)

py_library(
    name = "types",
    srcs = ["types.py"],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        "//third_party/py/numpy",
        "//third_party/tensorflow/python:dtypes",
        "//xla:xla_data_proto_py",
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
