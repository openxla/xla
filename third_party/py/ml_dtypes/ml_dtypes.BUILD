load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")

package(
    default_visibility = [":__subpackages__"],
    licenses = ["notice"],
)

exports_files(["LICENSE"])

cc_library(
    name = "float8",
    hdrs = ["include/float8.h"],
    visibility = ["//visibility:public"],
    deps = ["@eigen3"],
)

pybind_extension(
    name = "_custom_floats",
    srcs = [
        "_src/common.h",
        "_src/custom_float.h",
        "_src/dtypes.cc",
        "_src/int4.h",
        "_src/numpy.cc",
        "_src/numpy.h",
        "_src/ufuncs.h",
    ],
    deps = [
        ":float8",
        "@eigen3",
    ],
)

py_library(
    name = "ml_dtypes",
    srcs = [
        "__init__.py",
        "_finfo.py",
        "_iinfo.py",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":_custom_floats",
    ],
)
