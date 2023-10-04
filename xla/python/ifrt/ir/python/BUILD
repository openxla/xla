load("//xla:strict.default.bzl", "py_strict_library", "py_strict_test")
load("@llvm-project//mlir:tblgen.bzl", "gentbl_filegroup", "td_library")
load("@tsl//tsl:tsl.default.bzl", "tsl_pybind_extension")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

tsl_pybind_extension(
    name = "ifrt_ir_support",
    srcs = ["ifrt_ir_support.cc"],
    deps = [
        "//xla/python:status_casters",
        "//xla/python/ifrt/ir",
        "//xla/python/ifrt/support:sharding_conversions",
        "@com_google_absl//absl/status",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:CAPIIRHeaders",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MLIRBindingsPythonHeaders",
        "@pybind11",
    ],
)

td_library(
    name = "ifrt_ir_ops_td_files",
    srcs = [
        "ifrt_ir.td",
    ],
    deps = [
        "//xla/python/ifrt/ir:ifrt_td",
    ],
)

gentbl_filegroup(
    name = "ifrt_ir_ops_gen",
    tbl_outs = [
        (
            [
                "-gen-python-op-bindings",
                "-bind-dialect=ifrt",
            ],
            "_ifrt_ir_ops_gen.py",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "ifrt_ir.td",
    deps = [
        ":ifrt_ir_ops_td_files",
        "//xla/mlir_hlo:hlo_ops_td_files",
    ],
)

py_strict_library(
    name = "ifrt_ir",
    srcs = [
        "_ods_common.py",
        "ifrt_ir.py",
        ":ifrt_ir_ops_gen",
    ],
    visibility = ["//xla/python/ifrt:friends"],
    deps = [
        ":ifrt_ir_support",
        "//third_party/py/mlir",
        "//third_party/py/mlir:core",
        "//third_party/py/mlir:ir",
    ],
)

py_strict_test(
    name = "ifrt_ir_test",
    srcs = ["ifrt_ir_test.py"],
    python_version = "PY3",
    deps = [
        ":ifrt_ir",
        "//third_party/py/jax",
        "//third_party/py/mlir:func_dialect",
        "//third_party/py/mlir:ir",
        "@absl_py//absl/testing:absltest",
    ],
)
