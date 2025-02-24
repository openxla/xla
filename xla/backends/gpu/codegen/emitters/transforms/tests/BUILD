load("//xla:lit.bzl", "lit_test_suite")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

lit_test_suite(
    name = "tests",
    srcs = glob(["*.mlir"]),
    cfg = "//xla:lit.cfg.py",
    tags_override = {
        "convert_float_cuda.mlir": ["cuda-only"],
        "convert_float_rocm.mlir": ["rocm-only"],
    },
    tools = [
        "//xla/codegen/tools:emitters_opt",
        "@llvm-project//llvm:FileCheck",
    ],
)
