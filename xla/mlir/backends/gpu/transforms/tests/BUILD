load("@tsl//tsl:tsl.default.bzl", "filegroup")
load("//xla:glob_lit_test.bzl", "glob_lit_tests")

package(
    # copybara:uncomment default_applicable_licenses = ["//third_party/tensorflow:license"],
    licenses = ["notice"],
)

glob_lit_tests(
    data = [":test_utilities"],
    driver = "//xla:run_lit.sh",
    test_file_exts = ["mlir"],
)

# Bundle together all of the test utilities that are used by tests.
filegroup(
    name = "test_utilities",
    testonly = True,
    data = [
        "//xla/mlir/backends/gpu:xla-gpu-opt",
        "@llvm-project//llvm:FileCheck",
        "@llvm-project//mlir:run_lit.sh",
    ],
)
