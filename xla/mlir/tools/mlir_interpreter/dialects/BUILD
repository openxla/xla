load("@tsl//tsl/platform:rules_cc.bzl", "cc_library")
load("//xla/tsl:tsl.bzl", "if_windows")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [":friends"],
    licenses = ["notice"],
)

package_group(
    name = "friends",
    includes = [
        "//xla:friends",
    ],
)

cc_library(
    name = "dialects",
    srcs = glob(
        [
            "*.cc",
        ],
        exclude = ["util.cc"],
    ),
    deps = [
        ":find_symbol",
        ":dialect_utils",
        "//xla/mlir/tools/mlir_interpreter/framework",
        "//xla/mlir_hlo",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AffineDialect",
        "@llvm-project//mlir:AffineUtils",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:BufferizationDialect",
        "@llvm-project//mlir:ComplexDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LinalgDialect",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:VectorDialect",
    ],
    alwayslink = 1,
)

cc_library(
    name = "dialect_utils",
    srcs = [
        "util.cc",
    ],
    hdrs = [
        "comparators.h",
        "cwise_math.h",
        "util.h",
    ],
    deps = [
        "//xla/mlir/tools/mlir_interpreter/framework",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:ViewLikeInterface",
    ],
)

cc_library(
    name = "symbol_finder_header",
    hdrs = ["symbol_finder.h"],
)

cc_library(
    name = "symbol_finder_linux",
    srcs = ["symbol_finder_linux.cc"],
    deps = [
        "@com_google_absl//absl/status:statusor",
        ":symbol_finder_header", 
    ],
    tags = ["manual"]
)

cc_library(
    name = "symbol_finder_windows",
    srcs = ["symbol_finder_windows.cc"],
    deps = [
        "@com_google_absl//absl/status:statusor",
        ":symbol_finder_header", 
    ],
    tags = ["manual"]
)

cc_library(
    name = "find_symbol",
    deps = if_windows([":symbol_finder_windows"], [":symbol_finder_linux"])
)
