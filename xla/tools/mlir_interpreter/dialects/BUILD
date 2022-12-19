load("@tsl/platform:rules_cc.bzl", "cc_library")

package(
    default_visibility = ["//xla:internal"],
    licenses = ["notice"],
)

cc_library(
    name = "dialects",
    srcs = [
        "affine.cc",
        "arith.cc",
        "bufferization.cc",
        "builtin.cc",
        "gml_st.cc",
        "linalg.cc",
        "math.cc",
        "memref.cc",
        "mhlo.cc",
        "scf.cc",
        "tensor.cc",
    ],
    deps = [
        ":comparators",
        ":util",
        "//xla/mlir_hlo",
        "//xla/mlir_hlo:gml_st",
        "//xla/tools/mlir_interpreter:interpreter",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AffineDialect",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:BufferizationDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LinalgDialect",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:TensorDialect",
        "@tsl/platform:logging",
    ],
    alwayslink = 1,
)

cc_library(
    name = "util",
    srcs = ["util.cc"],
    hdrs = ["util.h"],
    deps = [
        "//xla/tools/mlir_interpreter:interpreter",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:ViewLikeInterface",
    ],
)

cc_library(
    name = "comparators",
    hdrs = ["comparators.h"],
    deps = ["@llvm-project//llvm:Support"],
)
