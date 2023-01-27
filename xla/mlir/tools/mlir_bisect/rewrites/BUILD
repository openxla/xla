load("@tsl//tsl/platform:rules_cc.bzl", "cc_library")

cc_library(
    name = "rewrites",
    srcs = [
        "func.cc",
        "general.cc",
        "gml_st.cc",
        "scf.cc",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//xla/mlir/tools/mlir_bisect:bisect_lib",
        "//xla/mlir/tools/mlir_replay/public:execution_trace_utils",
        "//xla/mlir_hlo:gml_st",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:Support",
    ],
    alwayslink = 1,
)
