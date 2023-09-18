# Description:
#   Stubs for dynamically loading CUDA.

load(
    "//tsl/platform/default:cuda_build_defs.bzl",
    "cuda_rpath_flags",
    "if_cuda_is_configured",
)
load(
    "//tsl/platform:build_config.bzl",
    "tsl_cc_test",
)
load(
    "//tsl/platform:rules_cc.bzl",
    "cc_library",
)
load("//tsl/cuda:stub.bzl", "cuda_stub")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

cuda_stub(
    name = "cublas",
    srcs = ["cublas.symbols"],
)

cc_library(
    name = "cublas_stub",
    srcs = if_cuda_is_configured(["cublas_stub.cc", "cublas.tramp.S"]),
    linkopts = if_cuda_is_configured(cuda_rpath_flags(
        "nvidia/cublas/lib",
    )),
    textual_hdrs = ["cublas.inc"],
    visibility = ["//visibility:public"],
    deps = if_cuda_is_configured([
        "@com_google_absl//absl/container:flat_hash_set",
        "@local_config_cuda//cuda:cuda_headers",
        "//tsl/platform:dso_loader",
        "//tsl/platform:env",
    ]),
    local_defines=[
        "IMPLIB_EXPORT_SHIMS=1",
    ],
)

alias(
    name = "cublas_lib",
    actual = select({
        "//tsl:oss": ":cublas_stub",
        "//conditions:default": "@local_config_cuda//cuda:cublas",
    }),
    visibility = ["//visibility:public"],
)

cuda_stub(
    name = "cublasLt",
    srcs = ["cublasLt.symbols"],
)

cc_library(
    name = "cublas_lt_stub",
    srcs = if_cuda_is_configured(["cublasLt_stub.cc", "cublasLt.tramp.S"]),
    textual_hdrs = ["cublasLt.inc"],
    visibility = ["//visibility:public"],
    deps = if_cuda_is_configured([
        "@local_config_cuda//cuda:cuda_headers",
        "//tsl/platform:dso_loader",
        "//tsl/platform:env",
    ]),
    local_defines=[
        "IMPLIB_EXPORT_SHIMS=1",
    ],
)

alias(
    name = "cublas_lt_lib",
    actual = select({
        "//tsl:oss": ":cublas_lt_stub",
        "//conditions:default": "@local_config_cuda//cuda:cublasLt",
    }),
    visibility = ["//visibility:public"],
)

cuda_stub(
    name = "cuda",
    srcs = ["cuda.symbols"],
)

cc_library(
    name = "cuda_stub",
    srcs = if_cuda_is_configured(["cuda_stub.cc", "cuda.tramp.S"]),
    textual_hdrs = ["cuda.inc"],
    visibility = ["//visibility:public"],
    deps = if_cuda_is_configured([
        "@local_config_cuda//cuda:cuda_headers",
        "//tsl/platform:dso_loader",
        "//tsl/platform:env",
    ]),
    local_defines=[
        "IMPLIB_EXPORT_SHIMS=1",
    ],
)

cuda_stub(
    name = "cudart",
    srcs = ["cudart.symbols"],
)

cc_library(
    name = "cudart_stub",
    srcs = select({
        # include dynamic loading implementation only when if_cuda_is_configured and build dynamically
        "//tsl:is_cuda_enabled_and_oss": [
            "cudart_stub.cc",
            "cudart.tramp.S",
        ],
        "//conditions:default": [],
    }),
    linkopts = select({
        "//tsl:is_cuda_enabled_and_oss": cuda_rpath_flags("nvidia/cuda_runtime/lib"),
        "//conditions:default": [],
    }),
    textual_hdrs = ["cudart.inc"],
    visibility = ["//visibility:public"],
    deps = select({
        "//tsl:is_cuda_enabled_and_oss": [
            ":cuda_stub",
            "//tsl/platform:dso_loader",
            "//tsl/platform:env",
            "@local_config_cuda//cuda:cuda_headers",
        ],
        "//conditions:default": [],
    }),
    local_defines=[
        "IMPLIB_EXPORT_SHIMS=1",
    ],
)

filegroup(
    name = "cudnn_wrappers",
    srcs = glob(["cudnn_*.inc"]),
    visibility = ["//visibility:public"],
)

cuda_stub(
    name = "cudnn",
    srcs = ["cudnn.symbols"],
)

cc_library(
    name = "cudnn_stub",
    srcs = if_cuda_is_configured(["cudnn_stub.cc", "cudnn.tramp.S"]),
    linkopts = if_cuda_is_configured(cuda_rpath_flags("nvidia/cudnn/lib")),
    textual_hdrs = ["cudnn.inc"],
    visibility = ["//visibility:public"],
    deps = if_cuda_is_configured([
        ":cudnn_version",
        "@local_config_cuda//cuda:cudnn_header",
        "//tsl/platform:dso_loader",
        "//tsl/platform:env",
    ]),
    local_defines=[
        "IMPLIB_EXPORT_SHIMS=1",
    ],
)

cc_library(
    name = "nccl_rpath_stub",
    linkopts = if_cuda_is_configured(cuda_rpath_flags("nvidia/nccl/lib")),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "tensorrt_rpath_stub",
    linkopts = if_cuda_is_configured(cuda_rpath_flags("tensorrt")),
    visibility = ["//visibility:public"],
)

alias(
    name = "cudnn_lib",
    actual = select({
        "//tsl:oss": ":cudnn_stub",
        "//conditions:default": "@local_config_cuda//cuda:cudnn",
    }),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "cudnn_version",
    srcs = ["cudnn_version.cc"],
    hdrs = ["cudnn_version.h"],
    visibility = ["//visibility:public"],
    deps = [
        "@com_google_absl//absl/strings",
    ],
)

tsl_cc_test(
    name = "cudnn_version_test",
    srcs = ["cudnn_version_test.cc"],
    deps = [
        ":cudnn_version",
        "//tsl/platform:test",
        "//tsl/platform:test_main",
    ],
)

cuda_stub(
    name = "cufft",
    srcs = ["cufft.symbols"],
)

cc_library(
    name = "cufft_stub",
    srcs = if_cuda_is_configured(["cufft_stub.cc", "cufft.tramp.S"]),
    linkopts = if_cuda_is_configured(cuda_rpath_flags("nvidia/cufft/lib")),
    textual_hdrs = ["cufft.inc"],
    visibility = ["//visibility:public"],
    deps = if_cuda_is_configured([
        "@local_config_cuda//cuda:cuda_headers",
        "//tsl/platform:dso_loader",
        "//tsl/platform:env",
    ]),
    local_defines=[
        "IMPLIB_EXPORT_SHIMS=1",
    ],
)

alias(
    name = "cufft_lib",
    actual = select({
        "//tsl:oss": ":cufft_stub",
        "//conditions:default": "@local_config_cuda//cuda:cufft",
    }),
    visibility = ["//visibility:public"],
)


cuda_stub(
    name = "cupti",
    srcs = ["cupti.symbols"],
)

cc_library(
    name = "cupti_stub",
    srcs = if_cuda_is_configured(["cupti_stub.cc", "cupti.tramp.S"]),
    data = if_cuda_is_configured(["@local_config_cuda//cuda:cupti_dsos"]),
    linkopts = if_cuda_is_configured(cuda_rpath_flags("nvidia/cuda_cupti/lib")),
    textual_hdrs = ["cupti.inc"],
    visibility = ["//visibility:public"],
    deps = if_cuda_is_configured([
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cupti_headers",
        "//tsl/platform:dso_loader",
        "//tsl/platform:env",
    ]),
    local_defines=[
        "IMPLIB_EXPORT_SHIMS=1",
    ],
)

cuda_stub(
    name = "cusolver",
    srcs = ["cusolver.symbols"],
)

cc_library(
    name = "cusolver_stub",
    srcs = if_cuda_is_configured(["cusolver_stub.cc", "cusolver.tramp.S"]),
    linkopts = if_cuda_is_configured(cuda_rpath_flags("nvidia/cusolver/lib")),
    textual_hdrs = ["cusolver.inc"],
    visibility = ["//visibility:public"],
    deps = if_cuda_is_configured([
        "@local_config_cuda//cuda:cuda_headers",
        "//tsl/platform:dso_loader",
        "//tsl/platform:env",
    ]),
    local_defines=[
        "IMPLIB_EXPORT_SHIMS=1",
    ],
)

alias(
    name = "cusolver_lib",
    actual = select({
        "//tsl:oss": ":cusolver_stub",
        "//conditions:default": "@local_config_cuda//cuda:cusolver",
    }),
    visibility = ["//visibility:public"],
)

cuda_stub(
    name = "cusparse",
    srcs = ["cusparse.symbols"],
)

cc_library(
    name = "cusparse_stub",
    srcs = if_cuda_is_configured(["cusparse_stub.cc", "cusparse.tramp.S"]),
    linkopts = if_cuda_is_configured(cuda_rpath_flags("nvidia/cusparse/lib")),
    textual_hdrs = ["cusparse.inc"],
    visibility = ["//visibility:public"],
    deps = if_cuda_is_configured([
        "@local_config_cuda//cuda:cuda_headers",
        "//tsl/platform:dso_loader",
        "//tsl/platform:env",
    ]),
    local_defines=[
        "IMPLIB_EXPORT_SHIMS=1",
    ],
)

alias(
    name = "cusparse_lib",
    actual = select({
        "//tsl:oss": ":cusparse_stub",
        "//conditions:default": "@local_config_cuda//cuda:cusparse",
    }),
    visibility = ["//visibility:public"],
)
