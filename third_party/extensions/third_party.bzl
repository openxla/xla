"""Module extension for third party dependencies."""

load("//third_party:repo.bzl", "tf_http_archive")
load("//third_party/benchmark:workspace.bzl", benchmark = "repo")
load("//third_party/dlpack:workspace.bzl", dlpack = "repo")
load("//third_party/ducc:workspace.bzl", ducc = "repo")
load("//third_party/eigen3:workspace.bzl", eigen3 = "repo")
load("//third_party/farmhash:workspace.bzl", farmhash = "repo")
load("//third_party/fmt:workspace.bzl", fmt = "repo")
load("//third_party/FP16:workspace.bzl", FP16 = "repo")
load("//third_party/gemmlowp:workspace.bzl", gemmlowp = "repo")
load("//third_party/gloo:workspace.bzl", gloo = "repo")
load("//third_party/highwayhash:workspace.bzl", highwayhash = "repo")
load("//third_party/hwloc:workspace.bzl", hwloc = "repo")
load("//third_party/implib_so:workspace.bzl", implib_so = "repo")
load("//third_party/llvm:workspace.bzl", llvm = "repo")
load("//third_party/mpitrampoline:workspace.bzl", mpitrampoline = "repo")
load("//third_party/nanobind:workspace.bzl", nanobind = "repo")
load("//third_party/nasm:workspace.bzl", nasm = "repo")
load("//third_party/nvshmem:workspace.bzl", nvshmem = "repo")
load("//third_party/py/ml_dtypes:workspace.bzl", ml_dtypes = "repo")
load("//third_party/raft:workspace.bzl", raft = "repo")
load("//third_party/rapids_logger:workspace.bzl", rapids_logger = "repo")
load("//third_party/rmm:workspace.bzl", rmm = "repo")
load("//third_party/robin_map:workspace.bzl", robin_map = "repo")
load("//third_party/shardy:workspace.bzl", shardy = "repo")
load("//third_party/spdlog:workspace.bzl", spdlog = "repo")
load("//third_party/stablehlo:workspace.bzl", stablehlo = "repo")
load("//third_party/tensorrt:workspace.bzl", tensorrt = "repo")
load("//third_party/triton:workspace.bzl", triton = "repo")
load("//third_party/uv:workspace.bzl", uv = "repo")

def _third_party_extension_impl(mctx):
    FP16()
    benchmark()
    dlpack()
    ducc()
    eigen3()
    farmhash()
    fmt()
    gemmlowp()
    gloo()
    highwayhash()
    hwloc()
    implib_so()
    ml_dtypes()
    mpitrampoline()
    nanobind()
    nasm()
    nvshmem()
    raft()
    rapids_logger()
    rmm()
    robin_map()
    shardy()
    spdlog()
    stablehlo()
    tensorrt()
    triton()
    uv()
    llvm(name = "llvm-raw")
    tf_http_archive(
        name = "onednn",
        sha256 = "071f289dc961b43a3b7c8cbe8a305290a7c5d308ec4b2f586397749abdc88296",
        urls = ["https://storage.googleapis.com/mirror.tensorflow.org/github.com/oneapi-src/oneDNN/archive/refs/tags/v3.7.3.tar.gz", "https://github.com/oneapi-src/oneDNN/archive/refs/tags/v3.7.3.tar.gz"],
        strip_prefix = "oneDNN-3.7.3",
        patch_file = ["//third_party/mkl_dnn:setting_init.patch"],
        build_file = "//third_party/mkl_dnn:mkldnn_v1.BUILD",
    )
    tf_http_archive(
        name = "XNNPACK",
        sha256 = "d6871c9e499924d0efe6c759b976615f8704804d4fda782626db130abe0bc599",
        urls = ["https://storage.googleapis.com/mirror.tensorflow.org/github.com/google/XNNPACK/archive/dd9be413f1a49957f0c7617caf315b64566c3ed2.zip", "https://github.com/google/XNNPACK/archive/dd9be413f1a49957f0c7617caf315b64566c3ed2.zip"],
        strip_prefix = "XNNPACK-dd9be413f1a49957f0c7617caf315b64566c3ed2",
    )
    tf_http_archive(
        name = "pthreadpool",
        sha256 = "516ba8d05c30e016d7fd7af6a7fc74308273883f857faf92bc9bb630ab6dba2c",
        urls = ["https://storage.googleapis.com/mirror.tensorflow.org/github.com/google/pthreadpool/archive/c2ba5c50bb58d1397b693740cf75fad836a0d1bf.zip", "https://github.com/google/pthreadpool/archive/c2ba5c50bb58d1397b693740cf75fad836a0d1bf.zip"],
        strip_prefix = "pthreadpool-c2ba5c50bb58d1397b693740cf75fad836a0d1bf",
    )
    tf_http_archive(
        name = "FXdiv",
        sha256 = "3d7b0e9c4c658a84376a1086126be02f9b7f753caa95e009d9ac38d11da444db",
        urls = ["https://storage.googleapis.com/mirror.tensorflow.org/github.com/Maratyszcza/FXdiv/archive/63058eff77e11aa15bf531df5dd34395ec3017c8.zip", "https://github.com/Maratyszcza/FXdiv/archive/63058eff77e11aa15bf531df5dd34395ec3017c8.zip"],
        strip_prefix = "FXdiv-63058eff77e11aa15bf531df5dd34395ec3017c8",
    )
    tf_http_archive(
        name = "cpuinfo",
        sha256 = "a074e612706113048f1bb2937e7af3c5b57a037ce048d3cfaaca2931575819d2",
        urls = ["https://storage.googleapis.com/mirror.tensorflow.org/github.com/pytorch/cpuinfo/archive/e4cadd02a8b386c38b84f0a19eddacec3f433baa.zip", "https://github.com/pytorch/cpuinfo/archive/e4cadd02a8b386c38b84f0a19eddacec3f433baa.zip"],
        strip_prefix = "cpuinfo-e4cadd02a8b386c38b84f0a19eddacec3f433baa",
    )
    # Intel openMP that is part of LLVM sources.
    tf_http_archive(
        name = "llvm_openmp",
        build_file = "//third_party/llvm_openmp:BUILD.bazel",
        patch_file = ["//third_party/llvm_openmp:openmp_switch_default_patch.patch"],
        sha256 = "d19f728c8e04fb1e94566c8d76aef50ec926cd2f95ef3bf1e0a5de4909b28b44",
        strip_prefix = "openmp-10.0.1.src",
        urls = ["https://storage.googleapis.com/mirror.tensorflow.org/github.com/llvm/llvm-project/releases/download/llvmorg-10.0.1/openmp-10.0.1.src.tar.xz", "https://github.com/llvm/llvm-project/releases/download/llvmorg-10.0.1/openmp-10.0.1.src.tar.xz"],
    )
    tf_http_archive(
        name = "cudnn_frontend_archive",
        sha256 = "257b3b7f8a99abc096094abc9e5011659117b647d55293bcd2c5659f9181b99e",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/NVIDIA/cudnn-frontend/archive/refs/tags/v1.13.0.zip",
            "https://github.com/NVIDIA/cudnn-frontend/archive/refs/tags/v1.13.0.zip",
        ],
        strip_prefix = "cudnn-frontend-1.13.0",
        patch_file = ["//third_party:cudnn_frontend_header_fix.patch"],
        build_file = "//third_party:cudnn_frontend.BUILD",
    )
    tf_http_archive(
        name = "absl_py",
        sha256 = "8a3d0830e4eb4f66c4fa907c06edf6ce1c719ced811a12e26d9d3162f8471758",
        urls = ["https://storage.googleapis.com/mirror.tensorflow.org/github.com/abseil/abseil-py/archive/refs/tags/v2.1.0.tar.gz", "https://github.com/abseil/abseil-py/archive/refs/tags/v2.1.0.tar.gz"],
        strip_prefix = "abseil-py-2.1.0",
    )

third_party_extension = module_extension(
    implementation = _third_party_extension_impl,
)
