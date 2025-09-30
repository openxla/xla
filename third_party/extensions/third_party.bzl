"""Module extension for third party dependencies."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")
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

def _third_party_ext_impl(mctx):
    benchmark()
    dlpack()
    ducc()
    eigen3()
    farmhash()
    fmt()
    FP16()
    gemmlowp()
    gloo()
    highwayhash()
    hwloc()
    implib_so()
    llvm(name = "llvm-raw")
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
    # TODO: share the definitions with the WORKSPACE build
    tf_http_archive(
        name = "onednn",
        build_file = "//third_party/mkl_dnn:mkldnn_v1.BUILD",
        patch_file = ["//third_party/mkl_dnn:setting_init.patch"],
        sha256 = "071f289dc961b43a3b7c8cbe8a305290a7c5d308ec4b2f586397749abdc88296",
        strip_prefix = "oneDNN-3.7.3",
        urls = tf_mirror_urls("https://github.com/oneapi-src/oneDNN/archive/refs/tags/v3.7.3.tar.gz"),
    )
    tf_http_archive(
        name = "XNNPACK",
        sha256 = "d36a005c707c0cf26696acfb5ef27d55a37551a49ed2eeb5979815a61138f07d",
        strip_prefix = "XNNPACK-ea1906f8df2faf8172da1b341c563bf9115581dd",
        urls = tf_mirror_urls("https://github.com/google/XNNPACK/archive/ea1906f8df2faf8172da1b341c563bf9115581dd.zip"),
    )
    tf_http_archive(
        name = "pthreadpool",
        sha256 = "8b1d13195842c9b7e8ef5aa7d9b44ca4168a41b8ae97b4e50db4fcc562211f5b",
        strip_prefix = "pthreadpool-d561aae9dfeab38ff595a0ae3e6bbd90b862c5f8",
        urls = tf_mirror_urls("https://github.com/google/pthreadpool/archive/d561aae9dfeab38ff595a0ae3e6bbd90b862c5f8.zip"),
    )
    tf_http_archive(
        name = "FXdiv",
        sha256 = "3d7b0e9c4c658a84376a1086126be02f9b7f753caa95e009d9ac38d11da444db",
        strip_prefix = "FXdiv-63058eff77e11aa15bf531df5dd34395ec3017c8",
        urls = tf_mirror_urls("https://github.com/Maratyszcza/FXdiv/archive/63058eff77e11aa15bf531df5dd34395ec3017c8.zip"),
    )
    tf_http_archive(
        name = "cpuinfo",
        sha256 = "c0254ce97f7abc778dd2df0aaca1e0506dba1cd514fdb9fe88c07849393f8ef4",
        strip_prefix = "cpuinfo-8a9210069b5a37dd89ed118a783945502a30a4ae",
        urls = tf_mirror_urls("https://github.com/pytorch/cpuinfo/archive/8a9210069b5a37dd89ed118a783945502a30a4ae.zip"),
    )
    # Intel openMP that is part of LLVM sources.
    tf_http_archive(
        name = "llvm_openmp",
        build_file = "//third_party/llvm_openmp:BUILD.bazel",
        patch_file = ["//third_party/llvm_openmp:openmp_switch_default_patch.patch"],
        sha256 = "d19f728c8e04fb1e94566c8d76aef50ec926cd2f95ef3bf1e0a5de4909b28b44",
        strip_prefix = "openmp-10.0.1.src",
        urls = tf_mirror_urls("https://github.com/llvm/llvm-project/releases/download/llvmorg-10.0.1/openmp-10.0.1.src.tar.xz"),
    )
    tf_http_archive(
        name = "cudnn_frontend_archive",
        build_file = "//third_party:cudnn_frontend.BUILD",
        patch_file = ["//third_party:cudnn_frontend_header_fix.patch"],
        sha256 = "257b3b7f8a99abc096094abc9e5011659117b647d55293bcd2c5659f9181b99e",
        strip_prefix = "cudnn-frontend-1.13.0",
        urls = tf_mirror_urls("https://github.com/NVIDIA/cudnn-frontend/archive/refs/tags/v1.13.0.zip"),
    )

third_party_ext = module_extension(
    implementation = _third_party_ext_impl,
)
