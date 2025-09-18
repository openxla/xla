"""Module extension for third party dependencies."""

load("@xla//third_party:repo.bzl", "tf_http_archive")
load("@xla//third_party/py/ml_dtypes:workspace.bzl", ml_dtypes = "repo")
load("@xla//third_party/highwayhash:workspace.bzl", highwayhash = "repo")
load("@xla//third_party/stablehlo:workspace.bzl", stablehlo = "repo")
load("@xla//third_party/farmhash:workspace.bzl", farmhash = "repo")
load("@xla//third_party/shardy:workspace.bzl", shardy = "repo")
load("@xla//third_party/ducc:workspace.bzl", ducc = "repo")
load("@xla//third_party/hwloc:workspace.bzl", hwloc = "repo")
load("@xla//third_party/eigen3:workspace.bzl", eigen3 = "repo")
load("@xla//third_party/llvm:workspace.bzl", llvm = "repo")

def _third_party_extension_impl(mctx):
    ml_dtypes()
    highwayhash()
    stablehlo()
    farmhash()
    shardy()
    ducc()
    hwloc()
    eigen3()
    llvm(name = "llvm-raw")
    tf_http_archive(
        name = "onednn",
        sha256 = "071f289dc961b43a3b7c8cbe8a305290a7c5d308ec4b2f586397749abdc88296",
        urls = ["https://storage.googleapis.com/mirror.tensorflow.org/github.com/oneapi-src/oneDNN/archive/refs/tags/v3.7.3.tar.gz", "https://github.com/oneapi-src/oneDNN/archive/refs/tags/v3.7.3.tar.gz"],
        strip_prefix = "oneDNN-3.7.3",
        patch_file = ["@xla//third_party/mkl_dnn:setting_init.patch"],
        build_file = "@xla//third_party/mkl_dnn:mkldnn_v1.BUILD",
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

third_party_extension = module_extension(
    implementation = _third_party_extension_impl,
)
