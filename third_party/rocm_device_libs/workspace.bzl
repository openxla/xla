"""Provides the repository macro to import Rocm-Device-Libs"""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    """Imports Rocm-Device-Libs."""
    LLVM_COMMIT = "0cf1859d038376421b4cd597e3df90d37cfca06e"
    LLVM_SHA256 = "0374d1efa0f049d2d1c24c4d86029b006cb5594cc0a1b6a18c49fb094c29cd29"

    tf_http_archive(
        name = "rocm_device_libs",
        sha256 = LLVM_SHA256,
        strip_prefix = "llvm-project-{commit}/amd/device-libs".format(commit = LLVM_COMMIT),
        urls = tf_mirror_urls("https://github.com/ROCm/llvm-project/archive/{commit}.tar.gz".format(commit = LLVM_COMMIT)),
        build_file = "//third_party/rocm_device_libs:rocm_device_libs.BUILD",
        patch_file = [
            "//third_party/rocm_device_libs:prepare_builtins.patch",
        ],
        link_files = {
            "//third_party/rocm_device_libs:build_defs.bzl": "build_defs.bzl",
        },
    )
