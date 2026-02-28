"""oneAPI Deep Neural Network Library (oneDNN)"""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    tf_http_archive(
        name = "onednn",
        build_file = "//third_party/mkl_dnn:mkldnn_v1.BUILD",
        # Rename the gtest module in oneDNN's third_party to avoid
        # conflict with Google's gtest module
        patch_file = ["//third_party/mkl_dnn:setting_init.patch",
                      "//third_party/mkl_dnn:rename_gtest.patch"],
        sha256 = "04df98b18300daf6c3aa7cc2d5e7ce8a8f430fed1787151daed0254d8dd4e64e",
        strip_prefix = "oneDNN-3.11",
        urls = tf_mirror_urls("https://github.com/oneapi-src/oneDNN/archive/refs/tags/v3.11.tar.gz"),
    )
