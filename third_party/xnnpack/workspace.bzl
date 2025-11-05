"""XNNPACK is a highly optimized library of floating-point neural network inference operators for ARM, WebAssembly, and x86 platforms."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    # LINT.IfChange
    tf_http_archive(
        name = "XNNPACK",
        sha256 = "be7f061e99dc42fee46605e6c513adf21e89ead9e6aa189593e1b4c4d6cde82e",
        strip_prefix = "XNNPACK-ddc8e1a7f08b70e416df3f5385f7678052c3deb4",
        urls = tf_mirror_urls("https://github.com/google/XNNPACK/archive/ddc8e1a7f08b70e416df3f5385f7678052c3deb4.zip"),
    )
    # LINT.ThenChange(//tensorflow/lite/tools/cmake/modules/xnnpack.cmake)
