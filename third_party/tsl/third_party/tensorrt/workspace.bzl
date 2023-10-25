"""Provides the repository macro to import TensorRT Open Source components."""

load("//third_party:repo.bzl", "tf_http_archive")

def repo(name = "tensorrt_oss_archive"):
    """Imports TensorRT Open Source Components."""
    TRT_OSS_COMMIT = "e4da0731366154b5b2874d835bc1dde95b00ecf9"
    TRT_OSS_SHA256 = "b43292619136588449528385c9ff9b399b81be2339f91e483595875006554150"

    tf_http_archive(
        name = name,
        sha256 = TRT_OSS_SHA256,
        strip_prefix = "TensorRT-{commit}".format(commit = TRT_OSS_COMMIT),
        urls = [
            # TODO: Google Mirror "https://storage.googleapis.com/...."
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/NVIDIA/TensorRT/archive/{commit}.tar.gz".format(commit = TRT_OSS_COMMIT),
            "https://github.com/NVIDIA/TensorRT/archive/{commit}.tar.gz".format(commit = TRT_OSS_COMMIT),
        ],
        build_file = "//third_party/tensorrt/plugin:BUILD",
        patch_file = ["//third_party/tensorrt/plugin:tensorrt_oss.patch"],
    )
