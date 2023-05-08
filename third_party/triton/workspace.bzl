"""Provides the repository macro to import Triton."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    """Imports Triton."""

    tf_http_archive(
        name = "dlfcn-win32",
        build_file = "@xla//third_party/triton:BUILD.dlfcn-win32",
        sha256 = "4f611c4372eef7f0179a33f76f84d54857c4fe676b60b654c6c5d91a6d4dad55",
        strip_prefix = "dlfcn-win32-1.3.1/src",
        urls = tf_mirror_urls("https://github.com/dlfcn-win32/dlfcn-win32/archive/refs/tags/v1.3.1.zip"),
    )

    TRITON_COMMIT = "1627e0c27869b4098e5fa720717645c1baaf5972"
    TRITON_SHA256 = "574436dab7c65f185834bd80c1d92167bacb7471b0c25906db60686835c46e21"

    tf_http_archive(
        name = "triton",
        sha256 = TRITON_SHA256,
        strip_prefix = "triton-{commit}".format(commit = TRITON_COMMIT),
        urls = tf_mirror_urls("https://github.com/openxla/triton/archive/{commit}.tar.gz".format(commit = TRITON_COMMIT)),
        # For temporary changes which haven't landed upstream yet.
        patch_file = [
            "//third_party/triton:cl526173620.patch",
            "//third_party/triton:cl528701873.patch",
        ],
    )
