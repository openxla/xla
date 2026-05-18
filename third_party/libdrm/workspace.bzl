"""Loads libdrm headers for ROCm compatibility."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    """Import libdrm headers."""
    # libdrm 2.4.120 - a recent stable version
    tf_http_archive(
        name = "libdrm",
        build_file = "//third_party/libdrm:libdrm.BUILD",
        sha256 = "3bca436867da471c8d2a0afe0c6e1bf6b47f4b4b9e7f6288d89b7b80b16c302f",
        strip_prefix = "libdrm-libdrm-2.4.120",
        urls = tf_mirror_urls(
            "https://gitlab.freedesktop.org/mesa/drm/-/archive/libdrm-2.4.120/drm-libdrm-2.4.120.tar.gz",
        ),
    )
