"""Loads the ml_dtypes library, for bfloat16, float8, int4 types."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    tf_http_archive(
        name = "ml_dtypes",
        strip_prefix = "ml_dtypes-e88f16b3ce844a364c39589b27a9ff964ef5c62b",
        sha256 = "",
        urls = tf_mirror_urls("https://github.com/jax-ml/ml_dtypes/archive/e88f16b3ce844a364c39589b27a9ff964ef5c62b.tar.gz"),
        build_file = "//third_party/py/ml_dtypes:ml_dtypes.BUILD",
    )
