"""Provides the repo macro to import ml_dtypes.

ml_dtypes provides machine-learning-specific data-types like bfloat16,
float8 varieties, and int4.
"""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    ML_DTYPES_COMMIT = "91c79111973ee3a3a5eeb2a25629af655b0e1df5"
    ML_DTYPES_SHA256 = ""
    tf_http_archive(
        name = "ml_dtypes",
        build_file = "//third_party/py/ml_dtypes:ml_dtypes.BUILD",
        link_files = {
            "//third_party/py/ml_dtypes:ml_dtypes.tests.BUILD": "tests/BUILD.bazel",
            "//third_party/py/ml_dtypes:LICENSE": "LICENSE",
        },
        sha256 = ML_DTYPES_SHA256,
        strip_prefix = "ml_dtypes-{commit}/ml_dtypes".format(commit = ML_DTYPES_COMMIT),
        patch_file = ["//third_party/py/ml_dtypes:ml_dtypes.patch"],
        urls = tf_mirror_urls("https://github.com/jax-ml/ml_dtypes/archive/{commit}.tar.gz".format(commit = ML_DTYPES_COMMIT)),
    )
