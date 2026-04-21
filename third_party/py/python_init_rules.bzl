"""Hermetic Python initialization. Consult the WORKSPACE on how to use it."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def python_init_rules(extra_patches = []):
    """Defines (doesn't setup) the rules_python repository.

    Args:
      extra_patches: list of labels. Additional patches to apply after the default
        set of patches.
    """

    tf_http_archive(
        name = "rules_cc",
        patch_file = [
            "@xla//third_party/py:rules_cc_protobuf.patch",
        ],
        urls = tf_mirror_urls("https://github.com/bazelbuild/rules_cc/archive/refs/tags/0.1.0.tar.gz"),
        strip_prefix = "rules_cc-0.1.0",
        sha256 = "4b12149a041ddfb8306a8fd0e904e39d673552ce82e4296e96fac9cbf0780e59",
    )

    tf_http_archive(
        name = "com_google_protobuf",
        patch_file = [
            "@xla//third_party/protobuf:protobuf.patch",
            "@xla//third_party/protobuf:protobuf_arena.patch",
        ],
        repo_mapping = {
            "@abseil-cpp": "@com_google_absl",
            "@protobuf_pip_deps": "@pypi",
        },
        sha256 = "6e09bbc950ba60c3a7b30280210cd285af8d7d8ed5e0a6ed101c72aff22e8d88",
        strip_prefix = "protobuf-6.31.1",
        urls = tf_mirror_urls("https://github.com/protocolbuffers/protobuf/archive/refs/tags/v6.31.1.zip"),
    )

    tf_http_archive(
        name = "rules_python",
        patch_file = [
            "@xla//third_party/py:rules_python_scope.patch",
            "@xla//third_party/py:rules_python_freethreaded.patch",
        ],
        sha256 = "29c9420b5f8fa4e51fd475517186c07a5b0c2269163ce6537e63e5344faca01a",
        strip_prefix = "rules_python-2.0.0-rc3",
        urls = tf_mirror_urls("https://github.com/bazel-contrib/rules_python/releases/download/2.0.0-rc3/rules_python-2.0.0-rc3.tar.gz"),
    )
