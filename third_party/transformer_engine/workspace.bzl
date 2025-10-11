"""Loads the TransformerEngine library."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    tf_http_archive(
        name = "transformer_engine",
        strip_prefix = "TransformerEngine-2.8",
        sha256 = "00cf069f5fba8228805360c57dbd8fda768e2a76d4a545f3be19a30c4d25bfc2",
        urls = tf_mirror_urls("https://github.com/NVIDIA/TransformerEngine/archive/refs/tags/v2.8.tar.gz"),
        build_file = "//third_party/transformer_engine:transformer_engine.BUILD",
        patch_file = ["//third_party/transformer_engine:cuda_path.patch"],
    )
