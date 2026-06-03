"""Provides the repository macro to import Triton."""

load("//third_party/triton:common/series.bzl", "common_patch_list")
load("//third_party/triton:intel_xpu/workspace.bzl", "triton_archive")
load("//third_party/triton:oss_only/series.bzl", "oss_only_patch_list")

def repo():
    """Imports Triton."""

    TRITON_COMMIT = "4768da5e8228dfbda8e0b7a61101f87d953341bd"
    TRITON_SHA256 = "ba9a4a2643c16ae75da9a6eaa3658cb8d4284c7f590db1343aef28951c35c777"
    triton_archive(
        name = "triton",
        sha256 = TRITON_SHA256,
        triton_commit = TRITON_COMMIT,
        patch_file = common_patch_list + oss_only_patch_list,
    )
