"""NVSHMEM - NVIDIA Shared Memory"""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    tf_http_archive(
        name = "nvshmem",
        strip_prefix = "nvshmem_src_3.0.6-4",
        sha256 = "4f435fdee320a365dd19d24b9f74df69b69886d3902ec99b16b553d485b18871",
        urls = tf_mirror_urls("https://developer.download.nvidia.com/compute/redist/nvshmem/3.0.6/source/nvshmem_src_3.0.6-4.txz"),
        build_file = "//third_party/nvshmem:nvshmem.BUILD",
    )
