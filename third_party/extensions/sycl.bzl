"""Module extension for sycl."""

load("@xla//third_party/gpus:sycl_configure.bzl", "sycl_configure")

def _sycl_extension_impl(mctx):
    sycl_configure(name = "local_config_sycl")

sycl_extension = module_extension(
    implementation = _sycl_extension_impl,
)
