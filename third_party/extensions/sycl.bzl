"""Module extension for sycl."""

load("@xla//third_party/gpus:sycl_configure.bzl", "sycl_configure")

sycl_extension = module_extension(
    implementation = lambda mctx: sycl_configure(name = "local_config_sycl"),
)
