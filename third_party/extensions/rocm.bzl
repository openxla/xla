"""Module extension for rocm."""

load("@xla//third_party/gpus:rocm_configure.bzl", "rocm_configure")

def _rocm_extension_impl(mctx):
    rocm_configure(name = "local_config_rocm")

rocm_extension = module_extension(
    implementation = _rocm_extension_impl,
)
