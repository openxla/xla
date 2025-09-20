"""Module extension for tensorrt."""

load("@xla//third_party/tensorrt:tensorrt_configure.bzl", "tensorrt_configure")

def _tensorrt_extension_impl(mctx):
    tensorrt_configure(name = "local_config_tensorrt")

tensorrt_extension = module_extension(
    implementation = _tensorrt_extension_impl,
)
