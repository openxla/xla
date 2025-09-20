"""Module extension to initialize RBE configs."""

load("//tools/toolchains/remote_config:configs.bzl", "initialize_rbe_configs")

def _rbe_config_ext_impl(mctx):
    initialize_rbe_configs()

rbe_config_ext = module_extension(
    implementation = _rbe_config_ext_impl,
)
