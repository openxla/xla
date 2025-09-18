"""Module extension for remote_execution."""

load("@xla//tools/toolchains/remote:configure.bzl", "remote_execution_configure")

def _remote_execution_extension_impl(mctx):
    remote_execution_configure(name = "local_config_remote_execution")

remote_execution_extension = module_extension(
    implementation = _remote_execution_extension_impl,
)
