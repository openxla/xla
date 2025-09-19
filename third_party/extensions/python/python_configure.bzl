load("//third_party/py:python_configure.bzl", "python_configure")

def _python_configure_ext_impl(mctx):
    python_configure(name = "local_config_python")

python_configure_ext = module_extension(
    implementation = _python_configure_ext_impl,
)
