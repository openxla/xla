load("//third_party/py:python_init_pip.bzl", "python_init_pip")

def _python_init_pip_ext_impl(mctx):
    python_init_pip()

python_init_pip_ext = module_extension(
    implementation = _python_init_pip_ext_impl,
)
