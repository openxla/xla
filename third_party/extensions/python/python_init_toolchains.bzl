load("//third_party/py:python_init_toolchains.bzl", "python_init_toolchains")

def _python_init_toolchains_ext_impl(mctx):
    python_init_toolchains(register_toolchains=False)

python_init_toolchains_ext = module_extension(
    implementation = _python_init_toolchains_ext_impl,
)
