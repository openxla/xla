load("//third_party/py:python_init_pip.bzl", "python_init_pip")

python_init_pip_ext = module_extension(
    implementation = lambda mctx: python_init_pip(),
)
