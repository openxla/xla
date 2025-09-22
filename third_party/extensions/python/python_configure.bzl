load("//third_party/py:python_configure.bzl", "python_configure")

python_configure_ext = module_extension(
    implementation = lambda mctx: python_configure(name = "local_config_python"),
)
