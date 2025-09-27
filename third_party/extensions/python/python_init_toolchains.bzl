load("//third_party/py:python_init_toolchains.bzl", "python_init_toolchains")

python_init_toolchains_ext = module_extension(
    implementation = lambda mctx: python_init_toolchains(register_toolchains=False),
)
