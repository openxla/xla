load("@pypi//:requirements.bzl", "install_deps")

install_deps_ext = module_extension(
    implementation = lambda mctx: install_deps(),
)
