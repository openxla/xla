load("@pypi//:requirements.bzl", "install_deps")

def _install_deps_ext_impl(mctx):
    install_deps()

install_deps_ext = module_extension(
    implementation = _install_deps_ext_impl,
)
