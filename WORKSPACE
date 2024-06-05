# buildifier: disable=load-on-top
workspace(name = "xla")

# Initialize the XLA repository and all dependencies.
#
# The cascade of load() statements and xla_workspace?() calls works around the
# restriction that load() statements need to be at the top of .bzl files.
# E.g. we can not retrieve a new repository with http_archive and then load()
# a macro from that repository in the same file.

# Initialize hermetic Python
load("//third_party/py:python_init_rules.bzl", "python_init_rules")

python_init_rules()

load("//third_party/py:python_init_repositories.bzl", "python_init_repositories")

python_init_repositories(
    requirements = {
        "3.11": "//:requirements_lock_3_11.txt",
    },
)

#load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
#
#http_archive(
#    name = "local_config_nccl",
#    urls = ["https://developer.download.nvidia.com/compute/redist/nccl/v2.21.5/nccl_2.21.5-1+cuda12.4_x86_64.txz"],
#    strip_prefix = "nccl_2.21.5-1+cuda12.4_x86_64",
#    sha256 = "d9c53eb3929a45447eb155bc5ace739b8f8ec28578e7cdf44001bd42f0f8a170",  # Ensure this matches the actual SHA256 of the file
#)
#

load("//third_party/py:python_init_toolchains.bzl", "python_init_toolchains")

python_init_toolchains()

load("//third_party/py:python_init_pip.bzl", "python_init_pip")

python_init_pip()

load("@pypi//:requirements.bzl", "install_deps")

install_deps()

load(":workspace4.bzl", "xla_workspace4")

xla_workspace4()

load(":workspace3.bzl", "xla_workspace3")

xla_workspace3()

load(":workspace2.bzl", "xla_workspace2")

xla_workspace2()

load(":workspace1.bzl", "xla_workspace1")

xla_workspace1()

load(":workspace0.bzl", "xla_workspace0")

xla_workspace0()
