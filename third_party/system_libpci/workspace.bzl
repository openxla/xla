"""Exposes the ROCm CI image's system pciutils (libpci) headers to Bazel.

pciutils ships in the ROCm build image (rocm/tensorflow-build) but not in the
hermetic sysroot, so under --config=rocm_ci the compiler cannot find
<pci/pci.h> (used by MORI's src/application/topology/pci.cpp). Rather than
vendor pciutils -- whose public headers include a generated pci/config.h that
is awkward to reproduce -- we reference the image's already-installed copy via
new_local_repository. This keeps the headers as declared Bazel inputs (so the
absolute-include check passes) while still "using the one from the container".

libpci.so itself is linked from the image via @roc_mori//:libpci (-lpci),
resolved locally (rocm_clang_hermetic links with CppLink=local).

Only referenced by @roc_mori//:libpci, which is only built for ROCm targets, so
the /usr/include/pci path is only required when actually building MORI.
"""

def repo():
    """Registers @system_libpci pointing at the image's /usr/include/pci."""
    native.new_local_repository(
        name = "system_libpci",
        path = "/usr/include/pci",
        build_file = "//third_party/system_libpci:system_libpci.BUILD",
    )
