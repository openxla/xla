"""Build file overlay for the ROCm CI image's system pciutils (libpci) headers.

The repository root is the image's /usr/include/pci directory (see
//third_party/system_libpci:workspace.bzl), so the headers are exposed back
under the pci/ prefix (<pci/pci.h>, <pci/config.h>, ...). Only headers are
provided here; libpci.so itself is linked from the image via @roc_mori//:libpci
(-lpci).
"""

load("@rules_cc//cc:defs.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # GPL-2.0-or-later (pciutils; headers only, linked dynamically)

cc_library(
    name = "pci_headers",
    hdrs = glob(["*.h"]),
    include_prefix = "pci",
    visibility = ["//visibility:public"],
)
