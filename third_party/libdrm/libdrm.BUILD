"""BUILD file for libdrm headers."""

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # MIT-style license

# Export just the headers needed by ROCm
# ROCm expects #include <libdrm/drm.h>, but libdrm sources have include/drm/drm.h
# So we remap: include/drm -> libdrm
cc_library(
    name = "drm_headers",
    hdrs = glob([
        "include/drm/*.h",
    ]),
    include_prefix = "libdrm",
    strip_include_prefix = "include/drm",
    visibility = ["//visibility:public"],
)
