licenses(["notice"])

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "ducc",
    srcs = [
        "src/ducc0/fft/fft1d.h",
        "src/ducc0/infra/aligned_array.h",
        "src/ducc0/infra/error_handling.h",
        "src/ducc0/infra/mav.h",
        "src/ducc0/infra/misc_utils.h",
        "src/ducc0/infra/simd.h",
        "src/ducc0/infra/threading.cc",
        "src/ducc0/infra/threading.h",
        "src/ducc0/infra/useful_macros.h",
        "src/ducc0/math/cmplx.h",
        "src/ducc0/math/unity_roots.h",
        "src/ducc0/external/ducc0_custom_lowlevel_threading.h"
    ],
    hdrs = [
        "src/ducc0/fft/fft.h",
    ],
    copts = [
        "-fexceptions",
        "-ffast-math",
    ],
    features = ["-use_header_modules"],
    include_prefix = "ducc",
    includes = [
        "src",
        "src/ducc0/external",
    ],
    defines = ["DUCC0_NO_LOWLEVEL_THREADING"]
)