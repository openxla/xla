package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

exports_files(["LICENSE"])

# The DUCC FFT source files are dual-licensed as BSD 3 clause and GPLv2.
# We choose BSD 3 clause.
DUCC_SOURCES = [
    "google/ducc0_custom_lowlevel_threading.h",
    "google/fft.cc",
    "src/ducc0/infra/aligned_array.h",
    "src/ducc0/infra/error_handling.h",
    "src/ducc0/infra/misc_utils.h",
    "src/ducc0/infra/simd.h",
    "src/ducc0/infra/threading.cc",
    "src/ducc0/infra/useful_macros.h",
    "src/ducc0/math/cmplx.h",
    "src/ducc0/math/unity_roots.h",
]

DUCC_HEADERS = [
    "google/fft.h",
    "src/ducc0/fft/fft.h",
    "src/ducc0/fft/fft1d_impl.h",
    "src/ducc0/fft/fftnd_impl.h",
    "src/ducc0/infra/mav.h",
    "src/ducc0/infra/threading.h",
]

cc_library(
    name = "fft",
    srcs = DUCC_SOURCES,
    hdrs = DUCC_HEADERS,
    copts = [
        "-frtti",
        "-fexceptions",
        "-ffp-contract=fast",
    ],
    features = ["-use_header_modules"],
    include_prefix = "ducc",
    includes = [
        ".",  # Needed for google/-relative paths.
        "google",  # Needed for finding ducc0_custom_lowlevel_threading.h.
        "src",  # Needed for internal headers.
    ],
    licenses = ["notice"],
    local_defines = [
        # Use custom TSL/Eigen threading.
        "DUCC0_CUSTOM_LOWLEVEL_THREADING=1",
    ],
    deps = [
        # Required for custom threadpool usage:
        "@tsl//third_party/eigen3",
        "@tsl//tsl/platform:mutex",
    ],
)

# Export source files needed for mobile builds, which do not use granular targets.
filegroup(
    name = "mobile_srcs_no_runtime",
    srcs = DUCC_SOURCES,
    # The above files are covered under BSD-3.
    licenses = ["notice"],
)

filegroup(
    name = "mobile_hdrs_no_runtime",
    srcs = DUCC_HEADERS,
    # The above files are covered under BSD-3.
    licenses = ["notice"],
)
