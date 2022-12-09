load("//tsl/platform:rules_cc.bzl", "cc_library")
load("//tsl:tsl.default.bzl", "get_compatible_with_portable")

package(
    # copybara:uncomment default_applicable_licenses = ["//third_party/tensorflow:license"],
    default_visibility = ["//visibility:public"],
    features = [
        # Required since headers are not self-contained.
        "-parse_headers",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "fixedpoint",
    hdrs = [
        "FixedPoint.h",
        "MatMatProduct.h",
        "MatMatProductAVX2.h",
        "MatMatProductNEON.h",
        "MatVecProduct.h",
        "PacketMathAVX.h",
        "PacketMathAVX2.h",
        "PacketMathAVX512.h",
        "TypeCastingAVX2.h",
        "TypeCastingAVX512.h",
    ],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "//third_party/eigen3",
        "//tsl/framework:fixedpoint_types",
    ],
)

filegroup(
    name = "xla_cpu_runtime_hdrs",
    srcs = [
        "FixedPoint.h",
        "MatMatProduct.h",
        "MatMatProductAVX2.h",
        "MatMatProductNEON.h",
        "MatVecProduct.h",
        "PacketMathAVX.h",
        "PacketMathAVX2.h",
        "PacketMathAVX512.h",
        "TypeCastingAVX2.h",
        "TypeCastingAVX512.h",
    ],
    compatible_with = get_compatible_with_portable(),
    visibility = [
        "//third_party/tensorflow:__subpackages__",
        "//tsl:internal",
    ],
)

# Files needed for core:mobile_srcs_no_runtime.
filegroup(
    name = "mobile_srcs_only_runtime",
    srcs = [
        "FixedPoint.h",
        "MatMatProduct.h",
        "MatMatProductAVX2.h",
        "MatMatProductNEON.h",
        "MatVecProduct.h",
        "PacketMathAVX.h",
        "PacketMathAVX2.h",
        "PacketMathAVX512.h",
        "TypeCastingAVX2.h",
        "TypeCastingAVX512.h",
    ],
    compatible_with = get_compatible_with_portable(),
)

filegroup(
    name = "mobile_srcs_no_runtime",
    srcs = [
        "FixedPoint.h",
        "MatMatProduct.h",
        "MatMatProductAVX2.h",
        "MatMatProductNEON.h",
        "MatVecProduct.h",
        "PacketMathAVX.h",
        "PacketMathAVX2.h",
        "PacketMathAVX512.h",
        "TypeCastingAVX2.h",
        "TypeCastingAVX512.h",
    ],
    compatible_with = get_compatible_with_portable(),
)
