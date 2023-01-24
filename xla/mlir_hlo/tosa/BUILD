# MHLO -> TOSA bridge.
package(
    # copybara:uncomment default_applicable_licenses = ["//third_party/tensorflow:license"],
    default_visibility = [
        ":internal",
        "//xla/mlir_hlo/tosa:__subpackages__",
    ],
    licenses = ["notice"],
)

package_group(
    name = "internal",
    packages = [],
)

cc_binary(
    name = "mhlo-tosa-opt",
    srcs = ["mhlo_tosa_opt.cc"],
    deps = [
        "//xla/mlir_hlo:hlo_dialect_registration",
        "//xla/mlir_hlo/tosa/transforms:MHLOTOSATransforms",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AllPassesAndDialects",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MlirOptLib",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TosaDialect",
        "@stablehlo//:register",
    ],
)
