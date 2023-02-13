package_group(
    name = "friends",
    packages = [
        "//xla/mlir/...",
        # copybara:uncomment_begin(google-only)
        # # TODO(ezhulenev): Clean up dependencies that are leforvers from Autofusion project.
        # "@tf_runtime//...",
        # copybara:uncomment_end(google-only)
        # TODO(ecg): Targets depending on mlir must be under xla/mlir
        "//xla/service/cpu/...",
    ],
)

package(
    # copybara:uncomment default_applicable_licenses = ["//third_party/tensorflow:license"],
    default_visibility = [":friends"],
    licenses = ["notice"],
)
