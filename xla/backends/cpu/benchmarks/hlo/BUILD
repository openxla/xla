# HLO files for end-to-end benchmarks.

package(
    default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

exports_files(
    [
        "gemma2_2b_keras_jax.hlo",
        "gemma3_1b_flax_call.hlo",
        "gemma3_1b_flax_sample_loop.hlo",
        "huge/gemma3_4b_text_keras_jax_batch1_in8_out100.hlo",
        "huge/gemma4_2b_bf16.hlo",
    ],
    visibility = [
        "//xla:__subpackages__",
    ],
)

filegroup(
    name = "hlo_files",
    testonly = True,
    srcs = glob(["**/*.hlo"]),
    visibility = [
        "//xla:__subpackages__",
    ],
)
