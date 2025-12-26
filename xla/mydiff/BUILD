cc_binary(
    name = "my_diff",
    srcs = ["my_diff.cc"],
    deps = [
        "//xla/hlo/dialect/hlo/parser:hlo_parser",
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        # "//xla/hlo/transforms:hlo_module_diff",  # ❌ 这个不存在，先别写
    ],
)
