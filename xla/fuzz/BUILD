# Description:
#   libFuzzer harnesses for openxla/xla parser and proto-deserialization
#   surfaces. Built by the OSS-Fuzz integration under projects/xla/ in
#   google/oss-fuzz, and runnable locally with -fsanitize=fuzzer flags
#   passed via --copt and --linkopt.

load("//xla/tsl/platform:rules_cc.bzl", "cc_binary")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_binary(
    name = "hlo_parser_fuzz",
    srcs = ["hlo_parser_fuzz.cc"],
    tags = [
        "fuzz_target",
        "nobuilder",
        "no_oss",
    ],
    deps = [
        "//xla/hlo/parser:hlo_parser",
        "//xla/service:hlo_module_config",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_binary(
    name = "hlo_proto_fuzz",
    srcs = ["hlo_proto_fuzz.cc"],
    tags = [
        "fuzz_target",
        "nobuilder",
        "no_oss",
    ],
    deps = [
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_module_config",
        "//xla/service:hlo_proto_cc",
    ],
)
