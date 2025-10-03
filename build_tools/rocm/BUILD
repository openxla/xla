load("@bazel_skylib//rules:common_settings.bzl", "string_flag")

string_flag(
    name = "sanitizer",
    build_setting_default = "none",
    values = [
        "none",
        "asan",
        "tsan",
    ],
)

config_setting(
    name = "asan",
    flag_values = {"//build_tools/rocm:sanitizer": "asan"},
)

config_setting(
    name = "tsan",
    flag_values = {"//build_tools/rocm:sanitizer": "tsan"},
)

filegroup(
    name = "sanitizer_ignore_lists",
    srcs = select({
        ":asan": [
            "asan_ignore_list.txt",
            "lsan_ignore_list.txt",
        ],
        ":tsan": ["tsan_ignore_list.txt"],
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"],
)

genrule(
    name = "san_wrapper_script",
    srcs = [":sanitizer_ignore_lists"],
    outs = ["san_wrapper.sh"],
    cmd = """
      echo '#!/bin/bash' > $@
      echo 'exec "$$@"' >> $@
      chmod +x $@
    """,
)

# this wrapper ensures the test target
# take into account any changes in the ignore list files
sh_binary(
    name = "sanitizer_wrapper",
    srcs = [":san_wrapper_script"],
    data = [":sanitizer_ignore_lists"],
    visibility = ["//visibility:public"],
)
