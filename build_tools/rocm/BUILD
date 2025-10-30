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
    name = "exclusive_wrapper_script",
    outs = ["exclusive_wrapper.sh"],
    cmd = """
      echo '#!/bin/bash' > $@
      echo 'exec {lock_fd}>/var/lock/gpulock || exit 1' >> $@
      echo 'flock "$$lock_fd"' >> $@
      echo '"$$@"' >> $@
      echo 'return_code=$$?' >> $@
      echo 'flock -u "$$lock_fd"' >> $@
      echo 'exit $$return_code' >> $@
      chmod +x $@
    """,
)

# this wrapper ensures the test target
# take into account any changes in the ignore list files
sh_binary(
    name = "exclusive_local_wrapper",
    srcs = [":exclusive_wrapper_script"],
    visibility = ["//visibility:public"],
)

# this wrapper ensures the test target
# take into account any changes in the ignore list files
sh_binary(
    name = "sanitizer_wrapper",
    srcs = [":exclusive_wrapper_script"],
    data = [":sanitizer_ignore_lists"],
    visibility = ["//visibility:public"],
)
