filegroup(
    name="sanitizer_ignore_lists",
    srcs = [
        "asan_ignore_list.txt",
        "lsan_ignore_list.txt",
        "tsan_ignore_list.txt",
    ],
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
