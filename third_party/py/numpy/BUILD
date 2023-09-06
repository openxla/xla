licenses(["restricted"])

package(default_visibility = ["//visibility:public"])

alias(
    name = "numpy",
    actual = "@local_config_python//:numpy",
)

alias(
    name = "headers",
    actual = "@local_config_python//:numpy_headers",
)

genrule(
    name = "dummy",
    outs = ["tf_numpy_dummy.py"],
    cmd = "touch $@",
    visibility = ["//visibility:private"],
)
