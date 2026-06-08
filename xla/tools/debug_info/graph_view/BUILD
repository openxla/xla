load("//xla:pytype.bzl", "pytype_strict_binary")
load("//xla/tsl:tsl.bzl", "if_google")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:public"],
)

pytype_strict_binary(
    name = "glslminifier",
    srcs = ["glslminifier.py"],
)

genrule(
    name = "generate_shaders_ts",
    srcs = [
        "node.vert.glsl",
        "node.frag.glsl",
        "node_simple.frag.glsl",
        "edge.vert.glsl",
        "edge.frag.glsl",
    ],
    outs = ["shaders.ts"],
    cmd = (
        "printf 'export const NODE_VS = `' > $@ && $(location :glslminifier) $(location node.vert.glsl) >> $@ && echo '`;' >> $@ && " +
        "printf 'export const NODE_FS = `' >> $@ && $(location :glslminifier) $(location node.frag.glsl) >> $@ && echo '`;' >> $@ && " +
        "printf 'export const NODE_FS_SIMPLE = `' >> $@ && $(location :glslminifier) $(location node_simple.frag.glsl) >> $@ && echo '`;' >> $@ && " +
        "printf 'export const EDGE_VS = `' >> $@ && $(location :glslminifier) $(location edge.vert.glsl) >> $@ && echo '`;' >> $@ && " +
        "printf 'export const EDGE_FS = `' >> $@ && $(location :glslminifier) $(location edge.frag.glsl) >> $@ && echo '`;' >> $@"
    ),
    compatible_with = if_google(["//buildenv/target:libtpu"]),
    tools = [":glslminifier"],
)

filegroup(
    name = "graph_renderer_srcs",
    srcs = [
        "graph_renderer.ts",
        "shaders.ts",
    ],
    visibility = ["//visibility:public"],
)
