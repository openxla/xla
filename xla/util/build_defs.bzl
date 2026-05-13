"""Contains embed_files build rule."""

load("@bazel_skylib//lib:new_sets.bzl", "sets")
load("@com_google_protobuf//bazel/common:proto_info.bzl", "ProtoInfo")
load("//xla/tsl:package_groups.bzl", "DEFAULT_LOAD_VISIBILITY")
load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

visibility(DEFAULT_LOAD_VISIBILITY)

def embed_files(name, srcs, cpp_namespace = "", compatible_with = None, **kwargs):
    """Compiles srcs into a cc_library with functions returning embedded file data.

    Example:
        embed_files(
            name = "embed_some_file",
            srcs = ["file1.txt", "file2.txt"],
            cpp_namespace = "my_namespace",
        )

    will generate a cc_library with the following functions:

        const std::string& get_file1();
        const std::string& get_file2();

    Args:
        name: name for the generated cc_library target
        srcs: files to embed
        cpp_namespace: If set, the generated code will be wrapped in this namespace
        compatible_with: The `compatible_with` attribute to pass to the generated targets.
        **kwargs: keyword arguments passed onto the generated cc_library() rule.
    """

    namespace_open = ""
    namespace_close = ""
    if cpp_namespace:
        namespace_open = "namespace " + cpp_namespace + " { "
        namespace_close = "}  // namespace " + cpp_namespace + "\n"

    native.genrule(
        name = name + "_gen",
        srcs = srcs,
        outs = [
            name + ".cc",
            name + ".h",
        ],
        tools = ["@xxd//:xxd"],
        cmd = """
            HDR_OUT=$(location {name}.h)
            CC_OUT=$(location {name}.cc)
            GUARD="{guard}"

            # 1. Start Header File
            echo "#ifndef $${{GUARD}}" > "$${{HDR_OUT}}"
            echo "#define $${{GUARD}}" >> "$${{HDR_OUT}}"
            echo "#include <string>" >> "$${{HDR_OUT}}"
            echo "" >> "$${{HDR_OUT}}"
            echo "{namespace_open}" >> "$${{HDR_OUT}}"

            # 2. Start CC File
            # Include standard headers FIRST to avoid namespace issues if header is malformed
            echo "#include <cstddef>" > "$${{CC_OUT}}"
            echo "#include <string>" >> "$${{CC_OUT}}"
            echo '#include "{name}.h"' >> "$${{CC_OUT}}"
            echo "" >> "$${{CC_OUT}}"
            echo "{namespace_open}" >> "$${{CC_OUT}}"

            # 3. Iterate over source files
            for src in $(SRCS); do
                # Extract filename without path
                FILENAME=$$(basename "$${{src}}")
                # Extract stem (filename without extension)
                STEM=$$(echo "$${{FILENAME}}" | sed 's/\\.[^.]*$$//')
                # Create C++ identifier safe names
                SAFE_STEM=$$(echo "$${{STEM}}" | sed 's/[^a-zA-Z0-9_]/_/g')
                FUNC_NAME="get_$${{SAFE_STEM}}"
                VAR_NAME="$${{SAFE_STEM}}_data"

                # Header: Add function declaration
                echo "const std::string& $${{FUNC_NAME}}();" >> "$${{HDR_OUT}}"

                # CC: Embed data using xxd
                $(location @xxd//:xxd) -i "$${{src}}" | \
                sed -e "s/^unsigned char [^[]*/static const unsigned char $${{VAR_NAME}}/" \
                    -e "s/^unsigned int .*_len/static const size_t $${{VAR_NAME}}_size/" \
                    >> "$${{CC_OUT}}"
                echo "" >> "$${{CC_OUT}}"

                # CC: Define the accessor function
                echo "const std::string& $${{FUNC_NAME}}() {{" >> "$${{CC_OUT}}"
                echo "  static const std::string* const kInstance = new std::string(" >> "$${{CC_OUT}}"
                echo "      reinterpret_cast<const char*>($${{VAR_NAME}}), $${{VAR_NAME}}_size);" >> "$${{CC_OUT}}"
                echo "  return *kInstance;" >> "$${{CC_OUT}}"
                echo "}}" >> "$${{CC_OUT}}"
                echo "" >> "$${{CC_OUT}}"
            done

            # 4. Finish Header File
            echo "{namespace_close}" >> "$${{HDR_OUT}}"
            echo "{namespace_close}" >> "$${{CC_OUT}}"
            echo "#endif  // $${{GUARD}}" >> "$${{HDR_OUT}}"
        """.format(
            name = name,
            guard = name.upper() + "_H_",
            namespace_open = namespace_open,
            namespace_close = namespace_close,
        ),
        compatible_with = compatible_with,
    )

    cc_library(
        name = name,
        srcs = [name + ".cc"],
        hdrs = [name + ".h"],
        compatible_with = compatible_with,
        **kwargs
    )

def _descriptor_set_list(deps, descriptor_set):
    """Makes a list of distinct FileDescriptorSet files."""

    descriptor_set_set = sets.make()
    for dep in deps:
        # Silently drop deps without ProtoInfo.
        if ProtoInfo in dep:
            for descriptor_set in dep[ProtoInfo].transitive_descriptor_sets.to_list():
                sets.insert(descriptor_set_set, descriptor_set)
    if descriptor_set != None:
        sets.insert(descriptor_set_set, descriptor_set)
    return sets.to_list(descriptor_set_set)

def _run_protoc_impl(ctx):
    """Rule to translate text to binary using protocol_compiler."""

    proto_descriptor_sets = _descriptor_set_list(ctx.attr.deps, ctx.file.descriptor_set)

    descriptor_set_in = ("--descriptor_set_in=%s" %
                         ":".join([file.path for file in proto_descriptor_sets]))

    if len(ctx.outputs.outs) != 1:
        fail("Expected exactly one output")
    out = ctx.outputs.outs[0]

    protoc_args = [
        "--encode=%s" % ctx.attr.proto_name,
        "--deterministic_output",
        descriptor_set_in,
    ]
    redirect = [
        "< %s" % ctx.file.src.path,
        "> %s" % out.path,
    ]

    # If command line will be long, use flag file
    if len(descriptor_set_in) > 20000:
        # Unfortunately, we can't use Starlark's flag file support,
        # because we're not using the Args object (because we need
        # to specify redirection using some of the "args")
        flagfile = ctx.actions.declare_file(ctx.attr.name + ".flagfile")
        ctx.actions.write(flagfile, "\n".join(protoc_args))

        ctx.actions.run_shell(
            outputs = ctx.outputs.outs,
            inputs = [ctx.file.src, flagfile] + proto_descriptor_sets,
            tools = [ctx.executable._tool],
            command = " ".join([ctx.executable._tool.path, "@%s" % flagfile.path] + redirect),
            mnemonic = "ProtoDataCompilerFlagfile",
            use_default_shell_env = False,
        )

    else:
        # No flag file necessary
        ctx.actions.run_shell(
            outputs = ctx.outputs.outs,
            inputs = [ctx.file.src] + proto_descriptor_sets,
            tools = [ctx.executable._tool],
            command = " ".join([ctx.executable._tool.path] + protoc_args + redirect),
            mnemonic = "ProtoDataCompiler",
            use_default_shell_env = False,
        )

    return DefaultInfo(runfiles = ctx.runfiles(files = ctx.outputs.outs))

def _run_protoc_rule():
    return rule(
        # Consider removing output_to_genfiles once integrated, is preferred but unclear if it is
        # supported by XLA's use cases.
        output_to_genfiles = True,
        attrs = {
            "src": attr.label(allow_single_file = True, mandatory = True),
            "outs": attr.output_list(mandatory = True),
            "deps": attr.label_list(allow_files = True, default = []),
            "descriptor_set": attr.label(allow_single_file = True),
            "proto_name": attr.string(mandatory = True),
            "_tool": attr.label(
                default = "@com_google_protobuf//:protoc",
                executable = True,
                cfg = "exec",
            ),
        },
        implementation = _run_protoc_impl,
    )

_run_protoc = _run_protoc_rule()

def text_to_binary_proto(
        name,
        src,
        proto_name,
        proto_deps = None,
        out = None,
        descriptor_set = None,
        **kwargs):
    """Converts a protocol buffer in text format into binary format using protoc_minimal.

    This is a stripped-down version of proto_data that only supports protobuf output.
    """

    _run_protoc(
        name = name,
        src = src,
        outs = [out or (name + ".binarypb")],
        deps = proto_deps or [],
        descriptor_set = descriptor_set,
        proto_name = proto_name,
        **kwargs
    )
