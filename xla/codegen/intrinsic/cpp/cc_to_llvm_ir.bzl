"""
A rule to compile a C++ file to a header containing LLVM IR.

This rule is critical for generating LLVM IR bitcode that is embedded into the XLA compiler.
It uses a hermetic Clang binary to ensure consistent compilation across different build environments
(OSS, Google, Stargate) and to bypass potential issues with the default toolchain wrappers.
"""

load("@rules_cc//cc:find_cc_toolchain.bzl", "find_cc_toolchain", "use_cc_toolchain")
load("@rules_cc//cc/common:cc_common.bzl", "cc_common")
load("@rules_cc//cc/common:cc_info.bzl", "CcInfo")
load("//xla/tsl:package_groups.bzl", "DEFAULT_LOAD_VISIBILITY")
load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

visibility(DEFAULT_LOAD_VISIBILITY)

def to_camel_case(s):
    """Converts a snake_case or kebab-case string to CamelCase."""
    return "".join([p.capitalize() for p in s.replace("-", "_").split("_")])

def _get_builtin_flags(headers):
    """Generates -isystem flags for Clang builtin headers."""
    builtin_include_dirs = {}
    wrappers = ["openmp_wrappers", "cuda_wrappers", "llvm_libc_wrappers"]
    for h in headers:
        # Check if any wrapper string is in the dirname
        is_wrapper = False
        for w in wrappers:
            if w in h.dirname:
                is_wrapper = True
                break
        if not is_wrapper:
            builtin_include_dirs[h.dirname] = True

    return ["-isystem" + d for d in sorted(builtin_include_dirs.keys())]

def _get_header_category(path):
    """Determines the category of a header path."""
    if not path:
        return None
    if "third_party/stl" in path or "cuda_wrappers" in path:
        return None

    if any([x in path for x in ["libcxx/include", "libc++/include", "c++/v1"]]):
        return "libcxx"
    if "libc/include" in path:
        return "libc"
    if any([x in path for x in ["/usr/include", "/usr/local/include", "grte"]]):
        return "system"
    return "other"

def _process_toolchain_flags(flags, categories, src_path):
    """Processes toolchain flags: extracts sysroot, filters, and categorizes."""
    sysroot = None
    skip = False

    for i, arg in enumerate(flags):
        if skip:
            skip = False
            continue

        if arg.startswith("--sysroot="):
            sysroot = arg.split("=", 1)[1].strip()
            continue

        # Filter compilation/dependency flags and source file
        if arg in ["-c", "-S", "-emit-llvm"] or arg == src_path:
            continue
        if arg == "-o":
            skip = True
            continue
        if arg in ["-MD", "-MF", "-MP", "-MT"]:
            if arg in ["-MF", "-MT"]:
                skip = True
            continue

        # Handle includes (split or joined)
        current_args = [arg]
        path = arg

        if arg in ["-isystem", "-iquote", "-I"]:
            if i + 1 < len(flags):
                path = flags[i + 1].strip()
                current_args = [arg, path]
                skip = True

            # Handle OSS case with leading space in joined arg (e.g. "-isystem external/...")
        elif arg.startswith("-isystem ") or arg.startswith("-iquote ") or arg.startswith("-I "):
            parts = arg.split(" ", 1)
            flag = parts[0]
            path = parts[1].strip()
            current_args = [flag, path]

        cat = _get_header_category(path)
        if cat:
            categories[cat].extend(current_args)

    return sysroot

def _cc_ir_header_impl(ctx):
    """Rule implementation that generates IR for multiple features and embeds them in a header."""
    cc_toolchain = find_cc_toolchain(ctx)
    output_header = ctx.outputs.out_header
    temp_ir_output = ctx.actions.declare_file(ctx.label.name + ".ll")

    filtered_compiler_files = depset([
        f
        for f in cc_toolchain.all_files.to_list()
        if f.extension not in ["o", "a"] and "startup_libs" not in f.path
    ])

    # Configure features to disable conflicting ones (sanitizers, etc.).
    feature_configuration = cc_common.configure_features(
        ctx = ctx,
        cc_toolchain = cc_toolchain,
        unsupported_features = ctx.disabled_features + [
            "thin_lto",
            "per_object_debug_info",
            "module_maps",
            "use_header_modules",
            "layering_check",
            "parse_headers",
            "fdo_optimize",
            "fdo_instrument",
            "asan",
            "msan",
            "tsan",
            "ubsan",
        ],
    )

    # Get toolchain flags
    compile_variables = cc_common.create_compile_variables(
        feature_configuration = feature_configuration,
        cc_toolchain = cc_toolchain,
        user_compile_flags = ctx.fragments.cpp.cxxopts + ctx.fragments.cpp.copts,
    )
    command_line = cc_common.get_memory_inefficient_command_line(
        feature_configuration = feature_configuration,
        action_name = "c++-compile",
        variables = compile_variables,
    )

    # Initialize categories with builtin flags
    categories = {
        "libcxx": [],
        "builtin": _get_builtin_flags(ctx.files._clang_headers),
        "libc": [],
        "system": [],
        "other": [],
    }

    # Process toolchain flags
    sysroot = _process_toolchain_flags(command_line, categories, ctx.file.src.path)
    # fail("DEBUG: categories={}".format(categories))

    # Process dependency flags
    compilation_contexts = [dep[CcInfo].compilation_context for dep in ctx.attr.deps]
    merged_context = cc_common.merge_compilation_contexts(compilation_contexts = compilation_contexts)

    dep_flags = (
        ["-iquote" + p for p in merged_context.quote_includes.to_list()] +
        ["-I" + p for p in merged_context.includes.to_list()] +
        ["-isystem" + p for p in merged_context.system_includes.to_list()] +
        ["-iquote."]
    )

    for arg in dep_flags:
        cat = _get_header_category(arg)
        if cat:
            categories[cat].append(arg)

    # Explicitly add libc++ path from sysroot if present to ensure precedence
    if sysroot:
        categories["libcxx"].insert(0, "-isystem" + sysroot + "/usr/include/c++/v1")

        # Add sysroot/include to system category to ensure libc headers are found
        # This fixes math.h errors in Google repo where sysroot has include/ but not usr/include/
        categories["system"].append("-isystem" + sysroot + "/include")

    # Construct arguments
    args = ctx.actions.args()
    args.add("-v")
    args.add("-pthread")

    # Order: Libc++ -> Builtin -> Libc -> System -> Others
    for cat in ["libcxx", "builtin", "libc", "system", "other"]:
        args.add_all(categories[cat])

    args.add_all([
        "-S",
        "-emit-llvm",
        "-O3",
        "-DNDEBUG",
        "-mprefer-vector-width=512",
        "-DEIGEN_VECTORIZE_GENERIC",
        "-fno-builtin",
        "-Wno-psabi",
        "-std=c++17",
        "-o",
        temp_ir_output.path,
        ctx.file.src.path,
    ])

    clang_executable = ctx.executable._clang_binary

    additional_inputs = []

    ctx.actions.run(
        executable = clang_executable,
        arguments = [args],
        tools = [clang_executable],
        inputs = depset(
            [ctx.file.src] + ctx.files._clang_headers,
            transitive = [
                dep[CcInfo].compilation_context.headers
                for dep in ctx.attr.deps
            ] + [filtered_compiler_files] + additional_inputs,
        ),
        outputs = [temp_ir_output],
        mnemonic = "CompileLlvmIr",
        progress_message = "Compiling %s to LLVM IR" % ctx.label.name,
        env = {"LC_ALL": "C"},
    )

    # Generate the final C++ header file.
    python_script = """
import sys

def main():
    if len(sys.argv) != 5:
        print("Usage: script.py <input_ll_file> <output_header_file> <variable_name> <namespace>")
        sys.exit(1)

    input_path, output_path, variable_name, namespace = sys.argv[1:5]
    
    with open(input_path, 'rb') as f:
        content = f.read()

    if content.startswith(b'\\xef\\xbb\\xbf'):
        content = content[3:]
        
    ir_content = content.decode('utf-8', errors='ignore')

    header_content = '''#pragma once

// This file is generated by the cc_to_llvm_ir_header rule. Do not edit.

namespace {namespace} {{

// LLVM IR compiled for the current architecture
inline constexpr char {variable_name}[] = R"IR(
{ir_content}
)IR";

}}  // namespace {namespace}
'''.format(namespace=namespace, variable_name=variable_name, ir_content=ir_content)

    with open(output_path, 'w') as f:
        f.write(header_content)

if __name__ == "__main__":
    main()
"""

    script_file = ctx.actions.declare_file(ctx.label.name + "_gen.py")
    ctx.actions.write(output = script_file, content = python_script)

    variable_name = "k{}Ir".format(to_camel_case(ctx.attr.base_name))

    ctx.actions.run_shell(
        inputs = [script_file, temp_ir_output],
        outputs = [output_header],
        command = "python {} {} {} {} {}".format(
            script_file.path,
            temp_ir_output.path,
            output_header.path,
            variable_name,
            ctx.attr.namespace,
        ),
        mnemonic = "GenerateLlvmIrHeader",
        progress_message = "Generating LLVM IR header %s" % output_header.short_path,
    )

    return [DefaultInfo(files = depset([output_header]))]

_cc_ir_header_rule = rule(
    implementation = _cc_ir_header_impl,
    attrs = {
        "src": attr.label(allow_single_file = True, mandatory = True),
        "deps": attr.label_list(providers = [CcInfo]),
        "out_header": attr.output(mandatory = True),
        "base_name": attr.string(mandatory = True, doc = "The base name of the generated IR variables."),
        "namespace": attr.string(default = "llvm_ir", doc = "The C++ namespace for the generated IR variables."),
        # No "Label()" wrapper - copybara expects exact strings here.
        "_clang_headers": attr.label(default = "@llvm-project//clang:builtin_headers_gen"),
        "_clang_binary": attr.label(
            default = "@llvm-project//clang:clang",
            executable = True,
            cfg = "exec",
        ),
    },
    toolchains = use_cc_toolchain(),
    fragments = ["cpp"],
)

def cc_ir_header(name, src, deps, **kwargs):
    """A macro that generates an IR header and wraps it in a cc_library.

    Args:
      name: The name of the generated cc_library.
      src: The C++ source file to compile.
      deps: The C++ dependencies of the source file.
      **kwargs: Additional arguments to pass to the generated cc_library.
    """
    out_header = name + ".h"
    generator_name = name + "_generator"

    _cc_ir_header_rule(
        base_name = name,
        name = generator_name,
        tags = ["manual"],
        src = src,
        deps = deps,
        out_header = out_header,
        # copybara_removed compatible_with = ["//buildenv/target:non_prod"],
        **kwargs
    )

    cc_library(
        name = name,
        hdrs = [":" + out_header],
        **kwargs
    )
