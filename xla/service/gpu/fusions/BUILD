cc_library(
    name = "in_place_dynamic_update_slice",
    srcs = ["in_place_dynamic_update_slice.cc"],
    hdrs = ["in_place_dynamic_update_slice.h"],
    deps = [
        ":fusion_emitter",
        "//xla/service/gpu:ir_emission_utils",
        "//xla/service/gpu:launch_dimensions",
        "//xla/service/llvm_ir:dynamic_update_slice_util",
        "//xla/service/llvm_ir:fused_ir_emitter",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:ir_headers",
    ],
)

cc_library(
    name = "copy",
    srcs = ["copy.cc"],
    hdrs = ["copy.h"],
    deps = [
        ":fusion_emitter",
        "//xla/service/gpu:gpu_executable",
        "//xla/service/gpu:ir_emitter_context",
    ],
)

cc_library(
    name = "fusion_emitter",
    srcs = ["fusion_emitter.cc"],
    hdrs = ["fusion_emitter.h"],
    visibility = ["//xla/service/gpu:__subpackages__"],
    deps = [
        "//xla/hlo/ir:hlo",
        "//xla/mlir_hlo:lhlo",
        "//xla/service:elemental_ir_emitter",
        "//xla/service/gpu:gpu_executable",
        "//xla/service/gpu:hlo_to_ir_bindings",
        "//xla/service/gpu:ir_emitter_context",
        "//xla/service/gpu:kernel_reuse_cache",
        "//xla/service/gpu:launch_dimensions",
        "//xla/service/gpu:target_util",
        "//xla/service/gpu:thunk",
        "//xla/service/llvm_ir:buffer_assignment_util",
        "//xla/service/llvm_ir:ir_array",
        "//xla/service/llvm_ir:llvm_util",
        "//xla/translate/mhlo_to_hlo:location_exporter",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MemRefDialect",
    ],
)

cc_library(
    name = "fusions",
    srcs = ["fusions.cc"],
    hdrs = ["fusions.h"],
    visibility = ["//xla/service/gpu:__subpackages__"],
    deps = [
        ":copy",
        ":fusion_emitter",
        ":in_place_dynamic_update_slice",
        ":input_slices",
        ":loop",
        ":transpose",
        "//xla/hlo/ir:hlo",
        "//xla/mlir_hlo:lhlo",
        "//xla/service:elemental_ir_emitter",
        "//xla/service/gpu:hlo_fusion_analysis",
        "//xla/service/gpu:ir_emission_utils",
        "//xla/service/gpu:ir_emitter_context",
    ],
)

cc_library(
    name = "loop",
    srcs = ["loop.cc"],
    hdrs = ["loop.h"],
    deps = [
        ":fusion_emitter",
        "//xla/hlo/ir:hlo",
        "//xla/mlir_hlo:lhlo",
        "//xla/service:elemental_ir_emitter",
        "//xla/service/gpu:hlo_fusion_analysis",
        "//xla/service/gpu:ir_emitter_context",
        "//xla/service/gpu:parallel_loop_emitter",
        "//xla/service/llvm_ir:fused_ir_emitter",
        "@llvm-project//llvm:ir_headers",
    ],
)

cc_library(
    name = "tiling_util",
    srcs = ["tiling_util.cc"],
    hdrs = ["tiling_util.h"],
    visibility = ["//xla/service/gpu:__subpackages__"],
    deps = [
        "//xla/service/gpu:hlo_fusion_analysis",
        "//xla/service/gpu:ir_emission_utils",
        "//xla/service/gpu:target_util",
        "//xla/service/llvm_ir:ir_array",
        "//xla/service/llvm_ir:kernel_support_library",
        "@llvm-project//llvm:ir_headers",
    ],
)

cc_library(
    name = "transpose",
    srcs = ["transpose.cc"],
    hdrs = ["transpose.h"],
    deps = [
        ":fusion_emitter",
        ":tiling_util",
        "//xla:permutation_util",
        "//xla/hlo/ir:hlo",
        "//xla/mlir_hlo:lhlo",
        "//xla/service:elemental_ir_emitter",
        "//xla/service/gpu:hlo_fusion_analysis",
        "//xla/service/gpu:ir_emitter_context",
        "//xla/service/gpu:parallel_loop_emitter",
        "//xla/service/gpu:target_util",
        "//xla/service/llvm_ir:fused_ir_emitter",
        "//xla/service/llvm_ir:ir_array",
        "//xla/service/llvm_ir:llvm_util",
        "@llvm-project//llvm:ir_headers",
    ],
)

cc_library(
    name = "input_slices",
    srcs = ["input_slices.cc"],
    hdrs = ["input_slices.h"],
    deps = [
        ":fusion_emitter",
        "//xla/service:elemental_ir_emitter",
        "//xla/service/gpu:hlo_fusion_analysis",
        "//xla/service/gpu:ir_emission_utils",
        "//xla/service/gpu:parallel_loop_emitter",
        "//xla/service/llvm_ir:fused_ir_emitter",
        "//xla/service/llvm_ir:kernel_support_library",
        "//xla/service/llvm_ir:llvm_util",
        "@llvm-project//llvm:ir_headers",
    ],
)
