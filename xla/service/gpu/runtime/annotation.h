#ifndef XLA_SERVICE_GPU_GPU_ANNOTATION_H_
#define XLA_SERVICE_GPU_GPU_ANNOTATION_H_

#include "absl/container/flat_hash_map.h"
#include "tsl/profiler/lib/nvtx_utils.h"
#include "xla/hlo/ir/hlo_module.h"

namespace xla {
namespace gpu {
// Prepared information for the top level NVTX/profiler range covering an
// HloModule
struct ModuleAnnotation {
  ModuleAnnotation(std::string module_name, int module_id);
  ModuleAnnotation(HloModule const& mod);
  std::string_view longest_op_name_prefix() const;
  nvtxStringHandle_t NVTXRegisteredTitle() const;
  std::string_view Title() const;

 private:
  std::string longest_prefix{}, title_str{};
  nvtxStringHandle_t title{};
};

// Prepared information for a kernel/thunk/fusion/... within an HloModule
struct KernelAnnotation {
  KernelAnnotation(ModuleAnnotation const& module_annotaion,
                   HloInstruction const& inst);
  nvtxStringHandle_t NVTXRegisteredTitle() const;
  std::string_view Title() const;

 private:
  std::string title_str{};
  nvtxStringHandle_t title{};
};
// Parsed/prepared information for an HloModule that gets propagated to NVTX
// ranges/profilers/... at execution time.
struct ModuleAnnotations {
  ModuleAnnotations(HloModule const&);
  ModuleAnnotation top_level;
  absl::flat_hash_map<std::string_view, KernelAnnotation> kernels{};
};
}  // namespace gpu
}  // namespace xla

#endif
