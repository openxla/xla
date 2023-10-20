#include "xla/service/gpu/runtime/annotation.h"

#include "xla/hlo/ir/dfs_hlo_visitor_with_default.h"

namespace xla {
namespace gpu {

namespace {
nvtxStringHandle_t registerString(const char* str) {
  auto domain = tsl::profiler::nvtx::GetNVTXDomain();
  if (!domain) {
    // NVTX not enabled, so don't bother registering strings with it
    return {};
  }
  std::string buffer{};
  constexpr auto max_length = 65330;
  if (auto const length = std::strlen(str); length >= max_length) {
    // nvbugs 4340868
    std::string_view suffix{"\n[truncated]\n"};
    buffer.reserve(max_length);
    buffer.assign(str, str + length - suffix.size());
    buffer.append(suffix);
    str = buffer.c_str();
  }
  return nvtxDomainRegisterStringA(*domain, str);
}

template <typename Visitor>
Status visit_inst_and_called_but_not_operands(Visitor& visitor,
                                              HloInstruction const& inst) {
  // Visit the given instruction, and the things it calls, but not its operands.
  TF_RETURN_IF_ERROR(visitor.DefaultAction(&inst));
  for (HloComputation const* called : inst.called_computations()) {
    HloInstruction const* const root = called->root_instruction();
    TF_RETURN_IF_ERROR(root->Accept(&visitor, false /* call_finish_visit */,
                                    true /* ignore_control_predecessors */,
                                    true /* cross_computation */));
  }
  return OkStatus();
}

// Split `a` and `b` by `delim` into two lists of possibly-empty tokens, then
// rejoin the first N of those lists that match by `delim`. Note: it is
// unspecified which argument the return value points into.
std::string_view longest_prefix(std::string_view a, std::string_view b,
                                char delim = '/') {
  if (a.size() > b.size()) a.swap(b);  // allow assumption that b is longer
  for (auto start_a = a.begin(), iter_a = start_a, start_b = b.begin(),
            iter_b = start_b;
       ; ++iter_a, ++iter_b) {
    if (iter_a == a.end() && (iter_b == b.end() || *iter_b == delim)) {
      // reached both ends without finding a mismatch, or reached the end of `a`
      // and not `b` but it was the end of the chunk in `b`
      return a;
    }
    if (*iter_a != *iter_b) {
      // mismatch in this chunk
      return {a.begin(),
              static_cast<std::size_t>(std::distance(a.begin(), start_a))};
    }
    if (*iter_a == delim) {
      // end of this chunk, start the next one
      start_a = iter_a;
      start_b = iter_b;
    }
  }
}

// Find the longest prefix among instructions' op_name metadata
// Chunk this by delimiting slashes, i.e. given a/b/cat and a/b/cabbage, the
// longest prefix is a/b not a/b/ca
class OpNamePrefixVisitor : public ConstDfsHloVisitorWithDefault {
 public:
  Status DefaultAction(HloInstruction const* inst) final {
    auto const& op_name = inst->metadata().op_name();
    if (!op_name.empty()) {
      prefix = prefix ? longest_prefix(*prefix, op_name) : op_name;
    }
    return OkStatus();
  }
  std::string_view longest_op_name_prefix() const {
    return prefix.value_or(std::string_view{});
  }

 private:
  std::optional<std::string_view> prefix{};
};

std::string_view get_longest_op_name_prefix(HloInstruction const& inst,
                                            bool include_operands) {
  OpNamePrefixVisitor visitor{};
  if (!(include_operands
            ? inst.Accept(&visitor, false /* call_finish_visit */,
                          true /* ignore_control_predecessors */,
                          true /* cross_computation */)
            : visit_inst_and_called_but_not_operands(visitor, inst))
           .ok()) {
    return "[error]";
  }
  return visitor.longest_op_name_prefix();
}

std::string make_title(HloModule const& mod, std::string_view longest_prefix) {
  if (longest_prefix.empty()) {
    return absl::StrFormat("XlaModule:#hlo_module=%s,program_id=%d#",
                           mod.name(), mod.unique_id());
  }
  return absl::StrFormat("XlaModule:#prefix=%s,hlo_module=%s,program_id=%d#",
                         longest_prefix, mod.name(), mod.unique_id());
}
}  // namespace

ModuleAnnotation::ModuleAnnotation(std::string module_name_, int module_id_)
    : longest_prefix{},
      title_str{
          module_id_ >= 0
              ? absl::StrFormat("XlaModule:#hlo_module=%s,program_id=%d",
                                module_name_, module_id_)
              : absl::StrFormat("XlaModule:#hlo_module=%s", module_name_)},
      title{registerString(title_str.c_str())} {}

ModuleAnnotation::ModuleAnnotation(HloModule const& mod)
    : longest_prefix{get_longest_op_name_prefix(
          *mod.entry_computation()->root_instruction(), true)},
      title_str{make_title(mod, longest_prefix)},
      title{registerString(title_str.c_str())} {}

std::string_view ModuleAnnotation::longest_op_name_prefix() const {
  return longest_prefix;
}

std::string_view ModuleAnnotation::Title() const { return title_str; }

nvtxStringHandle_t ModuleAnnotation::NVTXRegisteredTitle() const {
  return title;
}

namespace {
std::string make_kernel_name(std::string_view prefix,
                             HloInstruction const& inst) {
  // Sometimes an instruction doesn't have metadata, but the computations that
  // it calls do have metadata. Consider all of those metadata op_name entries
  // and attach the longest prefix to this launch.
  std::string_view op_name = get_longest_op_name_prefix(inst, false);
  if (op_name.empty()) {
    return absl::StrFormat("Thunk:#hlo_op=%s#", inst.name());
  } else {
    // remove the prefix that's in the parent module annotation
    if (op_name.substr(0, prefix.size()) != prefix) {
      std::string msg{op_name};
      msg += " did not start with ";
      msg += prefix;
      throw std::runtime_error(std::move(msg));
    }
    auto short_name = op_name.substr(prefix.size());
    // remove the leading / if there is one (prefix might be an empty string)
    if (!short_name.empty() && short_name.front() == '/') {
      short_name = short_name.substr(1);
    }
    return absl::StrFormat("Thunk:#name=%s,hlo_op=%s#", short_name,
                           inst.name());
  }
}
}  // namespace

KernelAnnotation::KernelAnnotation(ModuleAnnotation const& module_annotation,
                                   HloInstruction const& inst)
    : title_str{make_kernel_name(module_annotation.longest_op_name_prefix(),
                                 inst)},
      title{registerString(title_str.c_str())} {}

std::string_view KernelAnnotation::Title() const { return title_str; }

nvtxStringHandle_t KernelAnnotation::NVTXRegisteredTitle() const {
  return title;
}

ModuleAnnotations::ModuleAnnotations(HloModule const& mod) : top_level{mod} {
  // loop through `mod` and populate `kernels` (string -> KernelAnnotation map)
  // with the information we want to attach to individual kernels.
  for (HloComputation const* computation :
       mod.computations()) {  // top-level blocks in the module
    for (HloInstruction const* inst :
         computation->instructions()) {  // statements within block
      // working assumption: only custom calls and fusions end up with NVTX
      // ranges named after them. bad assumption [at least partially]: cuda
      // graph launches are not handled correctly
      switch (inst->opcode()) {
        case HloOpcode::kCustomCall:
        case HloOpcode::kFusion: {
          // e.g. inst.name is "fusion.6", inst.opcode is "kFusion" and called
          // is ["fused_computation.5"], in which case the content of
          // "fused_computation.5" ends up under an NVTX range called
          // "fusion.6". We want to construct a useful annotation for that NVTX
          // range based on the content of `inst`, including `called` etc.
          // FIXME: using try_emplace here was sensitive to
          // https://github.com/abseil/abseil-cpp/issues/388.
          auto const [iter, inserted] =
              kernels.insert({inst->name(), {top_level, *inst}});
          if (!inserted) {
            throw std::runtime_error(
                absl::StrCat("Name collision: ", inst->name()));
          }
        } break;
        default:
          break;
      }
    }
  }
}
}  // namespace gpu
}  // namespace xla
