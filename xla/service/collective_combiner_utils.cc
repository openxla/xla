#include "xla/service/collective_combiner_utils.h"

#include "absl/strings/str_cat.h"

namespace xla {

// Merges op metadata from a list of instructions to be combined
// as much as possible.
OpMetadata MergeOpMetadata(absl::Span<HloInstruction* const> to_combine) {
  OpMetadata op_metadata;
  std::string source_info;
  for (const HloInstruction* instr : to_combine) {
    const auto& metadata = instr->metadata();
    absl::StrAppend(&source_info, metadata.source_file(), ":",
                    metadata.source_line(), ",");
  }
  source_info = absl::StripSuffix(source_info, ",");
  op_metadata.set_source_file(source_info);
  return op_metadata;
}

std::string MaybeMergeBackendConfigString(
    absl::Span<HloInstruction* const> to_combine) {
  if (to_combine.empty()) {
    return "";
  }

  std::string result = to_combine.front()->raw_backend_config_string();
  for (const auto* instr : to_combine) {
    if (instr->raw_backend_config_string() != result) {
      return "";
    }
  }
  return result;
}
}  // namespace xla
