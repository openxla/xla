/* Copyright 2026 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "xla/tools/hlo_isolation/hlo_isolation_api.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <optional>
#include <queue>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/btree_map.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/flags/commandlineflag.h"
#include "absl/flags/reflection.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/match.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "absl/strings/strip.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "google/protobuf/repeated_ptr_field.h"
#include "re2/re2.h"
#include "tsl/platform/path.h"
#include "xla/comparison_util.h"
#include "xla/error_spec.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/pass/hlo_pass_pipeline.h"
#include "xla/hlo/transforms/defuser.h"
#include "xla/hlo/transforms/despecializer.h"
#include "xla/hlo/transforms/simplifiers/hlo_memory_scheduler.h"
#include "xla/literal.h"
#include "xla/literal_comparison.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/primitive_util.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/hlo_runner_interface.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tests/test_utils.h"
#include "xla/tools/hlo_decomposer.h"
#include "xla/tools/hlo_dump/hlo_dump_utils.h"
#include "xla/tools/hlo_isolation/hlo_inf_nan_intent_analyzer.h"
#include "xla/tools/hlo_isolation/hlo_isolation.pb.h"
#include "xla/tools/hlo_module_loader.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/test.h"
#include "xla/xla_data.pb.h"

using ::xla::hlo_isolation::ModuleIsolationOptions;
using ::xla::hlo_isolation::PipelineIsolationOptions;

namespace xla {
namespace hlo_isolation {

namespace {

constexpr absl::string_view kResultsFilename =
    "hlo_isolation_test_results.pbtxt";

absl::Status InitIsolatorOptions(ModuleIsolationOptions& options) {
  if (!options.run_module_fn) {
    options.run_module_fn =
        [run_hlo_passes = options.run_hlo_passes](
            std::unique_ptr<HloModule> m, HloRunnerInterface* r,
            absl::Span<const Literal> i,
            const RunModuleOptions& run_opts) -> absl::StatusOr<Literal> {
      RunModuleOptions run_opts_copy = run_opts;
      run_opts_copy.run_hlo_passes = run_hlo_passes;
      return RunModule(std::move(m), r, i, run_opts_copy);
    };
  }
  if (!options.on_mismatch_fn) {
    options.on_mismatch_fn = [](const HloModule& module,
                                const Literal& /*test_output*/,
                                const Literal& /*reference_output*/,
                                const absl::Status& compare_status) {
      ADD_FAILURE() << compare_status.message();
      LOG(ERROR) << compare_status.message();
      auto* env = tsl::Env::Default();
      std::string outdir;
      std::string filename;
      if (tsl::io::GetTestUndeclaredOutputsDir(&outdir)) {
        filename = tsl::io::JoinPath(
            outdir, absl::StrCat("failed-module-", module.name(), ".txt"));
      } else {
        filename = tsl::io::GetTempFilename(
            absl::StrCat("failed-module-", module.name(), ".txt"));
      }
      CHECK_OK(tsl::WriteStringToFile(env, filename, module.ToString()));
      LOG(INFO) << "Wrote failed HLO module to " << filename;
    };
  }
  if (!options.make_fake_arguments_fn) {
    options.make_fake_arguments_fn =
        [use_dataflow_based_input_generation =
             options.use_dataflow_based_input_generation](
            const HloModule& module) -> absl::StatusOr<std::vector<Literal>> {
      if (use_dataflow_based_input_generation) {
        return MakeDataflowConstrainedArguments(&module);
      }
      return MakeFakeArguments(&module);
    };
  }
  if (!options.estimate_module_size_fn) {
    options.estimate_module_size_fn = [](const HloModule& module) -> int64_t {
      int64_t total_size = 0;
      for (const auto* param :
           module.entry_computation()->parameter_instructions()) {
        total_size += ShapeUtil::ByteSizeOf(param->shape());
      }
      total_size += ShapeUtil::ByteSizeOf(
          module.entry_computation()->root_instruction()->shape());
      return total_size;
    };
  }
  return absl::OkStatus();
}

absl::Status ValidatePipelineOptions(const PipelineIsolationOptions& options) {
  if (options.shard_index >= 0) {
    if (options.num_shards <= 0) {
      return absl::InvalidArgumentError(
          "num_shards must be strictly positive when shard_index is "
          "specified.");
    }
    if (options.shard_index >= options.num_shards) {
      return absl::InvalidArgumentError(
          "shard_index must be less than num_shards.");
    }
  }
  return absl::OkStatus();
}

void WriteLiteralToTempFile(const LiteralSlice& literal,
                            const std::string& module_name,
                            const std::string& name) {
  auto* env = tsl::Env::Default();
  std::string text_filename;
  std::string outdir;
  std::string prefix = absl::StrCat(module_name, "-", name);
  if (tsl::io::GetTestUndeclaredOutputsDir(&outdir)) {
    std::string filename =
        tsl::io::JoinPath(outdir, absl::StrCat("failed-", prefix));
    text_filename = absl::StrCat(filename, ".txt");
  } else {
    text_filename = tsl::io::GetTempFilename(absl::StrCat(prefix, ".txt"));
  }
  CHECK_OK(tsl::WriteStringToFile(env, text_filename, literal.ToString()));
  LOG(INFO) << "Wrote Literal to " << prefix << " text: " << text_filename;
}

void WriteResults(const std::vector<HloIsolationTestResult>& pipeline_results) {
  auto* env = tsl::Env::Default();
  std::string outdir;
  std::string filename;
  if (tsl::io::GetTestUndeclaredOutputsDir(&outdir)) {
    filename = tsl::io::JoinPath(outdir, kResultsFilename);
  } else {
    filename = tsl::io::GetTempFilename(std::string(kResultsFilename));
  }
  HloIsolationTestSummary results_proto;
  for (const auto& res : pipeline_results) {
    *results_proto.add_results() = res;
  }
  absl::Status status = tsl::WriteTextProto(env, filename, results_proto);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to write results proto to " << filename << ": "
               << status;
  } else {
    LOG(INFO) << "Wrote test results to " << filename;
  }
}

void ApplyBoundingBoxToMismatch(
    const absl::flat_hash_map<
        int64_t, numerics::debug_info::MismatchBoundingBox>& computed_bboxes,
    NumericMismatch& mismatch) {
  int64_t idx = mismatch.output_shape_index();
  auto it = computed_bboxes.find(idx);
  if (it != computed_bboxes.end()) {
    const auto& bbox = it->second;
    for (int64_t dim : bbox.tensor_shape) {
      mismatch.add_tensor_dimensions(dim);
    }
    for (int64_t val : bbox.box_min) {
      mismatch.add_mismatch_box_min(val);
    }
    for (int64_t val : bbox.box_max) {
      mismatch.add_mismatch_box_max(val);
    }
    if (mismatch.top_mismatch_index().empty() &&
        !bbox.top_mismatch_coords.empty()) {
      for (int64_t coord : bbox.top_mismatch_coords.front()) {
        mismatch.add_top_mismatch_index(coord);
      }
    }
    mismatch.set_mismatch_count(bbox.mismatch_count);
    mismatch.set_total_elements(bbox.total_elements);
    if (bbox.total_elements > 0 &&
        mismatch.percentage_of_elems_exceeding_both_errors() == 0.0 &&
        bbox.mismatch_count > 0) {
      const double pct = 100.0 * static_cast<double>(bbox.mismatch_count) /
                         static_cast<double>(bbox.total_elements);
      mismatch.set_percentage_of_elems_exceeding_both_errors(pct);
      mismatch.set_percentage_of_elems_exceeding_abs_error(pct);
      mismatch.set_percentage_of_elems_exceeding_rel_error(pct);
    }
  }
}

auto MakeMiscompareCallback(
    std::string literal_prefix,
    absl::flat_hash_map<int64_t, numerics::debug_info::MismatchBoundingBox>*
        computed_bboxes = nullptr) {
  return [literal_prefix = std::move(literal_prefix), computed_bboxes](
             const LiteralSlice& expected, const LiteralSlice& actual,
             const LiteralSlice& mismatches, const ShapeIndex& shape_index,
             const literal_comparison::ErrorBuckets& /*error_buckets*/) {
    std::string escaped_shape_index = absl::StrReplaceAll(
        shape_index.ToString(), {{",", "_"}, {"{", ""}, {"}", ""}});
    std::string shape_suffix =
        escaped_shape_index.empty()
            ? ""
            : absl::StrCat("-shape-", escaped_shape_index);
    WriteLiteralToTempFile(expected, literal_prefix,
                           absl::StrCat("expected", shape_suffix));
    WriteLiteralToTempFile(actual, literal_prefix,
                           absl::StrCat("actual", shape_suffix));
    WriteLiteralToTempFile(mismatches, literal_prefix,
                           absl::StrCat("mismatches", shape_suffix));
    if (computed_bboxes != nullptr) {
      int64_t idx = shape_index.empty() ? 0 : shape_index.front();
      (*computed_bboxes)[idx] =
          numerics::debug_info::ComputeBoundingBoxFromLiteralMask(mismatches);
    }
  };
}

bool IsBf16Subnormal(uint16_t val) {
  return (val & 0x7F80) == 0 && (val & 0x007F) != 0;
}

bool IsF16Subnormal(uint16_t val) {
  return (val & 0x7C00) == 0 && (val & 0x03FF) != 0;
}

bool Is16BitFloatZero(uint16_t val) { return (val & 0x7FFF) == 0; }

bool IsF8Subnormal(PrimitiveType type, uint8_t val) {
  switch (type) {
    case F8E5M2:
    case F8E5M2FNUZ:
      return (val & 0x7C) == 0 && (val & 0x03) != 0;
    case F8E4M3:
    case F8E4M3FN:
    case F8E4M3FNUZ:
    case F8E4M3B11FNUZ:
      return (val & 0x78) == 0 && (val & 0x07) != 0;
    case F8E3M4:
      return (val & 0x70) == 0 && (val & 0x0F) != 0;
    case F8E8M0FNU:
    default:
      return false;
  }
}

bool IsF8Zero(PrimitiveType type, uint8_t val) {
  switch (type) {
    case F8E5M2FNUZ:
    case F8E4M3FNUZ:
    case F8E4M3B11FNUZ:
      return val == 0x00;
    case F8E5M2:
    case F8E4M3:
    case F8E4M3FN:
    case F8E3M4:
      return (val & 0x7F) == 0;
    case F8E8M0FNU:
    default:
      return false;
  }
}

bool IsF32Subnormal(uint32_t val) {
  return (val & 0x7F800000) == 0 && (val & 0x007FFFFF) != 0;
}

bool IsF32Zero(uint32_t val) { return (val & 0x7FFFFFFF) == 0; }

template <typename UintT, typename IsSubnormalFn, typename IsZeroFn>
std::optional<std::string> ScanForSubnormalFlush(
    absl::string_view type_name, absl::PadSpec hex_pad,
    const uint8_t* expected_ptr, const uint8_t* actual_ptr,
    int64_t size_in_bytes, IsSubnormalFn is_subnormal, IsZeroFn is_zero) {
  const int64_t num_elements = size_in_bytes / sizeof(UintT);
  for (int64_t i = 0; i < num_elements; ++i) {
    UintT exp_v, act_v;
    std::memcpy(&exp_v, expected_ptr + i * sizeof(UintT), sizeof(UintT));
    std::memcpy(&act_v, actual_ptr + i * sizeof(UintT), sizeof(UintT));
    if (is_subnormal(exp_v) && is_zero(act_v)) {
      return absl::StrCat(
          "SUBNORMAL FLUSH-TO-ZERO DETECTED at element ", i,
          ": Expected subnormal ", type_name, " 0x", absl::Hex(exp_v, hex_pad),
          ", but actual produced flushed zero 0x", absl::Hex(act_v, hex_pad));
    }
  }
  return std::nullopt;
}

std::optional<std::string> DetectSubnormalFlushInArray(
    PrimitiveType element_type, const uint8_t* expected_ptr,
    const uint8_t* actual_ptr, int64_t size_in_bytes) {
  if (element_type == BF16) {
    return ScanForSubnormalFlush<uint16_t>(
        "BF16", absl::kZeroPad4, expected_ptr, actual_ptr, size_in_bytes,
        IsBf16Subnormal, Is16BitFloatZero);
  }
  if (element_type == F16) {
    return ScanForSubnormalFlush<uint16_t>("F16", absl::kZeroPad4, expected_ptr,
                                           actual_ptr, size_in_bytes,
                                           IsF16Subnormal, Is16BitFloatZero);
  }
  if (primitive_util::IsF8Type(element_type)) {
    std::string type_name = absl::AsciiStrToUpper(
        primitive_util::LowercasePrimitiveTypeName(element_type));
    return ScanForSubnormalFlush<uint8_t>(
        type_name, absl::kZeroPad2, expected_ptr, actual_ptr, size_in_bytes,
        [element_type](uint8_t v) { return IsF8Subnormal(element_type, v); },
        [element_type](uint8_t v) { return IsF8Zero(element_type, v); });
  }
  if (element_type == F32) {
    return ScanForSubnormalFlush<uint32_t>("F32", absl::kZeroPad8, expected_ptr,
                                           actual_ptr, size_in_bytes,
                                           IsF32Subnormal, IsF32Zero);
  }
  return std::nullopt;
}

std::optional<std::string> DetectSubnormalFlushInLiteral(
    const LiteralSlice& expected, const LiteralSlice& actual) {
  if (expected.shape().IsTuple() && actual.shape().IsTuple()) {
    const int64_t count = std::min(expected.shape().tuple_shapes_size(),
                                   actual.shape().tuple_shapes_size());
    for (int64_t i = 0; i < count; ++i) {
      if (auto flush = DetectSubnormalFlushInLiteral(
              LiteralSlice(expected, {i}), LiteralSlice(actual, {i}));
          flush.has_value()) {
        return flush;
      }
    }
    return std::nullopt;
  }
  if (expected.shape().IsArray() && actual.shape().IsArray() &&
      expected.shape().element_type() == actual.shape().element_type() &&
      expected.size_bytes() == actual.size_bytes()) {
    return DetectSubnormalFlushInArray(
        expected.shape().element_type(),
        static_cast<const uint8_t*>(expected.untyped_data()),
        static_cast<const uint8_t*>(actual.untyped_data()),
        expected.size_bytes());
  }
  return std::nullopt;
}

template <typename UintT, typename IsSubnormalFn>
bool ArrayContainsSubnormal(const uint8_t* ptr, int64_t size_in_bytes,
                            IsSubnormalFn is_subnormal) {
  const int64_t num_elements = size_in_bytes / sizeof(UintT);
  for (int64_t i = 0; i < num_elements; ++i) {
    UintT val;
    std::memcpy(&val, ptr + i * sizeof(UintT), sizeof(UintT));
    if (is_subnormal(val)) {
      return true;
    }
  }
  return false;
}

bool LiteralContainsSubnormal(const LiteralSlice& literal) {
  if (literal.shape().IsTuple()) {
    for (int64_t i = 0; i < literal.shape().tuple_shapes_size(); ++i) {
      if (LiteralContainsSubnormal(LiteralSlice(literal, {i}))) {
        return true;
      }
    }
    return false;
  }
  if (!literal.shape().IsArray()) {
    return false;
  }
  const PrimitiveType element_type = literal.shape().element_type();
  const auto* ptr = static_cast<const uint8_t*>(literal.untyped_data());
  const int64_t size_in_bytes = literal.size_bytes();
  if (element_type == BF16) {
    return ArrayContainsSubnormal<uint16_t>(ptr, size_in_bytes,
                                            IsBf16Subnormal);
  }
  if (element_type == F16) {
    return ArrayContainsSubnormal<uint16_t>(ptr, size_in_bytes, IsF16Subnormal);
  }
  if (primitive_util::IsF8Type(element_type)) {
    return ArrayContainsSubnormal<uint8_t>(
        ptr, size_in_bytes,
        [element_type](uint8_t v) { return IsF8Subnormal(element_type, v); });
  }
  if (element_type == F32) {
    return ArrayContainsSubnormal<uint32_t>(ptr, size_in_bytes, IsF32Subnormal);
  }
  return false;
}

bool InputsContainSubnormal(absl::Span<const Literal> inputs) {
  for (const Literal& input : inputs) {
    if (LiteralContainsSubnormal(LiteralSlice(input))) {
      return true;
    }
  }
  return false;
}

absl::StatusOr<Literal> CoerceActualLiteralToExpectedShape(
    const Literal& actual, const Shape& expected_shape) {
  Shape host_expected = ShapeUtil::DeviceShapeToHostShape(expected_shape);
  if (ShapeUtil::Compatible(host_expected, actual.shape())) {
    return actual.Clone();
  }
  if (actual.shape().IsArray() && host_expected.IsArray() &&
      ShapeUtil::SameDimensions(host_expected, actual.shape()) &&
      actual.shape().element_type() == F32) {
    const PrimitiveType expected_type = host_expected.element_type();
    if (primitive_util::IsFloatingPointType(expected_type)) {
      return actual.Convert(expected_type);
    }
    if (primitive_util::IsIntegralType(expected_type)) {
      const int bit_width = primitive_util::BitWidth(expected_type);
      if (bit_width == 32) {
        return actual.BitcastConvert(host_expected);
      }
      if (bit_width < 32) {
        ABSL_ASSIGN_OR_RETURN(
            Literal as_u32, actual.BitcastConvert(ShapeUtil::ChangeElementType(
                                actual.shape(), U32)));
        return as_u32.Convert(expected_type);
      }
    }
  }
  return absl::InvalidArgumentError(
      absl::StrCat("Reference literal shape ", expected_shape.ToString(),
                   " is not compatible with actual literal shape ",
                   actual.shape().ToString()));
}

// Compares `test_output` against `reference_output` and records a NumericCheck
// named `check_name` on `result`.
//
// Failing literals are dumped as `failed-<literal_prefix>-{expected,actual,
// mismatches}...`. Callers must give checks that can run on the same module
// distinct prefixes, so that a later check cannot overwrite the artifacts of an
// earlier one.
//
// On mismatch the (formatted) failure message is appended to
// `mismatch_messages` instead of being reported immediately: a single failing
// check is not by itself conclusive, because RunIsolationTestOnModule accepts
// the module as correct if *any* of its reference checks passes.
absl::Status CompareOutputs(const HloModule& module, const Literal& test_output,
                            const Literal& reference_output,
                            HloIsolationTestResult& result,
                            const ModuleIsolationOptions& options,
                            absl::string_view check_name,
                            absl::string_view literal_prefix,
                            std::vector<std::string>& mismatch_messages) {
  ErrorSpec error_spec(options.abs_error_bound, options.rel_error_bound);
  absl::flat_hash_map<int64_t, numerics::debug_info::MismatchBoundingBox>
      computed_bboxes;
  absl::Status status = literal_comparison::Near(
      reference_output, test_output, error_spec, true,
      MakeMiscompareCallback(std::string(literal_prefix), &computed_bboxes));
  std::optional<std::string> subnormal_flush = DetectSubnormalFlushInLiteral(
      LiteralSlice(reference_output), LiteralSlice(test_output));
  if (subnormal_flush.has_value()) {
    if (status.ok()) {
      status = literal_comparison::Near(
          reference_output, test_output, ErrorSpec(0, 0), true,
          MakeMiscompareCallback(std::string(literal_prefix),
                                 &computed_bboxes));
    }
    status = absl::InternalError(
        absl::StrCat(status.message(), "\n", *subnormal_flush));
  }
  NumericCheck* numeric_check = result.add_numeric_checks();
  numeric_check->set_name(check_name);
  numeric_check->set_expected_contains_inf_or_nan(
      LiteralContainsInfOrNan(reference_output));
  numeric_check->set_actual_contains_inf_or_nan(
      LiteralContainsInfOrNan(test_output));
  if (!status.ok()) {
    status = absl::InternalError(
        absl::StrFormat("Value mismatch in check %s for module %s\n\n%s",
                        check_name, module.name(), status.message()));
    mismatch_messages.push_back(std::string(status.message()));
    absl::StatusOr<std::vector<NumericMismatch>> top_mismatches =
        ExtractAndEnrichTopMismatches(std::string(status.message()), &module);
    if (top_mismatches.ok()) {
      for (NumericMismatch& mismatch : *top_mismatches) {
        ApplyBoundingBoxToMismatch(computed_bboxes, mismatch);
      }
    }
    PopulateNumericCheckMismatches(numeric_check, top_mismatches);
  }
  return status;
}

// Best-effort re-check of a stage-1 mismatch with excess precision disabled on
// both sides. Returns true if the re-check succeeded and matched.
bool RetryWithoutExcessPrecision(const HloModule& module,
                                 HloRunnerInterface* test_runner,
                                 absl::Span<const Literal> input_data,
                                 HloIsolationTestResult& result,
                                 const ModuleIsolationOptions& options,
                                 std::vector<std::string>& mismatch_messages) {
  HloModuleConfig exact_config = module.config();
  exact_config.mutable_debug_options().set_xla_allow_excess_precision(false);
  std::unique_ptr<HloModule> exact_module = module.Clone("exact", exact_config);
  std::unique_ptr<HloModule> exact_defused_module =
      module.Clone("exact-defused", exact_config);
  absl::Status defuse_status = DefuseModule(exact_defused_module.get());
  if (!defuse_status.ok()) {
    LOG(WARNING) << "Could not defuse module " << module.name()
                 << " with excess precision disabled: " << defuse_status;
    return false;
  }
  absl::StatusOr<Literal> exact_test_output = options.run_module_fn(
      std::move(exact_module), test_runner, input_data, {});
  absl::StatusOr<Literal> exact_defused_output = options.run_module_fn(
      std::move(exact_defused_module), test_runner, input_data, {});
  if (!exact_test_output.ok() || !exact_defused_output.ok()) {
    LOG(WARNING) << "Could not re-run module " << module.name()
                 << " with excess precision disabled: "
                 << (exact_test_output.ok() ? exact_defused_output.status()
                                            : exact_test_output.status());
    return false;
  }
  return CompareOutputs(
             module, *exact_test_output, *exact_defused_output, result, options,
             "TPU_VS_DEFUSED_TPU_NO_EXCESS_PRECISION",
             absl::StrCat(module.name(), "-exact"), mismatch_messages)
      .ok();
}

}  // namespace

std::vector<HloOutputCallback> CreateDumpHloOutputCallbacks(
    HloModule* module, std::shared_ptr<ExpectedLiteralsMap> expected_literals,
    const std::function<void(absl::string_view, Literal*)>&
        eval_literal_mutator) {
  std::vector<HloOutputCallback> reference_callbacks;
  int64_t next_id = 1000000;
  for (auto* computation : module->computations()) {
    for (auto* instruction : computation->instructions()) {
      int64_t hlo_id = next_id++;
      xla::FrontendAttributes frontend_attributes =
          instruction->frontend_attributes();
      (*frontend_attributes.mutable_map())["_xla_tag"] = std::to_string(hlo_id);
      instruction->set_frontend_attributes(frontend_attributes);

      HloOutputCallback hlo_cb;
      hlo_cb.callback_id = hlo_id;
      hlo_cb.num_operands = 1;
      hlo_cb.callback =
          [hlo_name = std::string(instruction->name()),
           module_name = std::string(module->name()), expected_literals,
           eval_literal_mutator](
              int64_t replica_id, int64_t partition_id,
              absl::Span<std::shared_ptr<const Literal> const> literals) {
            if (literals.empty() || !literals[0]) {
              LOG(ERROR) << "HloOutputCallback called with empty or null "
                            "literals for op "
                         << hlo_name << " within fusion " << module_name;
              return;
            }
            std::shared_ptr<const Literal> stored_literal = literals[0];
            if (eval_literal_mutator) {
              Literal mutated_literal = literals[0]->Clone();
              eval_literal_mutator(hlo_name, &mutated_literal);
              stored_literal =
                  std::make_shared<Literal>(std::move(mutated_literal));
            }
            if (expected_literals != nullptr) {
              (*expected_literals)[hlo_name] = stored_literal;
            }
          };
      reference_callbacks.push_back(std::move(hlo_cb));
    }
  }
  return reference_callbacks;
}

std::vector<HloOutputCallback> CreateComparisonHloOutputCallbacks(
    HloModule* test_module_clone,
    const absl::flat_hash_map<GroupKey, std::vector<std::string>>& ref_groups,
    std::shared_ptr<ExpectedLiteralsMap> expected_literals,
    const HloModule& original_module, const ModuleIsolationOptions& options,
    std::shared_ptr<absl::Mutex> result_mutex,
    HloIsolationTestResult* test_result) {
  int64_t next_hlo_id = 1;
  std::vector<xla::HloOutputCallback> dynamic_cbs;

  absl::btree_map<GroupKey, std::vector<std::string>> test_groups;
  for (const auto* computation : test_module_clone->computations()) {
    for (const auto* instruction : computation->MakeInstructionPostOrder()) {
      GroupKey key(instruction->opcode(), instruction->shape().ToString());
      test_groups[key].push_back(std::string(instruction->name()));
    }
  }

  absl::flat_hash_map<std::string, std::string> test_name_to_ref_name;
  for (const auto& [key, test_names] : test_groups) {
    auto it = ref_groups.find(key);
    if (it != ref_groups.end()) {
      const auto& ref_names = it->second;
      size_t map_size = std::min(test_names.size(), ref_names.size());
      for (size_t i = 0; i < map_size; ++i) {
        test_name_to_ref_name[test_names[i]] = ref_names[i];
      }
    }
  }

  for (auto* computation : test_module_clone->computations()) {
    for (auto* instruction : computation->instructions()) {
      int64_t hlo_id = next_hlo_id++;
      xla::FrontendAttributes frontend_attributes;
      (*frontend_attributes.mutable_map())["_xla_tag"] = std::to_string(hlo_id);
      instruction->add_frontend_attributes(frontend_attributes);

      std::string op_name(instruction->name());
      std::string ref_op_name = op_name;
      if (auto it = test_name_to_ref_name.find(op_name);
          it != test_name_to_ref_name.end()) {
        ref_op_name = it->second;
      }

      double abs_error = options.abs_error_bound;
      double rel_error = options.rel_error_bound;

      xla::HloOutputCallback dynamic_cb;
      dynamic_cb.callback_id = hlo_id;
      dynamic_cb.num_operands = 1;
      dynamic_cb.callback = [op_name, ref_op_name, expected_literals,
                             module_name = original_module.name(), abs_error,
                             rel_error, result_mutex, test_result](
                                int64_t replica_id, int64_t partition_id,
                                absl::Span<std::shared_ptr<const Literal> const>
                                    literals) {
        if (literals.empty() || !literals[0]) {
          LOG(ERROR)
              << "HloOutputCallback called with empty or null literals for "
                 "op "
              << op_name << " within fusion " << module_name;
          return;
        }

        std::shared_ptr<const Literal> expected_literal_ptr;
        if (expected_literals != nullptr) {
          absl::MutexLock lock(*result_mutex);
          auto it = expected_literals->find(ref_op_name);
          if (it != expected_literals->end()) {
            expected_literal_ptr = it->second;
            expected_literals->erase(it);
          }
        }

        if (expected_literal_ptr == nullptr) {
          LOG(WARNING) << "No reference literal found in memory for op "
                       << ref_op_name << " within fusion " << module_name;
          return;
        }

        Literal expected_literal = expected_literal_ptr->Clone();
        absl::StatusOr<Literal> actual_literal_or =
            CoerceActualLiteralToExpectedShape(*literals[0],
                                               expected_literal.shape());
        if (!actual_literal_or.ok()) {
          LOG(WARNING) << actual_literal_or.status().message() << " for op "
                       << op_name << " within fusion " << module_name;
          return;
        }
        const Literal& actual_literal = *actual_literal_or;

        absl::flat_hash_map<int64_t, numerics::debug_info::MismatchBoundingBox>
            computed_bboxes;
        xla::ErrorSpec error_spec(static_cast<float>(abs_error),
                                  static_cast<float>(rel_error));
        std::string cb_prefix = absl::StrCat(module_name, "-", op_name);
        absl::Status matched = xla::literal_comparison::Near(
            /*expected=*/expected_literal,
            /*actual=*/actual_literal,
            /*error=*/error_spec,
            /*detailed_message=*/true,
            /*miscompare_callback=*/
            MakeMiscompareCallback(cb_prefix, &computed_bboxes));
        std::optional<std::string> subnormal_flush =
            DetectSubnormalFlushInLiteral(LiteralSlice(expected_literal),
                                          LiteralSlice(actual_literal));
        if (subnormal_flush.has_value()) {
          if (matched.ok()) {
            matched = xla::literal_comparison::Near(
                /*expected=*/expected_literal,
                /*actual=*/actual_literal,
                /*error=*/xla::ErrorSpec(0, 0),
                /*detailed_message=*/true,
                /*miscompare_callback=*/
                MakeMiscompareCallback(cb_prefix, &computed_bboxes));
          }
          matched = absl::InternalError(
              absl::StrCat(matched.message(), "\n", *subnormal_flush));
        }

        if (!matched.ok()) {
          std::string error_message = absl::StrFormat(
              "FusionDebugger: Mismatch found in op \"%s\" within fusion "
              "\"%s\"\n%s",
              op_name, module_name, matched.message());
          ADD_FAILURE() << error_message;
          LOG(ERROR) << error_message;

          absl::MutexLock lock(*result_mutex);
          NumericCheck* numeric_check = test_result->add_numeric_checks();
          numeric_check->set_name(absl::StrCat("FusionDebugger:", op_name));
          numeric_check->set_expected_contains_inf_or_nan(
              LiteralContainsInfOrNan(expected_literal));
          numeric_check->set_actual_contains_inf_or_nan(
              LiteralContainsInfOrNan(actual_literal));

          absl::StatusOr<std::vector<NumericMismatch>> top_mismatches =
              ExtractTopMismatches(std::string(matched.message()),
                                   actual_literal.shape().IsTuple());
          if (top_mismatches.ok()) {
            for (NumericMismatch& mismatch : *top_mismatches) {
              ApplyBoundingBoxToMismatch(computed_bboxes, mismatch);
            }
          }
          PopulateNumericCheckMismatches(numeric_check, top_mismatches);
        }
      };

      dynamic_cbs.push_back(std::move(dynamic_cb));
    }
  }
  return dynamic_cbs;
}

void PopulateNumericCheckMismatches(
    NumericCheck* numeric_check,
    const absl::StatusOr<std::vector<NumericMismatch>>& top_mismatches) {
  if (!top_mismatches.ok()) {
    LOG(ERROR) << "Failed to extract top relative error mismatch: "
               << top_mismatches.status();
  } else if (top_mismatches->empty()) {
    LOG(ERROR) << "No top relative error mismatches found.";
  } else {
    numeric_check->clear_top_mismatches();
    for (const NumericMismatch& mismatch : *top_mismatches) {
      *numeric_check->add_top_mismatches() = mismatch;
    }
    *numeric_check->mutable_top_mismatch() = *std::max_element(
        top_mismatches->begin(), top_mismatches->end(),
        [](const NumericMismatch& a, const NumericMismatch& b) {
          return a.rel_error() < b.rel_error();
        });
  }
}

absl::StatusOr<Literal> RunModule(std::unique_ptr<HloModule> module,
                                  HloRunnerInterface* runner,
                                  absl::Span<const Literal> input_data,
                                  const RunModuleOptions& options) {
  if (!options.run_hlo_passes && !module->has_schedule()) {
    ABSL_RETURN_IF_ERROR(HloTrivialScheduler().Run(module.get()).status());
  }

  absl::FlagSaver flag_saver;
  if (options.use_fusion_debugger) {
    auto set_flag = [](absl::string_view name, absl::string_view value) {
      if (absl::CommandLineFlag* flag = absl::FindCommandLineFlag(name)) {
        std::string error;
        flag->ParseFrom(value, &error);
      }
    };
    set_flag("xla_tpu_enable_fusion_debugger", "true");
    set_flag("xla_tpu_hlo_graph_fusion_debug", "true");
    set_flag("xla_tpu_fusion_debug_use_log_callback", "true");
  }

  std::vector<HloOutputCallback> reference_callbacks;
  if (options.use_fusion_debugger && options.hlo_output_callbacks.empty()) {
    reference_callbacks = CreateDumpHloOutputCallbacks(
        module.get(), options.expected_literals, options.eval_literal_mutator);
  }

  ABSL_ASSIGN_OR_RETURN(
      std::unique_ptr<OpaqueExecutable> executable,
      runner->CreateExecutable(std::move(module), options.run_hlo_passes));

  HloRunnerInterface::ReplicatedExecuteOptions exec_options;
  exec_options.arguments.reserve(input_data.size());
  for (const auto& arg : input_data) {
    exec_options.arguments.push_back(&arg);
  }
  exec_options.hlo_output_callbacks = !reference_callbacks.empty()
                                          ? reference_callbacks
                                          : options.hlo_output_callbacks;

  auto results_or =
      runner->ExecuteReplicatedWithExecutable(executable.get(), exec_options);

  ABSL_ASSIGN_OR_RETURN(auto results, std::move(results_or));
  if (results.empty()) {
    return absl::InternalError(
        "No results returned from ExecuteReplicatedWithExecutable");
  }
  return std::move(results[0]);
}

absl::StatusOr<HloIsolationTestResult> RunIsolationTestOnModule(
    const HloModule& module, HloRunnerInterface* test_runner,
    HloRunnerInterface* reference_runner, ModuleIsolationOptions options,
    absl::Span<const Literal> input_data) {
  HloIsolationTestResult result;
  result.set_module_name(module.name());
  result.set_module_contains_constant_inf_or_nan(
      ModuleContainsConstantInfOrNan(module));
  InfNanIntentOptions intent_options;
  intent_options.reject_unconstrained_ops = options.reject_unconstrained_ops;
  result.set_is_intentional_inf_nan(
      IsIntentionalInfNan(module, intent_options));

  ABSL_RETURN_IF_ERROR(InitIsolatorOptions(options));

  std::vector<Literal> local_inputs;
  if (input_data.empty()) {
    ABSL_ASSIGN_OR_RETURN(local_inputs, options.make_fake_arguments_fn(module));
    input_data = local_inputs;
  }

  // Run a series of checks on the module. If any one of them passes, consider
  // the module to be correct immediately.
  // 1. TPU vs defused TPU
  // 2. TPU vs interpreter
  // 3. Try another input (e.g., uniform(0.1, 1))
  //   - Repeat TPU vs defused TPU
  //   - Repeat TPU vs interpreter

  auto run_module = [&](std::unique_ptr<HloModule> m, HloRunnerInterface* r,
                        absl::Span<const Literal> i,
                        const RunModuleOptions& run_opts = {}) {
    return options.run_module_fn(std::move(m), r, i, run_opts);
  };

  auto log_failure = [](absl::string_view prefix, const absl::Status& status,
                        absl::string_view module_name) {
    std::string message =
        absl::StrCat(prefix, status.ToString(), " for module: ", module_name);
    ADD_FAILURE() << message;
    LOG(ERROR) << message;
  };

  // Run test runner.
  absl::StatusOr<Literal> test_output =
      run_module(module.Clone(""), test_runner, input_data);
  if (!test_output.ok()) {
    result.set_state(State::FAILURE);
    result.set_reason("TEST_RUNNER_FAILURE");
    log_failure("Test runner failed: ", test_output.status(), module.name());
    return result;
  }
  const Literal& test_literal = *test_output;

  // Messages of all the reference checks that mismatched. Reporting is deferred
  // until the module is definitively classified as a numeric failure, because
  // any single passing check is enough to accept the module.
  std::vector<std::string> mismatch_messages;
  absl::StatusOr<Literal> defused_output(absl::UnknownError("not run"));

  // Reports every mismatch collected so far, exactly once. Must be called
  // before returning any non-SUCCESS result, so that `on_mismatch_fn` side
  // effects (notably dumping the failing module for later repro) still happen
  // when a later stage aborts with a runner error. `defused_output` is
  // preferred as the "expected" side because it is the reference that ran on
  // the test platform; when it is unavailable the test output is passed in its
  // place. Neither literal is used by any current on_mismatch_fn.
  auto report_mismatches = [&] {
    if (mismatch_messages.empty()) {
      return;
    }
    options.on_mismatch_fn(
        module, test_literal,
        defused_output.ok() ? *defused_output : test_literal,
        absl::InternalError(absl::StrJoin(mismatch_messages, "\n\n")));
    mismatch_messages.clear();
  };

  auto fail_runner = [&](absl::string_view reason, absl::string_view prefix,
                         const absl::Status& status,
                         absl::string_view module_name) {
    result.set_state(State::FAILURE);
    result.set_reason(std::string(reason));
    log_failure(prefix, status, module_name);
    report_mismatches();
    return result;
  };

  // Run defused test runner.
  //
  // Defusing materializes every value that the fusion kept in registers or in
  // VMEM, so the defused module can require orders of magnitude more HBM than
  // the fusion under test. When that (or a timeout) happens the defused
  // reference is simply unavailable for this module; it is not evidence of a
  // miscompile, so fall through to the interpreter reference instead.
  std::unique_ptr<HloModule> defused_module = module.Clone("defused");
  ABSL_RETURN_IF_ERROR(DefuseModule(defused_module.get()));
  defused_output =
      run_module(std::move(defused_module), test_runner, input_data);
  if (!defused_output.ok()) {
    if (!absl::IsResourceExhausted(defused_output.status()) &&
        !absl::IsDeadlineExceeded(defused_output.status())) {
      return fail_runner("DEFUSED_TEST_RUNNER_FAILURE",
                         "Test runner failed for defused module: ",
                         defused_output.status(), module.name());
    }
    LOG(WARNING) << "Skipping the defused reference for module "
                 << module.name()
                 << " because the defused module could not be run: "
                 << defused_output.status();
  }

  const bool inputs_have_subnormals =
      reference_runner != nullptr && InputsContainSubnormal(input_data);
  if (defused_output.ok()) {
    // Compare Test vs Defused Test.
    // When the inputs contain subnormal floats and an interpreter reference is
    // available, do not short-circuit on a matching defused run because the
    // backend may flush subnormals to zero in both fused and defused execution.
    absl::Status compare_status =
        CompareOutputs(module, test_literal, *defused_output, result, options,
                       "TPU_VS_DEFUSED_TPU", module.name(), mismatch_messages);
    if (compare_status.ok()) {
      if (!inputs_have_subnormals) {
        result.set_state(State::SUCCESS);
        result.set_reason("STAGE_1_DEFUSED_TPU_SUCCESS");
        return result;
      }
    } else if (options.retry_without_excess_precision &&
               module.config().debug_options().xla_allow_excess_precision() &&
               RetryWithoutExcessPrecision(module, test_runner, input_data,
                                           result, options,
                                           mismatch_messages) &&
               !inputs_have_subnormals) {
      result.set_state(State::SUCCESS);
      result.set_reason("STAGE_1B_NO_EXCESS_PRECISION_SUCCESS");
      return result;
    }
  }

  // Potentially skip reference run.
  if (options.max_module_size_bytes > 0) {
    int64_t size = options.estimate_module_size_fn(module);
    if (size > options.max_module_size_bytes) {
      LOG(INFO) << "Skipping reference run for module: " << module.name()
                << " due to large size: " << size;
      reference_runner = nullptr;
    }
  }
  if (reference_runner != nullptr && ModuleContainsLargeKeyValueSort(module)) {
    LOG(INFO) << "Skipping reference run for module: " << module.name()
              << " due to large key value sort";
    reference_runner = nullptr;
  }

  // The defused run exhausted device memory (or timed out), which means the
  // module's materialized intermediates do not fit. The interpreter reference
  // materializes those same intermediates on the host and is orders of
  // magnitude slower, so escalating to it does not produce an answer -- it
  // just turns a fast failure into a test timeout. Observed on
  // broadcast_select_fusion.87 (b/524252856): the defused module needs 834 GB
  // against 94.74 GB of HBM, and the interpreter then spent >27 minutes inside
  // HloEvaluator::HandleBroadcast before the shard was killed.
  //
  // Report the module as unverified rather than guessing.
  //
  // TODO(b/524252856): with a pre-flight peak-memory estimate for the defused
  // module we could tell "too big for HBM but fine on the host" apart from
  // "too big for anything" and still use the interpreter for the former.
  if (!defused_output.ok()) {
    LOG(WARNING) << "No usable reference for module: " << module.name();
    result.set_state(State::SKIPPED);
    result.set_reason("DEFUSED_REFERENCE_UNAVAILABLE");
    return result;
  }

  if (reference_runner) {
    std::unique_ptr<HloModule> despecialized_module =
        module.Clone("despecialized");
    Despecializer despecializer;
    ABSL_RETURN_IF_ERROR(
        despecializer.Run(despecialized_module.get()).status());
    std::string despecialized_module_name = despecialized_module->name();

    // Run the reference runner.
    absl::StatusOr<Literal> reference_output = run_module(
        std::move(despecialized_module), reference_runner, input_data);
    if (!reference_output.ok()) {
      return fail_runner("REFERENCE_RUNNER_FAILURE",
                         "Reference runner failed: ", reference_output.status(),
                         despecialized_module_name);
    }

    // Compare Test vs Reference.
    absl::Status compare_status =
        CompareOutputs(module, test_literal, *reference_output, result, options,
                       "TPU_VS_INTERPRETER", module.name(), mismatch_messages);
    if (compare_status.ok()) {
      result.set_state(State::SUCCESS);
      result.set_reason("STAGE_2_INTERPRETER_SUCCESS");
      return result;
    }

    // If there was a mismatch then we should re-run the failing HLO module
    // with the fusion debugger enabled - to isolate the mismatch.
    // Run the reference runner again with despecialization to dump reference
    // binaries.
    std::unique_ptr<HloModule> debug_despecialized_module =
        module.Clone("despecialized");
    ABSL_RETURN_IF_ERROR(
        despecializer.Run(debug_despecialized_module.get()).status());
    std::string debug_despecialized_module_name =
        debug_despecialized_module->name();

    using GroupKey = std::pair<HloOpcode, std::string>;
    absl::flat_hash_map<GroupKey, std::vector<std::string>> ref_groups;
    for (const auto* computation : debug_despecialized_module->computations()) {
      for (const auto* instruction : computation->MakeInstructionPostOrder()) {
        GroupKey key(instruction->opcode(), instruction->shape().ToString());
        ref_groups[key].push_back(std::string(instruction->name()));
      }
    }

    auto expected_literals = std::make_shared<ExpectedLiteralsMap>();

    RunModuleOptions reference_opts;
    reference_opts.use_fusion_debugger = true;
    reference_opts.expected_literals = expected_literals;
    absl::StatusOr<Literal> debug_reference_output =
        run_module(std::move(debug_despecialized_module), reference_runner,
                   input_data, reference_opts);
    if (!debug_reference_output.ok()) {
      return fail_runner(
          "REFERENCE_RUNNER_FAILURE",
          "Reference runner failed (with fusion debugger enabled): ",
          debug_reference_output.status(), debug_despecialized_module_name);
    }

    std::shared_ptr<absl::Mutex> result_mutex = std::make_shared<absl::Mutex>();
    HloIsolationTestResult* test_result = &result;

    std::unique_ptr<HloModule> test_module_clone = module.Clone("");

    std::vector<xla::HloOutputCallback> dynamic_cbs =
        CreateComparisonHloOutputCallbacks(test_module_clone.get(), ref_groups,
                                           expected_literals, module, options,
                                           result_mutex, test_result);

    RunModuleOptions retry_opts;
    retry_opts.hlo_output_callbacks = dynamic_cbs;
    retry_opts.use_fusion_debugger = true;
    retry_opts.expected_literals = expected_literals;
    absl::StatusOr<Literal> retry_test_output = run_module(
        std::move(test_module_clone), test_runner, input_data, retry_opts);
    if (!retry_test_output.ok()) {
      return fail_runner("TEST_RUNNER_FAILURE_ON_RETRY",
                         "Test runner failed on retry (with fusion debugger): ",
                         retry_test_output.status(), module.name());
    }
  }

  result.set_state(State::FAILURE);
  result.set_reason("NUMERIC_MISMATCH");

  // Every reference check disagreed with the module under test.
  report_mismatches();

  std::vector<numerics::debug_info::MismatchDetails> all_mismatch_details =
      ExtractMismatchDetails(module, result);
  if (!all_mismatch_details.empty()) {
    auto html_path_or =
        numerics::debug_info::DumpHloModuleMismatchWithGraphData(
            module, all_mismatch_details,
            absl::StrCat("failed-module-", module.name(), ".html"));
    if (html_path_or.ok()) {
      LOG(INFO) << "Wrote failed HLO module HTML to " << *html_path_or;
    }
  }

  return result;
}

absl::StatusOr<std::vector<HloIsolationTestResult>> RunIsolationPipeline(
    const HloModule& input_module, HloRunnerInterface* test_runner,
    HloRunnerInterface* reference_runner, PipelineIsolationOptions options) {
  ABSL_RETURN_IF_ERROR(ValidatePipelineOptions(options));
  ABSL_RETURN_IF_ERROR(InitIsolatorOptions(options.module_options));

  ABSL_ASSIGN_OR_RETURN(
      std::vector<std::unique_ptr<HloModule>> modules,
      DecomposeHloModule(input_module, /*deduplicate_modules=*/true));

  // Sort submodules by name and fingerprint to ensure deterministic sharding.
  std::sort(modules.begin(), modules.end(),
            [](const std::unique_ptr<HloModule>& a,
               const std::unique_ptr<HloModule>& b) {
              if (a->name() != b->name()) {
                return a->name() < b->name();
              }
              return a->GetFingerprint128() < b->GetFingerprint128();
            });

  std::vector<HloIsolationTestResult> pipeline_results;
  int64_t filtered_module_index = 0;
  for (int i = 0; i < modules.size(); ++i) {
    auto& module = modules[i];

    bool is_filtered = false;
    std::string skip_reason;
    if (!options.filter_by_name.empty() &&
        !RE2::FullMatch(module->name(), options.filter_by_name)) {
      is_filtered = true;
      skip_reason = "NO_MATCH_FILTER_BY_NAME";
    } else if (!options.skip_by_name.empty() &&
               RE2::FullMatch(module->name(), options.skip_by_name)) {
      is_filtered = true;
      skip_reason = "MATCH_SKIP_BY_NAME";
    } else {
      bool has_matching_opcode = false;
      bool has_skipped_opcode = false;
      for (const auto* computation : module->computations()) {
        for (const auto* instruction : computation->instructions()) {
          std::string opcode_str(HloOpcodeString(instruction->opcode()));
          if (options.filter_by_opcode.empty() ||
              RE2::FullMatch(opcode_str, options.filter_by_opcode)) {
            has_matching_opcode = true;
          }
          if (!options.skip_by_opcode.empty() &&
              RE2::FullMatch(opcode_str, options.skip_by_opcode)) {
            has_skipped_opcode = true;
          }
        }
      }
      if (!has_matching_opcode || has_skipped_opcode) {
        is_filtered = true;
        skip_reason = "NO_MATCH_FILTER_BY_OPCODE";
        if (has_skipped_opcode) {
          skip_reason = "MATCH_SKIP_BY_OPCODE";
        }
      }
    }

    if (is_filtered) {
      LOG(INFO) << "Module " << module->name() << " skipped: " << skip_reason;
      HloIsolationTestResult skipped_result;
      skipped_result.set_module_name(module->name());
      skipped_result.set_state(State::SKIPPED);
      skipped_result.set_reason(skip_reason);
      pipeline_results.push_back(std::move(skipped_result));
      WriteResults(pipeline_results);
      continue;
    }

    // Sharding check happens after all filters are cleared
    if (options.shard_index >= 0 && options.num_shards > 0) {
      if (filtered_module_index % options.num_shards != options.shard_index) {
        filtered_module_index++;
        continue;
      }
      filtered_module_index++;
    }

    // Execute module
    absl::StatusOr<HloIsolationTestResult> result_or = RunIsolationTestOnModule(
        *module, test_runner, reference_runner, options.module_options);
    if (!result_or.ok()) {
      LOG(ERROR) << "Failed to run isolation test on module " << module->name()
                 << ": " << result_or.status();
      HloIsolationTestResult failed_result;
      failed_result.set_module_name(module->name());
      failed_result.set_state(State::FAILURE);
      failed_result.set_reason(result_or.status().message());
      pipeline_results.push_back(std::move(failed_result));
      WriteResults(pipeline_results);
      continue;
    }
    HloIsolationTestResult main_result = std::move(*result_or);
    std::vector<HloIsolationTestResult> fusion_debug_results;

    // Unbundle Fusion Debugger checks into standalone test results.
    // When the TPU fusion debugger identifies failing sub-instructions within a
    // fusion, it attaches `FusionDebugger:<op_name>` checks to `main_result`.
    // To ensure CI dashboards report each failing sub-instruction as an
    // independent test failure, we extract them into standalone test results
    // named `<module_name>-<op_name>` while preserving standard checks in
    // `main_result`.
    auto* mutable_checks = main_result.mutable_numeric_checks();
    google::protobuf::RepeatedPtrField<NumericCheck> module_level_checks;
    for (int j = 0; j < mutable_checks->size(); ++j) {
      NumericCheck* check = mutable_checks->Mutable(j);
      absl::string_view check_name = check->name();
      if (absl::StartsWith(check_name, "FusionDebugger:")) {
        std::string op_name(absl::StripPrefix(check_name, "FusionDebugger:"));
        HloIsolationTestResult fusion_result;
        fusion_result.set_module_name(
            absl::StrCat(main_result.module_name(), "-", op_name));
        fusion_result.set_state(State::FAILURE);
        fusion_result.set_reason("NUMERIC_MISMATCH");
        if (main_result.has_shard_index()) {
          fusion_result.set_shard_index(main_result.shard_index());
        }
        if (main_result.has_module_contains_constant_inf_or_nan()) {
          fusion_result.set_module_contains_constant_inf_or_nan(
              main_result.module_contains_constant_inf_or_nan());
        }
        if (main_result.has_is_intentional_inf_nan()) {
          fusion_result.set_is_intentional_inf_nan(
              main_result.is_intentional_inf_nan());
        }

        NumericCheck* new_check = fusion_result.add_numeric_checks();
        *new_check = std::move(*check);
        new_check->set_name("TPU_VS_INTERPRETER");

        fusion_debug_results.push_back(std::move(fusion_result));
      } else {
        *module_level_checks.Add() = std::move(*check);
      }
    }
    mutable_checks->Swap(&module_level_checks);

    pipeline_results.push_back(std::move(main_result));
    for (auto&& f_result : fusion_debug_results) {
      pipeline_results.push_back(std::move(f_result));
    }
    WriteResults(pipeline_results);
  }

  return pipeline_results;
}

absl::StatusOr<std::vector<HloIsolationTestResult>> RunIsolationPipeline(
    const std::string& input_path, HloRunnerInterface* test_runner,
    HloRunnerInterface* reference_runner, PipelineIsolationOptions options) {
  ABSL_ASSIGN_OR_RETURN(std::unique_ptr<HloModule> loaded_module,
                        LoadModuleFromFile(input_path));
  return RunIsolationPipeline(*loaded_module, test_runner, reference_runner,
                              options);
}

absl::Status DefuseModule(HloModule* module) {
  HloPassPipeline pipeline("defuser");
  pipeline.AddPass<HloDescheduler>();
  pipeline.AddPass<Defuser>();
  pipeline.AddPass<HloTrivialScheduler>();
  return pipeline.Run(module).status();
}

absl::StatusOr<NumericMismatch> ParseMismatchLine(absl::string_view line) {
  std::string actual_str, expected_str, index_str, rel_error_str, abs_error_str;
  if (RE2::PartialMatch(
          line,
          R"(actual\s+([^,]+),\s+expected\s+([^,]+),\s+index\s+(.+?),\s+rel error\s+([^,]+),\s+abs error\s+(.+))",
          &actual_str, &expected_str, &index_str, &rel_error_str,
          &abs_error_str)) {
    double actual_double, expected_double, rel_error_double;
    if (!absl::SimpleAtod(actual_str, &actual_double) ||
        !absl::SimpleAtod(expected_str, &expected_double) ||
        !absl::SimpleAtod(rel_error_str, &rel_error_double)) {
      return absl::InvalidArgumentError(
          absl::StrCat("Failed to parse numeric values from line: ", line));
    }
    NumericMismatch data;
    data.set_actual(actual_double);
    data.set_expected(expected_double);
    data.set_rel_error(rel_error_double);
    std::string clean_indices =
        absl::StrReplaceAll(index_str, {{"{", ""}, {"}", ""}, {" ", ""}});
    for (absl::string_view idx_part :
         absl::StrSplit(clean_indices, ',', absl::SkipEmpty())) {
      int64_t coord;
      if (absl::SimpleAtoi(idx_part, &coord)) {
        data.add_top_mismatch_index(coord);
      }
    }
    return data;
  }
  return absl::InvalidArgumentError(
      absl::StrCat("Failed to match line: ", line));
}

absl::StatusOr<std::vector<NumericMismatch>> ExtractTopMismatches(
    std::string error_message, bool is_tuple) {
  std::stringstream ss(error_message);
  std::string line;
  std::vector<NumericMismatch> mismatches;
  std::optional<int64_t> shape_index;
  if (!is_tuple) {
    shape_index = 0;
  }
  std::optional<NumericMismatch> current_mismatch;
  std::optional<double> parsed_mismatch_percentage;

  bool parsed_abs_percentage = false;
  bool parsed_rel_percentage = false;

  while (std::getline(ss, line)) {
    if (!shape_index.has_value()) {
      std::string parsed_shape_index_str;
      if (RE2::PartialMatch(line, R"(Array at shape index\s*\{\s*(\d+))",
                            &parsed_shape_index_str)) {
        int64_t idx;
        if (!absl::SimpleAtoi(parsed_shape_index_str, &idx)) {
          return absl::InvalidArgumentError(
              absl::StrCat("Failed to parse shape index from line: ", line));
        }
        shape_index = idx;
        continue;
      }
    }

    if (!parsed_mismatch_percentage.has_value()) {
      std::string parsed_mismatch_percentage_str;
      if (RE2::PartialMatch(line, R"(Mismatch count\s*\d+\s*\(([^%]+)%\))",
                            &parsed_mismatch_percentage_str)) {
        double percentage;
        if (!absl::SimpleAtod(parsed_mismatch_percentage_str, &percentage)) {
          return absl::InvalidArgumentError(absl::StrCat(
              "Failed to parse mismatch percentage from line: ", line));
        }
        parsed_mismatch_percentage = percentage;
        continue;
      }
    }

    absl::StatusOr<NumericMismatch> parsed = ParseMismatchLine(line);
    if (!current_mismatch.has_value() && parsed.ok()) {
      CHECK(shape_index.has_value());
      parsed->set_output_shape_index(*shape_index);
      current_mismatch = std::move(*parsed);
      current_mismatch->set_percentage_of_elems_exceeding_both_errors(
          parsed_mismatch_percentage.value_or(0.0));
      continue;
    }

    std::string percentage_str;
    if (!parsed_abs_percentage &&
        RE2::PartialMatch(
            line,
            R"(Elements exceeding abs error bound[^:]*:\s*\d+\s*\(([^%]+)%\))",
            &percentage_str)) {
      CHECK(current_mismatch.has_value());
      double percentage;
      if (absl::SimpleAtod(percentage_str, &percentage)) {
        current_mismatch->set_percentage_of_elems_exceeding_abs_error(
            percentage);
        parsed_abs_percentage = true;
      }
    } else if (
        !parsed_rel_percentage &&
        RE2::PartialMatch(
            line,
            R"(Elements exceeding rel error bound[^:]*:\s*\d+\s*\(([^%]+)%\))",
            &percentage_str)) {
      CHECK(current_mismatch.has_value());
      double percentage;
      if (absl::SimpleAtod(percentage_str, &percentage)) {
        current_mismatch->set_percentage_of_elems_exceeding_rel_error(
            percentage);
        parsed_rel_percentage = true;
      }
    }

    if (current_mismatch.has_value() && parsed_abs_percentage &&
        parsed_rel_percentage && parsed_mismatch_percentage.has_value()) {
      mismatches.push_back(std::move(*current_mismatch));
      parsed_abs_percentage = false;
      parsed_rel_percentage = false;
      parsed_mismatch_percentage = std::nullopt;
      shape_index = std::nullopt;
      current_mismatch = std::nullopt;
    }
  }
  if (current_mismatch.has_value()) {
    mismatches.push_back(std::move(*current_mismatch));
  }
  if (mismatches.empty()) {
    std::string index_str, expected_str, actual_str;
    if (RE2::PartialMatch(
            error_message,
            R"(first mismatch at array index\s*\{([^}]*)\}:\s*expected value:\s*([^\s]+)\s*actual value:\s*([^\s]+))",
            &index_str, &expected_str, &actual_str)) {
      double expected_double = 0.0;
      double actual_double = 0.0;
      if (absl::SimpleAtod(expected_str, &expected_double) &&
          absl::SimpleAtod(actual_str, &actual_double)) {
        NumericMismatch data;
        data.set_output_shape_index(shape_index.value_or(0));
        data.set_expected(expected_double);
        data.set_actual(actual_double);
        const double abs_diff = std::abs(actual_double - expected_double);
        const double rel_err = expected_double != 0.0
                                   ? abs_diff / std::abs(expected_double)
                                   : (actual_double != 0.0 ? 1.0 : 0.0);
        data.set_rel_error(rel_err);
        for (absl::string_view idx_part :
             absl::StrSplit(index_str, ',', absl::SkipEmpty())) {
          int64_t coord;
          if (absl::SimpleAtoi(absl::StripAsciiWhitespace(idx_part), &coord)) {
            data.add_top_mismatch_index(coord);
          }
        }
        mismatches.push_back(std::move(data));
      }
    }
  }
  return mismatches;
}

absl::StatusOr<NumericMismatch> ExtractTopRelativeErrorMismatch(
    std::string error_message) {
  ABSL_ASSIGN_OR_RETURN(std::vector<NumericMismatch> mismatches,
                        ExtractTopMismatches(error_message, false));
  if (mismatches.empty()) {
    return absl::NotFoundError(
        "Could not find top relative error mismatch in the error message.");
  }
  NumericMismatch top_relative_error_mismatch = mismatches.front();
  for (const auto& mismatch : mismatches) {
    if (mismatch.rel_error() > top_relative_error_mismatch.rel_error()) {
      top_relative_error_mismatch = mismatch;
    }
  }
  return top_relative_error_mismatch;
}

absl::StatusOr<std::vector<bool>> DetectReducesInModuleOutput(
    const HloModule* module) {
  const Shape& output_shape = module->result_shape();
  int64_t num_outputs = 1;
  if (output_shape.IsTuple()) {
    num_outputs = output_shape.tuple_shapes().size();
  }
  std::vector<bool> reduce_in_output(num_outputs, false);
  std::unique_ptr<HloModule> defused_module = module->Clone("defused");
  ABSL_RETURN_IF_ERROR(DefuseModule(defused_module.get()));

  auto bfs = [&reduce_in_output](HloModule* module,
                                 int64_t output_index) -> void {
    absl::flat_hash_set<const HloInstruction*> visited;
    std::queue<const HloInstruction*> q;
    if (module->result_shape().IsTuple()) {
      if (module->entry_computation()->root_instruction()->operands().size() >
          output_index) {
        q.push(module->entry_computation()->root_instruction()->operand(
            output_index));
      }
    } else {
      q.push(module->entry_computation()->root_instruction());
    }
    while (!q.empty()) {
      const HloInstruction* current = q.front();
      q.pop();
      if (visited.contains(current)) {
        continue;
      }
      visited.insert(current);
      if (current->opcode() == HloOpcode::kReduce) {
        reduce_in_output[output_index] = true;
      }
      for (const HloInstruction* operand : current->operands()) {
        if (operand->opcode() == HloOpcode::kGetTupleElement) {
          int64_t tuple_index = operand->tuple_index();
          const HloInstruction* tuple = operand->operand(0);
          if (tuple->operands().size() > tuple_index) {
            const HloInstruction* tuple_element = tuple->operand(tuple_index);
            visited.insert(tuple);
            visited.insert(operand);
            q.push(tuple_element);
          }
        } else {
          q.push(operand);
        }
      }
    }
  };

  for (int64_t i = 0; i < num_outputs; ++i) {
    bfs(defused_module.get(), i);
  }
  return reduce_in_output;
}

namespace {

numerics::debug_info::MismatchDetails ConvertToMismatchDetails(
    absl::string_view target_instruction_name, const NumericMismatch& m,
    bool is_tuple) {
  numerics::debug_info::MismatchDetails details;
  details.target_instruction_name = std::string(target_instruction_name);
  if (is_tuple && m.has_output_shape_index()) {
    details.output_shape_index = m.output_shape_index();
  }
  details.actual = m.actual();
  details.expected = m.expected();
  details.rel_error = m.rel_error();
  if (m.has_percentage_of_elems_exceeding_abs_error()) {
    details.percentage_of_elems_exceeding_abs_error =
        m.percentage_of_elems_exceeding_abs_error();
  }
  if (m.has_percentage_of_elems_exceeding_rel_error()) {
    details.percentage_of_elems_exceeding_rel_error =
        m.percentage_of_elems_exceeding_rel_error();
  }
  if (m.has_percentage_of_elems_exceeding_both_errors()) {
    details.percentage_of_elems_exceeding_both_errors =
        m.percentage_of_elems_exceeding_both_errors();
  }
  if (m.has_result_of_reduce()) {
    details.result_of_reduce = m.result_of_reduce();
  }
  if (!m.tensor_dimensions().empty() || !m.top_mismatch_index().empty()) {
    numerics::debug_info::MismatchBoundingBox bbox;
    bbox.tensor_shape.assign(m.tensor_dimensions().begin(),
                             m.tensor_dimensions().end());
    bbox.box_min.assign(m.mismatch_box_min().begin(),
                        m.mismatch_box_min().end());
    bbox.box_max.assign(m.mismatch_box_max().begin(),
                        m.mismatch_box_max().end());
    if (!m.top_mismatch_index().empty()) {
      bbox.top_mismatch_coords.push_back(std::vector<int64_t>(
          m.top_mismatch_index().begin(), m.top_mismatch_index().end()));
    }
    bbox.mismatch_count = m.mismatch_count();
    bbox.total_elements = m.total_elements();
    details.bounding_box = std::move(bbox);
  }
  return details;
}

}  // namespace

std::vector<numerics::debug_info::MismatchDetails> ExtractMismatchDetails(
    const HloModule& module, const HloIsolationTestResult& result) {
  std::vector<numerics::debug_info::MismatchDetails> all_details;
  bool is_tuple = module.result_shape().IsTuple();

  // 1. Extract parent fusion mismatch details from primary parent check
  // (prefer TPU_VS_INTERPRETER, fallback to TPU_VS_DEFUSED_TPU).
  const NumericCheck* parent_check = nullptr;
  for (const auto& check : result.numeric_checks()) {
    if (check.name() == "TPU_VS_INTERPRETER") {
      parent_check = &check;
      break;
    }
    if (check.name() == "TPU_VS_DEFUSED_TPU" && parent_check == nullptr) {
      parent_check = &check;
    }
  }

  std::string root_name;
  if (const HloComputation* entry = module.entry_computation()) {
    if (const HloInstruction* root = entry->root_instruction()) {
      root_name = root->name();
    }
  }

  if (parent_check != nullptr && !root_name.empty()) {
    for (const auto& m : parent_check->top_mismatches()) {
      all_details.push_back(ConvertToMismatchDetails(root_name, m, is_tuple));
    }
  }

  // 2. Extract Fusion Debugger sub-instruction mismatches.
  absl::flat_hash_map<std::string, const HloInstruction*> name_to_instr;
  for (const HloComputation* comp : module.computations()) {
    for (const HloInstruction* instr : comp->instructions()) {
      name_to_instr[instr->name()] = instr;
    }
  }

  for (const auto& check : result.numeric_checks()) {
    if (absl::StartsWith(check.name(), "FusionDebugger:")) {
      std::string op_name(absl::StripPrefix(check.name(), "FusionDebugger:"));
      bool is_op_tuple = false;
      if (auto it = name_to_instr.find(op_name); it != name_to_instr.end()) {
        is_op_tuple = it->second->shape().IsTuple();
      }
      for (const auto& m : check.top_mismatches()) {
        all_details.push_back(
            ConvertToMismatchDetails(op_name, m, is_op_tuple));
      }
    }
  }

  return all_details;
}

absl::StatusOr<std::vector<NumericMismatch>> ExtractAndEnrichTopMismatches(
    std::string error_message, const HloModule* module) {
  bool is_tuple = module->result_shape().IsTuple();
  int64_t num_outputs =
      is_tuple ? module->result_shape().tuple_shapes().size() : 1;

  ABSL_ASSIGN_OR_RETURN(std::vector<NumericMismatch> mismatches,
                        ExtractTopMismatches(error_message, is_tuple));
  ABSL_ASSIGN_OR_RETURN(std::vector<bool> reduce_in_output,
                        DetectReducesInModuleOutput(module));
  for (NumericMismatch& mismatch : mismatches) {
    int output_index = mismatch.output_shape_index();
    if (output_index >= num_outputs) {
      return absl::InternalError(
          absl::StrCat("Invalid output index: ", output_index));
    }
    mismatch.set_result_of_reduce(reduce_in_output[output_index]);
  }
  return mismatches;
}

int64_t GetFusionCountInNestedFusion(const HloInstruction* fusion_instr) {
  int64_t num_fusions = 0;
  if (fusion_instr->IsOutputFusion() || fusion_instr->IsLoopFusion()) {
    for (auto* instr :
         fusion_instr->fused_instructions_computation()->instructions()) {
      if (instr->IsOutputFusion() || instr->IsLoopFusion()) {
        auto cur_count = GetFusionCountInNestedFusion(instr);
        if (cur_count > num_fusions) {
          num_fusions = cur_count;
        }
      }
    }
  }
  if (fusion_instr->IsLoopFusion() || fusion_instr->IsOutputFusion()) {
    num_fusions += 1;
  }
  return num_fusions;
}

bool ModuleContainsLargeKeyValueSort(const HloModule& module) {
  for (const HloComputation* computation : module.computations()) {
    for (const HloInstruction* instruction : computation->instructions()) {
      if (instruction->opcode() == HloOpcode::kSort &&
          instruction->operand_count() > 1 &&
          instruction->operand(0)->shape().element_type() ==
              PrimitiveType::BF16 &&
          ShapeUtil::ElementsIn(instruction->operand(0)->shape()) >=
              (1 << 14)) {
        return true;
      }
    }
  }
  return false;
}

bool ModuleTestsFloatsForEquality(const HloModule& module) {
  for (const HloComputation* computation : module.computations()) {
    for (const auto* instruction : computation->instructions()) {
      if (instruction->opcode() == HloOpcode::kCompare &&
          (instruction->comparison_direction() == ComparisonDirection::kEq ||
           instruction->comparison_direction() == ComparisonDirection::kNe) &&
          ShapeUtil::ElementIsFloating(instruction->operand(0)->shape())) {
        return true;
      }
    }
  }
  return false;
}

bool ComputationHasRng(const HloComputation* computation) {
  for (const HloInstruction* instruction :
       computation->MakeInstructionPostOrder()) {
    if (instruction->opcode() == HloOpcode::kRng) {
      return true;
    }
  }
  return false;
}

}  // namespace hlo_isolation
}  // namespace xla
