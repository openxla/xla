/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/python/ifrt/remap_plan.h"

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/call_once.h"
#include "absl/base/optimization.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/hash/hash.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "xla/pjrt/pjrt_layout.h"
#include "xla/python/ifrt/array.h"
#include "xla/python/ifrt/array_spec.h"
#include "xla/python/ifrt/client.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/device_list.h"
#include "xla/python/ifrt/remap_plan.pb.h"
#include "xla/python/ifrt/serdes_version.h"
#include "xla/python/ifrt/shape.h"
#include "xla/python/ifrt/sharding.h"
#include "xla/status_macros.h"

namespace xla {
namespace ifrt {

namespace {

absl::StatusOr<RemapPlan::InputDeviceRange> InputDeviceRangeFromProto(
    Client* client, const RemapPlanProto::InputDevices& proto) {
  RemapPlan::InputDeviceRange range;
  range.in_array = proto.in_array();
  ABSL_ASSIGN_OR_RETURN(range.input_devices,
                   DeviceList::FromProto(client, proto.device_list()));
  return range;
}

void InputDeviceToOutputToProto(
    SerDesVersion version, int out_array,
    absl::Span<const RemapPlan::InputDeviceRange> input_devices,
    RemapPlanProto::InputDevicesForOutput& proto) {
  proto.set_out_array(out_array);
  for (const RemapPlan::InputDeviceRange& input : input_devices) {
    RemapPlanProto::InputDevices* input_proto = proto.add_input_devices();
    input_proto->set_in_array(input.in_array);
    input.input_devices->ToProto(*input_proto->mutable_device_list(), version);
  }
}

// Verifies that [start, end) with `step` addresses shard indices within [0,
// num_shards).
absl::Status CheckInterval(int64_t num_shards, int64_t start, int64_t end,
                           int64_t step) {
  if (step <= 0) {
    return absl::InvalidArgumentError(
        absl::StrFormat("step must be positive, but is %d", step));
  }
  if (start < 0 || start >= num_shards) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "start must be in [0, %d), but is %d", num_shards, start));
  }
  if (end < start || end > num_shards) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "end must be in [%d, %d], but is %d", start, num_shards, end));
  }
  if (start < end) {
    const int64_t last_index = end - 1 - (end - 1 - start) % step;
    if (last_index >= num_shards) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "interval addresses shard %d, which is out of range [0, %d)",
          last_index, num_shards));
    }
  }
  return absl::OkStatus();
}

// Converts MappingProto entries directly into `input_devices_for_output_map`
// for backward compatibility with older serialized RemapPlanProtos.
absl::StatusOr<
    absl::flat_hash_map<int, std::vector<RemapPlan::InputDeviceRange>>>
InputDevicesForOutputMapFromMappings(Client* client,
                                     absl::Span<const ArraySpec> input_specs,
                                     absl::Span<const ArraySpec> output_specs,
                                     const RemapPlanProto& proto) {
  struct InputDevicesBuilder {
    int in_array;
    std::vector<Device*> devices;
  };
  absl::flat_hash_map<int, std::vector<InputDevicesBuilder>> output_to_inputs;
  absl::flat_hash_map<int, absl::flat_hash_map<int, int>> out_to_in_to_idx;

  for (int64_t i = 0; i < proto.mappings_size(); ++i) {
    const RemapPlanProto::MappingProto& mapping = proto.mappings(i);
    if (mapping.in_array() < 0 || mapping.in_array() >= input_specs.size()) {
      return absl::InvalidArgumentError(
          absl::StrFormat("mappings[%d].in_array must be in [0, %d], but is %d",
                          i, input_specs.size() - 1, mapping.in_array()));
    }
    if (mapping.out_array() < 0 || mapping.out_array() >= output_specs.size()) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "mappings[%d].out_array must be in [0, %d], but is %d", i,
          output_specs.size() - 1, mapping.out_array()));
    }

    const int64_t num_intervals = mapping.from_start_size();
    TF_RET_CHECK(mapping.from_end_size() == num_intervals);
    TF_RET_CHECK(mapping.from_step_size() == num_intervals);
    TF_RET_CHECK(mapping.to_start_size() == num_intervals);
    TF_RET_CHECK(mapping.to_end_size() == num_intervals);
    TF_RET_CHECK(mapping.to_step_size() == num_intervals);

    const DeviceListRef& in_devices =
        input_specs[mapping.in_array()].sharding->devices();
    const int64_t in_shards_count = in_devices->AddressableDeviceList()->size();
    const int64_t out_shards_count = output_specs[mapping.out_array()]
                                         .sharding->devices()
                                         ->AddressableDeviceList()
                                         ->size();

    auto& in_to_idx = out_to_in_to_idx[mapping.out_array()];
    auto it = in_to_idx.find(mapping.in_array());
    if (it == in_to_idx.end()) {
      const int idx = output_to_inputs[mapping.out_array()].size();
      InputDevicesBuilder builder;
      builder.in_array = mapping.in_array();
      output_to_inputs[mapping.out_array()].push_back(std::move(builder));
      in_to_idx.emplace(mapping.in_array(), idx);
      it = in_to_idx.find(mapping.in_array());
    }
    std::vector<Device*>& collected_devices =
        output_to_inputs[mapping.out_array()][it->second].devices;

    for (int s = 0; s < num_intervals; ++s) {
      const int64_t from_start = mapping.from_start(s);
      const int64_t from_end = mapping.from_end(s);
      const int64_t from_step = mapping.from_step(s);
      const int64_t to_start = mapping.to_start(s);
      const int64_t to_end = mapping.to_end(s);
      const int64_t to_step = mapping.to_step(s);

      ABSL_RETURN_IF_ERROR(
          CheckInterval(in_shards_count, from_start, from_end, from_step));
      ABSL_RETURN_IF_ERROR(
          CheckInterval(out_shards_count, to_start, to_end, to_step));

      const int64_t from_steps =
          (from_end - from_start + from_step - 1) / from_step;
      const int64_t to_steps = (to_end - to_start + to_step - 1) / to_step;
      if (from_steps != to_steps) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "mappings[%d].from[%d] and mappings[%d].to[%d] must have the "
            "same number of steps, but were %d and %d",
            i, s, i, s, from_steps, to_steps));
      }

      int64_t index = from_start;
      while (index < from_end) {
        TF_RET_CHECK(index >= 0 && index < in_devices->size());
        collected_devices.push_back(in_devices->devices()[index]);
        index += from_step;
      }
    }
  }

  absl::flat_hash_map<int, std::vector<RemapPlan::InputDeviceRange>>
      input_devices_for_output_map;
  for (auto& [out_array, builders] :
       // NOLINTNEXTLINE(*-custom-deterministic-iteration-order)
       output_to_inputs) {
    TF_RET_CHECK(out_array >= 0 && out_array < output_specs.size());
    const DeviceListRef& out_devices =
        output_specs[out_array].sharding->devices();
    std::vector<RemapPlan::InputDeviceRange>& ranges =
        input_devices_for_output_map[out_array];
    ranges.reserve(builders.size());

    for (auto& builder : builders) {
      TF_RET_CHECK(builder.in_array >= 0 &&
                   builder.in_array < input_specs.size());
      const DeviceListRef& in_devices =
          input_specs[builder.in_array].sharding->devices();
      TF_RET_CHECK(builder.devices.size() <= out_devices->size());
      TF_RET_CHECK(builder.devices.size() <= in_devices->size());

      DeviceListRef interval_device_list;
      if (builder.devices.size() == in_devices->size() &&
          absl::c_equal(builder.devices, in_devices->devices())) {
        interval_device_list = in_devices;
      } else if (builder.devices.size() == out_devices->size() &&
                 absl::c_equal(builder.devices, out_devices->devices())) {
        interval_device_list = out_devices;
      } else {
        ABSL_ASSIGN_OR_RETURN(interval_device_list,
                         client->MakeDeviceList(std::move(builder.devices)));
      }
      ranges.push_back({builder.in_array, std::move(interval_device_list)});
    }
  }
  return input_devices_for_output_map;
}

bool CheckOneInputForOneOutput(const xla::ifrt::RemapPlan& plan) {
  for (const auto& [out_array, inputs] :
       // NOLINTNEXTLINE(*-custom-deterministic-iteration-order)
       plan.input_devices_for_output_map()) {
    int first_in_array = -1;
    for (const auto& input : inputs) {
      if (first_in_array == -1) {
        first_in_array = input.in_array;
      } else if (first_in_array != input.in_array) {
        return false;
      }
    }
  }
  return true;
}

// Validates array-level consistency between an input array spec and an output
// array spec.
absl::Status CheckArraySpecConsistency(int in_array, const ArraySpec& in_spec,
                                       int out_array,
                                       const ArraySpec& out_spec) {
  if (in_spec.dtype != out_spec.dtype) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Input and output must have the same dtype: %v (input %d) vs. %v "
        "(output %d)",
        in_spec.dtype, in_array, out_spec.dtype, out_array));
  }

  if (in_spec.sharding == nullptr) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Input array %d has null sharding", in_array));
  }
  if (out_spec.sharding == nullptr) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Output array %d has null sharding", out_array));
  }

  ABSL_ASSIGN_OR_RETURN(const Shape in_shard_shape,
                   in_spec.sharding->GetShardShape(in_spec.shape));
  ABSL_ASSIGN_OR_RETURN(const Shape out_shard_shape,
                   out_spec.sharding->GetShardShape(out_spec.shape));
  if (in_shard_shape != out_shard_shape) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Input and output must have the same shard shape: %v (input %d) vs. %v "
        "(output %d)",
        in_shard_shape, in_array, out_shard_shape, out_array));
  }

  if (in_spec.sharding->memory_kind() != out_spec.sharding->memory_kind()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Input and output must have the same memory kind: %v (input %d) vs. %v "
        "(output %d)",
        in_spec.sharding->memory_kind(), in_array,
        out_spec.sharding->memory_kind(), out_array));
  }

  const std::shared_ptr<const xla::PjRtLayout>& in_layout = in_spec.layout;
  const std::shared_ptr<const xla::PjRtLayout>& out_layout = out_spec.layout;
  if (in_layout != out_layout &&
      (in_layout == nullptr || out_layout == nullptr ||
       *in_layout != *out_layout)) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Input and output must have the same layout: %s (input %d) vs. %s "
        "(output %d)",
        in_layout != nullptr ? in_layout->ToString() : "<nullptr>", in_array,
        out_layout != nullptr ? out_layout->ToString() : "<nullptr>",
        out_array));
  }

  return absl::OkStatus();
}

}  // namespace

absl::Status RemapPlan::ValidateArraySpecsUncached() const {
  const int num_inputs = rep_->input_specs.size();
  if (num_inputs == 0) {
    return absl::InvalidArgumentError("Must have at least one input");
  }

  const int num_outputs = rep_->output_specs.size();
  if (num_outputs == 0) {
    return absl::InvalidArgumentError("Must have at least one output");
  }

  for (int i = 0; i < num_inputs; ++i) {
    if (rep_->input_specs[i].sharding == nullptr) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Input array %d has null sharding", i));
    }
  }
  for (int i = 0; i < num_outputs; ++i) {
    if (rep_->output_specs[i].sharding == nullptr) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Output array %d has null sharding", i));
    }
  }

  if (rep_->input_devices_for_output_map.empty()) {
    return absl::InvalidArgumentError(
        "Must have at least one mapping in `input_devices_for_output_map`");
  }

  absl::flat_hash_set<std::pair<int, int>> checked_pairs;
  // NOLINTNEXTLINE(*-custom-deterministic-iteration-order)
  for (const auto& [out_array, inputs] : rep_->input_devices_for_output_map) {
    if (out_array < 0 || out_array >= num_outputs) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Output buffer index %d in `input_devices_for_output_map` is out "
          "of range [0, %d]",
          out_array, num_outputs - 1));
    }
    for (const InputDeviceRange& range : inputs) {
      if (range.in_array < 0 || range.in_array >= num_inputs) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "Input buffer index %d in `input_devices_for_output_map` is out "
            "of range [0, %d]",
            range.in_array, num_inputs - 1));
      }
      if (checked_pairs.insert({range.in_array, out_array}).second) {
        ABSL_RETURN_IF_ERROR(CheckArraySpecConsistency(
            range.in_array, rep_->input_specs[range.in_array], out_array,
            rep_->output_specs[out_array]));
      }
    }
  }

  return absl::OkStatus();
}

absl::Status RemapPlan::ValidateArrayShardMappingsUncached() const {
  const int num_inputs = rep_->input_specs.size();
  const int num_outputs = rep_->output_specs.size();
  TF_RET_CHECK(num_inputs > 0);
  TF_RET_CHECK(num_outputs > 0);

  if (rep_->input_devices_for_output_map.size() != num_outputs) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "`input_devices_for_output_map` has %d outputs, but expected %d "
        "outputs",
        rep_->input_devices_for_output_map.size(), num_outputs));
  }
  std::vector<absl::flat_hash_set<Device*>> in_device_sets;
  in_device_sets.reserve(num_inputs);
  for (int i = 0; i < num_inputs; ++i) {
    const xla::ifrt::DeviceList* in_devices =
        rep_->input_specs[i].sharding->devices()->AddressableDeviceList();
    in_device_sets.push_back(absl::flat_hash_set<Device*>(
        in_devices->devices().begin(), in_devices->devices().end()));
  }
  // NOLINTNEXTLINE(*-custom-deterministic-iteration-order)
  for (const auto& [out_array, inputs] : rep_->input_devices_for_output_map) {
    TF_RET_CHECK(out_array >= 0 && out_array < num_outputs);
    for (const InputDeviceRange& range : inputs) {
      TF_RET_CHECK(range.in_array >= 0 && range.in_array < num_inputs);
      if (range.input_devices == nullptr) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "Output buffer index %d in `input_devices_for_output_map` has "
            "null input_devices for input array %d",
            out_array, range.in_array));
      }

      const absl::flat_hash_set<Device*>& in_device_set =
          in_device_sets[range.in_array];
      for (Device* device :
           range.input_devices->AddressableDeviceList()->devices()) {
        if (!in_device_set.contains(device)) {
          return absl::InvalidArgumentError(absl::StrFormat(
              "Output buffer index %d in `input_devices_for_output_map` "
              "references device %s from input array %d that is not in the "
              "input array's addressable device list",
              out_array, device->DebugString(), range.in_array));
        }
      }
    }
  }

  return absl::OkStatus();
}

absl::Status RemapPlan::ValidateArraySpecs() const {
  absl::call_once(rep_->validate_array_specs_once, [this]() {
    rep_->validate_array_specs_status = ValidateArraySpecsUncached();
  });
  return rep_->validate_array_specs_status;
}

absl::Status RemapPlan::Validate() const {
  ABSL_RETURN_IF_ERROR(ValidateArraySpecs());
  absl::call_once(rep_->validate_array_shard_mappings_once, [this]() {
    rep_->validate_array_shard_mappings_status =
        ValidateArrayShardMappingsUncached();
  });
  return rep_->validate_array_shard_mappings_status;
}

absl::StatusOr<RemapPlan> RemapPlan::FromProto(Client* client,
                                               const RemapPlanProto& proto) {
  const SerDesVersionNumber version_number(proto.version_number());
  if (version_number != SerDesVersionNumber(0)) {
    return absl::FailedPreconditionError(absl::StrCat(
        "Unsupported ", version_number, " for RemapPlan deserialization"));
  }

  std::vector<ArraySpec> input_specs;
  input_specs.reserve(proto.input_specs_size());
  for (const auto& input_spec_proto : proto.input_specs()) {
    ABSL_ASSIGN_OR_RETURN(ArraySpec input_spec,
                     ArraySpec::FromProto(client, input_spec_proto));
    input_specs.push_back(std::move(input_spec));
  }

  std::vector<ArraySpec> output_specs;
  output_specs.reserve(proto.output_specs_size());
  for (const auto& output_spec_proto : proto.output_specs()) {
    ABSL_ASSIGN_OR_RETURN(ArraySpec output_spec,
                     ArraySpec::FromProto(client, output_spec_proto));
    output_specs.push_back(std::move(output_spec));
  }

  absl::flat_hash_map<int, std::vector<InputDeviceRange>>
      input_devices_for_output_map;
  input_devices_for_output_map.reserve(proto.input_devices_for_output_size());
  for (const auto& inputs_for_output_proto : proto.input_devices_for_output()) {
    std::vector<InputDeviceRange>& input_ranges =
        input_devices_for_output_map[inputs_for_output_proto.out_array()];
    for (const auto& inputs_range_proto :
         inputs_for_output_proto.input_devices()) {
      ABSL_ASSIGN_OR_RETURN(auto devices,
                       InputDeviceRangeFromProto(client, inputs_range_proto));
      input_ranges.push_back(std::move(devices));
    }
  }

  if (input_devices_for_output_map.size() < output_specs.size() &&
      !proto.mappings().empty()) {
    ABSL_ASSIGN_OR_RETURN(auto computed_map,
                     InputDevicesForOutputMapFromMappings(client, input_specs,
                                                          output_specs, proto));
    if (input_devices_for_output_map.empty()) {
      input_devices_for_output_map = std::move(computed_map);
    } else {
      for (auto& [out_array, input_ranges] :
           // NOLINTNEXTLINE(*-custom-deterministic-iteration-order)
           computed_map) {
        input_devices_for_output_map.try_emplace(out_array,
                                                 std::move(input_ranges));
      }
    }
  }

  return RemapPlan(std::move(input_specs), std::move(output_specs),
                   std::move(input_devices_for_output_map));
}

absl::Status RemapPlan::ToProto(RemapPlanProto& proto,
                                SerDesVersion version) const {
  if (version.version_number() < SerDesVersionNumber(0)) {
    return absl::FailedPreconditionError(
        absl::StrCat("Unsupported ", version.version_number(),
                     " for RemapPlan serialization"));
  }

  proto.Clear();
  proto.set_version_number(SerDesVersionNumber(0).value());

  proto.mutable_input_specs()->Reserve(rep_->input_specs.size());
  for (const auto& input_spec : rep_->input_specs) {
    ABSL_RETURN_IF_ERROR(input_spec.ToProto(*proto.add_input_specs(), version));
  }
  proto.mutable_output_specs()->Reserve(rep_->output_specs.size());
  for (const auto& output_spec : rep_->output_specs) {
    ABSL_RETURN_IF_ERROR(output_spec.ToProto(*proto.add_output_specs(), version));
  }

  proto.mutable_input_devices_for_output()->Reserve(
      rep_->input_devices_for_output_map.size());
  for (const auto& [out_array, input_devices] :
       // NOLINTNEXTLINE(*-custom-deterministic-iteration-order)
       rep_->input_devices_for_output_map) {
    InputDeviceToOutputToProto(version, out_array, input_devices,
                               *proto.add_input_devices_for_output());
  }

  return absl::OkStatus();
}

std::string RemapPlan::DebugString() const {
  auto format_array_specs = [](absl::Span<const ArraySpec> array_specs) {
    return absl::StrCat("[", absl::StrJoin(array_specs, ","), "]");
  };
  auto format_output_to_inputs =
      [](const absl::flat_hash_map<int, std::vector<InputDeviceRange>>&
             output_to_inputs) {
        return absl::StrCat(
            "[",
            absl::StrJoin(
                output_to_inputs, ",",
                [](std::string* out, const auto& output_to_inputs) {
                  const auto& [out_array, input_devices] = output_to_inputs;
                  absl::StrAppend(
                      out, "o", out_array, ":{",
                      absl::StrJoin(
                          input_devices, ",",
                          [](std::string* out, const InputDeviceRange& range) {
                            absl::StrAppend(out, "i", range.in_array, ":#",
                                            range.input_devices->size());
                          }),
                      "}");
                }),
            "]");
      };
  return absl::StrCat(
      "RemapPlan(input_specs=", format_array_specs(rep_->input_specs),
      ",output_specs=", format_array_specs(rep_->output_specs), ",output_map=",
      format_output_to_inputs(rep_->input_devices_for_output_map), ")");
}

absl::Status RemapPlan::CheckArrayCopySemantics(
    xla::ifrt::ArrayCopySemantics semantics) const {
  if (semantics != xla::ifrt::ArrayCopySemantics::kDonateInput) {
    if (!CheckOneInputForOneOutput(*this)) {
      return absl::InvalidArgumentError(
          "kDonateInput is required if multiple inputs are mapped to one "
          "output");
    }
  }
  return absl::OkStatus();
}

void RemapPlan::Hash(absl::HashState state) const {
  uint64_t hash = rep_->hash.load(std::memory_order_relaxed);
  if (hash == Rep::kUnsetHash) {
    hash = absl::HashOf(rep_->input_specs, rep_->output_specs,
                        rep_->input_devices_for_output_map);
    if (ABSL_PREDICT_FALSE(hash == Rep::kUnsetHash)) {
      ++hash;
    }
    rep_->hash.store(hash, std::memory_order_relaxed);
  }
  absl::HashState::combine(std::move(state), hash);
}

}  // namespace ifrt
}  // namespace xla
