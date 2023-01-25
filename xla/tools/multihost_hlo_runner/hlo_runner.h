/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_TOOLS_MULTIHOST_HLO_RUNNER_HLO_RUNNER_H_
#define TENSORFLOW_COMPILER_XLA_TOOLS_MULTIHOST_HLO_RUNNER_HLO_RUNNER_H_

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/container/btree_map.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/service/computation_placer.h"
#include "xla/status.h"
#include "xla/xla_data.pb.h"

namespace xla {

// Supported input formats for the input HLO module.
enum class InputFormat {
  kText,                 // Text format.
  kProtoText,            // Protobuf text format.
  kProtoBinary,          // Protobuf binary format.
  kSnapshotProtoBinary,  // HloSnapshot protobuf binary format. Can be dumped by
                         // TensorFlow by setting the environment variable
                         // xla_dump_hlo_snapshots.
};

bool AbslParseFlag(absl::string_view text, InputFormat* input_format,
                   std::string* error);
std::string AbslUnparseFlag(InputFormat input_format);

// MultiHostHloRunner takes an HLO module as input and runs the HLO module on a
// single or multiple hosts, with various options (e.g. SPMD). The HLO module
// could be a pre- or post-optimizations.
class MultiHostHloRunner {
 public:
  using LogicalIdToDeviceIdMap = xla::Array2D<int>;
  using LiteralVec = std::vector<Literal>;
  using PerDeviceLiteralVecType = absl::btree_map<int, LiteralVec>;
  using PerDeviceIndexVecType = absl::btree_map<int, std::vector<int>>;
  enum class LogOutputMode { kLogOutput, kNotLogOutput };

  enum class HloPassesMode {
    // Only call the XLA compiler's RunBackend to compile the module. This is
    // used to run a post-optimization HLO module (dumped as
    // 'xxx.after_optimizations.hlo.xxx').
    kRunXLABackendOnly,
    // Calls Compile (i.e., both RunHloPasses and RunBackend) to compile the
    // module, but disables all HLO passes.
    kDisableAllHloPasses,
    // Standard XLA compilation by calling Compile (or both RunHloPasses and
    // RunBackend). This is used to run a pre-optimizations module.
    kStandardCompile
  };

  enum class SpmdMode { kUseSpmdPartitioning, kNotUseSpmdPartitioning };

  enum class SpmdPartitionedMode {
    kIsSpmdPartitionedModule,
    kIsNotSpmdPartitionedModule
  };

  enum class XlaTextDumpMode { kDumpAsText, kNotDumpAsText };

  enum class XlaProtoDumpMode { kDumpAsProto, kNotDumpAsProto };

  enum class ModuleArgumentMode {
    // Use device ID (casted to proper type) as arguments.
    kUseDeviceIdAsInput,
    // Use random values are arguments.
    kUseRandomInputs,
    // Use random values are arguments, and different local devices share the
    // same argument values.
    kUseSharedRandomInputs
  };

  enum class ModuleOutputMode {
    // Return output from all devices.
    kReturnOutputs,
    // Do not return output from any device.
    kNotReturnOutputs,
    // Return the output only from the logical device 0.
    kReturnDevice0Outputs
  };

  enum class DeviceType {
    // Only GPU is supported for now
    kGpu
  };

  static StatusOr<std::unique_ptr<PjRtClient>> GetDeviceClient(
      DeviceType device_type, bool use_tfrt_client = false);

  ~MultiHostHloRunner() = default;

  // Config options for MultiHostHloRunner.
  struct Options {
    std::optional<size_t> num_replicas = 1;
    std::optional<size_t> num_partitions = 1;
    LogOutputMode log_output_mode = LogOutputMode::kNotLogOutput;
    HloPassesMode hlo_passes_mode = HloPassesMode::kStandardCompile;
    SpmdMode spmd_mode = SpmdMode::kNotUseSpmdPartitioning;
    SpmdPartitionedMode spmd_partitioned_mode =
        SpmdPartitionedMode::kIsNotSpmdPartitionedModule;
    std::string xla_dump_to = std::string();
    XlaTextDumpMode xla_text_dump_mode = XlaTextDumpMode::kNotDumpAsText;
    XlaProtoDumpMode xla_proto_dump_mode = XlaProtoDumpMode::kNotDumpAsProto;
    ModuleArgumentMode module_argument_mode =
        ModuleArgumentMode::kUseRandomInputs;
    // Defines which devices' output is returned to the caller when an HLO
    // module is executed.
    ModuleOutputMode module_output_mode = ModuleOutputMode::kReturnOutputs;
    // Whether flatten while loops.
    bool flatten_while_loop = false;
    // If flatten while loops, execute all while loops with this number of
    // iterations.
    int while_execution_count = 0;
    bool remove_infeed_outfeed = true;
    size_t num_repeats = 1;
    // TODO(b/233393955): we encounter runtime errors when using the PjRt SE
    // client. We will remove the option of using PjRt SE client after it is
    // deprecated, which seems to be a close target.
    bool use_tfrt_client = true;
    int32_t task_id = 0;
    // The xla::ExecutionOptions which is used to initialized the HLO runner
    // when provided. When a more specific configuration option,
    // e.g., text_dump_mode, is set, the value in execution options are
    // ignored by the MultiHostHloRunner.
    std::optional<ExecutionOptions> execution_options = std::nullopt;
  };

  // Factory method using a client pointer.
  static StatusOr<std::unique_ptr<MultiHostHloRunner>> CreateMultiHostHloRunner(
      const Options& options, std::shared_ptr<PjRtClient> client);

  // Factory method using a client enum.
  static StatusOr<std::unique_ptr<MultiHostHloRunner>> CreateMultiHostHloRunner(
      const Options& options, DeviceType device_type);

  absl::Span<PjRtDevice* const> local_devices() const;
  absl::Span<PjRtDevice* const> devices() const;
  PjRtClient* client() const;

  // Parses and runs the given HLO module text. The HLO module is run with the
  // provided arguments if the arguments map is not empty.
  StatusOr<PerDeviceLiteralVecType> ParseAndRun(
      absl::string_view hlo_text,
      const PerDeviceLiteralVecType& arguments = {}) const;

  // Parses and runs the given HLO module text. The module arguments are
  // provided by `argument_literals`. The arguments per device is defined by
  // the `per_device_index_vec`, which should contain a vector of indices for
  // each local device. This means different devices may use the same argument
  // literals. This is essential to run HLO modules with large arguments (e.g.,
  // models with large weights).
  StatusOr<PerDeviceLiteralVecType> ParseAndRun(
      absl::string_view hlo_text, const LiteralVec& argument_literals,
      const PerDeviceIndexVecType& per_device_index_vec) const;

  // Loads an HLO module from hlo_file according to input_format and run it.
  // The HLO module is run with the provided arguments if the arguments map is
  // not empty. Otherwise, use arguments from the HLO file or fake arguments.
  // The hlo file might be a HLO snapshot and thus contain arguments, otherwrse
  // it is run with fake arguments.
  StatusOr<PerDeviceLiteralVecType> LoadAndRun(
      absl::Span<const std::string> hlo_files, InputFormat input_format,
      const PerDeviceLiteralVecType& arguments = {}) const;

  // Loads an HLO module from hlo_file according to input_format and run it.
  // The module arguments are provided by `argument_literals`. The arguments per
  // device is defined by the `per_device_index_vec`, which should contain a
  // vector of indices for each local device. This means different device may
  // use the same argument literals. This is essential to run HLO modules with
  // large arguments (e.g., models with large weights).
  StatusOr<PerDeviceLiteralVecType> LoadAndRun(
      absl::Span<const std::string> hlo_files, InputFormat input_format,
      const LiteralVec& argument_literals,
      const PerDeviceIndexVecType& per_device_index_vec) const;

  // Compiles and runs the given HLO module with the given arguments for each
  // device. The given arguments is a map from device ID to a list of arguments.
  // If the arguments map is empty, the HLO module is run with fake arguments.
  StatusOr<PerDeviceLiteralVecType> CompileAndRun(
      HloModule* hlo_module,
      const PerDeviceLiteralVecType& arguments = {}) const;

  // Compiles and runs the given HLO module with the given arguments for each
  // device. The module arguments are provided by `argument_literals`. The
  // arguments per device is defined by the `per_device_index_vec`, which should
  // contain a vector of indices for each local device. This means different
  // devices may use the same argument literals. This is essential to run HLO
  // modules with large arguments (e.g., models with large weights).
  StatusOr<PerDeviceLiteralVecType> CompileAndRun(
      HloModule* hlo_module, const LiteralVec& argument_literals,
      const PerDeviceIndexVecType& argument_indices) const;

  void SetDeviceAssignment(const DeviceAssignment& device_assignment);
  const DeviceAssignment& GetDeviceAssignment() const;

  // Compiles the HLO module.
  StatusOr<std::unique_ptr<PjRtLoadedExecutable>> Compile(
      HloModule* hlo_module) const;

  StatusOr<std::unique_ptr<PjRtLoadedExecutable>> Load(
      std::unique_ptr<PjRtExecutable> executable,
      const LoadOptions& load_options) const;

  // Runs the executable.
  StatusOr<PerDeviceLiteralVecType> Run(
      PjRtLoadedExecutable* executable,
      const PerDeviceLiteralVecType& arguments, int num_repeats) const;

  // Runs the executable, where the module arguments are provided through
  // a shared literal vector and per-device indices.
  StatusOr<PerDeviceLiteralVecType> Run(
      PjRtLoadedExecutable* executable, const LiteralVec& argument_literals,
      const PerDeviceIndexVecType& argument_indices, int num_repeats) const;

  StatusOr<LogicalIdToDeviceIdMap> CreateLogicalIdToDeviceIdMap();

  static StatusOr<LogicalIdToDeviceIdMap> CreateLogicalIdToDeviceIdMap(
      const DeviceAssignment& device_assignment,
      absl::Span<const int> device_ids);

  static StatusOr<std::unique_ptr<HloModule>> ReadModuleFromHloTextFile(
      absl::string_view hlo_file);
  static StatusOr<std::unique_ptr<HloModule>> ReadModuleFromBinaryProtoFile(
      absl::string_view hlo_file);
  static StatusOr<std::unique_ptr<HloModule>> ReadModuleFromTextProtoFile(
      absl::string_view hlo_file);

  struct HloModuleAndArguments {
    std::unique_ptr<HloModule> hlo_module;
    std::vector<Literal> arguments;
  };

  static StatusOr<HloModuleAndArguments> ReadModuleFromSnapshotBinaryProtoFile(
      absl::string_view hlo_file);
  static StatusOr<HloModuleAndArguments> LoadHloModuleAndArguments(
      absl::string_view hlo_file, InputFormat input_format);

  static StatusOr<std::unique_ptr<HloModule>> ReadModuleFromString(
      absl::string_view hlo_text);

  static StatusOr<std::unique_ptr<HloModule>> ReadModuleFromProto(
      const HloModuleProto& proto);

  // Sets the correct argument layouts for backend only compilation.
  // Otherwise Pjrt client will assume the default compact layouts which might
  // be different from the actual ones.
  void SetArgumentLayouts(const HloModule* hlo_module);

 private:
  MultiHostHloRunner(const Options& options,
                     std::shared_ptr<PjRtClient> client);

  size_t num_replicas() const {
    return compile_options_.executable_build_options.num_replicas();
  }

  size_t num_partitions() const {
    return compile_options_.executable_build_options.num_partitions();
  }
  Status PrepareHloModuleForCompilation(HloModule* hlo_module) const;
  CompileOptions GetCompileOptions(const HloModule& hlo_module) const;

  // Creates fake arguments to run the given executable.
  StatusOr<std::vector<std::vector<std::unique_ptr<PjRtBuffer>>>>
  CreateArgumentsOnDevice(const PjRtLoadedExecutable* executable,
                          bool flatten_arguments = false) const;

  // Creates argument buffers based on the given arguments map. Note that the
  // arguments might be invalid when arguments are destructed.
  StatusOr<std::vector<std::vector<std::unique_ptr<PjRtBuffer>>>>
  CopyArgumentsToDevice(absl::Span<PjRtDevice* const> addressable_devices,
                        const PerDeviceLiteralVecType& arguments) const;

  StatusOr<std::vector<std::vector<std::unique_ptr<PjRtBuffer>>>>
  CopyArgumentsToDevice(absl::Span<PjRtDevice* const> addressable_devices,
                        const LiteralVec& argument_literals,
                        const PerDeviceIndexVecType& argument_indices) const;

  StatusOr<PerDeviceLiteralVecType> RunInternal(
      PjRtLoadedExecutable* executable,
      std::function<
          StatusOr<std::vector<std::vector<std::unique_ptr<PjRtBuffer>>>>(bool)>
          create_argument_buffers_on_device,
      int num_repeats) const;

  StatusOr<PerDeviceLiteralVecType> FetchAndLogOutput(
      const std::vector<std::vector<std::unique_ptr<PjRtBuffer>>>&
          output_buffers) const;
  std::shared_ptr<PjRtClient> client_;
  CompileOptions compile_options_;
  const bool log_output_;
  const HloPassesMode hlo_passes_mode_;
  const bool use_spmd_partitioning_;
  const bool is_spmd_partitioned_module_;
  const ModuleArgumentMode module_argument_mode_;
  const ModuleOutputMode module_output_mode_;
  const bool flatten_while_loop_ = false;
  const int while_execution_count_ = 0;
  const bool remove_infeed_outfeed_ = true;
  const std::string exec_name_ = "";
  const size_t num_repeats_ = 0;
  const int32_t task_id_ = 0;
};

bool AbslParseFlag(absl::string_view text,
                   MultiHostHloRunner::DeviceType* device_type,
                   std::string* error);
std::string AbslUnparseFlag(MultiHostHloRunner::DeviceType device_type);

bool AbslParseFlag(absl::string_view text,
                   MultiHostHloRunner::ModuleArgumentMode* argument_mode,
                   std::string* error);
std::string AbslUnparseFlag(
    MultiHostHloRunner::ModuleArgumentMode argument_mode);

}  // namespace xla
#endif  // TENSORFLOW_COMPILER_XLA_TOOLS_MULTIHOST_HLO_RUNNER_HLO_RUNNER_H_
