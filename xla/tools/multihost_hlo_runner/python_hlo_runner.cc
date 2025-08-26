/* Copyright 2025 The OpenXLA Authors.

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


#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "nanobind/nanobind.h"
#include "nanobind/stl/shared_ptr.h"  // IWYU pragma: keep
#include "nanobind/stl/shared_ptr.h"  // IWYU pragma: keep
#include "nanobind/stl/string.h"  // IWYU pragma: keep
#include "nanobind/stl/string_view.h"  // IWYU pragma: keep
#include "nanobind/stl/vector.h"  // IWYU pragma: keep
#include "xla/python/logging.h"
#include "xla/python/nb_numpy.h"
#include "xla/python/types.h"
#include "xla/tools/multihost_hlo_runner/create_client.h"
#include "xla/tools/multihost_hlo_runner/functional_hlo_runner.h"
#include "xla/tools/multihost_hlo_runner/hlo_input_output_format.h"
#include "xla/tsl/python/lib/core/numpy.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/pjrt/plugin/xla_gpu/xla_gpu_allocator_config.h"
#include "xla/pjrt/plugin/xla_gpu/xla_gpu_client_options.h"
#include "xla/xla_data.pb.h"
#include "xla/pjrt/status_casters.h"
#include "xla/service/custom_call_target_registry.h"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/ffi.h"
#include "xla/ffi/ffi_api.h"

namespace nb = ::nanobind;

namespace xla {

namespace {

enum DeviceType {
  kHost = 0,
  kGpu  = 1,
};

struct HloRunnerConfig {
  InputFormat input_format = InputFormat::kProtoText;
  FunctionalHloRunner::ModuleOutputMode output_mode = FunctionalHloRunner::ModuleOutputMode::kReturnOutputs;
  bool should_run = true;
  bool enable_mock_nccl = false;
  std::string dump_output_literal_to = "";
  int task_id = 0;
  int num_nodes = 1;
  DeviceType device_type = DeviceType::kGpu;
  std::string address = "";
  int32_t num_replicas = -1;
  int32_t num_partitions = 1;
  bool log_output = false;
  FunctionalHloRunner::HloPassesMode hlo_pass_mode = FunctionalHloRunner::HloPassesMode::kStandardCompile;
  FunctionalHloRunner::SpmdMode spmd_mode = FunctionalHloRunner::SpmdMode::kNotUseSpmdPartitioning;
  bool is_spmd_partitioned_module = false;
  std::string xla_dump_to = "";
  bool xla_dump_as_text = false;
  bool xla_dump_as_proto = false;
  FunctionalHloRunner::ModuleArgumentMode hlo_argument_mode = FunctionalHloRunner::ModuleArgumentMode::kUseRandomInputs;
  int32_t while_execution_count = -1;
  bool remove_infeed_outfeed = true;
  bool compile_as_stablehlo = false;
  bool use_layouts_from_hlo_module = false;
  bool force_auto_layout = false;
  int32_t num_repeats = 1;
  std::string execution_options_path = "";
  int64_t gpu_client_initialization_timeout_sec = 300;
  float gpu_client_mem_fraction = GpuAllocatorConfig{}.memory_fraction;
  bool profile_execution = false;
  std::string xla_gpu_dump_xspace_to = "";
};

absl::StatusOr<FunctionalHloRunner::PreprocessingOptions>
PreprocessingOptionsFromFlags(const HloRunnerConfig& opts) {
  FunctionalHloRunner::PreprocessingOptions out;
  out.spmd_partitioned_mode =
      opts.is_spmd_partitioned_module
          ? FunctionalHloRunner::SpmdPartitionedMode::kIsSpmdPartitionedModule
          : FunctionalHloRunner::SpmdPartitionedMode::
                kIsNotSpmdPartitionedModule;
  out.while_execution_count =
      opts.while_execution_count > 0
          ? std::make_optional(opts.while_execution_count)
          : std::nullopt;
  out.remove_infeed_outfeed = opts.remove_infeed_outfeed;
  return out;
}

absl::StatusOr<FunctionalHloRunner::RunningOptions>
RunningOptionsFromFlags(const HloRunnerConfig& opts) {
  FunctionalHloRunner::RunningOptions out;
  out.module_argument_mode = opts.hlo_argument_mode;
  out.module_output_mode = opts.output_mode;
  out.num_repeats = static_cast<size_t>(opts.num_repeats);
  out.log_input_output_mode =
      opts.log_output ? FunctionalHloRunner::LogOutputMode::kLogOutput
                      : FunctionalHloRunner::LogOutputMode::kNotLogOutput;
  return out;
}

absl::StatusOr<FunctionalHloRunner::RawCompileOptions>
RawCompileOptionsFromFlags(const HloRunnerConfig& opts) {
  FunctionalHloRunner::RawCompileOptions out;
  out.hlo_passes_mode = opts.hlo_pass_mode;
  out.spmd_mode = opts.spmd_mode;
  if (!opts.execution_options_path.empty()) {
    TF_ASSIGN_OR_RETURN(
        out.execution_options,
        FunctionalHloRunner::LoadExecutionOptions(opts.execution_options_path));
  }
  out.num_replicas = opts.num_replicas < 0
                         ? std::nullopt
                         : std::optional<int>(opts.num_replicas);
  out.num_partitions = opts.num_partitions < 0
                           ? std::nullopt
                           : std::optional<int>(opts.num_partitions);
  out.xla_dump_to = opts.xla_dump_to;
  out.xla_text_dump_mode =
      opts.xla_dump_as_text
          ? FunctionalHloRunner::XlaTextDumpMode::kDumpAsText
          : FunctionalHloRunner::XlaTextDumpMode::kNotDumpAsText;
  out.xla_proto_dump_mode =
      opts.xla_dump_as_proto
          ? FunctionalHloRunner::XlaProtoDumpMode::kDumpAsProto
          : FunctionalHloRunner::XlaProtoDumpMode::kNotDumpAsProto;
  out.xla_gpu_dump_xspace_to = opts.xla_gpu_dump_xspace_to;
  return out;
}

// TODO: Return profile protos?
absl::Status RunHlos(const std::vector<std::string>& hlo_files, const HloRunnerConfig& opts) {

  TF_ASSIGN_OR_RETURN(
      FunctionalHloRunner::PreprocessingOptions preproc_options,
      PreprocessingOptionsFromFlags(opts));
  preproc_options.annotate_while_loop_trip_count = true;
  TF_ASSIGN_OR_RETURN(
      FunctionalHloRunner::RawCompileOptions raw_compile_options,
      RawCompileOptionsFromFlags(opts));
  TF_ASSIGN_OR_RETURN(FunctionalHloRunner::RunningOptions running_options,
                      RunningOptionsFromFlags(opts));

  // tsl::Flags::Parse() leaves unknown flags in argv, we assume that those are
  // HLO files to run. Note that argv[0] is the binary name and is excluded.
  if(hlo_files.size() == 0) {
    return absl::InvalidArgumentError("No HLO files provided.");
  }

  if(!opts.dump_output_literal_to.empty() && hlo_files.size() > 1) {
      return absl::InvalidArgumentError("Can only dump output literal when single input file is specified.");
  }

  if (opts.gpu_client_mem_fraction < 0.0 || opts.gpu_client_mem_fraction > 1.0) {
    return absl::InvalidArgumentError("Invalid GPU client memory fraction. Must be in range [0.0, 1.0]");
  }

  PjRtEnvironment env;
  std::unique_ptr<HLORunnerProfiler> hlo_runner_profiler;
  if (opts.device_type == DeviceType::kGpu) {
    GpuClientOptions gpu_options;
    gpu_options.node_id = opts.task_id;
    gpu_options.num_nodes = opts.num_nodes;
    gpu_options.enable_mock_nccl = opts.enable_mock_nccl;
    gpu_options.allocator_config.memory_fraction = opts.gpu_client_mem_fraction;
    TF_ASSIGN_OR_RETURN(
        env, GetPjRtEnvironmentForGpu(
                 opts.address, gpu_options,
                 absl::Seconds(opts.gpu_client_initialization_timeout_sec)));
    // Create a GPURunnerProfiler to profile GPU executions to save xspace data
    // to disk.
    if (env.client != nullptr && !opts.xla_gpu_dump_xspace_to.empty()) {
      TF_ASSIGN_OR_RETURN(hlo_runner_profiler,
                          HLORunnerProfiler::Create(opts.xla_gpu_dump_xspace_to,
                                                    /*keep_xspace=*/false));
      running_options.profiler = hlo_runner_profiler.get();
    }
  } else {
    QCHECK(opts.device_type == DeviceType::kHost) << "Invalid device type";
    TF_ASSIGN_OR_RETURN(env, GetPjRtEnvironmentForHostCpu());
    if (env.client != nullptr && !opts.xla_gpu_dump_xspace_to.empty()) {
      TF_ASSIGN_OR_RETURN(hlo_runner_profiler,
                          HLORunnerProfiler::Create(opts.xla_gpu_dump_xspace_to,
                                                    /*keep_xspace=*/false));
      running_options.profiler = hlo_runner_profiler.get();
    }
  } 
  CHECK(env.client != nullptr);

  std::vector<ExecutionProfile> execution_profiles;
  if (opts.profile_execution) {
    running_options.execution_profiles = &execution_profiles;
  }

  for (const auto& hlo_file : hlo_files) {
    execution_profiles.clear();
    if (opts.should_run) {
      std::cout << "\n** Running " << hlo_file << " **\n";
      TF_RETURN_IF_ERROR(FunctionalHloRunner::LoadAndRunAndDump(
          *env.client, GetDebugOptionsFromFlags(), preproc_options,
          raw_compile_options, running_options, hlo_file, opts.input_format,
          opts.dump_output_literal_to, opts.task_id));
    } else {
      std::cout << "\n** Compiling " << hlo_file << " **\n";
      TF_RETURN_IF_ERROR(FunctionalHloRunner::LoadAndCompile(
          *env.client, GetDebugOptionsFromFlags(), preproc_options,
          raw_compile_options, hlo_file, opts.input_format, opts.task_id));
    }
    for (int i = 0; i < execution_profiles.size(); ++i) {
      std::cout << "## Execution time, file=" << hlo_file << " repeat=" << i
                << " duration=" << execution_profiles[i].compute_time_ns()
                << "ns" << std::endl;
    }
  }
  return absl::OkStatus();
}
}

absl::Status PyRegisterCustomCallTarget(const std::string& fn_name,
                                        nb::object fn,
                                        const std::string& platform,
                                        int api_version,
                                        XLA_FFI_Handler_Traits traits) {
  
  // Register legacy custom call target (untyped void* API).
  if (api_version == 0) {
    if (traits != 0) {
      return absl::InvalidArgumentError(
          "Custom call target registration with traits is not supported for "
          "api_version=0");
    }

    nb::capsule capsule;
    if (!nb::try_cast<nb::capsule>(fn, capsule)) {
      return absl::InvalidArgumentError(
          "Custom call target registration with api_version=0 requires a "
          "PyCapsule fn object");
    }

    CustomCallTargetRegistry::Global()->Register(
        fn_name, static_cast<void*>(capsule.data()), platform);
    return absl::OkStatus();
  }

  // Register XLA FFI handler (typed API with explicit function signatures).
  if (api_version == 1) {
    nb::capsule capsule;
    if (nb::try_cast<nb::capsule>(fn, capsule)) {
      return ffi::TakeStatus(ffi::Ffi::RegisterStaticHandler(
          ffi::GetXlaFfiApi(), fn_name, platform,
          reinterpret_cast<XLA_FFI_Handler*>(
              static_cast<void*>(capsule.data()))));
    }

    nb::dict bundle;
    if (nb::try_cast<nb::dict>(fn, bundle)) {
      auto handler = [&](const char* name) -> absl::StatusOr<XLA_FFI_Handler*> {
        if (!bundle.contains(name)) return nullptr;

        nb::capsule capsule;
        if (!nb::try_cast<nb::capsule>(bundle[name], capsule)) {
          return absl::InvalidArgumentError(
              "Custom call target registration with api_version=1 requires a "
              "PyCapsule fn object for all dict keys");
        }

        return reinterpret_cast<XLA_FFI_Handler*>(capsule.data());
      };

      XLA_FFI_Handler_Bundle bundle;
      TF_ASSIGN_OR_RETURN(bundle.instantiate, handler("instantiate"));
      TF_ASSIGN_OR_RETURN(bundle.prepare, handler("prepare"));
      TF_ASSIGN_OR_RETURN(bundle.initialize, handler("initialize"));
      TF_ASSIGN_OR_RETURN(bundle.execute, handler("execute"));

      return ffi::TakeStatus(ffi::Ffi::RegisterStaticHandler(
          ffi::GetXlaFfiApi(), fn_name, platform, bundle, traits));
    }

    return absl::InvalidArgumentError(
        "Unsupported custom call target type for api_version=1");
  }

  return absl::UnimplementedError(absl::StrFormat(
      "API version %d is not supported by RegisterCustomCallTarget. "
      "Supported versions are 0 and 1.",
      api_version));
}

NB_MODULE(py_hlo_multihost_runner, m) {
  nb::exception<XlaRuntimeError>(m, "XlaRuntimeError", PyExc_RuntimeError);

  m.def("RunHlos", ThrowIfErrorWrapper(RunHlos));

  m.def("register_custom_call_target",
      [](nb::object fn_name_py, nb::object fn, const std::string& platform,
         int api_version, XLA_FFI_Handler_Traits traits) {
        std::string fn_name;
        if (!nb::try_cast<std::string>(fn_name_py, fn_name)) {
          nb::bytes bytes = nb::cast<nb::bytes>(fn_name_py);
          fn_name = std::string(bytes.c_str(), bytes.size());
        }
        ThrowIfError(PyRegisterCustomCallTarget(
            fn_name, std::move(fn), platform, api_version, traits));
      },
      nb::arg("fn_name"), nb::arg("fn"), nb::arg("platform"),
      nb::arg("api_version") = 0, nb::arg("traits") = 0);
  m.def(
      "custom_call_targets",
      [](const std::string& platform) -> nb::dict {
        nb::dict targets;
        for (const auto& [name, target] :
             CustomCallTargetRegistry::Global()->registered_symbols(platform)) {
          targets[nb::str(name.data(), name.size())] = nb::capsule(target);
        }

        auto ffi_handlers = ffi::StaticRegisteredHandlers(platform);
        if (!ffi_handlers.ok()) return targets;

        for (const auto& [name, registration] : *ffi_handlers) {
          nb::dict bundle;
          auto export_handler = [&](std::string_view name, XLA_FFI_Handler* h) {
            if (h != nullptr) {
              bundle[nb::str(name.data(), name.size())] =
                  nb::capsule(reinterpret_cast<void*>(h));
            }
          };
          export_handler("instantiate", registration.bundle.instantiate);
          export_handler("prepare", registration.bundle.prepare);
          export_handler("initialize", registration.bundle.initialize);
          export_handler("execute", registration.bundle.execute);
          targets[nb::str(name.data(), name.size())] = std::move(bundle);
        }
        return targets;
      },
      nb::arg("platform"));

  nb::class_<HloRunnerConfig>(m, "HloRunnerConfig")
    .def(nb::init<>())
    .def_rw("input_format", &HloRunnerConfig::input_format)
    .def_rw("output_mode", &HloRunnerConfig::output_mode)
    .def_rw("should_run", &HloRunnerConfig::should_run)
    .def_rw("enable_mock_nccl", &HloRunnerConfig::enable_mock_nccl)
    .def_rw("dump_output_literal_to", &HloRunnerConfig::dump_output_literal_to)
    .def_rw("task_id", &HloRunnerConfig::task_id)
    .def_rw("num_nodes", &HloRunnerConfig::num_nodes)
    .def_rw("device_type", &HloRunnerConfig::device_type)
    .def_rw("address", &HloRunnerConfig::address)
    .def_rw("num_replicas", &HloRunnerConfig::num_replicas)
    .def_rw("num_partitions", &HloRunnerConfig::num_partitions)
    .def_rw("log_output", &HloRunnerConfig::log_output)
    .def_rw("hlo_pass_mode", &HloRunnerConfig::hlo_pass_mode)
    .def_rw("spmd_mode", &HloRunnerConfig::spmd_mode)
    .def_rw("is_spmd_partitioned_module", &HloRunnerConfig::is_spmd_partitioned_module)
    .def_rw("xla_dump_to", &HloRunnerConfig::xla_dump_to)
    .def_rw("xla_dump_as_text", &HloRunnerConfig::xla_dump_as_text)
    .def_rw("xla_dump_as_proto", &HloRunnerConfig::xla_dump_as_proto)
    .def_rw("hlo_argument_mode", &HloRunnerConfig::hlo_argument_mode)
    .def_rw("while_execution_count", &HloRunnerConfig::while_execution_count)
    .def_rw("remove_infeed_outfeed", &HloRunnerConfig::remove_infeed_outfeed)
    .def_rw("compile_as_stablehlo", &HloRunnerConfig::compile_as_stablehlo)
    .def_rw("use_layouts_from_hlo_module", &HloRunnerConfig::use_layouts_from_hlo_module)
    .def_rw("force_auto_layout", &HloRunnerConfig::force_auto_layout)
    .def_rw("num_repeats", &HloRunnerConfig::num_repeats)
    .def_rw("gpu_client_initialization_timeout_sec", &HloRunnerConfig::gpu_client_initialization_timeout_sec)
    .def_rw("gpu_client_mem_fraction", &HloRunnerConfig::gpu_client_mem_fraction)
    .def_rw("profile_execution", &HloRunnerConfig::profile_execution)
    .def_rw("xla_gpu_dump_xspace_to", &HloRunnerConfig::xla_gpu_dump_xspace_to);


  nb::enum_<InputFormat>(m, "InputFormat")
        .value("Text", InputFormat::kText)
        .value("ProtoText", InputFormat::kProtoText)
        .value("ProtoBinary", InputFormat::kProtoBinary);
  // TODO: Other formats...

  nb::enum_<FunctionalHloRunner::ModuleOutputMode>(m, "ModuleOutputMode")
        .value("ReturnOutputs", FunctionalHloRunner::ModuleOutputMode::kReturnOutputs)
        .value("NotReturnOutputs", FunctionalHloRunner::ModuleOutputMode::kNotReturnOutputs)
        .value("ReturnDevice0Outputs", FunctionalHloRunner::ModuleOutputMode::kReturnDevice0Outputs);

  nb::enum_<FunctionalHloRunner::ModuleArgumentMode>(m, "ModuleArgumentMode")
        .value("UseDeviceIdAsInput", FunctionalHloRunner::ModuleArgumentMode::kUseDeviceIdAsInput)
        .value("UseRandomInputs", FunctionalHloRunner::ModuleArgumentMode::kUseRandomInputs)
        .value("UseSharedRandomInputs", FunctionalHloRunner::ModuleArgumentMode::kUseSharedRandomInputs)
        .value("UseZerosAsInput", FunctionalHloRunner::ModuleArgumentMode::kUseZerosAsInput)
        .value("Uninitialized", FunctionalHloRunner::ModuleArgumentMode::kUninitialized);

  nb::enum_<FunctionalHloRunner::HloPassesMode>(m, "HloPassesMode")
        .value("RunXLABackendOnly", FunctionalHloRunner::HloPassesMode::kRunXLABackendOnly)
        .value("DisableAllHloPasses", FunctionalHloRunner::HloPassesMode::kDisableAllHloPasses)
        .value("StandardCompile", FunctionalHloRunner::HloPassesMode::kStandardCompile);

  nb::enum_<FunctionalHloRunner::SpmdMode>(m, "SpmdMode")
        .value("UseSpmdPartitioning", FunctionalHloRunner::SpmdMode::kUseSpmdPartitioning)
        .value("UseShardyPartitioning", FunctionalHloRunner::SpmdMode::kUseShardyPartitioning)
        .value("NotUseSpmdPartitioning", FunctionalHloRunner::SpmdMode::kNotUseSpmdPartitioning);

  nb::enum_<DeviceType>(m, "DeviceType")
        .value("Host", DeviceType::kHost)
        .value("Gpu", DeviceType::kGpu);
}

}