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

#ifndef XLA_SERVICE_EXPORT_HLO_H_
#define XLA_SERVICE_EXPORT_HLO_H_

#include <memory>
#include <string>
#include <utility>

#include "xla/hlo/ir/hlo_module.h"
#include "xla/stream_executor/device_description.pb.h"

namespace xla {

class XSymbolUploader {
 public:
  virtual ~XSymbolUploader() = default;

  virtual std::string MaybeUploadUnoptimizedHloModule(
      HloModule* module,
      const stream_executor::GpuTargetConfigProto& gpu_target_config) {
    return "";
  }
};

class XsymbolUploaderRegistry {
 public:
  XsymbolUploaderRegistry()
      : xsymbol_uploader_(std::make_unique<XSymbolUploader>()) {}

  void Register(std::unique_ptr<XSymbolUploader> xsymbol_uploader) {
    xsymbol_uploader_ = std::move(xsymbol_uploader);
  }

  XSymbolUploader& Get() const { return *xsymbol_uploader_; }

 private:
  std::unique_ptr<XSymbolUploader> xsymbol_uploader_;
};

inline XsymbolUploaderRegistry& GetGlobalXsymbolUploaderRegistry() {
  static auto* const registry = new XsymbolUploaderRegistry;
  return *registry;
}

inline std::string MaybeUploadUnoptimizedGpuSymbolsToXSymbol(
    HloModule* module,
    const stream_executor::GpuTargetConfigProto& gpu_target_config) {
  return GetGlobalXsymbolUploaderRegistry()
      .Get()
      .MaybeUploadUnoptimizedHloModule(module, gpu_target_config);
}

}  // namespace xla

#endif  // XLA_SERVICE_EXPORT_HLO_H_
