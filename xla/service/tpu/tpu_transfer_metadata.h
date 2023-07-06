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

#ifndef XLA_SERVICE_TPU_TPU_TRANSFER_METADATA_H_
#define XLA_SERVICE_TPU_TPU_TRANSFER_METADATA_H_

#include "xla/service/transfer_manager.h"

namespace xla {

// Defines the metadata of a TPU device transfer.
class TpuTransferMetadata : public TransferManager::TransferMetadata {
 public:
  explicit TpuTransferMetadata(int32_t sync_flag_to_update)
      : sync_flag_to_update_(sync_flag_to_update) {}
  ~TpuTransferMetadata() override = default;
  // The sync flag to bump for the transfer.
  int32_t sync_flag_to_update() const { return sync_flag_to_update_; }

 private:
  const int32_t sync_flag_to_update_;
};

}  // namespace xla

#endif  // XLA_SERVICE_TPU_TPU_TRANSFER_METADATA_H_
