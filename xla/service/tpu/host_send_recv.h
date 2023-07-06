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

#ifndef XLA_SERVICE_TPU_HOST_SEND_RECV_H_
#define XLA_SERVICE_TPU_HOST_SEND_RECV_H_

#include <cstdint>

namespace xla::tpu::hostsendrecv {

// We use the following strings to identify host transfer directions.

// The device Id to use when an op is to be placed on the host.
inline constexpr int32_t kHostDeviceId = -1;

// IDs 1 and 2 are currently unused.

// Send a value to the host, with explicit completion syncflag.
// Operand: 24-bit ChannelID.
inline constexpr uint32_t kCommandSendSF = 0x03000000;

// Receive a value from the host, with explicit completion syncflag.
// Operand: 24-bit ChannelID.
inline constexpr uint32_t kCommandRecvSF = 0x04000000;

}  // namespace xla::tpu::hostsendrecv

#endif  //  XLA_SERVICE_TPU_HOST_SEND_RECV_H_
