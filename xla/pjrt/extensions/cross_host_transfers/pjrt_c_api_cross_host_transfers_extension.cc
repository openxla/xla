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

#include "xla/pjrt/extensions/cross_host_transfers/pjrt_c_api_cross_host_transfers_extension.h"

#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/future.h"
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_helpers.h"
#include "xla/pjrt/c/pjrt_c_api_wrapper_impl.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/shape.h"

namespace pjrt {

PJRT_Error* PJRT_Transfers_PJRT_Client_CrossHostReceiveBuffers(
    PJRT_Transfers_PJRT_Client_CrossHostReceiveBuffers_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_Transfers_PJRT_Client_CrossHostReceiveBuffers_Args",
      PJRT_Transfers_PJRT_Client_CrossHostReceiveBuffers_Args_STRUCT_SIZE,
      args->struct_size));

  std::vector<xla::Shape> shapes;
  shapes.reserve(args->num_shapes);
  for (int i = 0; i < args->num_shapes; ++i) {
    PJRT_ASSIGN_OR_RETURN(
        xla::Shape shape,
        pjrt::BuildXlaShapeFromC(args->element_types[i], args->num_dims[i],
                                 args->shape_num_dims[i], args->layouts[i]));
    shapes.push_back(std::move(shape));
  }

  std::vector<xla::PjRtGlobalDeviceId> src_global_device_ids;
  src_global_device_ids.reserve(args->num_shapes);
  for (int i = 0; i < args->num_shapes; ++i) {
    src_global_device_ids.push_back(args->src_global_device_ids[i]);
  }

  PJRT_ASSIGN_OR_RETURN(
      std::vector<std::unique_ptr<xla::PjRtBuffer>> buffers,
      args->client->client->CrossHostReceiveBuffers(
          absl::MakeSpan(shapes), args->device->device, src_global_device_ids,
          args->transfer_key));

  for (int i = 0; i < buffers.size(); ++i) {
    args->buffers[i] = new PJRT_Buffer{std::move(buffers[i]), args->client};
  }
  return nullptr;
}

PJRT_Error* PJRT_Transfers_PJRT_Client_CrossHostSendBuffers(
    PJRT_Transfers_PJRT_Client_CrossHostSendBuffers_Args* args) {
  std::vector<xla::PjRtBuffer*> buffers;
  buffers.reserve(args->num_buffers);
  for (int i = 0; i < args->num_buffers; ++i) {
    buffers.push_back(args->buffers[i]->buffer.get());
  }

  std::vector<xla::PjRtGlobalDeviceId> dst_global_device_ids;
  dst_global_device_ids.reserve(args->num_buffers);
  for (int i = 0; i < args->num_buffers; ++i) {
    dst_global_device_ids.push_back(args->dst_global_device_ids[i]);
  }

  std::vector<tsl::Future<>> send_futures =
      args->client->client->CrossHostSendBuffers(buffers, dst_global_device_ids,
                                                 args->transfer_key);

  for (int i = 0; i < buffers.size(); ++i) {
    args->send_events[i] = new PJRT_Event{std::move(send_futures[i])};
  }
  return nullptr;
}

PJRT_CrossHostTransfers_Extension CreateCrossHostTransfersExtension(
    PJRT_Extension_Base* next) {
  return PJRT_CrossHostTransfers_Extension{
      PJRT_Extension_Base{
          /*struct_size=*/PJRT_CrossHostTransfers_Extension_STRUCT_SIZE,
          /*type=*/PJRT_Extension_Type_CrossHostTransfers,
          /*next=*/next,
      },
      /*PJRT_Transfers_PJRT_Client_CrossHostReceiveBuffers=*/
      PJRT_Transfers_PJRT_Client_CrossHostReceiveBuffers,
      /*PJRT_Transfers_PJRT_Client_CrossHostSendBuffers=*/
      PJRT_Transfers_PJRT_Client_CrossHostSendBuffers};
}

}  // namespace pjrt
