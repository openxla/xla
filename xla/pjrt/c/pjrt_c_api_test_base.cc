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

#include "xla/pjrt/c/pjrt_c_api_test_base.h"

#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_wrapper_impl.h"

namespace xla {
namespace pjrt {
PjrtCApiTestBase::PjrtCApiTestBase() = default;

PjrtCApiTestBase::~PjrtCApiTestBase() { this->destroy_client(this->client_); }

void PjrtCApiTestBase::destroy_client(PJRT_Client* client) {
  PJRT_Client_Destroy_Args destroy_args = PJRT_Client_Destroy_Args{
      .struct_size = PJRT_Client_Destroy_Args_STRUCT_SIZE,
      .priv = nullptr,
      .client = client,
  };
  PJRT_Error* error = api_->PJRT_Client_Destroy(&destroy_args);
  CHECK_EQ(error, nullptr);
}

PJRT_Client* PjrtCApiTestBase::make_client() {
  PJRT_Client_Create_Args create_args = PJRT_Client_Create_Args{
      .struct_size = PJRT_Client_Create_Args_STRUCT_SIZE,
      .priv = nullptr,
      .client = nullptr,
  };
  PJRT_Error* error = api_->PJRT_Client_Create(&create_args);
  CHECK_EQ(error, nullptr);
  CHECK_NE(create_args.client, nullptr);
  return create_args.client;
}

void PjrtCApiTestBase::initialize() {
  client_ = make_client();
  cc_pjrt_client_ = client_->client.get();
}
}  // namespace pjrt
}  // namespace xla
