/*
 * Copyright 2023 The OpenXLA Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef XLA_PYTHON_IFRT_PROXY_COMMON_GRPC_CREDENTIALS_H_
#define XLA_PYTHON_IFRT_PROXY_COMMON_GRPC_CREDENTIALS_H_

#include <memory>

#include "grpcpp/security/credentials.h"  // from @com_github_grpc_grpc
#include "grpcpp/security/server_credentials.h"  // from @com_github_grpc_grpc

namespace xla {
namespace ifrt {
namespace proxy {

// Get credentials to use in the client gRPC.
std::shared_ptr<::grpc::ChannelCredentials> GetClientCredentials();

// Get credentials to use in the server gRPC.
std::shared_ptr<::grpc::ServerCredentials> GetServerCredentials();

}  // namespace proxy
}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_IFRT_PROXY_COMMON_GRPC_CREDENTIALS_H_
