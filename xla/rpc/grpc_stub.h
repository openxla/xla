/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#ifndef XLA_RPC_GRPC_STUB_H_
#define XLA_RPC_GRPC_STUB_H_

#include "xla/rpc/xla_service.grpc.pb.h"
#include "xla/service_interface.h"
#include "xla/xla_data.pb.h"

namespace xla {

class GRPCStub : public ServiceInterface {
 public:
  explicit GRPCStub(grpc::XlaService::Stub* stub) : grpc_stub_(stub) {}
  ~GRPCStub() override;

  Status TransferToClient(const TransferToClientRequest* request,
                          TransferToClientResponse* response) override;

  Status TransferToServer(const TransferToServerRequest* request,
                          TransferToServerResponse* response) override;

  Status TransferToInfeed(const TransferToInfeedRequest* request,
                          TransferToInfeedResponse* response) override;

  Status TransferFromOutfeed(const TransferFromOutfeedRequest* request,
                             TransferFromOutfeedResponse* response) override;

  Status ResetDevice(const ResetDeviceRequest* request,
                     ResetDeviceResponse* response) override;

  Status Compile(const CompileRequest* request,
                 CompileResponse* response) override;

  Status Execute(const ExecuteRequest* request,
                 ExecuteResponse* response) override;

  Status ExecuteGraphParallel(const ExecuteGraphParallelRequest* request,
                              ExecuteParallelResponse* response) override;

  Status WaitForExecution(const WaitForExecutionRequest* request,
                          WaitForExecutionResponse* response) override;

  Status DeconstructTuple(const DeconstructTupleRequest* request,
                          DeconstructTupleResponse* response) override;

  Status GetComputationGraphStats(const ComputationGraphStatsRequest* request,
                                  ComputationStatsResponse* response) override;

  Status GetShape(const GetShapeRequest* request,
                  GetShapeResponse* response) override;

  Status GetDeviceHandles(const GetDeviceHandlesRequest* request,
                          GetDeviceHandlesResponse* response) override;

  Status CreateChannelHandle(const CreateChannelHandleRequest* request,
                             CreateChannelHandleResponse* response) override;

  Status ComputeConstantGraph(const ComputeConstantGraphRequest* request,
                              ComputeConstantResponse* response) override;

  // Methods used by GlobalData.
  Status Unregister(const UnregisterRequest* request,
                    UnregisterResponse* response) override;

  grpc::XlaService::Stub* service() { return grpc_stub_; }

 private:
  grpc::XlaService::Stub* grpc_stub_;

  GRPCStub(const GRPCStub&) = delete;
  GRPCStub& operator=(const GRPCStub&) = delete;
};

}  // namespace xla

#endif  // XLA_RPC_GRPC_STUB_H_
