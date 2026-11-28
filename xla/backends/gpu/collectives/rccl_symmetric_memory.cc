/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/backends/gpu/collectives/rccl_symmetric_memory.h"

#include <memory>
#include <string>
#include <utility>

#include "absl/functional/any_invocable.h"
#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "rocm/rocm_config.h"  // defines TF_ROCM_VERSION
#include "xla/backends/gpu/collectives/rccl_errors.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/future.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/rocm/rocm_vmm_allocator.h"
#include "xla/tsl/concurrency/executor.h"
#include "xla/util.h"

#if (TF_ROCM_VERSION >= 100000)
// ROCm/RCCL 10.0+ exposes the host-side symmetric-memory pointer helper
// ncclGetPeerDevicePointer as an `NCCL_EXTERN_C __host__` symbol exported from
// librccl.so.
//
// These are declared in RCCL's nccl_device/core.h, but that header is NOT
// shipped by the ROCm install -- the installed RCCL dev package only provides
// rocm/include/rccl/rccl.h (nccl_device/ lives solely in the RCCL source
// tree).
//
// PROVENANCE: copied verbatim from RCCL hip-version_10.2.62640,
// nccl_device/core.h.
extern "C" {
ncclResult_t ncclGetPeerDevicePointer(ncclWindow_t window, size_t offset,
                                      int peer, void** outPtr);
}  // extern "C"
#endif  // TF_ROCM_VERSION >= 100000

namespace xla::gpu {

namespace {

#if (TF_ROCM_VERSION >= 100000)
// Only used by the destructor's ROCm/RCCL 10.0+ path (window deregistration).
// Guard the definition so it is not compiled as an unused static function on
// ROCm < 10.0 builds (which would trip -Wunused-function under -Werror).
Future<> Execute(absl::AnyInvocable<absl::Status() &&> f,
                 std::shared_ptr<tsl::Executor> executor) {
  return executor ? MakeFutureOn<void>(*executor, std::move(f))
                  : Future<>(std::move(f)());
}

constexpr absl::string_view kCuMemDisabledError =
    "RCCL symmetric memory (NCCL windows) requires NCCL_CUMEM_ENABLE=1. On "
    "ROCm this variable defaults to 0, which leaves the communicator's "
    "symmetricSupport disabled and makes "
    "ncclCommWindowRegister(NCCL_WIN_COLL_SYMMETRIC) silently fall back to a "
    "non-symmetric window. Export NCCL_CUMEM_ENABLE=1 before launching to "
    "enable symmetric memory, or disable the symmetric-memory collective "
    "path.";

// Hint printed when ncclCommWindowRegister rejects a buffer.
constexpr absl::string_view kSymmetricMemoryAllocatorHint =
    "Only VMM-backed buffers can be registered as RCCL symmetric windows. "
    "Re-run with --xla_gpu_enable_allocator_spatial_partitioning=false, so "
    "that collective memory is allocated through ncclMemAlloc, or with "
    "--xla_gpu_enable_allocator_spatial_partitioning=true "
    "--xla_gpu_command_buffer_update_mode=SKIP_TEMP to use the VMM allocator.";
#endif  // TF_ROCM_VERSION >= 100000

}  // namespace

RcclSymmetricMemory::RcclSymmetricMemory(
    ncclComm_t comm, ncclWindow_t win, stream_executor::DeviceAddressBase addr,
    std::shared_ptr<tsl::Executor> executor)
    : comm_(comm), win_(win), addr_(addr), executor_(std::move(executor)) {}

absl::StatusOr<std::unique_ptr<RcclSymmetricMemory>>
RcclSymmetricMemory::Create(ncclComm_t comm,
                            stream_executor::DeviceAddressBase addr,
                            std::shared_ptr<tsl::Executor> executor,
                            bool symmetric_window) {
  VLOG(3) << absl::StrFormat(
      "Create RCCL symmetric memory on comm=%p from: ptr=%p; size=%ld; "
      "symmetric_window=%v",
      comm, addr.opaque(), addr.size(), symmetric_window);
#if (TF_ROCM_VERSION >= 100000)
  // Require NCCL_CUMEM_ENABLE=1 before touching the window-registration path.
  // Without it RCCL's comm->symmetricSupport is false and the registration
  // below silently downgrades to a non-symmetric window.
  if (!stream_executor::gpu::IsNcclCuMemEnabled()) {
    LOG_FIRST_N(WARNING, 1) << kCuMemDisabledError;
    return absl::FailedPreconditionError(kCuMemDisabledError);
  }

  // ncclCommWindowRegister is available in ROCm/RCCL 10.0+.
  ncclWindow_t win = nullptr;
  int win_flags = symmetric_window ? NCCL_WIN_COLL_SYMMETRIC : NCCL_WIN_DEFAULT;
  if (absl::Status status = XLA_RCCL_STATUS(ncclCommWindowRegister(
          comm, addr.opaque(), addr.size(), &win, win_flags));
      !status.ok()) {
    // Carry the hint in the status, not just in the log: the status is what
    // reaches the user, the log line may be buried in other ranks' output.
    absl::Status annotated(
        status.code(),
        absl::StrCat(status.message(), " ", kSymmetricMemoryAllocatorHint));
    LOG(ERROR) << absl::StrFormat(
        "Failed to register an RCCL symmetric window on comm=%p (ptr=%p, "
        "size=%ld): %s",
        comm, addr.opaque(), addr.size(), annotated.message());
    return annotated;
  }

  // Validate that the window is genuinely symmetric / load-store accessible.
  //
  // ncclCommWindowRegister(NCCL_WIN_COLL_SYMMETRIC) can return ncclSuccess even
  // when the runtime silently downgrades the registration to a non-symmetric
  // window (e.g. when the device/communicator does not actually support LSA
  // symmetric memory: the RCCL log shows "windowRegisterNonSym" and
  // "Symmetric memory is not supported"). In that case the window is NOT usable
  // with ncclGetPeerDevicePointer, and the collective kernel would later crash
  // deep inside RCCL ("Could not find communicator matching window").
  void* self_peer_addr = nullptr;
  ncclResult_t probe_status =
      ncclGetPeerDevicePointer(win, 0, /*peer=*/0, &self_peer_addr);
  if (probe_status != ncclSuccess || self_peer_addr == nullptr) {
    // The window is not a usable symmetric/LSA window. Deregister it and fail
    // cleanly rather than leaking the window or crashing later.
    XLA_RCCL_LOG_IF_ERROR(ncclCommWindowDeregister(comm, win));
    const char* rccl_err = probe_status != ncclSuccess
                               ? ncclGetErrorString(probe_status)
                               : "null peer pointer";
    return absl::FailedPreconditionError(absl::StrFormat(
        "ncclCommWindowRegister(NCCL_WIN_COLL_SYMMETRIC) succeeded on comm=%p "
        "(ptr=%p, size=%ld) but the window is not load/store accessible "
        "(ncclGetPeerDevicePointer probe failed: %s). The device/communicator "
        "does not support symmetric (LSA) memory for this clique, so the "
        "Triton "
        "one-shot collective kernel cannot be used. Fall back to a "
        "non-symmetric collective strategy (e.g. the library collective or the "
        "rendezvous path).",
        comm, addr.opaque(), addr.size(), rccl_err));
  }

  return absl::WrapUnique(
      new RcclSymmetricMemory(comm, win, addr, std::move(executor)));
#else
  // Symmetric-memory windows require ncclCommWindowRegister (ROCm/RCCL
  // >= 10.0).
  return absl::UnimplementedError(
      "RCCL symmetric memory requires ncclCommWindowRegister (ROCm/RCCL "
      ">= 10.0)");
#endif  // TF_ROCM_VERSION >= 100000
}

RcclSymmetricMemory::~RcclSymmetricMemory() {
#if (TF_ROCM_VERSION >= 100000)
  // ncclCommWindowDeregister is available in ROCm/RCCL 10.0+.
  //
  // comm_ is a non-owning ncclComm_t. If this window outlived its
  // RcclCommunicator (whose destructor calls ncclCommDestroy), the
  // ncclCommWindowDeregister() below would be a use-after-free. Callers (the
  // collective clique) must destroy all windows before destroying the
  // communicator.

  // The deregistration below is posted to executor_ and awaited. If this
  // destructor ran on executor_'s thread (e.g. released inside another Execute
  // callback), that would self-deadlock.
  VLOG(3) << absl::StrFormat(
      "Destroy %v with addr=%p, size=%ld executor=%p (awaiting window "
      "deregistration on executor)",
      *this, addr_.opaque(), addr_.size(), executor_.get());

  absl::Status status =
      Execute(
          [&] {
            if (win_ != nullptr) {
              XLA_RCCL_LOG_IF_ERROR(ncclCommWindowDeregister(comm_, win_));
            }
            return absl::OkStatus();
          },
          executor_)
          .Await();
  if (!status.ok()) {
    LOG(ERROR) << "Failed to destroy RCCL symmetric memory: " << status;
  }
#endif  // TF_ROCM_VERSION >= 100000
}

stream_executor::DeviceAddressBase RcclSymmetricMemory::addr() const {
  return addr_;
}

absl::StatusOr<stream_executor::DeviceAddressBase>
RcclSymmetricMemory::peer_addr(RankId peer) const {
#if (TF_ROCM_VERSION >= 100000)
  // ncclGetPeerDevicePointer is available in ROCm/RCCL 10.0+.
  void* peer_addr = nullptr;
  XLA_RCCL_RETURN_IF_ERROR(
      ncclGetPeerDevicePointer(win_, 0, peer.value(), &peer_addr));
  if (peer_addr) {
    return stream_executor::DeviceAddressBase(peer_addr, addr_.size());
  }
  return absl::FailedPreconditionError(absl::StrFormat(
      "Peer rank %d is not load/store accessible from this rank",
      peer.value()));
#else
  return absl::UnimplementedError(
      "Peer address requires ncclGetPeerDevicePointer (ROCm/RCCL >= 10.0)");
#endif  // TF_ROCM_VERSION >= 100000
}

std::string RcclSymmetricMemory::ToString() const {
  return absl::StrFormat(
      "RcclSymmetricMemory(comm=%p, win=%p, ptr=%p, size=%ld)", comm_, win_,
      addr_.opaque(), addr_.size());
}

RcclSymmetricMemory::PackedKernelArg RcclSymmetricMemory::PackKernelArg()
    const {
  return win_;
}

}  // namespace xla::gpu
