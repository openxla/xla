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

#include <cstddef>
#include <memory>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "rocm/include/hip/hip_runtime.h"
#include "rocm/include/rccl/rccl.h"
#include "xla/stream_executor/device_address.h"

namespace xla::gpu {
namespace {

namespace se = stream_executor;

using ::absl_testing::StatusIs;
using ::testing::HasSubstr;

// RcclSymmetricMemory::Create() validates that the registered window is
// genuinely symmetric / load/store accessible (LSA): on ROCm/RCCL >= 10.0 it
// probes the window with ncclGetPeerDevicePointer and returns
// FailedPrecondition (after deregistering the window) when the
// device/communicator does not support symmetric memory for this clique. A
// single-rank communicator may or may not back an LSA window depending on the
// device/runtime, so the tests below skip (via SKIP_IF_SYMMETRIC_MEMORY_
// UNAVAILABLE) when Create() reports FailedPrecondition rather than failing.
// When Create() does succeed, the window is guaranteed to be non-null (a null
// or non-LSA window would have failed the probe), so there is no separate
// null-window case to handle.
//
// A macro is used so that the GTEST_SKIP()'s `return` exits the enclosing
// test body.
#define SKIP_IF_SYMMETRIC_MEMORY_UNAVAILABLE(status)                      \
  do {                                                                    \
    const absl::Status& _s = (status);                                    \
    if (_s.code() == absl::StatusCode::kFailedPrecondition) {             \
      GTEST_SKIP() << "Symmetric (LSA) memory is not supported for this " \
                      "single-rank communicator: "                        \
                   << _s;                                                 \
    }                                                                     \
  } while (0)

// Test fixture that creates a single-rank RCCL communicator and allocates a
// small GPU buffer. Tests are skipped if no ROCm device or RCCL is available.
class RcclSymmetricMemoryTest : public ::testing::Test {
 protected:
  void SetUp() override {
#if (TF_ROCM_VERSION < 100000)
    // RcclSymmetricMemory::Create requires ncclCommWindowRegister (ROCm/RCCL
    // >= 10.0) and now returns UnimplementedError on older toolkits, so there
    // is nothing meaningful to exercise here.
    GTEST_SKIP() << "RCCL symmetric memory requires ROCm/RCCL >= 10.0";
#endif  // TF_ROCM_VERSION < 100000
    // Skip if no ROCm/HIP GPU is present.
    int device_count = 0;
    if (hipGetDeviceCount(&device_count) != hipSuccess || device_count < 1) {
      GTEST_SKIP() << "No ROCm/HIP GPU device available";
    }
    if (hipSetDevice(0) != hipSuccess) {
      GTEST_SKIP() << "hipSetDevice(0) failed";
    }

    // Initialise a single-rank RCCL communicator on device 0.
    int device_ids[] = {0};
    ncclResult_t nccl_err = ncclCommInitAll(&comm_, 1, device_ids);
    if (nccl_err != ncclSuccess) {
      GTEST_SKIP() << "RCCL communicator initialisation failed: "
                   << ncclGetErrorString(nccl_err);
    }

    // Allocate a 4 KiB symmetric memory buffer via RCCL. ncclMemAlloc ensures
    // the buffer is mapped at the same virtual address on all ranks, which is
    // required for ncclCommWindowRegister with NCCL_WIN_COLL_SYMMETRIC.
    constexpr size_t kBufSize = 4096;
    buf_size_ = kBufSize;
    if (ncclMemAlloc(&buf_ptr_, buf_size_) != ncclSuccess) {
      ncclCommDestroy(comm_);
      comm_ = nullptr;
      GTEST_SKIP() << "ncclMemAlloc(" << kBufSize << ") failed";
    }
  }

  void TearDown() override {
    if (buf_ptr_ != nullptr) {
      (void)ncclMemFree(buf_ptr_);
      buf_ptr_ = nullptr;
    }
    if (comm_ != nullptr) {
      ncclCommDestroy(comm_);
      comm_ = nullptr;
    }
  }

  ncclComm_t comm_ = nullptr;
  void* buf_ptr_ = nullptr;
  size_t buf_size_ = 0;
};

// Verifies that RcclSymmetricMemory::Create succeeds for a valid buffer when
// the device/communicator supports symmetric (LSA) memory. If this single-rank
// communicator does not back an LSA window, Create() returns FailedPrecondition
// and the test is skipped.
TEST_F(RcclSymmetricMemoryTest, CreateSucceeds) {
  se::DeviceAddressBase addr(buf_ptr_, buf_size_);
  auto result = RcclSymmetricMemory::Create(comm_, addr, /*executor=*/nullptr);
  SKIP_IF_SYMMETRIC_MEMORY_UNAVAILABLE(result.status());
  ASSERT_OK_AND_ASSIGN(auto symm_mem, std::move(result));
  ASSERT_NE(symm_mem, nullptr);
  // A successful Create() always yields a usable (non-null) window; a null or
  // non-LSA window would have failed the ncclGetPeerDevicePointer probe.
  EXPECT_NE(symm_mem->win(), nullptr);
}

// Verifies that addr() returns exactly the pointer and size that were
// registered. Skipped when symmetric memory is unavailable for this comm.
TEST_F(RcclSymmetricMemoryTest, AddrMatchesRegisteredBuffer) {
  se::DeviceAddressBase addr(buf_ptr_, buf_size_);
  auto result = RcclSymmetricMemory::Create(comm_, addr, /*executor=*/nullptr);
  SKIP_IF_SYMMETRIC_MEMORY_UNAVAILABLE(result.status());
  ASSERT_OK_AND_ASSIGN(auto symm_mem, std::move(result));
  EXPECT_EQ(symm_mem->addr().opaque(), buf_ptr_);
  EXPECT_EQ(symm_mem->addr().size(), buf_size_);
}

// Verifies that ToString() contains key diagnostic fields. Skipped when
// symmetric memory is unavailable for this comm.
TEST_F(RcclSymmetricMemoryTest, ToStringContainsExpectedFields) {
  se::DeviceAddressBase addr(buf_ptr_, buf_size_);
  auto result = RcclSymmetricMemory::Create(comm_, addr, /*executor=*/nullptr);
  SKIP_IF_SYMMETRIC_MEMORY_UNAVAILABLE(result.status());
  ASSERT_OK_AND_ASSIGN(auto symm_mem, std::move(result));
  const std::string str = symm_mem->ToString();
  EXPECT_THAT(str, HasSubstr("RcclSymmetricMemory"));
  EXPECT_THAT(str, HasSubstr("comm="));
  EXPECT_THAT(str, HasSubstr("win="));
  EXPECT_THAT(str, HasSubstr("ptr="));
}

// Verifies that PackKernelArg() returns a non-null handle, and that the
// win() accessor returns the same underlying ncclWindow_t.
//
// A successful Create() always yields a non-null, load/store-accessible window
// (the ncclGetPeerDevicePointer probe in Create() rejects null / non-symmetric
// windows). When this single-rank communicator does not back an LSA window,
// Create() returns FailedPrecondition and the test is skipped.
TEST_F(RcclSymmetricMemoryTest, PackKernelArgReturnsValidWindowHandle) {
  se::DeviceAddressBase addr(buf_ptr_, buf_size_);
  auto result = RcclSymmetricMemory::Create(comm_, addr, /*executor=*/nullptr);
  SKIP_IF_SYMMETRIC_MEMORY_UNAVAILABLE(result.status());
  ASSERT_OK_AND_ASSIGN(auto symm_mem, std::move(result));
  // PackKernelArg() returns a void* (PackedKernelArg) wrapping the window.
  auto packed = symm_mem->PackKernelArg();
  EXPECT_NE(packed, nullptr);
  // win() returns the typed ncclWindow_t handle directly.
  EXPECT_NE(symm_mem->win(), nullptr);
}

// Verifies that multimem_addr() reports multimem as unsupported.
//
// No AMD GPU provides multicast memory, so RCCL always reports
// comm->nvlsSupport == 0. RcclSymmetricMemory does not override
// multimem_addr(); the base class returns Unimplemented.
TEST_F(RcclSymmetricMemoryTest, MultimemAddrIsUnimplemented) {
  se::DeviceAddressBase addr(buf_ptr_, buf_size_);
  auto create_result =
      RcclSymmetricMemory::Create(comm_, addr, /*executor=*/nullptr);
  SKIP_IF_SYMMETRIC_MEMORY_UNAVAILABLE(create_result.status());
  ASSERT_OK_AND_ASSIGN(auto symm_mem, std::move(create_result));
  EXPECT_THAT(symm_mem->multimem_addr(),
              StatusIs(absl::StatusCode::kUnimplemented));
}

// Verifies that two independent windows created from the same communicator
// on different buffers have distinct ncclWindow_t handles. Skipped when
// symmetric memory is unavailable for this comm.
TEST_F(RcclSymmetricMemoryTest, TwoWindowsHaveDistinctHandles) {
  void* buf2_ptr = nullptr;
  ASSERT_EQ(ncclMemAlloc(&buf2_ptr, buf_size_), ncclSuccess);

  se::DeviceAddressBase addr1(buf_ptr_, buf_size_);
  se::DeviceAddressBase addr2(buf2_ptr, buf_size_);

  auto result1 =
      RcclSymmetricMemory::Create(comm_, addr1, /*executor=*/nullptr);
  auto result2 =
      RcclSymmetricMemory::Create(comm_, addr2, /*executor=*/nullptr);
  if (result1.status().code() == absl::StatusCode::kFailedPrecondition ||
      result2.status().code() == absl::StatusCode::kFailedPrecondition) {
    (void)ncclMemFree(buf2_ptr);
    GTEST_SKIP() << "Symmetric (LSA) memory is not supported for this "
                    "single-rank communicator; skipping distinctness check.";
  }
  ASSERT_OK_AND_ASSIGN(auto symm1, std::move(result1));
  ASSERT_OK_AND_ASSIGN(auto symm2, std::move(result2));

  // Successful Create() guarantees non-null windows, so the handles must be
  // distinct for distinct buffers.
  EXPECT_NE(symm1->win(), symm2->win());
  EXPECT_EQ(symm1->addr().opaque(), buf_ptr_);
  EXPECT_EQ(symm2->addr().opaque(), buf2_ptr);

  (void)ncclMemFree(buf2_ptr);
}

}  // namespace
}  // namespace xla::gpu
