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

#include "xla/stream_executor/memory_reservation.h"

#include <cstddef>
#include <utility>
#include <vector>

#include "absl/cleanup/cleanup.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/memory_allocation.h"
#include "tsl/platform/errors.h"

namespace stream_executor {

// ScopedMapping

void MemoryReservation::ScopedMapping::UnmapAndLogIfError(
    MemoryReservation* reservation, size_t reservation_offset, size_t size) {
  absl::Status status = reservation->UnMap(reservation_offset, size);
  if (!status.ok()) {
    LOG(ERROR) << "ScopedMapping: failed to unmap reservation range: "
               << status.message();
  }
}

MemoryReservation::ScopedMapping::ScopedMapping(MemoryReservation* reservation,
                                                size_t reservation_offset,
                                                size_t size)
    : reservation_(reservation),
      reservation_offset_(reservation_offset),
      size_(size) {}

MemoryReservation::ScopedMapping::~ScopedMapping() {
  if (reservation_ == nullptr) {
    return;
  }
  UnmapAndLogIfError(reservation_, reservation_offset_, size_);
}

MemoryReservation::ScopedMapping::ScopedMapping(ScopedMapping&& other) noexcept
    : reservation_(other.reservation_),
      reservation_offset_(other.reservation_offset_),
      size_(other.size_) {
  other.reservation_ = nullptr;
}

MemoryReservation::ScopedMapping& MemoryReservation::ScopedMapping::operator=(
    ScopedMapping&& other) noexcept {
  if (this != &other) {
    if (reservation_ != nullptr) {
      UnmapAndLogIfError(reservation_, reservation_offset_, size_);
    }
    reservation_ = other.reservation_;
    reservation_offset_ = other.reservation_offset_;
    size_ = other.size_;
    other.reservation_ = nullptr;
  }
  return *this;
}

DeviceAddressBase MemoryReservation::ScopedMapping::mapped_address() const {
  return reservation_->address().GetByteSlice(reservation_offset_, size_);
}

// MemoryReservation::MapTo

absl::StatusOr<MemoryReservation::ScopedMapping> MemoryReservation::MapTo(
    size_t reservation_offset, size_t allocation_offset, size_t size,
    MemoryAllocation& allocation) {
  TF_RETURN_IF_ERROR(
      Map(reservation_offset, allocation_offset, size, allocation));

  auto cleanup = absl::MakeCleanup([&] {
    absl::Status unmap_status = UnMap(reservation_offset, size);
    if (!unmap_status.ok()) {
      LOG(ERROR) << "MapTo: failed to unmap after failure: "
                 << unmap_status.message();
    }
  });

  TF_RETURN_IF_ERROR(SetAccess(reservation_offset, size));

  std::move(cleanup).Cancel();
  return ScopedMapping(this, reservation_offset, size);
}

absl::StatusOr<MemoryReservation::ScopedMapping> MemoryReservation::MapTo(
    absl::Span<const MappingDescriptor> mappings) {
  if (mappings.empty()) {
    return absl::InvalidArgumentError("MapTo: mappings must not be empty");
  }

  size_t start_offset = mappings[0].reservation_offset;
  size_t expected_offset = start_offset;
  for (const MappingDescriptor& desc : mappings) {
    if (desc.reservation_offset != expected_offset) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "MapTo: mappings are not contiguous. Expected reservation_offset=%zu "
          "but got %zu",
          expected_offset, desc.reservation_offset));
    }
    expected_offset += desc.size;
  }

  size_t total_size = 0;

  auto cleanup = absl::MakeCleanup([&] {
    if (total_size > 0) {
      absl::Status unmap_status = UnMap(start_offset, total_size);
      if (!unmap_status.ok()) {
        LOG(ERROR) << "MapTo: failed to unmap after failure: "
                   << unmap_status.message();
      }
    }
  });

  for (const MappingDescriptor& desc : mappings) {
    TF_RETURN_IF_ERROR(Map(desc.reservation_offset, desc.allocation_offset,
                           desc.size, *desc.allocation));
    total_size += desc.size;
  }

  TF_RETURN_IF_ERROR(SetAccess(start_offset, total_size));

  std::move(cleanup).Cancel();
  return ScopedMapping(this, start_offset, total_size);
}

// ScopedMapping::Remap

absl::StatusOr<MemoryReservation::ScopedMapping>
MemoryReservation::ScopedMapping::Remap(
    absl::Span<const MemoryReservation::RemappingDescriptor> mappings) && {
  if (reservation_ == nullptr) {
    return absl::FailedPreconditionError("Remap: mapping is empty");
  }
  if (mappings.empty()) {
    return absl::InvalidArgumentError("Remap: mappings must not be empty");
  }

  MemoryReservation* reservation = reservation_;
  const size_t existing_reservation_offset = reservation_offset_;
  const size_t existing_size = size_;

  const size_t start_offset = mappings[0].reservation_offset;
  size_t expected_offset = start_offset;
  for (const MemoryReservation::RemappingDescriptor& desc : mappings) {
    if (desc.reservation_offset != expected_offset) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Remap: mappings are not contiguous. Expected "
                          "reservation_offset=%zu but got %zu",
                          expected_offset, desc.reservation_offset));
    }
    if (desc.allocation == nullptr) {
      return absl::InvalidArgumentError("Remap: allocation must not be null");
    }
    expected_offset += desc.size;
  }
  const size_t total_size = expected_offset - start_offset;
  if (start_offset != existing_reservation_offset ||
      total_size != existing_size) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Remap: mappings must cover the existing mapping range [%zu, %zu), got "
        "[%zu, %zu)",
        existing_reservation_offset,
        existing_reservation_offset + existing_size, start_offset,
        start_offset + total_size));
  }

  // Track per-slice mapped state for cleanup on failure. Every slice starts
  // mapped because this ScopedMapping owns the full range. Changed slices are
  // temporarily unmapped before being mapped to their new allocation.
  std::vector<bool> slice_mapped(mappings.size(), true);

  // Detach the prior ScopedMapping without invoking its destructor; the
  // per-slice unmaps below replace the full-range unmap it would do.
  reservation_ = nullptr;

  // On failure, unmap every slice that is still mapped so the reservation
  // is left in a clean (fully unmapped) state rather than partially mapped
  // with no RAII owner.
  auto cleanup = absl::MakeCleanup([&] {
    for (size_t k = 0; k < mappings.size(); ++k) {
      if (slice_mapped[k]) {
        absl::Status s = reservation->UnMap(mappings[k].reservation_offset,
                                            mappings[k].size);
        if (!s.ok()) {
          LOG(ERROR) << "Remap: cleanup failed to unmap slice at offset "
                     << mappings[k].reservation_offset << ": " << s.message();
        }
      }
    }
  });

  // Walk descriptors in order, processing maximal contiguous runs of slices
  // that require remapping. For each run we issue per-slice UnMap and Map,
  // then a single SetAccess covering the whole run.
  size_t i = 0;
  while (i < mappings.size()) {
    if (!mappings[i].remap_required) {
      ++i;
      continue;
    }
    // Coalesce: extend this run as long as the next descriptor is also
    // being remapped. Map/UnMap stay per-slice (cuMemMap requires per-handle
    // calls); SetAccess is the expensive call to coalesce.
    const size_t run_start = mappings[i].reservation_offset;
    size_t run_size = 0;
    size_t j = i;
    while (j < mappings.size() && mappings[j].remap_required) {
      const auto& dj = mappings[j];
      TF_RETURN_IF_ERROR(reservation->UnMap(dj.reservation_offset, dj.size));
      slice_mapped[j] = false;
      TF_RETURN_IF_ERROR(reservation->Map(dj.reservation_offset,
                                          dj.allocation_offset, dj.size,
                                          *dj.allocation));
      slice_mapped[j] = true;
      run_size += dj.size;
      ++j;
    }
    TF_RETURN_IF_ERROR(reservation->SetAccess(run_start, run_size));
    i = j;
  }

  std::move(cleanup).Cancel();
  return ScopedMapping(reservation, start_offset, total_size);
}

}  // namespace stream_executor
