# Command-Buffer Virtual Addresses for Temporary Buffers

GPU command buffers record the device addresses used by kernels and library
calls. If an address changes on a later execution, XLA must update the affected
command-buffer nodes before replaying them. This document describes the two
allocation modes available for controlling that behavior.

## Update modes

The `xla_gpu_command_buffer_update_mode` debug option has two values:

- `DYNAMIC_ALLOCATE` is the default. Every temporary and output allocation uses
  the client's normal allocator. Command-buffer nodes are updated whenever a
  referenced address changes.
- `VMM_PERSISTENT_TEMP` gives command-buffer-referenced preallocated temporary
  buffers stable virtual addresses backed by the VMM allocator. Parameters and
  live-out buffers continue to use their ordinary dynamic addresses.

The former names `ALWAYS_UPDATE`, `NEVER_UPDATE`, and
`CAPTURE_CMD_NEVER_UPDATE` have been removed and are rejected by flag and text
proto parsing.

## Why only temporary buffers?

Entry parameters and live-out buffers cross the executable boundary. Their
addresses are owned by the caller or by the result allocation path, so they can
legitimately change from run to run. The command buffer must continue updating
commands that reference those allocations.

Preallocated temporary buffers are owned by the executable's allocation path.
XLA can therefore assign them a stable virtual address without changing the
external buffer contract. Constants already have stable global addresses and do
not require VMM remapping.

Only nonzero temporary allocations referenced by a command buffer participate.
Unreferenced temporaries continue through the normal allocation path.

## How `VMM_PERSISTENT_TEMP` works

For each executable and device, XLA collects the referenced preallocated
temporaries and creates one virtual-address reservation. Each temporary receives
a deterministic offset in that reservation.

On each execution, its physical storage is allocated with:

```cpp
DeviceAddressVmmAllocator::Allocate(
    /* ... reservation and offset ... */,
    /*return_reservation_address=*/true);
```

The returned buffer points into the executable's reserved virtual range, so the
command buffer observes the same address on every execution. The physical
allocation and mapping still follow the execution buffer's lifetime; the
persistent property is the reserved virtual address.

The command-buffer address policy is then:

| Allocation kind                  | VMM reserved address | Treated as persistent |
|----------------------------------|----------------------|-----------------------|
| Referenced preallocated temp     | Yes                  | Yes                   |
| Referenced constant              | No                   | Yes                   |
| Entry parameter                  | No                   | No                    |
| Live-out/output allocation       | No                   | No                    |
| Unreferenced or zero-sized temp  | No                   | No                    |

Commands that reference a dynamic parameter or output still update normally.
Commands whose referenced addresses are all persistent can be replayed without
an address update.

## Allocator selection and fallback

Using this mode requires a `DeviceAddressVmmAllocator`. Selecting
`VMM_PERSISTENT_TEMP` globally causes the GPU client to select the VMM
allocator. A per-executable compile option can also request the mode.

If an executable requests `VMM_PERSISTENT_TEMP` while its client uses a
non-VMM allocator, XLA safely falls back to dynamic allocation and ordinary
command-buffer updates. This keeps the executable functional, but it does not
provide stable temporary addresses.

## Choosing a mode

Use `DYNAMIC_ALLOCATE` unless command-buffer update cost is significant and the
platform supports the VMM allocator. Use `VMM_PERSISTENT_TEMP` to stabilize
internal temporary addresses while preserving normal parameter and result
allocation semantics.
