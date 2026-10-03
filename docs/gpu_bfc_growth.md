# GPU BFC growth in a reserved virtual arena

Preallocated spatial GPU BFC pools grow by default when using device memory.
C++ clients can set `GpuAllocatorConfig::bfc_allow_growth=false` to keep the
initial shared pool fixed. The memory fraction or absolute size determines the
initial physical allocation, which must succeed in full. Total device memory,
rounded down to the mapping granularity, is the growth cap. Reserving this
capacity does not allocate physical memory or guarantee that growth will succeed.
Non-spatial pools, unified memory, and other allocator kinds retain their
existing behavior.

## Virtual reservation and physical backing

On CUDA, the existing `DeviceMemAllocator` reserves one VA range for the cap
before BFC's first allocation. `CudaMemoryReservation::Create` calls
`cuMemAddressReserve`; the entire range is initially unmapped. BFC first asks for
the configured preallocation size, rounded up to the mapping granularity.
`CudaRawMemoryAllocation::Create` obtains physical memory with `cuMemCreate`.
`MemoryReservation::MapTo` maps it at the reservation base with `cuMemMap`, then
grants access to the local device and P2P-capable peers with `cuMemSetAccess`.
The rest of the reservation stays unmapped and consumes no GPU physical memory.

An S(0) allocation first tries its owned holes and the central gap. On a miss,
BFC calls its existing `Extend` path, using normal growth sizing and backpedaling.
`DeviceMemAllocator` allocates additional physical backing and maps it immediately
after the mapped prefix. It never unmaps or moves existing backing. Only after
mapping and access setup succeed does it publish the new address and byte count
to BFC. Failure releases only the new backing (and unmaps the new slice if access
setup failed); the existing prefix and BFC accounting remain unchanged.

```text
                     fixed S(1) limit P                  VA cap C
Initial: | S(1) -> | gap | <- S(0) |....... unmapped ...........|
Growth:  | S(1) -> | gap | <- S(0) |  more S(0)  |.. unmapped ..|
         ^ base never moves                    ^ mapped end M
```

The three limits are distinct: reserved capacity C, physically mapped capacity
M, and the initial collective allocation limit P. Growth advances M, up to C;
it never changes P or the reservation base. A physical allocation may fail below
C if cuDNN, NCCL, another process, or other users consume available device memory.

## BFC region extension and collective safety

In reserved mode, `DeviceMemAllocator::SupportsCoalescing()` is true and each
successful extension is adjacent to the preceding prefix. BFC's existing
`RegionManager::AddOrExtendAllocationRegion` extends the region's end and handle
table. The new chunk is upper-owned and coalesces with eligible adjacent free
chunks. S(0) buffers can span physical mapping boundaries. S(1), including
alignment and padding, remains confined to the original allocation even when a
free span crosses P. A lower-end request never triggers growth.

Near the cap, BFC can request just enough additional memory to complete an
immediately safe free tail. Generic suballocators that do not reserve VA retain
separate-region growth: nonadjacent allocations stay upper-owned and cannot
coalesce. Only an explicitly unsupported reservation falls back to that path;
actual CUDA reservation failures are reported.

Allocation policy remains attached to the memory space: S(1) uses ascending
equal-size holes and exact splitting; S(0) uses descending equal-size holes and
the BFC heuristic for both holes and gap carves.

Ranks may grow independently. Optional automatic registration of ordinary
backing allocations is disabled for growing pools; collective-window
registration continues to use the fixed S(1) range. Existing buffers never
move or change backing. Each physical handle is retained until its mapping is
unmapped at teardown, then the VA reservation is released.

Growth adds no headroom or future collective-gap reservation. Live S(0)
buffers can still occupy initial capacity needed by a later S(1) request.
Freed buffers return to BFC; backing allocations remain until destruction.
The PjRt cross-host fabric descriptor carries one physical handle, so exporting
an S(0) buffer spanning mappings returns an explicit unsupported-transfer error.
Representing multiple handles in that protocol is a separate extension.
