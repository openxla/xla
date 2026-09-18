# GPU BFC growth through the existing suballocator

Preallocated spatial GPU BFC pools grow by default when using device memory.
C++ clients can set `GpuAllocatorConfig::bfc_allow_growth=false` to keep the
initial shared pool fixed. The memory fraction or absolute size determines the
initial allocation, which must succeed in full. Total device memory is the cap
across all regions. Non-spatial pools, unified memory, and other allocator kinds
retain their existing behavior.

An S(0) allocation first tries its owned holes and the central gap. On a miss,
BFC calls its existing `Extend` path and suballocator, using normal growth sizing
and backpedaling. Only upper-end/S(0) requests can grow after the initial
allocation. No new allocator class or virtual-address reservation path is added.

The existing GPU `DeviceMemAllocator` returns separate regions and does not
support coalescing. Extra regions therefore remain S(0)-only, even when their
addresses happen to be adjacent. They never join the collective central gap.
Each buffer must fit in one region; this path does not guarantee a contiguous
virtual arena.

```text
Initial: | S(1) -> | shared gap | <- S(0) |
Extra:   |             S(0) only         |
```

BFC also supports adjacent region extension when its suballocator advertises
`SupportsCoalescing()`. This capability permits merging; it does not promise
that allocations will be adjacent. Nonadjacent allocations remain separate
upper-owned regions. When capacity permits, BFC requests enough memory for the
whole buffer so a separate region can satisfy it. Near the cap, BFC can try a
smaller extension if it could complete an immediately safe free tail.

When an adjacent allocation extends the initial region, BFC updates that
region's end and coalesces free chunks normally. An S(0) buffer can then span
the old end. S(1), including alignment and padding, remains confined to the
original allocation even when a free span crosses that limit. Only the region
containing the initial allocation may have a central gap.

Allocation policy remains attached to the memory space: S(1) uses ascending
equal-size holes and exact splitting; S(0) uses descending equal-size holes and
the BFC heuristic for both holes and gap carves.

Ranks may grow independently. Optional automatic registration of ordinary
backing allocations is disabled for growing pools; collective-window
registration continues to use the fixed S(1) range. Existing buffers never
move or change backing.

Growth adds no headroom or future collective-gap reservation. Live S(0)
buffers can still occupy initial capacity needed by a later S(1) request.
Freed buffers return to BFC; backing allocations remain until destruction.
