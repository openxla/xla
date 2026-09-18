# Contiguous GPU BFC growth after allocation failure

Preallocated spatial GPU BFC pools grow by default when using device memory.
C++ clients can set `GpuAllocatorConfig::bfc_allow_growth=false` to keep the
initial shared pool fixed. Non-spatial pools, unified memory, and other
allocator kinds retain their existing allocation behavior.

On CUDA, reserve one virtual address range up to total device memory, rounded
down to mapping granularity. The memory fraction (or absolute memory setting)
determines the initial physical backing. That initial allocation must succeed
in full. Later growth maps new physical allocations at the current backed end;
existing mappings, backing handles, and buffer addresses never change.

S(1) uses the lower end and remains bounded by the initial allocation forever.
S(0) uses the upper end and may use all mapped capacity. Both spaces keep their
existing policies: S(1) splits exactly and prefers ascending equal-size holes;
S(0) keeps the BFC heuristic for both holes and gap carves and prefers descending
equal-size holes.

```text
Initial: | S(1) -> | gap | <- S(0) |             unmapped             |
Grown:   | S(1) -> | gap | <- S(0) | additional S(0) |    unmapped     |
         ^                       ^                 ^                ^
      fixed base         fixed S(1) limit     backed end        reserved cap
```

When neither an S(0) hole nor the central gap fits, BFC's existing `Extend` path
obtains more backing and extends the same allocation region. Immediately safe
free space at the old end contributes to the request, so only the missing
capacity must be allocated. Growth uses normal BFC sizing and backpedaling;
S(1) requests cannot trigger growth.

Free chunks can coalesce across physical mapping boundaries, allowing one
ordinary buffer to span multiple backing allocations. A coalesced central free
span may cross the original end; S(1) carving and allocation padding are checked
against the fixed initial limit. This avoids creating an artificial free-space
boundary while keeping the collective capacity unchanged. Other backends keep
the existing separate-region fallback when their suballocator cannot coalesce.

The collective registration path remains unchanged. Optional automatic
registration of default-space backing allocations is disabled because ranks
can grow independently. Custom suballocator visitors still run per physical
allocation and must not assume matching backing allocations across ranks.

Growth does not reserve future S(1) capacity or recover space occupied by live
S(0) buffers. It adds no device-headroom policy; physical allocation may fail
because other GPU users have consumed memory. Fragmentation between live
buffers can still cause OOM. Freeing buffers returns memory to BFC; backing
mappings are released only at allocator destruction.
