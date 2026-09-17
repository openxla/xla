# Spatial BFC growth after allocation failure

Preallocated spatial GPU BFC pools grow by default when using device memory.
C++ clients can set `GpuAllocatorConfig::bfc_allow_growth=false` to keep the
initial shared pool fixed. Non-spatial pools, unified memory, and other
allocator kinds retain their existing allocation behavior.

The memory fraction (or absolute memory setting) determines the initial shared
region. That region must be allocated in full. S(1) uses its lower end and S(0)
uses its upper end, with the existing hole-selection and splitting policies.
In particular, S(0) keeps the BFC heuristic for central-gap carves.

When neither an S(0) hole nor the central gap fits a request, BFC's existing
`Extend` path asks the existing suballocator for another region. Subsequent
growth uses normal BFC sizing and backpedaling on allocation failure, up to
total device memory. S(1) requests cannot trigger growth.

```text
Initial region:  | S(1) -->       shared gap       <-- S(0) |
Extra region:    |                 S(0) only                |
```

Extra regions can have unrelated addresses. Existing allocations never move,
and extra regions never join the central gap, even after every allocation is
freed. Regions stay separate even if their addresses happen to be adjacent.
An individual allocation must fit in one region.

The collective registration path remains unchanged. Optional automatic
registration of default-space backing allocations is disabled because ranks
can grow independently. Custom suballocator visitors still run and must not
assume that every rank allocates the same backing regions.

This is demand-triggered growth: it does not reserve future S(1) capacity or
recover space occupied by live S(0) buffers. It adds no device-headroom policy;
physical allocation may fail because other GPU users have consumed memory.
Freeing buffers returns memory to BFC without releasing the backing regions.
