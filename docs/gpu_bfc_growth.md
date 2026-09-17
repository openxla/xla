# Experimental spatial BFC growth on CUDA

The shared GPU BFC allocator can reserve a larger virtual address range and
append physical memory without changing existing addresses or mappings. Enable
it with `XLA_PJRT_GPU_BFC_ALLOW_GROWTH=true`. This mode requires CUDA, the BFC
allocator, preallocation, and spatial partitioning. It is incompatible with
unified memory and command-buffer modes that require the separate remapping VMM
allocator. Unsupported combinations return an error.

`XLA_PYTHON_CLIENT_MEM_FRACTION` (or the client's absolute memory limit) still
sets the initial shared allocation. The initial allocation must succeed in full;
it is never reduced in response to memory pressure on an individual rank.
As with fixed BFC preallocation, physical allocation occurs on the first request.

## Layout and invariants

```text
low address                  initial end                     reserved end
| S(1) -->  shared gap  <-- S(0) |       S(0) extensions             |
|------ initially mapped -------|--- mapped only when needed -------|
^ fixed anchor                  ^ fixed limit for S(1)
```

S(1) collective buffers remain in the initial prefix forever. Only S(0) requests
can grow the arena. Free extensions remain upper-owned, including after all
buffers are freed and coalesced. S(0) may allocate across the initial boundary;
S(1) may not. CUDA mappings are only appended: existing memory is never remapped,
relocated, or released while the allocator is alive.

The memory-space policies remain unchanged: S(1) uses ascending equal-size hole
selection and exact splitting; S(0) uses descending equal-size hole selection
and the BFC heuristic for both owned holes and central-gap carves. Best fit
continues to compare sizes first.

The reservation base is aligned for collective memory. Given the same S(1)
request/free sequence and alignments, successful S(1) allocations retain the same
offsets from this anchor regardless of S(0) growth. This does not synchronize
allocation ordering across concurrent executions or guarantee that S(1) requests
succeed when S(0) has already occupied the shared gap.

## Growth policy

| Environment variable                              | Default | Meaning                                      |
|---------------------------------------------------|---------|----------------------------------------------|
| `XLA_PJRT_GPU_BFC_ALLOW_GROWTH`                     | false   | Enable the experimental allocator.           |
| `XLA_PJRT_GPU_BFC_DEVICE_HEADROOM_BYTES`            | 1 GiB   | Keep this allowance outside the BFC arena.   |
| `XLA_PJRT_GPU_BFC_GROWTH_INCREMENT_BYTES`           | 64 MiB  | Initial candidate size for each extension.   |
| `XLA_PJRT_GPU_BFC_COLLECTIVE_GAP_RESERVE_BYTES`      | 0       | Shared gap S(0) carves must leave available.  |

Numeric settings take integer byte counts. C++ clients can instead populate
`GpuAllocatorConfig::bfc_growth`. Environment settings override that config.

The virtual cap is total device memory minus headroom, rounded down to mapping
granularity. Before every physical allocation, the allocator also checks current
free device memory and leaves the same headroom. These checks provide an allowance
for cuDNN, NCCL, and other users; they cannot prevent concurrent allocations from
consuming it. The initial shared allocation must fit below the cap.

Growth starts with the configured increment and doubles the candidate as needed
to fit the missing capacity, counting reusable tail space and alignment padding.
If physical allocation fails, BFC tries smaller extensions. A failed allocation
or mapping preserves all earlier mappings and allocation metadata. CUDA sizes
are rounded to mapping granularity. Freeing buffers returns capacity to BFC; it
does not shrink the physical arena.

A zero collective-gap reserve grows only when existing space cannot satisfy an
S(0) request. A nonzero reserve rejects S(0) gap carves that would leave too little
shared space, accounting for BFC's retained padding. Existing S(0) holes are
reused first. The reserve may cause earlier growth, or S(0) OOM if physical memory
is unavailable, even while some shared space remains. S(1) can consume the
reserve. It cannot reclaim space already occupied by live S(0) buffers.

## NCCL registration

Collective buffers retain the existing symmetric-window registration path in
the initial mapping. Growth never changes that mapping. Automatic registration
of default-space backing allocations is disabled in this mode: independent
per-rank extensions must not trigger collective registration callbacks.
Application-supplied suballocator visitors still run for each newly mapped region
and receive matching free callbacks at teardown; such visitors must not assume
all ranks grow together.

## Validation

CPU tests cover append-only mapping, rollback, cap enforcement, growth across
the initial boundary, S(1) confinement after coalescing, allocation-policy
preservation, gap reserves, timestamps, and randomized sizes and alignments.
The CUDA test uses one GPU to check that growth preserves live device data,
extensions are writable, S(1) cannot use them, and headroom checks reject growth.
A two-GPU NCCL test registers collective windows, grows only one rank, runs
all-reduce again, and registers another window at matching offsets. Larger
multi-host runs and workload-level fragmentation/performance remain validation
requirements before enabling this mode by default.
