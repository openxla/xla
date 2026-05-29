# NV Large-Scale Training Staging Branch

This branch (`nv-staging/2026-05-29`) consolidates unmerged PRs for NVIDIA large-scale training work while upstream review is pending.

## Branch Purpose

- Integrate PRs that are in review and waiting for upstream merge
- Provide a stable base for [JAX Toolbox](https://github.com/NVIDIA/JAX-Toolbox) builds
- Enable large-scale training development to continue unblocked

## Repository

- **Repo:** https://github.com/openxla/xla
- **Branch:** `nv-staging/2026-05-29`
- **Rolling alias:** `nv-staging/latest` (always points to the most recent snapshot)
- **JAX commit:** [8f711b489](https://github.com/jax-ml/jax/commit/8f711b489)
- **Upstream:** https://github.com/openxla/xla (`main`)

## Using This Branch

```bash
git clone git@github.com:openxla/xla.git
cd xla
git checkout nv-staging/2026-05-29   # this specific snapshot
# or
git checkout nv-staging/latest       # always the most recent snapshot
```

Each `nv-staging/<YYYY-MM-DD>` branch is an immutable snapshot. New snapshots are published as new dated branches; existing dated branches are not force-pushed. The `nv-staging/latest` branch is force-updated each time a new snapshot is published.

## Tracked PRs

### In review

| PR | Description |
|----|-------------|
| [#41903](https://github.com/openxla/xla/pull/41903) | Add device-initiated ragged all-to-all kernel using NCCL LSA/GIN |

### Merged upstream

| PR | Description |
|----|-------------|
| [#26196](https://github.com/openxla/xla/pull/26196) | Add LHS config to prioritize compute nodes over collective starts |
| [#33240](https://github.com/openxla/xla/pull/33240) | Add delayMoveToHost heuristic to GPU latency hiding scheduler |
| [#33269](https://github.com/openxla/xla/pull/33269) | Add flag to control async compute resource limitation |
| [#36224](https://github.com/openxla/xla/pull/36224) | Support all-reduce hoisting for scatter-based accumulation pattern |
| [#36441](https://github.com/openxla/xla/pull/36441) | Support all-reduce hoisting with scalar multiplication pattern |
| [#39302](https://github.com/openxla/xla/pull/39302) | Fix dynamic memcpy offset computation for host offloading with collective pipelining |
| [#39604](https://github.com/openxla/xla/pull/39604) | Add annotation to allow scheduling of custom communication kernels |
| [#40316](https://github.com/openxla/xla/pull/40316) | Do aliasing only when collective permute is in-place |
| [#40656](https://github.com/openxla/xla/pull/40656) | Wire deadlock prevention for async collective multi-streaming |
| [#40921](https://github.com/openxla/xla/pull/40921) | Move barrier out of loop in collective permute |
| [#41552](https://github.com/openxla/xla/pull/41552) | Add one-sided put path for RaggedAllToAll |
| [#41799](https://github.com/openxla/xla/pull/41799) | Add a knob for scheduling collectives as early as possible |

## Building from Source

To build JAX/XLA against this branch, check out the JAX commit listed in the [Repository](#repository) section above:

```bash
git clone https://github.com/jax-ml/jax.git && cd jax
git checkout 8f711b489

git clone git@github.com:openxla/xla.git
cd xla && git checkout nv-staging/2026-05-29 && cd ..
```

Then follow your normal build flow.

## Contact

- **Maintainer:** Sevin F. Varoglu (@sfvaroglu)
