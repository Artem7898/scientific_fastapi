# CPU measurements for the optional Numba backend

Measured on 2026-10-29 (Europe/Moscow) in the development cloud environment. These are local observations from five warm samples per workload; the small differences and descriptive p95 values do not establish a general performance advantage.

Hardware/runtime: AMD EPYC 9V74 80-Core Processor, x86-64 Linux 6.18.55/glibc 2.41, five CPUs in process affinity and a cgroup CPU quota of four cores. CPython 3.12.14 with the GIL enabled, NumPy 2.5.4, Numba 0.68.0, llvmlite 0.50.0, JAX/jaxlib 0.11.2. Native thread environment limits were one. No GPU was available.

## Actual simulation runtime

Each backend ran in a fresh child process. The first service call includes optional Numba import/JIT, PCG64, physics, final observables, source/environment recording, compressed NPZ, manifest and `fsync`. Common Python/NumPy/service imports and interpreter startup are excluded from that cold measurement.

The table shows median **warm full-service** time. This invokes `run_simulation`, writes and then validates real artifacts. HTTP, queue waiting, Redis, SSE, tracking and artifact reads are excluded. Source digest and Git inspection are included. The separate evolution profile includes PCG64 and the production step kernel, without per-step reduction. It is a separate run, not a decomposition of the full-service sample. The RNG-only profile is another diagnostic and must not be added to or subtracted from those samples.

| Particles × steps | NumPy full run | Numba full run | Observed NumPy / Numba ratio | NumPy evolution | Numba evolution |
| --- | ---: | ---: | ---: | ---: | ---: |
| [2,000 × 100](runtime-2000x100.json) | 16.281 ms | 15.419 ms | 1.056× | 4.786 ms | 4.610 ms |
| [100,000 × 100](runtime-100000x100.json) | 284.461 ms | 285.558 ms | 0.996× | 215.582 ms | 222.129 ms |
| [1,000,000 × 20](runtime-1000000x20.json) | 1,086.712 ms | 1,023.491 ms | 1.062× | 507.819 ms | 448.850 ms |

The corresponding first-run NumPy/Numba times were 0.029/0.478 s, 0.370/0.712 s and 1.194/1.520 s. Peak child RSS was approximately 30/160 MiB, 41/171 MiB and 151/282 MiB. RSS includes imports, JIT, the NumPy reference and validation arrays; it is not incremental engine allocation. Production evolution retains one current state and per-step noise, without a `steps × particles` noise buffer.

All persisted final arrays matched the seeded NumPy reference exactly on this machine; each backend also repeated its own seed exactly. The declared cross-backend acceptance tolerance is `rtol=atol=1e-11`, not a cross-platform bitwise guarantee. Parameters were `T=300`, `dt=0.001`, `seed=42` with float64 PCG64. Numerical equivalence does not replace the scientific reference tests.

Numba remains an explicit optional choice. The measured full-run differences were small, warm Numba was slightly slower for the middle workload, and first-run compilation/import and additional memory are material. RNG-only timings were close to evolution timings; artifact persistence also matters for large snapshots. These measurements do not support an order-of-magnitude speedup or a change to the NumPy default.

Reproduce one workload, choosing a new output path:

```bash
uv sync --frozen --extra numba
uv run --frozen --no-sync python scripts/benchmark_runtime.py \
  --particles 100000 --steps 100 --repeats 5 --seed 42 \
  --backends numpy,numba --require-backends --output /tmp/runtime-new.json
```

The JSON reports contain all samples, median/p95, the cold call, stage profiles, execution metadata and software/hardware provenance. They were measured from base commit `ab6f73c72194c1fe4a9c901ac7b633b2eb98517a` with `source_dirty=true`: the stage-three implementation was present in the working tree. Its package-source SHA-256 at measurement was `5ba42fd3bf233bdfed4b851153f96472d967486a76f2475945e1d71042cb60a8`; the base commit alone does not identify that code. The common workload metadata excludes backend selection, which is recorded separately for each measured backend.

## Frozen-input kernel experiment, including JAX

The [kernel report](kernel-20000x100x2.json) used two independent batches, each with 20,000 particles and 100 steps; five warm samples and the same PCG64-generated innovations. The Numba candidate reuses the production single-step kernel. Input generation, reference computation, numerical validation and warmup/compilation are excluded; process IPC and JAX transfers/synchronization are included.

| Backend | Median | p95 |
| --- | ---: | ---: |
| NumPy | 6.229 ms | 6.631 ms |
| Threads | 4.282 ms | 5.699 ms |
| Processes | 99.914 ms | 181.255 ms |
| Numba | 8.140 ms | 8.595 ms |
| JAX | 46.300 ms | 51.825 ms |

JAX actually used `cpu:0`, with `device_platforms=["cpu"]`, float64 enabled and `gpu_available=false`. Its maximum absolute error versus NumPy was approximately `1.21e-13`; all candidates passed the `1e-11` tolerance. This is a kernel experiment with pre-generated noise, not a production JAX backend or a GPU measurement. It gives no basis for claims about GPU speed. GPU integration remains conditional on a real GPU profile with recorded device, transfer/synchronization cost, dtype, RNG policy and full-runtime benefit.

```bash
uv sync --frozen --extra benchmark
uv run --frozen --no-sync python scripts/benchmark.py \
  --particles 20000 --steps 100 --repeats 5 --workers 2 --seed 42 \
  --backends numpy,threads,processes,numba,jax --output /tmp/kernel-new.json
```
