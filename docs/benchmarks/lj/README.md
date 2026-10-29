# Measured 2D Lennard–Jones engine performance

Measured on 2026-10-30 (Europe/Moscow) in the development cloud environment. Five warm samples were collected for the 20-step workloads, three for the 100-step comparison. Small-sample p95 is descriptive; all ratios refer to these CPU workloads.

Hardware/runtime: AMD EPYC 9V74 80-Core Processor, x86-64 Linux 6.18.55/glibc 2.41, five CPUs in affinity and a cgroup quota of four cores. CPython 3.12.14 with the GIL enabled; NumPy 2.5.4, Numba 0.68.0, llvmlite 0.50.0, LLVM 22.1.0. Native thread limits were one. Variants ran serially in fresh processes; no GPU result is inferred.

## Scope

All runs use `lennard-jones-2d-v1`, NVE velocity Verlet, `T_initial=epsilon=sigma=mass=1`, density `4/9`, square box sides `12`, `24`, `48`, `dt=0.002`, cutoff `2.5`, skin `0.3` and seed `42`.

The cold call includes optional import/JIT. A warm call still creates a new same-seed engine: lattice/velocities, initial force, neighbor construction/rebuilding, integration and terminal observables are timed. Common Python/NumPy/package imports, process startup, HTTP, queue wait, tracking, RDF, trajectory sampling and artifact I/O are excluded. This is engine wall time, not full API-job latency.

The final cell builder batches minimum-image geometry per particle. It retains the same sorted neighbor pairs, rebuild rule and pair counters as the earlier scalar builder. The [baseline report and raw JSON](baseline/README.md) preserve that earlier implementation's measurements.

## Final implementation: twenty-step calls

Median warm wall time includes initialization and cell construction:

| Particles | NumPy direct | NumPy Verlet | Numba direct | Numba Verlet | NumPy direct/Verlet |
| --- | ---: | ---: | ---: | ---: | ---: |
| [64](engine-64-batched-cells.json) | 53.335 ms | 4.161 ms | 0.854 ms | 2.584 ms | 12.82x |
| [256](engine-256-batched-cells.json) | 231.848 ms | 11.013 ms | 4.071 ms | 6.340 ms | 21.05x |
| [1024](engine-1024-batched-cells.json) | 1,218.096 ms | 40.006 ms | 41.925 ms | 21.209 ms | 30.45x |

Numba direct is faster for the two smaller calls. At 1024 particles, Numba Verlet is about 1.98x faster than Numba direct. Cell batching reduces the 1024-particle Numba-Verlet median from the scalar baseline's 55.094 ms to 21.209 ms, without changing the scientific workload. Separate-run timing variation also matters; samples and p95 are retained.

All twenty-step cases built a list once. The counters are identical before and after batching and apply to both arithmetic backends:

| Particles | Direct force checks | Verlet force checks | Neighbor candidate checks | Verlet total checks |
| --- | ---: | ---: | ---: | ---: |
| 64 | 42,336 | 5,376 | 1,120 | 6,496 |
| 256 | 685,440 | 21,504 | 4,480 | 25,984 |
| 1024 | 10,999,296 | 86,016 | 16,050 | 102,066 |

## A longer interval with rebuilds

The [1024-particle, 100-step comparison](engine-1024-100steps-batched-cells.json) retains the same numerical tolerance and uses three warm samples:

| Variant | Cold wall | Warm median | Warm p95 | Total pair checks | Neighbor builds |
| --- | ---: | ---: | ---: | ---: | ---: |
| Numba direct | 1,113.927 ms | 205.762 ms | 293.641 ms | 52,901,376 | 0 |
| Numba Verlet | 981.093 ms | 113.570 ms | 118.222 ms | 507,926 | 5 |

Numba Verlet is about 1.81x faster than Numba direct here. Against the scalar-cell baseline's 305.607 ms, the same Verlet workload takes 113.570 ms after batching. Cell construction still costs time; changing density, skin, particle count or duration changes the tradeoff. These data establish neither a universal crossover nor a universal scaling bound.

First-call Numba times across the final profiles span about 0.787–1.114 s. Peak child RSS is about 30–31 MiB for NumPy and 167–168 MiB for Numba, including imports/JIT and untimed reference/validation state. It is not incremental engine/list memory.

## Checks and provenance

Unwrapped positions, velocities, forces and terminal observables passed an untimed NumPy/direct reference at `rtol=atol=1e-10`. Every variant exactly repeated its own seeded arrays and observables. Maximum differences were about `2.73e-12` for observables and `2.46e-13` for forces. Short-horizon agreement does not guarantee close long chaotic trajectories or replace independent force, NVE-convergence and interacting canonical-NVT tests.

Raw reports retain all samples, cold calls, counters, state hashes and source/lock digests. The measurement base commit is `fbe25edc2a1fc3e30b1351bd6d034265ae7acf9b` with `source_dirty=true`: stage-five code was in the working tree. Final package-source SHA-256: `e3a45a8e33351ad1e9cf9bf726ac2aabe98c76bd514899450ceb65d2626d01f6`. Scalar baseline: `31d9a23607129c75ab49d56c40ef2b29e6904fca8b3b35ed1726a9a49049aad6`. The base commit alone does not identify either implementation.

Reproduce a final workload with a new output path:

```bash
uv sync --frozen --extra numba
uv run --frozen --no-sync python scripts/benchmark_lj.py \
  --particles 1024 --steps 20 --repeats 5 --seed 42 --density 0.4444444444444444 \
  --variants numpy-direct,numpy-verlet,numba-direct,numba-verlet \
  --require-backends --output /tmp/lj-engine-new.json
```

`--require-backends` makes an unavailable backend an error; a skipped variant is not a measured implementation. To reproduce scalar cell construction in a separate disposable checkout, apply [the baseline patch](baseline/scalar-cells.patch) and verify the baseline package-source digest before running the same workload. The patch restores only the scalar candidate-geometry calculation.
