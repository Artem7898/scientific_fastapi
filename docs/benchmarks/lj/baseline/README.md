# Baseline: scalar cell construction for the 2D Lennard–Jones engines

This report preserves the measurements taken before batching neighbor-cell candidates. The timings and raw JSON are unchanged. See the [current report](../README.md) for the final implementation and comparison.

Measured on 2026-10-30 (Europe/Moscow) in the development cloud environment. These are workload-specific CPU observations, with five warm samples for the 20-step workloads and three for the 100-step comparison. Small-sample p95 values are descriptive.

Hardware/runtime: AMD EPYC 9V74 80-Core Processor, x86-64 Linux 6.18.55/glibc 2.41, five CPUs in affinity, cgroup quota four cores. CPython 3.12.14 with the GIL enabled; NumPy 2.5.4, Numba 0.68.0, llvmlite 0.50.0, LLVM 22.1.0. Native thread environment limits were one, and variants ran serially. This is a CPU benchmark; no GPU result is inferred.

## Scope and numerical checks

Each variant ran in a fresh child process. The first call includes optional import/JIT. Warm calls create a new same-seed engine, initialize lattice/velocities, evaluate the initial force, construct/rebuild neighbors, integrate and calculate terminal observables. Common Python/NumPy/package imports and process startup are excluded. So are HTTP, queue wait, tracking, RDF, trajectory sampling and artifact I/O. The measurements are engine timings, not complete API-job latency.

All runs use `lennard-jones-2d-v1`, NVE velocity Verlet, `T_initial=epsilon=sigma=mass=1`, density `4/9`, square boxes of sides `12`, `24`, `48`, `dt=0.002`, `cutoff_radius=2.5`, `neighbor_skin=0.3`, and seed `42`. Direct and Verlet variants see identical initial conditions. Forces, velocities, unwrapped positions and terminal observables passed the NumPy/direct comparison at `rtol=atol=1e-10`. Each variant repeated its own final arrays and observables exactly. Numerical agreement over these short horizons does not guarantee closeness of long chaotic trajectories or replace the independent force/NVE/NVT tests.

## Twenty-step engine calls

Median warm wall times, including initialization and neighbor construction:

| Particles | NumPy direct | NumPy Verlet | Numba direct | Numba Verlet |
| --- | ---: | ---: | ---: | ---: |
| [64](engine-64.json) | 48.697 ms | 6.045 ms | 0.904 ms | 5.190 ms |
| [256](engine-256.json) | 229.909 ms | 21.291 ms | 3.589 ms | 15.731 ms |
| [1024](engine-1024.json) | 1,058.394 ms | 76.885 ms | 44.060 ms | 55.094 ms |

For the NumPy implementation, the observed direct/Verlet ratios are approximately 8.06x, 10.80x and 13.77x. Numba direct is faster than Numba Verlet in all three short complete engine calls: Python cell construction costs remain material even though compiled neighbor force evaluation is faster. At 1024 particles, Numba/direct spends a median 41.614 ms in evolution versus 3.756 ms for Numba/Verlet, but initialization costs 2.354 versus 51.182 ms. Those stage medians are reported separately and are not summed to reconstruct the median wall time.

The 20-step runs built their neighbor list once; they did not amortize repeated rebuilds. Pair checks explain the work reduction without claiming a universal scaling bound:

| Particles | Direct force pair checks | Verlet force checks | Neighbor candidate checks | Verlet total checks |
| --- | ---: | ---: | ---: | ---: |
| 64 | 42,336 | 5,376 | 1,120 | 6,496 |
| 256 | 685,440 | 21,504 | 4,480 | 25,984 |
| 1024 | 10,999,296 | 86,016 | 16,050 | 102,066 |

First-call Numba wall times were approximately 0.825–0.942 s for these workloads. Peak child RSS was approximately 30–31 MiB for NumPy and 167–168 MiB for Numba. RSS includes imports/JIT and the untimed reference/validation states, rather than incremental engine or neighbor-list memory.

## A longer interval with rebuilds

The [1024-particle, 100-step comparison](engine-1024-100steps.json) kept every numerical check and tolerance unchanged, with three warm samples:

| Variant | Cold wall | Warm median | Warm p95 | Total pair checks | Neighbor builds |
| --- | ---: | ---: | ---: | ---: | ---: |
| Numba direct | 1,054.694 ms | 243.981 ms | 257.603 ms | 52,901,376 | 0 |
| Numba Verlet | 1,112.061 ms | 305.607 ms | 308.339 ms | 507,926 | 5 |

The compiled direct implementation remains faster for this measured complete engine call. Four additional list rebuilds inside evolution make cell construction a relevant next optimization target; fewer pair checks alone do not establish lower wall time. Selecting `interaction_backend` should use the target particle count, density, runtime interval, cutoff/skin and chosen arithmetic backend. These results do not establish a crossover point for larger systems.

The largest absolute observable difference across the saved comparisons is about `2.73e-12`; force differences are at most about `2.46e-13`. All records retain the acceptance tolerance, final-state hashes, cold/warm samples, counters, execution metadata, source and lock hashes. The measurement base commit is `fbe25edc2a1fc3e30b1351bd6d034265ae7acf9b` with `source_dirty=true`: stage-five code was present in the working tree. Its package-source SHA-256 at measurement was `31d9a23607129c75ab49d56c40ef2b29e6904fca8b3b35ed1726a9a49049aad6`; the base commit alone does not identify that implementation.

To reconstruct this baseline in a separate disposable checkout, first apply the [scalar-cell patch](scalar-cells.patch). The complete package digest with this substitution was verified to equal the recorded `31d9a23607129c75ab49d56c40ef2b29e6904fca8b3b35ed1726a9a49049aad6`. Without that patch, the command below measures the current batched implementation instead.

```bash
git apply docs/benchmarks/lj/baseline/scalar-cells.patch
```

Then reproduce a workload with a new output path:

```bash
uv sync --frozen --extra numba
uv run --frozen --no-sync python scripts/benchmark_lj.py \
  --particles 1024 --steps 20 --repeats 5 --seed 42 --density 0.4444444444444444 \
  --variants numpy-direct,numpy-verlet,numba-direct,numba-verlet \
  --require-backends --output /tmp/lj-engine-new.json
```

`--require-backends` makes an unavailable optional backend an error. A skipped variant is not a measured or validated implementation.
