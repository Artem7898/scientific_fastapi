# Conditional LJ time-series uncertainty: calibration and limits

The primary time-series interval is an approximate **non-overlapping batch-means Student-t interval**. A moving-block percentile interval is retained as secondary, exploratory evidence. These methods assume a stationary, representative retained window and sufficiently weak dependence between long approximately Gaussian batch means. Passing diagnostics does not establish those assumptions for molecular dynamics.

The mean uses all selected regular frames. Complete prefix batches estimate variance, with `SE = sample_sd(batch_means) * sqrt(block_length / n)` and `df = complete_batches - 1`. The report identifies any remainder excluded from variance estimation. The default batch length is `max(ceil(8*g), ceil(n/50), ceil(n**(1/3)))`, with `8..50` complete batches. IAT has the convention `g >= 1`; the separate IAT-based SE is diagnostic.

## Fixed holdout design

The [design](lj-uncertainty-holdout-design.json) was written before scoring the redesigned method. It fixes four stationary Gaussian cases, `1001` observations and `160` independent trials per case:

```text
x[0] ~ Normal(0, 1)
x[t] = phi * x[t-1] + sqrt(1 - phi**2) * Normal(0, 1)
phi = 0.0, 0.3, 0.6, 0.8
```

Every case has known mean zero and marginal variance one. Noise and bootstrap seeds use separate, predetermined SeedSequence child streams. Analysis uses the default block policy, explicit stationarity, nominal 95% intervals and 1000 bootstrap draws. No cases, streams, options or sample counts were selected after inspecting these scores.

These are **synthetic statistical-method tests**, using a scalar schema fixture. They are not LJ trajectories or evidence of molecular equilibrium. A separate saved LJ example illustrates reporting, without a known thermodynamic truth or coverage claim.

The design SHA256 is `d73748a25d75699ebe63dfb5e4d2419f863a384df3a0a6f5eb97518447faf295`. Recorded implementation digests are:

- [lj_uncertainty.py](../../src/scientific_api/research/lj_uncertainty.py): `e4786270cb58d693e2d4c2df1dae0a996729224c48d086e23b1e9bf61dd673e5`.
- [student_t.py](../../src/scientific_api/research/student_t.py): `e33efa6620d404aa8e3ec7bf25adf3a444d2eda3a1d48b8602efe79308be5d97`.

## Results

The [recorded results](lj-uncertainty-holdout-results.json) include eligibility, coverage among reported intervals, all unavailable reasons and the fraction of all trials both reported and covering truth. NumPy was `2.5.4`.

| phi | Primary intervals reported | Primary hits/reported | Conditional coverage | 95% Wilson interval | Secondary exploratory MBB hits/reported | Naive IID hits/160 |
| --- | --- | --- | --- | --- | --- | --- |
| 0.0 | 159/160 | 154/159 | 96.86% | 92.85–98.65% | 152/159 | 153/160 |
| 0.3 | 155/160 | 146/155 | 94.19% | 89.33–96.92% | 143/155 | 138/160 |
| 0.6 | 145/160 | 141/145 | 97.24% | 93.12–98.92% | 136/145 | 108/160 |
| 0.8 | 152/160 | 142/152 | 93.42% | 88.31–96.39% | 133/152 | 82/160 |

The nominal 95% level lies within each primary interval's Monte Carlo Wilson bounds in these four cases. This is descriptive consistency in a finite Gaussian family, **not a coverage guarantee** for MD or arbitrary stationary processes. The Wilson bounds apply separately; no simultaneous multi-case coverage claim is made. Between 90.63% and 99.38% of trials produced intervals, so conditional coverage must be read together with eligibility.

All 29 withheld primary intervals remain visible: 26 flagged an unresolved positive correlation tail and three had too few long batches. Withheld trials are not counted as successful intervals. The secondary percentile bootstrap still under-covered at `phi=0.8`: `133/152 = 87.50%`, with a Wilson interval of `81.30–91.85%`. It remains explicitly exploratory.

A preliminary, separate diagnostic run also exposed undercoverage of the original shorter-block percentile approach (`130/147` reported intervals at `phi=0.6`). That finding motivated the primary batch-means policy; it is not presented as holdout validation of the redesign. Its records were retained separately, without changing the registered holdout cases or streams.

## Reproduce

Use the reviewed project environment and a new output directory:

```bash
uv run --no-sync python docs/validation/run_lj_uncertainty_holdout.py \
  --output-dir /tmp/lj-uncertainty-holdout
```

The runner accepts no alternative cases, seed or method options. It copies the fixed design, records the current method hashes, checks that source files did not change during scoring, and writes every trial plus a summary. The recorded raw-trial SHA256 was `d47c0c350109f6ce9f8d0a9fc5cf87d987885485631018dcd3f1d0b0454ea184`. File hashes include run timestamps; compare numerical records, design and implementation hashes when reproducing a new invocation.

Intervals are pointwise and conditional. They exclude physical-model discrepancy, finite-step/finite-size/cutoff effects, experiment calibration and unsaved frames. Long correlation modes can be absent from a finite window, and approximately Gaussian batch means remain an assumption. Use independent matched replicas and sensitivity analyses as additional evidence, rather than treating this calibration as an equilibrium certificate.
