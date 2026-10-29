# Changelog

## Unreleased

- Add the separate `langevin-free-v1` model with exact joint position/velocity transitions, independent initial/bath temperatures, mass, relaxation rate, and constant external acceleration.
- Preserve continuous coordinates across periodic boundaries and persist bounded sampled trajectories, terminal positions/velocities, transport diagnostics, and SHA256-linked model provenance.
- Report raw and unbiased centered MSD, drift, thermal temperature, and diffusion estimates with transient/bias gates and delete-one-particle jackknife uncertainty.
- Support model-specific simulation and ensemble contracts through the existing workers, retry, cancellation, SSE, and artifact workflows while retaining the original OU model.
- Keep diffusion explicitly unavailable for unsuitable observation windows and provide verified saved-trajectory reanalysis without rerunning physics.
- Add an explicit optional Numba CPU backend for simulations and ensembles, with persisted backend selection, lazy loading, and no implicit fallback from failed execution.
- Keep NumPy as the default reference and PCG64 innovations common to both implementations; record backend/kernel versions, device, dtype, arithmetic, and RNG policy in run manifests.
- Sample terminal observables without redundant per-step reductions, while retaining step-boundary cancellation and progress behavior.
- Add cold/warm benchmarks of the actual runtime and artifact path, numerical equivalence and repeatability checks, and mandatory optional-backend CI alongside core-only installation checks.
- Preserve default-NumPy ensemble idempotency across upgrades; stop older consumers before admitting payloads with backend selection, and preserve Redis/artifact volumes during the rollout.
- Analyze committed 2D velocity snapshots without rerunning the simulation, with Gaussian component and Rayleigh speed diagnostics.
- Generate optional PNG and three-page PDF reports, with source checksums, versioned analysis provenance, atomic artifact publication, and bounded request capacity.
- Preserve the simulation API and lightweight core; plotting requires the existing `research` extra.
- Separate HTTP producers and standalone workers with explicit execution roles and a shared Redis queue/artifact configuration.
- Add attempt-fenced task progress, bounded SSE replay, and cooperative cancellation without changing existing status polling.
- Accept reproducible ensembles atomically with saved seed schedules, idempotent admission, retained child contributions, and explicit all-success uncertainty rules.
- Preserve the OU physics and synchronous bounded-thread analysis API; recovery restarts from the seed rather than a checkpoint.
- Require draining/stopping old embedded API and workers before the new API admits ensembles into the same Redis namespace; persisted records remain supported after the transition.

## 2.0.0

See [release notes](docs/releases/v2.0.0.md) for migration guidance and the complete changes.

- Versioned exact OU thermostat in reduced units, seeded PCG64, uncertainty and reproducibility manifests.
- Strict finite contracts, consistent task IDs, bounded workers, leased Redis recovery and isolated attempt artifacts.
- Optional OpenTelemetry, MLflow, research and benchmark capabilities with a lightweight core.
- Frozen dependencies, real wheel/container packaging, verified research outputs and numerical benchmarks.
- Protected-main CI and an immutable, checksum-verified GitHub release workflow.

## 1.0.0

Existing GitHub release preceding the reproducibility and task-execution changes. Its tag is retained unchanged.
