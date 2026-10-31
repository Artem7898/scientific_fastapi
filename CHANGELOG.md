# Changelog

## Unreleased

- Add an independent bounded leased verification queue and dedicated verification workers for archived completed results, without rewriting source results, envelopes or provenance exports.
- Verify source integrity and independently recompute supported observables/invariants for the three models; allow an explicitly requested, resource-bounded same-seed rerun.
- Publish durable unsigned canonical JSON certificates with checks, honest verdicts and integrity digests; distinguish job completion from scientific verification outcomes.
- Separate simulation and verification worker roles, with explicit shared queue/rerun budgets and a verification-only Compose service; retain the lightweight dependency set.
- Add optional bounded deterministic checkpoints for all three models, with PCG64 state, sampled prefixes, model caches and LJ work counters preserved across continuation.
- Resume leased retries from the latest owner-fenced registered checkpoint; expose explicit integrity/compatibility failures instead of silently restarting incompatible state.
- Add checkpoint discovery, verified downloads and bounded manual resume for failed/cancelled standalone tasks while preserving physical payloads, existing result responses and ensemble accounting.
- Record checkpoint lineage in provenance and run manifests; keep checkpoint state downloads separate from the existing bounded scientific RO-Crate export.
- Archive lifecycle transitions and committed evidence atomically with queue state, with an untrimmed Redis journal independent of live-task TTL and a process-local memory equivalent.
- Add archived provenance, paginated task timelines, W3C PROV JSON-LD graphs, stable artifact occurrence identifiers and saved-analysis links without inventing historical events or user identities.
- Export bounded generic RO-Crate 1.1 ZIP packages from verified committed bytes; keep result licensing, signatures and scientific validation distinct from provenance integrity.
- Exercise provenance beyond live-state expiry, offline exports and restart persistence without new core dependencies.
- Add versioned model contracts and a JSON Schema catalog for the three scientific models, with explicit units, assumptions, invariants, diagnostics, backend semantics and artifact requirements.
- Publish a typed Result Envelope after each new successful simulation and saved-velocity analysis; retain existing status responses and manifest outputs.
- Verify manifest bytes, artifact checksums and contract bindings when reading envelopes; distinguish unavailable historical evidence from integrity conflicts without claiming authenticity or independent scientific validation.
- Exercise model catalogs and envelopes in core-only wheel and container checks without importing optional computing or plotting libraries.
- Add the periodic `lennard-jones-2d-v1` model with shifted-force interactions, NVE velocity Verlet and NVT BAOAB, while preserving the earlier OU and free-Langevin contracts.
- Validate direct forces against cell-built Verlet neighbor lists and optional Numba, with unwrapped displacement rebuilds and explicit pair-work/trajectory budgets.
- Persist signed energies, pressure, sampled trajectories, a finite-N 2D radial distribution, and model-specific provenance; keep correlated-frame summaries separate from independent-replica uncertainty.
- Extend queued model dispatch, ensembles, artifact downloads and cold/warm engine benchmarks to interacting particles.
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
- Preserve the OU physics and synchronous bounded-thread analysis API; recovery without a registered checkpoint restarts from the seed.
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
