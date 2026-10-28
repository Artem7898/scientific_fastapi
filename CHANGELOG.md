# Changelog

## Unreleased

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
