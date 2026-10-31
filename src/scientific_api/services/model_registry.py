"""Fixed scientific contracts, without dynamic plugins or optional engine imports."""

import hashlib
import importlib.util
import json
from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import Any

from ..domain.experiments import MODEL_PARAMETER_TYPES
from ..domain.langevin import LANGEVIN_MODEL_VERSION
from ..domain.lennard_jones import LJ_MODEL_VERSION
from ..domain.simulation import MODEL_VERSION
from ..schemas.model_contract import (
    BackendAvailability,
    BackendContract,
    ModelContract,
    ModelDiagnostic,
    ModelVersionPolicy,
    QuantityUnit,
    RequiredArtifact,
    ScientificInvariant,
    UncertaintyCapability,
)

MODEL_VERSIONS = tuple(MODEL_PARAMETER_TYPES)


def _json_schemas(model_version: str) -> tuple[dict[str, Any], dict[str, Any]]:
    # Generate on demand, and never cache mutable dicts returned to callers.
    from ..schemas.langevin import LangevinRequest, LangevinResponse
    from ..schemas.lennard_jones import LennardJonesRequest, LennardJonesResponse
    from ..schemas.simulation import SimulationRequest, SimulationResponse

    request, response = {
        MODEL_VERSION: (SimulationRequest, SimulationResponse),
        LANGEVIN_MODEL_VERSION: (LangevinRequest, LangevinResponse),
        LJ_MODEL_VERSION: (LennardJonesRequest, LennardJonesResponse),
    }[model_version]
    return request.model_json_schema(mode="validation"), response.model_json_schema(
        mode="serialization"
    )


def _backends(model_version: str) -> tuple[BackendContract, ...]:
    semantics = (
        "Explicit CPU float64 implementation with local NumPy PCG64 randomness; "
        "sampling consumes no extra random values. Same-backend seeded replay is "
        "scoped to the recorded implementation and compatible environment. "
        "Cross-backend bitwise equality is not promised."
    )
    if model_version == LJ_MODEL_VERSION:
        semantics += (
            " Both direct and cell-Verlet pair paths implement the same shifted-force "
            "potential; short-trajectory agreement is tested numerically, while long "
            "chaotic trajectories may diverge."
        )
    return (
        BackendContract(
            backend="numpy", dependency="numpy", optional_extra=None, semantics=semantics
        ),
        BackendContract(
            backend="numba",
            dependency="numba",
            optional_extra="numba",
            semantics=semantics
            + " Optional kernels use fastmath=False and no internal parallel pool; "
            "import or compilation failure never selects NumPy silently.",
        ),
    )


def backend_availability() -> tuple[BackendAvailability, ...]:
    """Report dependency discovery here, without importing or compiling engines.

    An API-only process cannot establish readiness of separately deployed workers.
    Even an installed dependency may fail to import or compile at execution time.
    """
    records = []
    for backend in ("numpy", "numba"):
        try:
            present = importlib.util.find_spec(backend) is not None
        except (ImportError, ValueError):
            present = False
        records.append(
            BackendAvailability(
                backend=backend,
                dependency_present=present,
                interpretation=(
                    "Dependency discovery in this process only; no import, JIT compilation, "
                    "device probe or remote-worker readiness check was performed."
                ),
            )
        )
    return tuple(records)


def _units(model_version: str) -> tuple[QuantityUnit, ...]:
    definitions = [
        ("temperature", "energy", "Reduced k_B=1; not Kelvin without a declared conversion"),
        ("energy", "energy", "Reduced kinetic and potential energy, as applicable"),
        ("time", "time", "Reduced simulation time; not seconds without a declared conversion"),
    ]
    if model_version == MODEL_VERSION:
        definitions.append(("mass", "mass", "Fixed reduced particle mass m=1"))
    else:
        definitions.extend(
            (
                ("length", "length", "Unwrapped dynamics; periodic terminal coordinates wrap"),
                ("mass", "mass", "Reduced mass supplied by the request"),
                ("velocity", "length/time", "Laboratory-frame velocity"),
                ("friction", "1/time", "Velocity relaxation rate, not mass-weighted drag"),
            )
        )
    if model_version == LANGEVIN_MODEL_VERSION:
        definitions.extend(
            (
                ("acceleration", "length/time^2", "Constant external acceleration"),
                ("diffusion", "length^2/time", "Two-dimensional centered-MSD slope divided by 4"),
                ("msd", "length^2", "Displacement from a single initial-time origin"),
            )
        )
    elif model_version == LJ_MODEL_VERSION:
        definitions.append(("pressure", "energy/length^2", "Mechanical two-dimensional pressure"))
    return tuple(
        QuantityUnit(quantity=name, unit="reduced", dimension=dimension, interpretation=meaning)
        for name, dimension, meaning in definitions
    )


def _artifacts(model_version: str) -> tuple[RequiredArtifact, ...]:
    records = [
        RequiredArtifact(name="manifest", filename="manifest.json", media_type="application/json"),
        RequiredArtifact(
            name="final_state", filename="final_state.npz", media_type="application/octet-stream"
        ),
    ]
    if model_version != MODEL_VERSION:
        records.append(
            RequiredArtifact(
                name="trajectory", filename="trajectory.npz", media_type="application/octet-stream"
            )
        )
        name = "transport" if model_version == LANGEVIN_MODEL_VERSION else "observables"
        records.append(
            RequiredArtifact(name=name, filename=f"{name}.json", media_type="application/json")
        )
    return tuple(records)


def _ensemble_uncertainty() -> UncertaintyCapability:
    return UncertaintyCapability(
        name="ensemble_standard_error",
        availability="conditional",
        method="Sample SD with ddof=1 divided by sqrt(M); null for M=1",
        independent_units="Independent logical replicas with persisted distinct spawned seeds",
        assumptions=("All requested replicas completed successfully (require_all)",),
        limitations=(
            "An empirical standard error is not a confidence interval",
            "No survivor-only aggregate, parameter uncertainty or model-discrepancy estimate",
        ),
    )


def _ou_metadata() -> dict[str, Any]:
    return {
        "name": "Equilibrium two-dimensional ideal-gas OU thermostat",
        "submission_path": "/api/v1/simulations/monte-carlo",
        "assumptions": (
            "Independent noninteracting particles with equilibrium Gaussian initial velocities",
            "Exact Ornstein-Uhlenbeck velocity transition with relaxation rate 1 and m=k_B=1",
            "No particle coordinates, boundaries, collisions or interparticle forces",
        ),
        "invariants": (
            ScientificInvariant(
                name="kinetic_observables",
                definition="K=sum(|v|^2)/2; T_hat=K/N; K and T_hat are finite and nonnegative",
                applicability="Every completed terminal snapshot",
                enforcement="runtime_and_reference_tests",
            ),
            ScientificInvariant(
                name="stationary_gaussian_distribution",
                definition="Each velocity component has mean 0 and variance T_target",
                applicability="Distribution across independent seeded equilibrium samples",
                enforcement="reference_tests",
            ),
        ),
        "diagnostics": (
            ModelDiagnostic(
                name="terminal_temperature_and_energy",
                location="result; final_state.npz",
                interpretation="Observed terminal values; a finite sample fluctuates around theory",
            ),
            ModelDiagnostic(
                name="velocity_distribution",
                location="optional saved-state analysis.json and PNG/PDF report",
                interpretation="Gaussian components/Rayleigh speed comparison; no dynamics rerun",
            ),
        ),
        "uncertainty_capabilities": (
            UncertaintyCapability(
                name="terminal_temperature_standard_error",
                availability="available",
                method="T_target / sqrt(N)",
                independent_units="Independent particles in one terminal snapshot",
                assumptions=("Equilibrium independent two-dimensional Gaussian velocities",),
                limitations=(
                    "An analytical standard error is not a confidence interval",
                    "Excludes parameter, time-series and physical-model uncertainty",
                ),
            ),
            _ensemble_uncertainty(),
        ),
        "limitations": (
            "The historical monte-carlo route executes OU dynamics, not a general Monte Carlo sampler",
            "This model cannot describe spatial diffusion, interacting fluids or external fields",
        ),
    }


def _langevin_metadata() -> dict[str, Any]:
    return {
        "name": "Free two-dimensional Langevin transport under constant acceleration",
        "submission_path": "/api/v1/simulations/langevin",
        "assumptions": (
            "Independent noninteracting particles; Gaussian initial velocities at one shared position",
            "Exact joint integrated OU position/velocity increments and lab-frame friction",
            "Constant acceleration; unbounded or periodic coordinates without walls",
            "Thermal moments remove sample flow and use 2*(N-1) degrees of freedom",
        ),
        "invariants": (
            ScientificInvariant(
                name="thermal_temperature",
                definition="m*sum(|v-mean(v)|^2)/(2*(N-1)) is finite and nonnegative",
                applicability="Every completed terminal snapshot",
                enforcement="runtime_and_reference_tests",
            ),
            ScientificInvariant(
                name="joint_transition_moments",
                definition="Position/velocity means, variances and covariance follow integrated OU theory",
                applicability="Independent-particle ensemble under the declared constant acceleration",
                enforcement="reference_tests",
            ),
            ScientificInvariant(
                name="coordinate_convention",
                definition="Dynamics and trajectory are unwrapped; periodic terminal positions lie in [0,L)",
                applicability="All runs; wrapping applies only for periodic boundary",
                enforcement="runtime_and_reference_tests",
            ),
        ),
        "diagnostics": (
            ModelDiagnostic(
                name="raw_and_centered_msd",
                location="transport.json",
                interpretation="Single-origin raw MSD and unbiased drift-centered displacement variance",
            ),
            ModelDiagnostic(
                name="diffusion_applicability",
                location="result.diffusion; transport.json",
                interpretation="Explicit available/unavailable status, fit window and finite-window bias gate",
            ),
        ),
        "uncertainty_capabilities": (
            UncertaintyCapability(
                name="thermal_temperature_standard_error",
                availability="available",
                method="[T_initial*exp(-2*gamma*t)+T_bath*(1-exp(-2*gamma*t))]/sqrt(N-1)",
                independent_units="Independent particle velocities at one terminal snapshot",
                assumptions=("Gaussian fluctuations about sample mean velocity",),
                limitations=(
                    "No confidence interval, parameter uncertainty or model-discrepancy estimate",
                ),
            ),
            UncertaintyCapability(
                name="diffusion_standard_error",
                availability="conditional",
                method="Delete-one-particle jackknife of the centered-MSD OLS slope divided by 4",
                independent_units="Particles; all sampled times of a particle remain together",
                assumptions=(
                    "At least three saved frames in an admissible late-time fit window",
                    "Analytical finite-window relative bias is at most 1 percent",
                ),
                limitations=(
                    "Unavailable with a reason when the fit-window gate fails",
                    "No independent-time-sample claim or confidence interval",
                ),
            ),
            _ensemble_uncertainty(),
        ),
        "limitations": (
            "Periodic wrapping does not model sedimentation, walls or hydrostatic equilibrium",
            "Free-particle diffusion theory is specific to this model and is not valid for LJ interactions",
        ),
    }


def _lj_metadata() -> dict[str, Any]:
    return {
        "name": "Periodic two-dimensional shifted-force Lennard-Jones fluid",
        "submission_path": "/api/v1/simulations/lennard-jones",
        "assumptions": (
            "Pairwise two-dimensional Lennard-Jones 12-6 interactions with minimum-image periodic boundaries",
            "Shifted-force cutoff: potential and radial force vanish at r_cut; no tail correction",
            "NVE velocity Verlet or NVT BAOAB with an exact lab-frame OU thermostat substep",
            "Regular lattice, zero initial COM velocity and fixed kinetic temperature are not a canonical sample",
            "No external field; equilibration is not inferred automatically",
        ),
        "invariants": (
            ScientificInvariant(
                name="mechanical_observables",
                definition="E=K+U; P=(2*K+virial)/(2*area); U, E and P may be negative",
                applicability="Every completed state, in NVE and NVT",
                enforcement="runtime_and_reference_tests",
            ),
            ScientificInvariant(
                name="pair_force_symmetry_and_cutoff",
                definition="Each i<j pair has opposite forces; U_sf and F_sf vanish at and beyond cutoff",
                applicability="Both direct and Verlet interactions, NumPy and Numba",
                enforcement="reference_tests",
            ),
            ScientificInvariant(
                name="nve_conservation",
                definition="Momentum conservation and second-order finite-step energy error; not exact energy conservation",
                applicability="NVE only; finite time-step and sampling resolution apply",
                enforcement="reference_tests",
            ),
        ),
        "diagnostics": (
            ModelDiagnostic(
                name="sampled_energy_changes",
                location="observables.json.energy_drift",
                interpretation="NVE sampled changes normalized by max(abs(E_initial),epsilon*N); not applicable to NVT",
            ),
            ModelDiagnostic(
                name="terminal_radial_distribution",
                location="observables.json.rdf",
                interpretation="Single terminal 2D snapshot with finite-N annular normalization; no equilibrium claim",
            ),
            ModelDiagnostic(
                name="interaction_work",
                location="result.diagnostics",
                interpretation="Actual force/candidate/analysis pair checks and neighbor rebuild counters",
            ),
        ),
        "uncertainty_capabilities": (
            UncertaintyCapability(
                name="single_run_confidence_interval",
                availability="unavailable",
                method="Not estimated by the terminal snapshot; correlated time series use the separate research API",
                independent_units="Not established by a single terminal interacting snapshot",
                assumptions=("A valid method must account for equilibration and correlations",),
                limitations=(
                    "Saved-frame sample SD is descriptive and is not an independent-sample SE",
                    "No T/sqrt(N) or free-OU diffusion formula is applied",
                    "RDF bins and pairs are not independent uncertainty units",
                ),
            ),
            UncertaintyCapability(
                name="time_series_mean_confidence_interval",
                availability="conditional",
                method="Conditional non-overlapping batch-means Student-t CI/SE with IAT/ESS diagnostics and secondary exploratory moving-block bootstrap; POST /api/v1/research/lj/analyses with kind=timeseries",
                independent_units="Regular-time blocks of a correlated saved observable series, not particles or individual frames",
                assumptions=(
                    "Explicit stationarity assumption and a suitable post-burn-in observation window",
                    "At least 128 regularly sampled frames, twenty effective samples and eight to fifty complete long batches",
                    "Approximately Gaussian weakly dependent batch means; batches span at least eight estimated IATs",
                    "Correlation, trend and numerical-resolution diagnostics must permit inference",
                ),
                limitations=(
                    "Diagnostics do not prove equilibration or mixing",
                    "Flat, trending, short or unresolved series can have unavailable intervals with reasons",
                    "The secondary percentile bootstrap is exploratory; tail-draw resolution does not determine primary Student-t availability",
                    "No physical-model, finite-step, finite-size, cutoff or RDF uncertainty is included",
                ),
            ),
            UncertaintyCapability(
                name="replica_mean_confidence_interval",
                availability="conditional",
                method="Student-t interval across whole-trajectory means; POST /api/v1/research/lj/analyses with kind=replicas",
                independent_units="Independent trajectories with distinct seeds and matched physical contracts",
                assumptions=(
                    "All requested sources are committed and no survivor-only subset is selected",
                    "At least eight matched replicas and approximately Gaussian trajectory means for the primary interval",
                    "Replicas share the relevant physics and physical observation window",
                ),
                limitations=(
                    "Descriptive between-replica SE needs at least two replicas",
                    "A secondary bootstrap interval is exploratory and does not prove coverage",
                    "No guarantee of equilibrium, model validity or parameter uncertainty",
                ),
            ),
            UncertaintyCapability(
                name="one_parameter_sensitivity",
                availability="conditional",
                method="Matched-seed descriptive changes and conditional paired-mean SE/Student-t intervals when varying time_step or cutoff_radius; POST /api/v1/research/lj/analyses with kind=sensitivity",
                independent_units="Matched trajectory groups over equal physical observation windows",
                assumptions=(
                    "Exactly one declared numerical or cutoff parameter is varied",
                    "Other physical settings match and seed groups are paired consistently",
                    "Paired-mean intervals require at least eight pairs and approximately normal independent contrasts",
                ),
                limitations=(
                    "Observed changes do not establish convergence, continuum limits or physical-model validity",
                    "The source simulation result and its original uncertainty declaration remain unchanged",
                ),
            ),
            _ensemble_uncertainty(),
        ),
        "limitations": (
            "Finite-step, finite-size and cutoff effects are not included in an uncertainty interval",
            "The admission time-step limit does not guarantee stability for every nonlinear trajectory",
            "No experimental material calibration, automatic equilibration test or three-dimensional tail correction",
        ),
    }


_MODEL_METADATA_FACTORIES: Mapping[str, Callable[[], dict[str, Any]]] = MappingProxyType(
    {
        MODEL_VERSION: _ou_metadata,
        LANGEVIN_MODEL_VERSION: _langevin_metadata,
        LJ_MODEL_VERSION: _lj_metadata,
    }
)


def get_model_contract(model_version: str) -> ModelContract:
    """Return an independent typed metadata/schema snapshot for a fixed model.

    Unknown versions raise KeyError; there is no import-by-name plugin fallback.
    The contract declares semantics. Runtime source identity remains separate.
    """
    if model_version not in MODEL_VERSIONS:
        raise KeyError(model_version)
    request_schema, result_schema = _json_schemas(model_version)
    metadata = _MODEL_METADATA_FACTORIES[model_version]()
    return ModelContract(
        model_version=model_version,
        request_schema=request_schema,
        result_schema=result_schema,
        schema_limitations=(
            "JSON Schema describes transport types and scalar constraints; domain validators also "
            "enforce coupled physics, sampling and resource limits at admission",
            "Model catalog schemas describe simulation requests and completed results; "
            "pending, failed and cancelled task states use the existing task-status contract",
        ),
        units=_units(model_version),
        backends=_backends(model_version),
        required_artifacts=_artifacts(model_version),
        version_policy=ModelVersionPolicy(
            implementation_identity="Run manifests separately record package-source SHA256 and available source commit/runtime versions",
            enforcement=(
                "Declarative version policy supported by reference/regression tests and code review; "
                "the metadata alone cannot prevent an implementation changing under the same version"
            ),
        ),
        **metadata,
    )


def list_model_contracts() -> tuple[ModelContract, ...]:
    return tuple(get_model_contract(version) for version in MODEL_VERSIONS)


def model_contract_sha256(contract: ModelContract) -> str:
    """Identify this canonical metadata snapshot, excluding runtime availability."""
    data = json.dumps(
        contract.model_dump(mode="json"), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(data).hexdigest()
