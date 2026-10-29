"""MSD and diffusion diagnostics for sampled unwrapped Langevin coordinates.

Particles are the independent experimental units. A diffusion fit retains all
time correlations within each particle when estimating its uncertainty; time
frames are never counted as independent replications.
"""

from __future__ import annotations

import hashlib
import io
import math
import os
import re
import stat
import zipfile
import zlib
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..domain.langevin import LANGEVIN_MODEL_VERSION, LangevinParameters
from ..models.langevin_engine import langevin_moments

ANALYSIS_VERSION = "langevin-transport-v1"
MAX_TRAJECTORY_BYTES = 64 * 1024 * 1024
MAX_ARCHIVE_BYTES = 128 * 1024 * 1024
MAX_HEADER_BYTES = 10_000
MIN_RELAXATION_TIMES = 5.0
MAX_RELATIVE_FINITE_WINDOW_BIAS = 0.01
DIFFUSION_METHOD = "OLS centered MSD with intercept; delete-one-particle jackknife SE"


def _sampling_steps(values: Sequence[int]) -> tuple[int, ...]:
    try:
        result = tuple(values)
    except TypeError as exc:
        raise ValueError("Sampling steps must be an integer sequence") from exc
    if (
        not 2 <= len(result) <= 1001
        or any(
            isinstance(value, bool) or not isinstance(value, (int, np.integer)) for value in result
        )
        or result[0] != 0
        or result[-1] > 10_000
        or any(left >= right for left, right in zip(result[:-1], result[1:], strict=True))
    ):
        raise ValueError("Sampling steps must start at zero and strictly increase within 0..10000")
    return tuple(int(value) for value in result)


def _validate_positions(positions: NDArray[np.float64], particles: int, frames: int) -> None:
    if (
        not isinstance(positions, np.ndarray)
        or positions.dtype != np.dtype(np.float64)
        or positions.shape != (frames, particles, 2)
        or positions.nbytes > MAX_TRAJECTORY_BYTES
    ):
        raise ValueError("Positions must be a bounded native float64 array with shape (frames,N,2)")
    if not np.isfinite(positions).all():
        raise ValueError("Positions must contain only finite unwrapped coordinates")


def load_trajectory_state(
    path: Path,
    *,
    expected_sha256: str,
    expected_particles: int,
    expected_sample_steps: Sequence[int],
    expected_time_step: float | None = None,
) -> dict[str, np.ndarray]:
    """Verify a bounded trajectory archive before allocating its claimed arrays.

    The authoritative manifest and safe path are the caller's responsibility.
    Hashing and decoding use the same bytes. Only the three known C-order NPY
    members are supported; pickle, nested names, duplicate members, encrypted
    archives and unsupported compression are rejected.
    """
    if type(expected_particles) is not int or not 3 <= expected_particles <= 1_000_000:
        raise ValueError("expected_particles must be an integer in 3..1000000")
    steps = _sampling_steps(expected_sample_steps)
    frames = len(steps)
    if frames * expected_particles * 2 * 8 > MAX_TRAJECTORY_BYTES:
        raise ValueError("Position trajectory exceeds the 64 MiB limit")
    if not isinstance(expected_sha256, str) or not re.fullmatch(
        r"[0-9a-fA-F]{64}", expected_sha256
    ):
        raise ValueError("expected_sha256 must be a SHA256 hexadecimal digest")
    if expected_time_step is not None and (
        isinstance(expected_time_step, bool)
        or not isinstance(expected_time_step, (float, int))
        or not math.isfinite(expected_time_step)
        or expected_time_step <= 0
    ):
        raise ValueError("expected_time_step must be finite and positive")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(descriptor, "rb") as stream:
        details = os.fstat(stream.fileno())
        if not stat.S_ISREG(details.st_mode):
            raise ValueError("Trajectory archive must be a regular file")
        if details.st_size > MAX_ARCHIVE_BYTES:
            raise ValueError("Trajectory archive exceeds the 128 MiB limit")
        raw = stream.read(MAX_ARCHIVE_BYTES + 1)
    if len(raw) > MAX_ARCHIVE_BYTES:
        raise ValueError("Trajectory archive exceeds the 128 MiB limit")
    if hashlib.sha256(raw).hexdigest() != expected_sha256.lower():
        raise ValueError("Trajectory archive SHA256 does not match its manifest")

    expected = {
        "positions.npy": ((frames, expected_particles, 2), np.dtype(np.float64)),
        "sample_steps.npy": ((frames,), np.dtype(np.int64)),
        "times.npy": ((frames,), np.dtype(np.float64)),
    }
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            members = archive.infolist()
            if len(members) != 3 or {member.filename for member in members} != set(expected):
                raise ValueError(
                    "Trajectory archive must contain only positions, sample_steps, times"
                )
            for member in members:
                if member.flag_bits & 1:
                    raise ValueError("Encrypted trajectory archives are not supported")
                if member.compress_type not in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}:
                    raise ValueError("Unsupported trajectory archive compression")
                expected_shape, expected_dtype = expected[member.filename]
                array_bytes = math.prod(expected_shape) * expected_dtype.itemsize
                if member.file_size > array_bytes + MAX_HEADER_BYTES + 12:
                    raise ValueError("Uncompressed trajectory member exceeds its size limit")
                with archive.open(member) as array_stream:
                    version = np.lib.format.read_magic(array_stream)
                    if version == (1, 0):
                        shape, fortran, dtype = np.lib.format.read_array_header_1_0(
                            array_stream,
                            max_header_size=MAX_HEADER_BYTES,  # pyright: ignore[reportCallIssue]
                        )
                    elif version == (2, 0):
                        shape, fortran, dtype = np.lib.format.read_array_header_2_0(
                            array_stream,
                            max_header_size=MAX_HEADER_BYTES,  # pyright: ignore[reportCallIssue]
                        )
                    else:
                        raise ValueError("Unsupported trajectory NPY format version")
                    if fortran or dtype != expected_dtype or shape != expected_shape:
                        raise ValueError(
                            "Trajectory array header disagrees with its declared contract"
                        )
                    if array_stream.tell() + array_bytes != member.file_size:
                        raise ValueError("Trajectory member payload does not match its header")
        with np.load(
            io.BytesIO(raw), allow_pickle=False, max_header_size=MAX_HEADER_BYTES
        ) as state:
            result = {name: state[name] for name in ("positions", "sample_steps", "times")}
    except (OSError, EOFError, RuntimeError, zipfile.BadZipFile, zlib.error, OverflowError) as exc:
        raise ValueError("Invalid trajectory archive") from exc
    _validate_positions(result["positions"], expected_particles, frames)
    if not np.array_equal(result["sample_steps"], np.asarray(steps, dtype=np.int64)):
        raise ValueError("Saved sampling steps disagree with their manifest")
    times = result["times"]
    if not np.isfinite(times).all() or times[0] != 0 or not (np.diff(times) > 0).all():
        raise ValueError("Saved times must start at zero and strictly increase")
    step_size = float(times[-1] / steps[-1]) if expected_time_step is None else expected_time_step
    expected_times = np.asarray(steps, dtype=np.float64) * step_size
    consistent_times = (
        np.allclose(times, expected_times, rtol=8 * np.finfo(np.float64).eps, atol=0.0)
        if expected_time_step is None
        else np.array_equal(times, expected_times)
    )
    if not consistent_times:
        raise ValueError("Saved times disagree with the sampling schedule and time step")
    return result


def _diffusion_fit(
    positions: NDArray[np.float64],
    steps: NDArray[np.int64],
    times: NDArray[np.float64],
    centered_msd: list[float],
    theoretical_centered_msd: list[float],
    parameters: LangevinParameters,
) -> dict[str, Any]:
    theoretical = parameters.bath_temperature / (parameters.mass * parameters.friction)
    requested_start = parameters.diffusion_fit_start_step
    requested_end = parameters.diffusion_fit_end_step
    # A hot initial distribution can relax much later than five bath time
    # constants. This bound controls the relative analytic MSD slope bias.
    temperature_contrast = abs(parameters.initial_temperature / parameters.bath_temperature - 1.0)
    minimum_relaxation_times = max(
        MIN_RELAXATION_TIMES,
        math.log((1.0 + temperature_contrast) / MAX_RELATIVE_FINITE_WINDOW_BIAS),
    )
    end = parameters.num_steps if requested_end is None else requested_end
    if requested_start is None:
        admissible = np.flatnonzero(parameters.friction * times >= minimum_relaxation_times)
        start = int(steps[admissible[0]]) if admissible.size else None
    else:
        start = requested_start
    selected = (
        np.flatnonzero((steps >= start) & (steps <= end)) if start is not None else np.array([])
    )
    indices = selected.astype(np.int64)
    result = {
        "status": "unavailable",
        "estimate": None,
        "standard_error": None,
        "theoretical": theoretical,
        "method": DIFFUSION_METHOD,
        "reason": None,
        "requested_fit_start_step": requested_start,
        "requested_fit_end_step": requested_end,
        "fit_start_step": int(steps[indices[0]]) if indices.size else None,
        "fit_end_step": int(steps[indices[-1]]) if indices.size else None,
        "sample_count": int(indices.size),
        "intercept": None,
        "positive_estimate": None,
        "minimum_relaxation_times": minimum_relaxation_times,
        "maximum_relative_finite_window_bias": MAX_RELATIVE_FINITE_WINDOW_BIAS,
        "expected_finite_window_estimate": None,
        "relative_finite_window_bias": None,
        "independent_units": "particles; correlated time frames stay within each unit",
        "uncertainty_scope": "sampling variability of the finite-window slope; excludes finite-time bias and model discrepancy",
    }
    if indices.size < 3:
        result["reason"] = "insufficient_diffusive_samples"
        return result
    fit_times = times[indices]
    centered_times = fit_times - float(np.mean(fit_times))
    denominator = float(np.dot(centered_times, centered_times))
    if not math.isfinite(denominator) or denominator <= 0:
        result["reason"] = "degenerate_fit_times"
        return result
    weights = centered_times / denominator
    theoretical_fit = np.asarray(theoretical_centered_msd)[indices]
    expected_estimate = float(np.dot(weights, theoretical_fit - theoretical_fit[0])) / 4.0
    relative_bias = (expected_estimate - theoretical) / theoretical
    result.update(
        expected_finite_window_estimate=expected_estimate,
        relative_finite_window_bias=relative_bias,
    )
    if not all(math.isfinite(value) for value in (expected_estimate, relative_bias)):
        raise ValueError("Theoretical diffusion fit is not finite")
    if parameters.friction * times[indices[0]] < minimum_relaxation_times:
        result["reason"] = "initial_transient_not_relaxed"
        return result
    if abs(relative_bias) > MAX_RELATIVE_FINITE_WINDOW_BIAS:
        result["reason"] = "finite_window_bias_exceeds_tolerance"
        return result
    # A single contribution per particle carries that particle's entire fit
    # trajectory. Delete-one MSD uses N/(N-1) to update the sample mean exactly.
    contributions = np.zeros(parameters.num_particles, dtype=np.float64)
    initial_displacement = positions[indices[0]] - positions[0]
    initial_residual = initial_displacement - np.mean(initial_displacement, axis=0)
    baseline_squares = np.einsum("ij,ij->i", initial_residual, initial_residual)
    for index, weight in zip(indices, weights, strict=True):
        displacement = positions[index] - positions[0]
        residual = displacement - np.mean(displacement, axis=0)
        contributions += float(weight) * (
            np.einsum("ij,ij->i", residual, residual) - baseline_squares
        )
    total = float(np.sum(contributions))
    particles = parameters.num_particles
    slope = total / (particles - 1)
    estimate = slope / 4.0
    deleted = (total - particles / (particles - 1) * contributions) / (particles - 2) / 4.0
    deleted_mean = float(np.mean(deleted))
    standard_error = math.sqrt(
        (particles - 1) / particles * float(np.sum((deleted - deleted_mean) ** 2))
    )
    intercept = float(np.mean(np.asarray(centered_msd)[indices])) - slope * float(
        np.mean(fit_times)
    )
    if not all(math.isfinite(value) for value in (estimate, standard_error, intercept)):
        raise ValueError("Diffusion diagnostics cannot be represented by finite float64")
    result.update(
        status="available",
        estimate=estimate,
        standard_error=standard_error,
        reason=None,
        intercept=intercept,
        positive_estimate=estimate > 0,
    )
    return result


def analyze_transport(
    positions: NDArray[np.float64],
    sample_steps: NDArray[np.int64],
    *,
    parameters: LangevinParameters,
) -> dict[str, Any]:
    """Analyze a bounded single-origin trajectory of independent particles.

    Raw MSD includes drift. Centered MSD is the unbiased trace of displacement
    covariance (ddof=1), so common acceleration and sample-mean drift are
    removed. A long-time fit is explicitly unavailable for a short trajectory;
    the known model D is reported separately from a measured estimate.
    """
    if not isinstance(parameters, LangevinParameters):
        raise TypeError("parameters must be LangevinParameters")
    expected_steps = parameters.sampling_steps()
    if (
        not isinstance(sample_steps, np.ndarray)
        or sample_steps.dtype != np.dtype(np.int64)
        or not np.array_equal(sample_steps, np.asarray(expected_steps, dtype=np.int64))
    ):
        raise ValueError("sample_steps must match the declared int64 sampling schedule")
    _validate_positions(positions, parameters.num_particles, len(expected_steps))
    if not np.array_equal(
        positions[0], np.broadcast_to(np.asarray(parameters.initial_position), positions[0].shape)
    ):
        raise ValueError("Initial trajectory coordinates disagree with the model parameters")
    times = sample_steps.astype(np.float64) * parameters.time_step
    means: list[list[float]] = []
    raw_msd: list[float] = []
    raw_standard_error: list[float] = []
    centered_msd: list[float] = []
    theoretical_means: list[list[float]] = []
    theoretical_raw: list[float] = []
    theoretical_centered: list[float] = []
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            for index, time in enumerate(times):
                displacement = positions[index] - positions[0]
                mean = np.mean(displacement, axis=0)
                squared = np.einsum("ij,ij->i", displacement, displacement)
                residual = displacement - mean
                means.append(mean.tolist())
                raw_msd.append(float(np.mean(squared)))
                raw_standard_error.append(
                    float(np.std(squared, ddof=1)) / math.sqrt(parameters.num_particles)
                )
                centered_msd.append(
                    float(np.einsum("ij,ij->", residual, residual)) / (parameters.num_particles - 1)
                )
                moments = langevin_moments(float(time), parameters)
                theoretical_mean = moments["mean_displacement"]
                theoretical_means.append(theoretical_mean)
                thermal = 2.0 * moments["position_variance_per_component"]
                theoretical_centered.append(thermal)
                theoretical_raw.append(
                    thermal + math.fsum(value * value for value in theoretical_mean)
                )
            diffusion = _diffusion_fit(
                positions, sample_steps, times, centered_msd, theoretical_centered, parameters
            )
    except (FloatingPointError, OverflowError) as exc:
        raise ValueError("Transport diagnostics cannot be represented by finite float64") from exc
    return {
        "analysis_version": ANALYSIS_VERSION,
        "model_version": LANGEVIN_MODEL_VERSION,
        "sample_size": parameters.num_particles,
        "dimensions": 2,
        "coordinate_convention": "unwrapped",
        "displacement_origin": "each particle at the initial frame (single time origin)",
        "sample_steps": sample_steps.tolist(),
        "times": times.tolist(),
        "mean_displacement": means,
        "msd": raw_msd,
        "msd_standard_error": raw_standard_error,
        "centered_msd": centered_msd,
        "centered_msd_method": "trace of unbiased displacement covariance; denominator N-1",
        "theory": {
            "mean_displacement": theoretical_means,
            "msd": theoretical_raw,
            "centered_msd": theoretical_centered,
            "diffusion_coefficient": parameters.bath_temperature
            / (parameters.mass * parameters.friction),
            "assumptions": "independent particles, constant acceleration, lab-frame linear drag, k_B=1",
        },
        "diffusion": diffusion,
        "limitations": [
            "Diffusion estimates depend on the thermostat friction and this free-particle model",
            "The finite-window slope can have transient bias; its jackknife SE does not include that bias",
            "Correlated time frames are not independent replicates; uncertainty deletes whole particles",
            "Periodic wrapping is excluded from displacement calculations",
        ],
    }
