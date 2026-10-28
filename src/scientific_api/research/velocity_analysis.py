"""Analyse a saved equilibrium 2D velocity snapshot without rerunning physics.

Only NumPy and the standard library are required. The model temperature is an
input, not a parameter fitted to the observations. Histogram theory uses exact
bin probabilities rather than point samples of a probability density.
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
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

ANALYSIS_VERSION = "velocity-distribution-v1"
MAX_PARTICLES = 1_000_000
MAX_STATE_BYTES = 32 * 1024 * 1024
MAX_HEADER_BYTES = 10_000
MAX_ARRAY_BYTES = MAX_PARTICLES * 2 * 8


def _validate_velocities(velocities: NDArray[np.float64]) -> None:
    if (
        not isinstance(velocities, np.ndarray)
        or velocities.dtype != np.dtype(np.float64)
        or velocities.ndim != 2
        or velocities.shape[1] != 2
        or not 1 <= velocities.shape[0] <= MAX_PARTICLES
    ):
        raise ValueError("velocities must be a native float64 array with shape (1..1000000, 2)")
    if not np.isfinite(velocities).all():
        raise ValueError("velocities must contain only finite values")


def load_velocity_state(
    path: Path,
    *,
    expected_sha256: str,
    expected_particles: int,
) -> NDArray[np.float64]:
    """Read once, verify and safely load a bounded simulation NPZ artifact.

    The caller validates the artifact's path and authoritative manifest. The
    archive and NPY header are checked before NumPy can allocate from a shape
    claimed by an untrusted file. Object arrays and additional members are not
    supported. A malformed or mismatching artifact raises ``ValueError``;
    filesystem failures remain ``OSError``.
    """
    if type(expected_particles) is not int or not 1 <= expected_particles <= MAX_PARTICLES:
        raise ValueError("expected_particles must be an integer in 1..1000000")
    if not isinstance(expected_sha256, str) or not re.fullmatch(
        r"[0-9a-fA-F]{64}", expected_sha256
    ):
        raise ValueError("expected_sha256 must be a SHA256 hexadecimal digest")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(descriptor, "rb") as stream:
        details = os.fstat(stream.fileno())
        if not stat.S_ISREG(details.st_mode):
            raise ValueError("Velocity archive must be a regular file")
        if details.st_size > MAX_STATE_BYTES:
            raise ValueError("Velocity archive exceeds the 32 MiB limit")
        raw = stream.read(MAX_STATE_BYTES + 1)
    if len(raw) > MAX_STATE_BYTES:
        raise ValueError("Velocity archive exceeds the 32 MiB limit")
    if hashlib.sha256(raw).hexdigest() != expected_sha256.lower():
        raise ValueError("Velocity archive SHA256 does not match its manifest")

    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            members = archive.infolist()
            if len(members) != 1 or members[0].filename != "velocities.npy":
                raise ValueError("Velocity archive must contain only velocities.npy")
            member = members[0]
            if member.flag_bits & 1:
                raise ValueError("Encrypted velocity archives are not supported")
            if member.compress_type not in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}:
                raise ValueError("Unsupported velocity archive compression")
            if member.file_size > MAX_ARRAY_BYTES + MAX_HEADER_BYTES + 12:
                raise ValueError("Uncompressed velocity array exceeds its size limit")
            with archive.open(member) as array_stream:
                version = np.lib.format.read_magic(array_stream)
                if version == (1, 0):
                    shape, _, dtype = np.lib.format.read_array_header_1_0(
                        # The supported NumPy keyword is missing from its public stubs.
                        array_stream,
                        max_header_size=MAX_HEADER_BYTES,  # pyright: ignore[reportCallIssue]
                    )
                elif version == (2, 0):
                    shape, _, dtype = np.lib.format.read_array_header_2_0(
                        array_stream,
                        max_header_size=MAX_HEADER_BYTES,  # pyright: ignore[reportCallIssue]
                    )
                else:
                    raise ValueError("Unsupported velocity NPY format version")
                if dtype != np.dtype(np.float64) or shape != (expected_particles, 2):
                    raise ValueError("Velocity array header does not match float64 (N, 2)")
                expected_bytes = expected_particles * 2 * 8
                if array_stream.tell() + expected_bytes != member.file_size:
                    raise ValueError("Velocity array payload does not match its header")
        with np.load(
            io.BytesIO(raw), allow_pickle=False, max_header_size=MAX_HEADER_BYTES
        ) as state:
            velocities = state["velocities"]
    except (OSError, EOFError, RuntimeError, zipfile.BadZipFile, zlib.error, OverflowError) as exc:
        raise ValueError("Invalid velocity archive") from exc
    _validate_velocities(velocities)
    return velocities


def _gaussian_bin_probability(left: float, right: float, sigma: float) -> float:
    # erfc avoids subtracting two CDF values close to one in the upper tail.
    scale = sigma * math.sqrt(2.0)
    if left >= 0:
        return max(0.0, 0.5 * (math.erfc(left / scale) - math.erfc(right / scale)))
    return max(0.0, 0.5 * (math.erfc(-right / scale) - math.erfc(-left / scale)))


def _rayleigh_bin_probability(left: float, right: float, sigma: float) -> float:
    low, high = left / sigma, right / sigma
    # Multiplication can yield infinity for extreme finite inputs; exp(-inf)
    # is the correct limiting tail probability, with no overflow exception.
    lower_exponent, upper_exponent = -0.5 * low * low, -0.5 * high * high
    if math.isinf(lower_exponent):
        return 0.0
    return math.exp(lower_exponent) * -math.expm1(upper_exponent - lower_exponent)


def _histogram(values: NDArray[np.float64], *, temperature: float, bins: int, speed: bool) -> dict:
    minimum, maximum = float(values.min()), float(values.max())
    sigma = math.sqrt(temperature)
    if speed:
        minimum = 0.0
        maximum = max(maximum, sigma)
    elif minimum == maximum:
        # A single particle or constant snapshot still needs positive-width
        # bins. Include representable room around a large nonzero constant.
        half_width = max(sigma, abs(minimum) * float(np.finfo(np.float64).eps) * bins)
        minimum, maximum = minimum - half_width, maximum + half_width
    counts, edges = np.histogram(values, bins=bins, range=(minimum, maximum))
    widths = np.diff(edges)
    if not np.isfinite(edges).all() or not np.isfinite(widths).all() or not (widths > 0).all():
        raise ValueError("Velocity range cannot be represented by finite positive-width bins")
    density = counts / (values.size * widths)
    probability = np.array(
        [
            (
                _rayleigh_bin_probability(float(left), float(right), sigma)
                if speed
                else _gaussian_bin_probability(float(left), float(right), sigma)
            )
            for left, right in zip(edges[:-1], edges[1:], strict=True)
        ],
        dtype=np.float64,
    )
    theoretical_density = probability / widths
    if not np.isfinite(density).all() or not np.isfinite(theoretical_density).all():
        raise ValueError("Histogram densities are not finite")
    coverage = min(1.0, max(0.0, math.fsum(probability)))
    return {
        "bin_edges": edges.tolist(),
        "counts": counts.tolist(),
        "density": density.tolist(),
        "theoretical_density": theoretical_density.tolist(),
        "theoretical_bin_probability": probability.tolist(),
        "theoretical_coverage": coverage,
        "theoretical_missing_mass": 1.0 - coverage,
    }


def analyze_velocities(
    velocities: NDArray[np.float64], *, target_temperature: float, bins: int = 32
) -> dict[str, Any]:
    """Return finite JSON-ready diagnostics for a 2D equilibrium snapshot.

    Component variances use ``ddof=0`` as descriptive population moments, even
    for N=1. Theoretical curves use the specified model temperature; no fitted
    temperature, confidence interval or hypothesis-test verdict is fabricated.
    """
    _validate_velocities(velocities)
    if (
        isinstance(target_temperature, bool)
        or not isinstance(target_temperature, (float, int))
        or not 0 < target_temperature <= 1e6
        or not math.isfinite(target_temperature)
    ):
        raise ValueError("target_temperature must be finite and in (0, 1000000]")
    if type(bins) is not int or not 2 <= bins <= 256:
        raise ValueError("bins must be an integer in 2..256")
    temperature = float(target_temperature)
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            vx, vy = velocities[:, 0], velocities[:, 1]
            speed = np.hypot(vx, vy)
            second_moment = float(np.mean(speed * speed))
            moments = {
                "vx_mean": float(np.mean(vx)),
                "vy_mean": float(np.mean(vy)),
                "vx_variance": float(np.var(vx, ddof=0)),
                "vy_variance": float(np.var(vy, ddof=0)),
                "speed_mean": float(np.mean(speed)),
                "speed_second_moment": second_moment,
                "temperature_estimate": 0.5 * second_moment,
            }
            histograms = {
                name: _histogram(values, temperature=temperature, bins=bins, speed=name == "speed")
                for name, values in (("vx", vx), ("vy", vy), ("speed", speed))
            }
    except (FloatingPointError, OverflowError) as exc:
        raise ValueError("Velocity diagnostics cannot be represented as finite float64") from exc
    if not all(math.isfinite(value) for value in moments.values()):
        raise ValueError("Velocity moments are not finite")
    return {
        "analysis_version": ANALYSIS_VERSION,
        "sample_size": int(velocities.shape[0]),
        "dimensions": 2,
        "target_temperature": temperature,
        "moments": moments,
        "histograms": histograms,
        "theory": {
            "units": "reduced (m = k_B = 1)",
            "components": {"distribution": "Normal", "mean": 0.0, "variance": temperature},
            "speed": {
                "distribution": "Rayleigh (2D Maxwell-Boltzmann)",
                "scale": math.sqrt(temperature),
                "mean": math.sqrt(temperature) * math.sqrt(math.pi / 2.0),
                "second_moment": 2.0 * temperature,
            },
            "temperature_standard_error": temperature / math.sqrt(velocities.shape[0]),
            "variance_ddof": 0,
            "histogram_theory": "Bin-averaged PDF from exact CDF differences; not renormalized",
        },
        "limitations": [
            "Equilibrium independent 2D velocities; no positions, collisions or interacting particles",
            "The target temperature is specified by the model, not fitted to this snapshot",
            "Histogram ranges depend on the sample; theoretical tail mass is reported separately",
            "Component variances are descriptive moments with ddof=0, not unbiased estimates",
            "The theoretical temperature standard error is not a confidence interval or model validation",
            "No hypothesis-test p-value or pass/fail scientific verdict is inferred",
        ],
    }
