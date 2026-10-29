"""Optional serial Numba force kernels for the shifted-force LJ model."""

import math

import numba
import numpy as np


@numba.njit(nogil=True, fastmath=False, cache=False)
def _pair(dx, dy, epsilon, sigma, cutoff, cutoff_potential, cutoff_force):
    radius_squared = dx * dx + dy * dy
    if radius_squared < (0.5 * sigma) ** 2:
        return 0.0, 0.0, 0.0, 0.0, True
    if radius_squared >= cutoff * cutoff:
        return 0.0, 0.0, 0.0, 0.0, False
    radius = math.sqrt(radius_squared)
    sixth = (sigma * sigma / radius_squared) ** 3
    potential = (
        4.0 * epsilon * (sixth * sixth - sixth)
        - cutoff_potential
        + (radius - cutoff) * cutoff_force
    )
    force = 24.0 * epsilon / radius * (2.0 * sixth * sixth - sixth) - cutoff_force
    fx, fy = force / radius * dx, force / radius * dy
    return fx, fy, potential, dx * fx + dy * fy, False


@numba.njit(nogil=True, fastmath=False, cache=False)
def forces_direct(positions, box, epsilon, sigma, cutoff):
    forces = np.zeros_like(positions)
    sixth = (sigma / cutoff) ** 6
    cutoff_potential = 4.0 * epsilon * (sixth * sixth - sixth)
    cutoff_force = 24.0 * epsilon / cutoff * (2.0 * sixth * sixth - sixth)
    potential = virial = 0.0
    for first in range(len(positions) - 1):
        for second in range(first + 1, len(positions)):
            dx = positions[first, 0] - positions[second, 0]
            dy = positions[first, 1] - positions[second, 1]
            dx -= box[0] * np.rint(dx / box[0])
            dy -= box[1] * np.rint(dy / box[1])
            fx, fy, energy, pair_virial, overlap = _pair(
                dx, dy, epsilon, sigma, cutoff, cutoff_potential, cutoff_force
            )
            if overlap:
                return forces, potential, virial, True
            forces[first, 0] += fx
            forces[first, 1] += fy
            forces[second, 0] -= fx
            forces[second, 1] -= fy
            potential += energy
            virial += pair_virial
    return forces, potential, virial, False


@numba.njit(nogil=True, fastmath=False, cache=False)
def forces_pairs(positions, box, epsilon, sigma, cutoff, pairs):
    forces = np.zeros_like(positions)
    sixth = (sigma / cutoff) ** 6
    cutoff_potential = 4.0 * epsilon * (sixth * sixth - sixth)
    cutoff_force = 24.0 * epsilon / cutoff * (2.0 * sixth * sixth - sixth)
    potential = virial = 0.0
    for index in range(len(pairs)):
        first, second = pairs[index, 0], pairs[index, 1]
        dx = positions[first, 0] - positions[second, 0]
        dy = positions[first, 1] - positions[second, 1]
        dx -= box[0] * np.rint(dx / box[0])
        dy -= box[1] * np.rint(dy / box[1])
        fx, fy, energy, pair_virial, overlap = _pair(
            dx, dy, epsilon, sigma, cutoff, cutoff_potential, cutoff_force
        )
        if overlap:
            return forces, potential, virial, True
        forces[first, 0] += fx
        forces[first, 1] += fy
        forces[second, 0] -= fx
        forces[second, 1] -= fy
        potential += energy
        virial += pair_virial
    return forces, potential, virial, False
