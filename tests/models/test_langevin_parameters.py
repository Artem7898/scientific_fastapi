"""Resource limits and durable JSON validation for the new physical model."""

from dataclasses import asdict

import pytest

from scientific_api.domain.langevin import LangevinParameters


def test_saved_json_vectors_normalize_without_changing_seed_or_samples():
    values = {
        "initial_temperature": 1.0,
        "bath_temperature": 2.0,
        "num_particles": 3,
        "num_steps": 7,
        "sample_stride": 3,
        "initial_mean_velocity": [1.0, -2.0],
        "initial_position": [-1.0, 2.0],
        "acceleration": [0.0, -0.2],
        "boundary": "periodic",
        "box_size": [3.0, 4.0],
        "seed": 2**64 - 1,
    }
    parameters = LangevinParameters(**values)
    assert parameters.initial_mean_velocity == (1.0, -2.0)
    assert parameters.box_size == (3.0, 4.0)
    assert parameters.seed == 2**64 - 1
    assert parameters.sampling_steps() == (0, 3, 6, 7)
    assert LangevinParameters(**asdict(parameters)) == parameters


@pytest.mark.parametrize(
    "changes,reason",
    [
        ({"initial_temperature": 0.0}, "initial_temperature"),
        ({"bath_temperature": float("nan")}, "bath_temperature"),
        ({"mass": True}, "mass"),
        ({"friction": 0.0}, "friction"),
        ({"time_step": float("inf")}, "time_step"),
        ({"num_particles": 2}, "num_particles"),
        ({"num_particles": True}, "num_particles"),
        ({"num_steps": 0}, "num_steps"),
        ({"seed": 2**64}, "seed"),
        ({"sample_stride": 0}, "sample_stride"),
        ({"sample_stride": True}, "sample_stride"),
        ({"backend": "jax"}, "backend"),
        ({"boundary": "walls"}, "boundary"),
        ({"acceleration": [0]}, "acceleration"),
        ({"acceleration": [0, True]}, "acceleration"),
        ({"initial_position": [0, float("inf")]}, "initial_position"),
        ({"initial_mean_velocity": [0, 1001.0]}, "initial_mean_velocity"),
        ({"boundary": "periodic"}, "box_size"),
        ({"box_size": [1.0, 1.0]}, "box_size"),
        ({"boundary": "periodic", "box_size": [0.0, 1.0]}, "box_size"),
        ({"boundary": "periodic", "box_size": [1.0, float("nan")]}, "box_size"),
        ({"diffusion_fit_start_step": -1}, "diffusion_fit_start_step"),
        ({"diffusion_fit_start_step": True}, "diffusion_fit_start_step"),
        ({"diffusion_fit_end_step": 1001}, "diffusion_fit_end_step"),
        ({"diffusion_fit_start_step": 500, "diffusion_fit_end_step": 400}, "fit start"),
        ({"num_particles": 1_000_000, "num_steps": 101}, "num_particles \\* num_steps"),
        ({"sample_stride": 1, "num_steps": 1001}, "1001 frames"),
        ({"num_particles": 1_000_000, "num_steps": 10, "sample_stride": 1}, "64 MiB"),
    ],
)
def test_invalid_or_excessive_inputs_fail_before_state_allocation(changes, reason):
    with pytest.raises(ValueError, match=reason):
        LangevinParameters(
            **{
                "initial_temperature": 1.0,
                "bath_temperature": 1.0,
                "num_particles": 3,
                **changes,
            }
        )


def test_large_particle_count_is_allowed_when_sample_budget_fits():
    parameters = LangevinParameters(
        initial_temperature=1.0,
        bath_temperature=1.0,
        num_particles=1_000_000,
        num_steps=10,
        sample_stride=10,
    )
    assert parameters.sampling_steps() == (0, 10)
