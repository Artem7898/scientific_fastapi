"""Model-specific physical and resource limits precede allocation."""

from dataclasses import asdict

import pytest

from scientific_api.domain.lennard_jones import LennardJonesParameters


def test_valid_defaults_and_json_round_trip_preserve_complete_physics():
    parameters = LennardJonesParameters(
        ensemble="nvt", bath_temperature=2.0, box_size=[12.0, 15.0], seed=2**64 - 1
    )
    assert parameters.friction == 1.0
    assert parameters.box_size == (12.0, 15.0)
    assert parameters.lattice_shape() == (8, 8)
    assert LennardJonesParameters(**asdict(parameters)) == parameters


@pytest.mark.parametrize(
    "changes,reason",
    [
        ({"num_particles": 2}, "num_particles"),
        ({"num_particles": True}, "num_particles"),
        ({"num_particles": 4097}, "num_particles"),
        ({"epsilon": float("nan")}, "epsilon"),
        ({"sigma": 0.0}, "sigma"),
        ({"mass": True}, "mass"),
        ({"initial_temperature": 100.1}, "initial_temperature"),
        ({"initial_temperature": 0.0}, "initial_temperature"),
        ({"time_step": 0.0101}, "time_step"),
        ({"time_step": 1e-7}, "time_step"),
        ({"time_step": float("inf")}, "time_step"),
        ({"backend": "jax"}, "backend"),
        ({"ensemble": "npt"}, "ensemble"),
        ({"interaction_backend": "auto"}, "interaction_backend"),
        ({"ensemble": "nve", "bath_temperature": 1.0}, "NVE"),
        ({"ensemble": "nve", "friction": 1.0}, "NVE"),
        ({"ensemble": "nvt"}, "bath_temperature"),
        ({"ensemble": "nvt", "bath_temperature": 1.0, "friction": 0.0}, "friction"),
        ({"ensemble": "nvt", "bath_temperature": True}, "bath_temperature"),
        ({"box_size": [12.0]}, "box_size"),
        ({"box_size": [True, 12.0]}, "box_size"),
        ({"box_size": [0.0, 12.0]}, "box_size"),
        ({"box_size": [float("inf"), 12.0]}, "box_size"),
        ({"box_size": [5.6, 12.0]}, "half"),
        ({"cutoff_radius": 1.9}, "cutoff_radius"),
        ({"neighbor_skin": 0.01}, "neighbor_skin"),
        ({"num_particles": 256, "box_size": [12.0, 12.0]}, "lattice"),
        ({"seed": 2**64}, "seed"),
        ({"seed": -1}, "seed"),
        ({"num_steps": 0}, "num_steps"),
        ({"num_steps": 10001}, "num_steps"),
        ({"sample_stride": False}, "sample_stride"),
        ({"equilibration_steps": 1000}, "equilibration_steps"),
        ({"rdf_bins": 7}, "rdf_bins"),
        ({"rdf_bins": 257}, "rdf_bins"),
        ({"num_steps": 1001, "sample_stride": 1}, "frames"),
        ({"num_particles": 4096, "box_size": (96.0, 96.0), "num_steps": 5000}, "20000000"),
        (
            {
                "num_particles": 512,
                "box_size": (36.0, 36.0),
                "num_steps": 763,
                "interaction_backend": "direct",
            },
            "pair checks",
        ),
    ],
)
def test_invalid_physics_and_resource_inputs_are_rejected(changes, reason):
    with pytest.raises(ValueError, match=reason):
        LennardJonesParameters(**changes)


def test_sampling_always_contains_initial_and_terminal_state_once():
    parameters = LennardJonesParameters(num_steps=7, sample_stride=3)
    assert parameters.sampling_steps() == (0, 3, 6, 7)
    assert LennardJonesParameters(num_steps=6, sample_stride=3).sampling_steps() == (0, 3, 6)
    assert LennardJonesParameters(num_steps=1, sample_stride=100).sampling_steps() == (0, 1)


def test_direct_pair_budget_includes_initial_force_and_terminal_rdf():
    parameters = LennardJonesParameters(
        num_particles=512, box_size=(36.0, 36.0), num_steps=762, interaction_backend="direct"
    )
    assert parameters.direct_pair_evaluations() == 512 * 511 // 2 * 764
    assert parameters.direct_pair_evaluations() <= 100_000_000


def test_time_step_limit_scales_with_lj_time_unit():
    parameters = LennardJonesParameters(
        epsilon=4.0,
        mass=4.0,
        sigma=2.0,
        box_size=(24.0, 24.0),
        cutoff_radius=5.0,
        neighbor_skin=0.6,
        time_step=0.02,
    )
    assert parameters.time_step == 0.02
    with pytest.raises(ValueError, match="time_step"):
        LennardJonesParameters(
            epsilon=4.0,
            mass=4.0,
            sigma=2.0,
            box_size=(24.0, 24.0),
            cutoff_radius=5.0,
            neighbor_skin=0.6,
            time_step=0.02001,
        )
