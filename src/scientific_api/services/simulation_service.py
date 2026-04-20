import uuid
import logging
from typing import Optional

from ..models.physics_engine import IdealGasSimulator
from ..schemas.simulation import SimulationRequest, SimulationResponse
from ..core.experiment_tracker import ExperimentTracker

logger = logging.getLogger(__name__)


def run_simulation(
        request: SimulationRequest,
        tracker: Optional[ExperimentTracker] = None  # Dependency Injection
) -> SimulationResponse:
    """
    Executes the simulation synchronously.

    Args:
        request: Validated payload from the API layer.
        tracker: Optional experiment tracker. If provided, logs params/metrics.
                 Defaults to None to keep the service testable without infrastructure.
    """
    task_id = uuid.uuid4().hex

    logger.info(
        f"Starting simulation {task_id}: "
        f"N={request.num_particles}, T0={request.initial_temperature}K"
    )

    # --- Инфраструктурный слой (безопасное логирование) ---
    if tracker:
        tracker.start_run(run_name=f"sim_{task_id}")
        tracker.log_params(request.model_dump())

    # --- Научное ядро (Pure Domain) ---
    try:
        simulator = IdealGasSimulator(
            num_particles=request.num_particles,
            initial_temp=request.initial_temperature,
            dt=request.time_step
        )

        final_temp, total_energy = 0.0, 0.0
        for _ in range(100):
            final_temp, total_energy = simulator.step()

    except Exception as e:
        if tracker:
            tracker.end_run()
        logger.error(f"Simulation {task_id} failed: {e}")
        raise  # Пробрасываем ошибку дальше к API слою

    # --- Фиксация результатов ---
    result = SimulationResponse(
        task_id=task_id,
        final_temperature=round(final_temp, 4),
        total_energy=round(total_energy, 4),
        status="completed"
    )

    if tracker:
        tracker.log_metrics({
            "final_temperature": result.final_temperature,
            "total_energy": result.total_energy
        })
        tracker.end_run()

    logger.info(f"Simulation {task_id} completed. Final T={final_temp:.2f}K")
    return result