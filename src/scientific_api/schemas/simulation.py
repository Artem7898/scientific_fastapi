from pydantic import BaseModel, Field, StrictFloat, StrictInt


class SimulationRequest(BaseModel):
    """
    Request payload for the Monte Carlo simulation.
    Uses strict types to prevent string coercion.
    """
    initial_temperature: StrictFloat = Field(
        ...,
        gt=0.0,
        le=1e6,
        description="Initial system temperature in Kelvin",
        examples=[300.0],
    )
    num_particles: StrictInt = Field(
        ...,
        gt=0,
        le=1e6,
        description="Number of particles in the simulation volume",
    )
    time_step: StrictFloat = Field(
        default=1e-3,
        gt=0.0,
        description="Integration time step in femtoseconds",
    )


class SimulationResponse(BaseModel):
    """
    Response containing the computed thermodynamic properties.
    """
    task_id: str = Field(..., description="Unique identifier for tracking reproducibility")
    final_temperature: StrictFloat
    total_energy: StrictFloat
    status: str