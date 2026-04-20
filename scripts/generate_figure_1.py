"""
Reproducible script for generating Figure 1: Temperature Convergence.
This script is executed by papermill. 
"""
import numpy as np
import matplotlib.pyplot as plt

# --- PAPERMILL PARAMETERS BLOCK ---
parameters = {
    "NUM_PARTICLES": 1000,
    "TIME_STEP": 0.001,
    "TEMP_RANGE_START": 100.0,
    "TEMP_RANGE_END": 1000.0,
    "NUM_POINTS": 50,
    "NUM_STEPS": 100,
}
# ----------------------------------

def main():
    num_particles = parameters["NUM_PARTICLES"]
    time_step = parameters["TIME_STEP"]
    temp_start = parameters["TEMP_RANGE_START"]
    temp_end = parameters["TEMP_RANGE_END"]
    num_points = parameters["NUM_POINTS"]
    num_steps = parameters["NUM_STEPS"]

    from scientific_api.models.physics_engine import IdealGasSimulator

    temperatures = np.linspace(temp_start, temp_end, num_points)
    final_temps = []

    print(f"Starting computation for {num_points} points (N={num_particles})...")

    for t0 in temperatures:
        simulator = IdealGasSimulator(
            num_particles=num_particles,
            initial_temp=float(t0),
            dt=time_step
        )
        
        for _ in range(num_steps):
            final_t, _ = simulator.step()
            
        final_temps.append(final_t)

    print("Computation finished. Rendering plot...")

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(8, 5))
    
    ax.plot(temperatures, final_temps, 'o', markersize=5, color='#1f77b4', alpha=0.7, label='Simulated $T_{final}$')
    ax.plot([temp_start, temp_end], [temp_start, temp_end], 
            '--', color='red', linewidth=2, label='Ideal $T_{init} = T_{final}$')

    ax.set_xlabel('Initial Temperature $T_0$ (K)', fontsize=14)
    ax.set_ylabel('Final Temperature $T_{final}$ (K)', fontsize=14)
    ax.set_title(f'Stochastic Temperature Drift (N={num_particles}, Steps={num_steps})', fontsize=16)
    ax.legend(fontsize=12)

    import os
    os.makedirs("artifacts", exist_ok=True)
    out_path = "artifacts/figure_1_temperature.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {out_path}")

if __name__ == "__main__":
    main()
