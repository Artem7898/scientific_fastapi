"""
Orchestrator for paper artifacts generation.
Run this script to generate all figures for the publication.
"""
import papermill as pm
from pathlib import Path

def main():
    print("=" * 50)
    print("Starting Reproducible Paper Pipeline")
    print("=" * 50)

    Path("artifacts").mkdir(exist_ok=True)

    try:
        # Вызываем Python-скрипт через papermill
        pm.execute_notebook(
            input_path="scripts/generate_figure_1.py",
            output_path="artifacts/figure_1_log.py", # Сохраняем лог выполнения
            parameters=dict(
                NUM_PARTICLES=5000,   # Переопределяем параметры для статьи
                TIME_STEP=0.001,
                TEMP_RANGE_START=100.0,
                TEMP_RANGE_END=1000.0,
                NUM_POINTS=50,
                NUM_STEPS=200
            )
        )
        print("\n[SUCCESS] Pipeline finished. Check /artifacts folder.")
    except Exception as e:
        print(f"\n[ERROR] Pipeline failed: {e}")
        exit(1)

if __name__ == "__main__":
    main()