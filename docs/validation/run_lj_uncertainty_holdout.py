"""Run the fixed registered Gaussian holdout; do not tune cases or stream seeds.

This calibrates statistical methods on synthetic scalars, not LJ dynamics.
The output retains eligibility, unavailable cases and conditional coverage.
"""

import argparse
import hashlib
import json
import math
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from scientific_api.domain.lennard_jones import LJ_MODEL_VERSION, LennardJonesParameters
from scientific_api.research import lj_uncertainty, student_t
from scientific_api.research.lj_analysis import SERIES_NAMES


def wilson(hits: int, count: int) -> list[float] | None:
    if not count:
        return None
    z = 1.959963984540054
    proportion = hits / count
    denominator = 1 + z * z / count
    center = (proportion + z * z / (2 * count)) / denominator
    half = (
        z * math.sqrt(proportion * (1 - proportion) / count + z * z / (4 * count**2)) / denominator
    )
    return [center - half, center + half]


def coverage(records: list[dict], key: str) -> dict:
    eligible = [record for record in records if record[key] is not None]
    hits = sum(record[key] for record in eligible)
    return {
        "reported": len(eligible),
        "total": len(records),
        "reported_fraction": len(eligible) / len(records),
        "hits": hits,
        "conditional_coverage": hits / len(eligible) if eligible else None,
        "conditional_wilson_95": wilson(hits, len(eligible)),
        "reported_and_cover_fraction_all_trials": hits / len(records),
    }


def source_hashes() -> dict[str, str]:
    return {
        Path(module.__file__).name: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
        for module in (lj_uncertainty, student_t)
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    arguments = parser.parse_args()
    design_path = Path(__file__).with_name("lj-uncertainty-holdout-design.json")
    design_bytes = design_path.read_bytes()
    design = json.loads(design_bytes)
    arguments.output_dir.mkdir(parents=True, exist_ok=False)
    (arguments.output_dir / "design.json").write_bytes(design_bytes)
    sources_before = source_hashes()
    started = datetime.now(UTC).isoformat()
    roots = np.random.SeedSequence(int(design["seed_entropy_decimal"])).spawn(len(design["cases"]))
    cases, raw_cases = [], []
    for case, root in zip(design["cases"], roots, strict=True):
        count, phi = case["samples"], case["phi"]
        parameters = LennardJonesParameters(num_particles=16, num_steps=count - 1, sample_stride=1)
        trials = []
        representative = None
        for index, stream in enumerate(root.spawn(case["independent_trials"])):
            noise_stream, bootstrap_stream = stream.spawn(2)
            generator = np.random.Generator(np.random.PCG64(noise_stream))
            values = np.empty(count, dtype=np.float64)
            values[0] = generator.normal()
            innovations = generator.normal(size=count - 1)
            for step, noise in enumerate(innovations, start=1):
                values[step] = phi * values[step - 1] + math.sqrt(1 - phi**2) * noise
            bootstrap_seed = int(bootstrap_stream.generate_state(1, dtype=np.uint64)[0])
            # Schema fixture only: these are Gaussian scalars, not LJ observations.
            series = {name: np.ones(count).tolist() for name in SERIES_NAMES}
            series["pressure"] = values.tolist()
            fixture = {
                "model_version": LJ_MODEL_VERSION,
                "sample_steps": list(range(count)),
                "series": series,
            }
            result = lj_uncertainty.analyze_lj_uncertainty(
                fixture,
                parameters,
                bootstrap_seed=bootstrap_seed,
                bootstrap_samples=design["analysis_options"]["bootstrap_samples"],
                confidence_level=design["analysis_options"]["confidence_level"],
                assume_stationary=design["analysis_options"]["assume_stationary"],
                block_length=design["analysis_options"]["block_length"],
                observable_names=tuple(design["analysis_options"]["observable_names"]),
            )
            metric = result["metrics"]["pressure"]
            naive_half = 1.959963984540054 * float(np.std(values, ddof=1)) / math.sqrt(count)
            primary = metric["confidence_interval"]
            secondary = metric.get("exploratory_bootstrap_percentile")
            trial = {
                "index": index,
                "bootstrap_seed_decimal": str(bootstrap_seed),
                "naive_covers_zero": bool(abs(float(np.mean(values))) <= naive_half),
                "primary_covers_zero": (
                    primary["lower"] <= 0 <= primary["upper"] if primary is not None else None
                ),
                "secondary_covers_zero": (
                    secondary["lower"] <= 0 <= secondary["upper"] if secondary is not None else None
                ),
                "metric": metric,
            }
            trials.append(trial)
            if representative is None and primary is not None:
                representative = {"trial_index": index, "values": values.tolist(), "metric": metric}
        case_result = {
            **case,
            "asymptotic_true_g": (1 + phi) / (1 - phi),
            "naive_iid_normal": coverage(trials, "naive_covers_zero"),
            "primary_batch_means_student_t": coverage(trials, "primary_covers_zero"),
            "secondary_exploratory_moving_blocks": coverage(trials, "secondary_covers_zero"),
            "primary_unavailable_reasons": dict(
                Counter(
                    item["metric"]["reason"]
                    for item in trials
                    if item["primary_covers_zero"] is None
                )
            ),
            "secondary_unavailable_reasons": dict(
                Counter(
                    item["metric"].get("exploratory_bootstrap_reason", item["metric"]["reason"])
                    for item in trials
                    if item["secondary_covers_zero"] is None
                )
            ),
        }
        cases.append(case_result)
        raw_cases.append({**case_result, "trials": trials, "representative": representative})
    if source_hashes() != sources_before:
        raise RuntimeError("Method source changed while evaluating the registered holdout")
    summary = {
        "schema_version": 1,
        "started_at_utc": started,
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "design_sha256": hashlib.sha256(design_bytes).hexdigest(),
        "method_sha256": sources_before,
        "python_numpy": {"numpy": np.__version__},
        "nominal_level": design["analysis_options"]["confidence_level"],
        "cases": cases,
        "limitations": design["limitations"],
    }
    raw = (json.dumps({**summary, "cases": raw_cases}, indent=2, allow_nan=False) + "\n").encode()
    (arguments.output_dir / "trial-results.json").write_bytes(raw)
    summary["trial_results_sha256"] = hashlib.sha256(raw).hexdigest()
    summary_bytes = (json.dumps(summary, indent=2, allow_nan=False) + "\n").encode()
    (arguments.output_dir / "summary.json").write_bytes(summary_bytes)
    print(summary_bytes.decode())


if __name__ == "__main__":
    main()
