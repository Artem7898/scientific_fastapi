"""Render saved-state analysis without importing plotting tools into the API core."""

from __future__ import annotations

import math
import os
import tempfile
import textwrap
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from threading import RLock
from typing import Any

from scientific_api import __version__

REPORT_VERSION = "velocity-report-v1"
_RENDER_LOCK = RLock()
_MOMENTS = (
    ("vx_mean", "Mean v_x"),
    ("vy_mean", "Mean v_y"),
    ("vx_variance", "Variance v_x (ddof=0)"),
    ("vy_variance", "Variance v_y (ddof=0)"),
    ("speed_mean", "Mean speed"),
    ("speed_second_moment", "Mean squared speed"),
    ("temperature_estimate", "Temperature estimate E/N"),
)


class ResearchDependencyError(RuntimeError):
    """The optional research dependencies required for rendering are unavailable."""


@dataclass(frozen=True)
class _Histogram:
    edges: tuple[float, ...]
    density: tuple[float, ...]
    theory: tuple[float, ...]


@dataclass(frozen=True)
class _Report:
    version: str
    sample_size: int
    temperature: float
    standard_error: float
    moments: dict[str, float]
    histograms: dict[str, _Histogram]
    source: dict[str, str]
    limitations: tuple[str, ...]


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number")
    return float(value)


def _text(value: Any, *, limit: int = 300) -> str:
    """Keep untrusted manifest labels bounded and outside MathText parsing."""
    value = " ".join(str(value).split())
    return value if len(value) <= limit else value[: limit - 3] + "..."


def _prepare(analysis: dict[str, Any]) -> _Report:
    try:
        sample_size = analysis["sample_size"]
        if isinstance(sample_size, bool) or not isinstance(sample_size, int) or sample_size < 1:
            raise ValueError("sample_size must be a positive integer")
        if analysis["dimensions"] != 2:
            raise ValueError("Velocity reports support two-dimensional snapshots only")
        temperature = _number(analysis["target_temperature"], "target_temperature")
        if temperature <= 0:
            raise ValueError("target_temperature must be positive")
        standard_error = _number(
            analysis["theory"]["temperature_standard_error"], "temperature_standard_error"
        )
        if standard_error < 0:
            raise ValueError("temperature_standard_error must be non-negative")
        moments = {key: _number(analysis["moments"][key], key) for key, _ in _MOMENTS}
        histograms = {}
        for component in ("vx", "vy", "speed"):
            values = analysis["histograms"][component]
            edges = tuple(_number(value, "bin_edges") for value in values["bin_edges"])
            density = tuple(_number(value, "density") for value in values["density"])
            theory = tuple(
                _number(value, "theoretical_density") for value in values["theoretical_density"]
            )
            if not 1 <= len(density) <= 500 or len(edges) != len(density) + 1:
                raise ValueError("Histograms require 1..500 bins and one extra bin edge")
            if len(theory) != len(density):
                raise ValueError("Theoretical and empirical histogram lengths must match")
            if any(left >= right for left, right in zip(edges, edges[1:])):
                raise ValueError("Histogram bin edges must be strictly increasing")
            if any(value < 0 for value in (*density, *theory)):
                raise ValueError("Histogram densities must be non-negative")
            histograms[component] = _Histogram(edges, density, theory)
        source = {
            key: _text(analysis["source"][key])
            for key in ("task_id", "attempt_id", "model_version", "state_sha256", "manifest_sha256")
        }
        limitations = analysis["limitations"]
        if not isinstance(limitations, list) or not all(isinstance(v, str) for v in limitations):
            raise ValueError("limitations must be a list of strings")
        return _Report(
            version=_text(analysis["analysis_version"]),
            sample_size=sample_size,
            temperature=temperature,
            standard_error=standard_error,
            moments=moments,
            histograms=histograms,
            source=source,
            limitations=tuple(_text(value, limit=400) for value in limitations[:12]),
        )
    except (KeyError, TypeError) as exc:
        raise ValueError("Incomplete velocity analysis for reporting") from exc


def _distribution_figure(Figure: Any, FigureCanvasAgg: Any, report: _Report) -> Any:
    figure = Figure(figsize=(12, 5), facecolor="white")
    FigureCanvasAgg(figure)
    figure.subplots_adjust(left=0.065, right=0.98, bottom=0.22, top=0.76, wspace=0.32)
    for index, (component, title) in enumerate(
        (("vx", "v_x component"), ("vy", "v_y component"), ("speed", "Speed magnitude")), 1
    ):
        axes = figure.add_subplot(1, 3, index)
        values = report.histograms[component]
        axes.stairs(
            values.density,
            values.edges,
            fill=True,
            color="#2563a6",
            alpha=0.4,
            label="Saved-state density",
        )
        axes.stairs(
            values.theory,
            values.edges,
            color="#ac3d24",
            linewidth=1.8,
            linestyle="--",
            label="Rayleigh bin average" if component == "speed" else "Gaussian bin average",
        )
        axes.set_title(title, fontsize=11, fontfamily="DejaVu Sans")
        axes.set_xlabel(
            "Speed (reduced units)" if component == "speed" else "Velocity (reduced units)",
            fontsize=9,
            fontfamily="DejaVu Sans",
        )
        axes.set_ylabel("Probability density", fontsize=9, fontfamily="DejaVu Sans")
        axes.tick_params(labelsize=8)
        for label in (*axes.get_xticklabels(), *axes.get_yticklabels()):
            label.set_fontfamily("DejaVu Sans")
        axes.set_ylim(bottom=0)
        axes.grid(alpha=0.2)
        axes.legend(loc="best", fontsize=8, prop={"family": "DejaVu Sans", "size": 8})
    figure.text(
        0.5,
        0.96,
        "Terminal velocity distributions",
        ha="center",
        va="top",
        fontsize=16,
        fontfamily="DejaVu Sans",
        parse_math=False,
    )
    figure.text(
        0.5,
        0.88,
        f"N={report.sample_size:,} | target T={report.temperature:.7g} | "
        f"model SE(T)={report.standard_error:.7g} | reduced units: m=k_B=1",
        ha="center",
        fontsize=10,
        fontfamily="DejaVu Sans",
        parse_math=False,
    )
    figure.text(
        0.5,
        0.82,
        f"Model: {_text(report.source['model_version'], limit=100)}",
        ha="center",
        fontsize=9,
        fontfamily="DejaVu Sans",
        parse_math=False,
    )
    figure.text(
        0.065,
        0.09,
        "Reference: independent 2D equilibrium velocities; theory is averaged over each bin.\n"
        "Finite plotted ranges omit theoretical tails. This is a saved-state comparison, not a new run.",
        fontsize=8,
        fontfamily="DejaVu Sans",
        linespacing=1.6,
        parse_math=False,
    )
    figure.text(
        0.065,
        0.025,
        f"Task: {_text(report.source['task_id'], limit=60)} | "
        f"attempt: {_text(report.source['attempt_id'], limit=60)}",
        fontsize=8,
        fontfamily="DejaVu Sans",
        parse_math=False,
    )
    return figure


def _text_pages(Figure: Any, title: str, paragraphs: list[str]) -> list[Any]:
    lines = []
    for paragraph in paragraphs:
        lines.extend(textwrap.wrap(paragraph, width=98) or [""])
        lines.append("")
    figures = []
    for start in range(0, max(1, len(lines)), 42):
        figure = Figure(figsize=(8.27, 11.69), facecolor="white")
        page_title = title if start == 0 else f"{title} (continued)"
        figure.text(
            0.075,
            0.94,
            page_title,
            fontsize=17,
            fontfamily="DejaVu Sans",
            va="top",
            parse_math=False,
        )
        figure.text(
            0.075,
            0.88,
            "\n".join(lines[start : start + 42]),
            fontsize=10,
            fontfamily="DejaVu Sans",
            va="top",
            linespacing=1.45,
            parse_math=False,
        )
        figure.text(
            0.075,
            0.045,
            f"scientific-api {__version__} | {REPORT_VERSION}",
            fontsize=8,
            fontfamily="DejaVu Sans",
            parse_math=False,
        )
        figures.append(figure)
    return figures


def _summary(report: _Report) -> list[str]:
    return [
        "Post-processing of a saved terminal velocity state. No simulation was executed to produce "
        "this report.",
        f"Sample size: {report.sample_size:,} independent particles; dimensions: 2.",
        f"Target temperature: {report.temperature:.10g} (reduced units).",
        *(f"{label}: {report.moments[key]:.10g}" for key, label in _MOMENTS),
        f"Model-predicted equilibrium temperature standard error: {report.standard_error:.10g} "
        "= T_target / sqrt(N). This is not an empirical confidence interval.",
        *(f"{key}: {value}" for key, value in report.source.items()),
        f"Analysis format: {report.version}; report format: {REPORT_VERSION}.",
    ]


def _caveats(report: _Report) -> list[str]:
    return [
        "Reference assumptions: independent two-dimensional equilibrium particle velocities, "
        "zero component means, and reduced units m = k_B = 1. No Kelvin or femtosecond conversion.",
        "Components follow Normal(mean=0, variance=T). Speed follows a Rayleigh distribution: "
        "p(s) = (s/T) exp(-s^2/(2T)) for s >= 0. Dashed histogram values are bin-averaged PDFs.",
        "This reference OU velocity model has no positions, interparticle forces, or collisions; "
        "it is not a full molecular dynamics simulation.",
        "Variances use ddof=0. Temperature is the mean squared speed divided by two. Finite "
        "histograms and empirical moments fluctuate even when the reference model is correct.",
        "The reported standard error describes one equilibrium snapshot under the reference "
        "model. It excludes model discrepancy, experimental uncertainty, and trajectory uncertainty.",
        "Visual agreement is not a goodness-of-fit test or proof of equilibration. No p-values or "
        "confidence intervals are inferred from these plots.",
        "SHA-256 checksums establish file integrity, not scientific correctness. Repeating numerical "
        "results requires the same parameters and software/hardware environment; PNG and PDF bytes "
        "need not match across environments.",
        *(f"Analysis limitation: {value}" for value in report.limitations),
    ]


def render_velocity_report(
    analysis: dict[str, Any], *, output_dir: Path, formats: tuple[str, ...]
) -> dict[str, Path]:
    """Write requested PNG/PDF reports from an analysis dictionary, without resimulation.

    Matplotlib imports and rendering are serialized because its font and rendering
    internals are not thread-safe. Explicit Figure/Agg objects avoid pyplot and
    changing the host application's global plotting backend or styles.
    """
    if not formats:
        return {}
    if any(value not in {"png", "pdf"} for value in formats):
        raise ValueError("Report formats must be png or pdf")
    report = _prepare(analysis)
    output_dir = Path(output_dir)
    requested = dict.fromkeys(formats)
    destinations = {
        name: output_dir / ("velocity_distribution.png" if name == "png" else "report.pdf")
        for name in requested
    }
    with _RENDER_LOCK:
        try:
            import matplotlib
            from matplotlib.backends.backend_agg import FigureCanvasAgg
            from matplotlib.backends.backend_pdf import PdfPages
            from matplotlib.figure import Figure
        except ImportError as exc:
            raise ResearchDependencyError(
                "Install scientific-api[research] to generate PNG and PDF reports"
            ) from exc

        if any(path.exists() for path in destinations.values()):
            raise FileExistsError(
                "Report output files already exist; existing reports are preserved"
            )
        output_dir.mkdir(parents=True, exist_ok=True)
        temporary_paths: dict[str, Path] = {}
        published: list[Path] = []
        figures = []
        try:
            distribution = _distribution_figure(Figure, FigureCanvasAgg, report)
            figures.append(distribution)
            for name in requested:
                descriptor, filename = tempfile.mkstemp(
                    prefix=".velocity-report-", suffix=f".{name}", dir=output_dir
                )
                os.close(descriptor)
                temporary_paths[name] = Path(filename)
            if "png" in requested:
                distribution.savefig(
                    temporary_paths["png"],
                    format="png",
                    dpi=160,
                    metadata={
                        "Software": f"scientific-api {__version__}; Matplotlib {matplotlib.__version__}",
                        "ReportVersion": REPORT_VERSION,
                        "AnalysisVersion": report.version,
                        "Description": "Saved-state velocity distributions in reduced units",
                        "SourceTaskId": report.source["task_id"],
                        "SourceAttemptId": report.source["attempt_id"],
                    },
                )
            if "pdf" in requested:
                figures.extend(
                    _text_pages(Figure, "Numerical summary and source provenance", _summary(report))
                )
                figures.extend(
                    _text_pages(Figure, "Scientific assumptions and limitations", _caveats(report))
                )
                with PdfPages(
                    temporary_paths["pdf"],
                    metadata={
                        "Title": "Saved-state velocity distribution report",
                        "Subject": "Reduced units; model-predicted uncertainty; checksums are not scientific proof",
                        "Author": "Scientific API Gateway",
                        "Creator": f"scientific-api {__version__}; {REPORT_VERSION}; Matplotlib {matplotlib.__version__}",
                        "CreationDate": None,
                        "ModDate": None,
                    },
                ) as pdf:
                    for figure in figures:
                        pdf.savefig(figure)
            for name, path in temporary_paths.items():
                os.replace(path, destinations[name])
                published.append(destinations[name])
        except BaseException:
            for path in published:
                path.unlink(missing_ok=True)
            raise
        finally:
            for path in temporary_paths.values():
                path.unlink(missing_ok=True)
            for figure in figures:
                figure.clear()
    return destinations
