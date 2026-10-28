"""Check real saved-state PNG/PDF reports and optional plotting isolation."""

from __future__ import annotations

import copy
import importlib.util
import json
import os
import re
import struct
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from scientific_api.research.reporting import render_velocity_report
from scientific_api.research.velocity_analysis import analyze_velocities

HAS_MATPLOTLIB = importlib.util.find_spec("matplotlib") is not None
requires_plotting = pytest.mark.skipif(not HAS_MATPLOTLIB, reason="Install the research extra")


@pytest.fixture
def analysis() -> dict[str, Any]:
    data = analyze_velocities(
        np.random.default_rng(42).normal(size=(64, 2)), target_temperature=1.0, bins=8
    )
    data["source"] = {
        "task_id": "source_task",
        "attempt_id": "a" * 32,
        "model_version": "ideal-gas-ou-v1",
        "state_sha256": "b" * 64,
        "manifest_sha256": "c" * 64,
    }
    return data


def _python(script: str, directory: Path, analysis: dict[str, Any]) -> subprocess.CompletedProcess:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[2] / "src")
    environment["MPLCONFIGDIR"] = str(directory / "matplotlib-cache")
    return subprocess.run(
        [sys.executable, "-c", script, str(directory / "reports")],
        input=json.dumps(analysis),
        text=True,
        capture_output=True,
        env=environment,
        cwd=directory,
        timeout=45,
        check=False,
    )


def _assert_png(path: Path) -> None:
    content = path.read_bytes()
    assert content.startswith(b"\x89PNG\r\n\x1a\n")
    width, height = struct.unpack(">II", content[16:24])
    assert (width, height) == (1920, 800)
    assert b"velocity-report-v1" in content
    assert b"velocity-distribution-v1" in content
    assert b"source_task" in content


def _assert_pdf(path: Path) -> None:
    content = path.read_bytes()
    assert content.startswith(b"%PDF-")
    assert len(re.findall(rb"/Type\s*/Page\b", content)) == 3
    assert b"velocity-report-v1" in content
    assert b"checksums are not scientific proof" in content
    assert b"/Type /Font" in content


def test_import_and_json_only_workflow_do_not_require_matplotlib(tmp_path: Path) -> None:
    process = _python(
        """
import builtins
import sys
from pathlib import Path
original_import = builtins.__import__
def blocked(name, *args, **kwargs):
    if name.startswith('matplotlib'):
        raise AssertionError('A JSON-only workflow tried to import Matplotlib')
    return original_import(name, *args, **kwargs)
builtins.__import__ = blocked
from scientific_api.research.reporting import render_velocity_report
output = Path(sys.argv[1])
assert render_velocity_report({}, output_dir=output, formats=()) == {}
assert not output.exists()
""",
        tmp_path,
        {},
    )
    assert process.returncode == 0, process.stdout + process.stderr


def test_missing_research_extra_has_actionable_error(
    tmp_path: Path, analysis: dict[str, Any]
) -> None:
    process = _python(
        """
import builtins
import json
import sys
from pathlib import Path
from scientific_api.research.reporting import ResearchDependencyError, render_velocity_report
original_import = builtins.__import__
def blocked(name, *args, **kwargs):
    if name.startswith('matplotlib'):
        raise ImportError('Optional plotting dependency unavailable')
    return original_import(name, *args, **kwargs)
builtins.__import__ = blocked
output = Path(sys.argv[1])
try:
    render_velocity_report(json.load(sys.stdin), output_dir=output, formats=('png',))
except ResearchDependencyError as error:
    assert 'scientific-api[research]' in str(error)
else:
    raise AssertionError('Missing plotting dependencies did not fail')
assert not output.exists()
""",
        tmp_path,
        analysis,
    )
    assert process.returncode == 0, process.stdout + process.stderr


@requires_plotting
def test_png_pdf_use_saved_analysis_and_preserve_input(
    tmp_path: Path, analysis: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    from scientific_api.services import simulation_service

    def no_resimulation(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("Reporting must not execute a simulation")

    monkeypatch.setattr(simulation_service, "run_simulation", no_resimulation)
    original = copy.deepcopy(analysis)
    files = render_velocity_report(analysis, output_dir=tmp_path, formats=("png", "pdf"))
    assert files == {
        "png": tmp_path / "velocity_distribution.png",
        "pdf": tmp_path / "report.pdf",
    }
    _assert_png(files["png"])
    _assert_pdf(files["pdf"])
    assert analysis == original
    assert {path.name for path in tmp_path.iterdir()} == {"velocity_distribution.png", "report.pdf"}


@requires_plotting
def test_plotting_backend_is_preserved_without_pyplot(
    tmp_path: Path, analysis: dict[str, Any]
) -> None:
    process = _python(
        """
import os
os.environ['MPLBACKEND'] = 'svg'
import json
import sys
from pathlib import Path
import matplotlib
from scientific_api.research.reporting import render_velocity_report
assert matplotlib.get_backend() == 'svg'
render_velocity_report(json.load(sys.stdin), output_dir=Path(sys.argv[1]), formats=('png', 'pdf'))
assert matplotlib.get_backend() == 'svg'
assert 'matplotlib.pyplot' not in sys.modules
""",
        tmp_path,
        analysis,
    )
    assert process.returncode == 0, process.stdout + process.stderr


@requires_plotting
def test_distinct_observations_change_plot_and_existing_reports_are_preserved(
    tmp_path: Path, analysis: dict[str, Any]
) -> None:
    first = render_velocity_report(analysis, output_dir=tmp_path / "first", formats=("png",))
    previous = first["png"].read_bytes()
    with pytest.raises(FileExistsError, match="preserved"):
        render_velocity_report(analysis, output_dir=tmp_path / "first", formats=("png",))
    assert first["png"].read_bytes() == previous

    changed = analyze_velocities(
        np.random.default_rng(43).normal(size=(64, 2)), target_temperature=1.0, bins=8
    )
    changed["source"] = analysis["source"]
    second = render_velocity_report(changed, output_dir=tmp_path / "second", formats=("png",))
    assert second["png"].read_bytes() != previous
    assert not (tmp_path / "first" / "report.pdf").exists()


@requires_plotting
def test_pdf_failure_does_not_publish_partial_reports(
    tmp_path: Path, analysis: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    from matplotlib.backends.backend_pdf import PdfPages

    def fail_pdf(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("PDF encoder failed")

    monkeypatch.setattr(PdfPages, "savefig", fail_pdf)
    with pytest.raises(RuntimeError, match="PDF encoder failed"):
        render_velocity_report(analysis, output_dir=tmp_path, formats=("png", "pdf"))
    assert list(tmp_path.iterdir()) == []


@requires_plotting
def test_concurrent_reports_are_complete(tmp_path: Path, analysis: dict[str, Any]) -> None:
    def render(index: int) -> dict[str, Path]:
        return render_velocity_report(
            analysis, output_dir=tmp_path / str(index), formats=("png", "pdf")
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(render, (0, 1)))
    for files in results:
        _assert_png(files["png"])
        _assert_pdf(files["pdf"])


@pytest.mark.parametrize("invalid", ["nonfinite", "edges", "length", "dimensions"])
def test_malformed_analysis_is_rejected_before_writing(
    tmp_path: Path, analysis: dict[str, Any], invalid: str
) -> None:
    if invalid == "nonfinite":
        analysis["histograms"]["vx"]["density"][0] = float("nan")
    elif invalid == "edges":
        analysis["histograms"]["vx"]["bin_edges"][1] = analysis["histograms"]["vx"]["bin_edges"][0]
    elif invalid == "length":
        analysis["histograms"]["vx"]["theoretical_density"].pop()
    else:
        analysis["dimensions"] = 3
    output = tmp_path / "reports"
    with pytest.raises(ValueError):
        render_velocity_report(analysis, output_dir=output, formats=("png",))
    assert not output.exists()


def test_unsupported_format_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="png or pdf"):
        render_velocity_report({}, output_dir=tmp_path / "reports", formats=("svg",))
