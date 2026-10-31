"""Run with an isolated Python and a core-only installed wheel outside checkout."""

import asyncio
import hashlib
import importlib.metadata
import importlib.util
import io
import json
import logging
import math
import os
import sys
import tempfile
import time
import zipfile
from dataclasses import asdict
from pathlib import Path


def main() -> None:
    optional_packages = ("numba", "llvmlite", "jax", "jaxlib", "matplotlib")
    assert all(importlib.util.find_spec(name) is None for name in optional_packages)

    import numpy as np

    import scientific_api
    from scientific_api.core.config import Settings
    from scientific_api.core.research_manager import ResearchManager
    from scientific_api.core.research_store import MemoryResearchStore
    from scientific_api.core.task_store import MemoryTaskStore
    from scientific_api.core.telemetry import Telemetry
    from scientific_api.core.verification_manager import VerificationManager
    from scientific_api.core.verification_store import MemoryVerificationStore
    from scientific_api.domain.checkpoint import checkpoint_model_version
    from scientific_api.domain.langevin import LangevinParameters
    from scientific_api.domain.lennard_jones import LennardJonesParameters
    from scientific_api.domain.simulation import SimulationParameters
    from scientific_api.main import app
    from scientific_api.models.computation import (
        BackendUnavailable,
        create_simulator,
        require_backend,
    )
    from scientific_api.models.langevin_engine import create_langevin_simulator
    from scientific_api.models.lennard_jones_engine import create_lj_simulator
    from scientific_api.research.lj_analysis import analyze_lj_observables
    from scientific_api.research.transport_analysis import analyze_transport, load_trajectory_state
    from scientific_api.schemas.lj_research import LJResearchRequest
    from scientific_api.services.analysis_service import create_velocity_analysis
    from scientific_api.services.checkpoint_service import CheckpointControl, load_checkpoint
    from scientific_api.services.lj_research_service import validate_lj_research_report
    from scientific_api.services.model_registry import (
        backend_availability,
        get_model_contract,
        list_model_contracts,
        model_contract_sha256,
    )
    from scientific_api.services.provenance_evidence import build_archive_evidence
    from scientific_api.services.provenance_export import export_ro_crate
    from scientific_api.services.provenance_service import ProvenanceService
    from scientific_api.services.result_envelope_service import read_result_envelope
    from scientific_api.services.simulation_service import run_simulation
    from scientific_api.services.verification_service import validate_verification_certificate

    expected = os.environ["EXPECTED_PACKAGE_VERSION"]
    assert scientific_api.__version__ == expected
    assert importlib.metadata.version("scientific-api") == expected
    assert app is not None

    contracts = list_model_contracts()
    assert {contract.model_version for contract in contracts} == {
        "ideal-gas-ou-v1",
        "langevin-free-v1",
        "lennard-jones-2d-v1",
    }
    for contract in contracts:
        assert contract.request_schema["additionalProperties"] is False
        assert contract.result_schema["additionalProperties"] is False
        assert contract.default_backend == "numpy"
        assert {backend.backend for backend in contract.backends} == {"numpy", "numba"}
    availability = {entry.backend: entry for entry in backend_availability()}
    assert availability["numpy"].dependency_present is True
    assert availability["numba"].dependency_present is False
    assert all(entry.worker_readiness == "not_checked" for entry in availability.values())

    def verify_envelope(result: dict, root: Path) -> None:
        envelope = read_result_envelope(result, artifacts_dir=root, require_present=True)
        assert envelope.integrity.status == "verified"
        assert envelope.authenticity == "unsigned"
        assert envelope.scientific_verification == "not_performed"
        assert envelope.result.model_dump(mode="json") == result
        assert envelope.model_contract is not None
        contract = get_model_contract(result["model_version"])
        assert envelope.model_contract.sha256 == model_contract_sha256(contract)
        assert envelope.manifest is not None
        assert (
            envelope.manifest.sha256
            == hashlib.sha256((root / result["artifacts"]["manifest"]).read_bytes()).hexdigest()
        )

    def verify_checkpoint_resume(parameters, reference: dict, root: Path) -> None:
        registered = []

        class InterruptedAfterCheckpoint(RuntimeError):
            pass

        def register(descriptor: dict) -> bool:
            registered.append(descriptor)
            if descriptor["completed_steps"] > 0:
                raise InterruptedAfterCheckpoint
            return True

        task_id = reference["task_id"] + "_checkpoint"
        control = CheckpointControl(
            parameters,
            task_id=task_id,
            attempt_id="interrupted",
            artifacts_dir=root,
            interval_steps=1,
            register=register,
        )
        try:
            run_simulation(
                parameters,
                task_id=task_id,
                attempt_id="interrupted",
                artifacts_dir=root,
                checkpoint_control=control,
            )
        except InterruptedAfterCheckpoint:
            pass
        else:
            raise AssertionError("The controlled checkpoint interruption did not occur")
        descriptor = registered[-1]
        assert 0 < descriptor["completed_steps"] < parameters.num_steps
        loaded = load_checkpoint(descriptor, parameters=parameters, artifacts_dir=root)
        resumed_control = CheckpointControl(
            parameters,
            task_id=task_id,
            attempt_id="resumed",
            artifacts_dir=root,
            loaded=loaded,
        )
        resumed = run_simulation(
            parameters,
            task_id=task_id,
            attempt_id="resumed",
            artifacts_dir=root,
            checkpoint_control=resumed_control,
        )
        verify_envelope(resumed, root)
        for name in reference.keys() - {"task_id", "attempt_id", "artifacts"}:
            assert resumed[name] == reference[name], name
        for name, relative in reference["artifacts"].items():
            if name == "manifest":
                continue
            original_path = root / relative
            resumed_path = root / resumed["artifacts"][name]
            if original_path.suffix == ".npz":
                with (
                    np.load(original_path, allow_pickle=False) as original,
                    np.load(resumed_path, allow_pickle=False) as continued,
                ):
                    assert set(original.files) == set(continued.files)
                    for array in original.files:
                        np.testing.assert_array_equal(original[array], continued[array])
            else:
                assert json.loads(original_path.read_bytes()) == json.loads(
                    resumed_path.read_bytes()
                )

    require_backend("numpy")
    simulator = create_simulator(SimulationParameters(initial_temperature=1.0, num_particles=4))
    simulator.advance()
    assert all(math.isfinite(value) for value in simulator.observables())
    assert simulator.execution_metadata()["backend"] == "numpy"

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        ou_parameters = SimulationParameters(initial_temperature=1.0, num_particles=8, num_steps=3)
        ou_result = run_simulation(
            ou_parameters,
            task_id="core_ou",
            attempt_id="wheel",
            artifacts_dir=root,
        )
        verify_envelope(ou_result, root)
        verify_checkpoint_resume(ou_parameters, ou_result, root)
        analysis = create_velocity_analysis(ou_result, artifacts_dir=root, bins=4, formats=())
        analysis_envelope = read_result_envelope(
            ou_result, artifacts_dir=root, analysis_id=analysis["analysis_id"], require_present=True
        )
        assert analysis_envelope.kind == "velocity_analysis"
        assert analysis_envelope.integrity.status == "verified"
        assert analysis_envelope.result.model_dump(mode="json") == analysis
        parameters = LangevinParameters(
            1.0, 1.0, 4, num_steps=4, time_step=0.1, sample_stride=2, seed=42
        )
        result = run_simulation(
            parameters, task_id="core_langevin", attempt_id="wheel", artifacts_dir=root
        )
        assert result["status"] == "completed"
        assert result["model_version"] == "langevin-free-v1"
        verify_envelope(result, root)
        verify_checkpoint_resume(parameters, result, root)
        assert set(result["artifacts"]) == {"manifest", "final_state", "trajectory", "transport"}
        manifest = json.loads((root / result["artifacts"]["manifest"]).read_text())
        assert manifest["model"]["version"] == "langevin-free-v1"
        assert manifest["parameters"]["backend"] == manifest["execution"]["backend"] == "numpy"
        records = {entry["path"]: entry for entry in manifest["artifacts"]}
        for path, record in records.items():
            assert hashlib.sha256((root / path).read_bytes()).hexdigest() == record["sha256"]
        trajectory = load_trajectory_state(
            root / result["artifacts"]["trajectory"],
            expected_sha256=records[result["artifacts"]["trajectory"]]["sha256"],
            expected_particles=parameters.num_particles,
            expected_sample_steps=parameters.sampling_steps(),
            expected_time_step=parameters.time_step,
        )
        assert trajectory["positions"].shape == (3, 4, 2)
        assert trajectory["sample_steps"].dtype == np.int64
        assert trajectory["times"].dtype == np.float64
        with np.load(root / result["artifacts"]["final_state"], allow_pickle=False) as state:
            assert set(state.files) == {"positions", "unwrapped_positions", "velocities"}
            np.testing.assert_array_equal(trajectory["positions"][-1], state["unwrapped_positions"])
            np.testing.assert_array_equal(state["positions"], state["unwrapped_positions"])
        observed = analyze_transport(
            trajectory["positions"], trajectory["sample_steps"], parameters=parameters
        )
        saved = json.loads((root / result["artifacts"]["transport"]).read_text())
        assert observed == saved
        assert saved["coordinate_convention"] == "unwrapped"
        assert saved["msd"][-1] == result["final_msd"]
        assert saved["centered_msd"][-1] == result["final_centered_msd"]
        assert saved["diffusion"]["status"] == "unavailable"
        assert saved["diffusion"]["estimate"] is None

        parameters = LennardJonesParameters(
            num_particles=36,
            box_size=(8.0, 8.0),
            initial_temperature=0.1,
            num_steps=4,
            time_step=0.001,
            sample_stride=1,
            seed=42,
        )
        result = run_simulation(
            parameters, task_id="core_lj", attempt_id="wheel", artifacts_dir=root
        )
        assert result["status"] == "completed"
        assert result["model_version"] == "lennard-jones-2d-v1"
        verify_envelope(result, root)
        verify_checkpoint_resume(parameters, result, root)
        assert result["potential_energy"] < 0 and result["total_energy"] < 0
        assert set(result["artifacts"]) == {"manifest", "final_state", "trajectory", "observables"}
        manifest = json.loads((root / result["artifacts"]["manifest"]).read_text())
        assert manifest["execution"]["backend"] == "numpy"
        assert manifest["model"]["integrator"] == "velocity-verlet"
        assert "shifted-force" in manifest["model"]["potential"]["name"]
        records = {entry["path"]: entry for entry in manifest["artifacts"]}
        for path, record in records.items():
            assert hashlib.sha256((root / path).read_bytes()).hexdigest() == record["sha256"]
        trajectory = load_trajectory_state(
            root / result["artifacts"]["trajectory"],
            expected_sha256=records[result["artifacts"]["trajectory"]]["sha256"],
            expected_particles=parameters.num_particles,
            expected_sample_steps=parameters.sampling_steps(),
            expected_time_step=parameters.time_step,
        )
        assert trajectory["positions"].shape == (5, 36, 2)
        with np.load(root / result["artifacts"]["final_state"], allow_pickle=False) as state:
            assert set(state.files) == {"positions", "unwrapped_positions", "velocities"}
            np.testing.assert_array_equal(trajectory["positions"][-1], state["unwrapped_positions"])
            assert ((state["positions"] >= 0) & (state["positions"] < 8)).all()
        saved = json.loads((root / result["artifacts"]["observables"]).read_text())
        series = {
            name: np.asarray(values, dtype=np.float64) for name, values in saved["series"].items()
        }
        observed = analyze_lj_observables(
            trajectory["positions"], trajectory["sample_steps"], series, parameters=parameters
        )
        assert observed == saved
        assert observed["series"]["total_energy"][-1] == result["total_energy"]
        assert observed["rdf"]["dimensions"] == 2
        assert observed["energy_drift"]["applicable"] is True

        store = MemoryTaskStore(ttl_seconds=0.01)
        payload = {
            "initial_temperature": 1.0,
            "num_particles": 8,
            "num_steps": 3,
            "time_step": 0.001,
            "seed": 2**64 - 1,
            "backend": "numpy",
        }
        store.submit("core_archived", payload, {})
        job = store.claim()
        assert job is not None
        archived_result = run_simulation(
            SimulationParameters(**payload),
            task_id=job.task_id,
            attempt_id=job.owner,
            artifacts_dir=root,
        )
        envelope = read_result_envelope(archived_result, artifacts_dir=root, require_present=True)
        assert store.finish(job, archived_result, evidence=build_archive_evidence(envelope))
        analysis = create_velocity_analysis(archived_result, artifacts_dir=root, bins=4, formats=())
        analysis_envelope = read_result_envelope(
            archived_result,
            artifacts_dir=root,
            analysis_id=analysis["analysis_id"],
            require_present=True,
        )
        store.register_analysis(
            job.task_id,
            job.owner,
            analysis["analysis_id"],
            analysis,
            evidence=build_archive_evidence(analysis_envelope),
        )
        time.sleep(0.02)
        assert store.get(job.task_id) is None
        provenance = ProvenanceService(store, root)
        archived = provenance.task(job.task_id)
        assert archived.coverage == "complete" and archived.storage == "memory"
        assert archived.record == archived_result
        assert archived.evidence_status == "verified"
        assert archived.parameters["seed"] == 2**64 - 1
        assert provenance.analysis(analysis["analysis_id"]).record == analysis
        assert archived.artifacts
        assert provenance.artifact(archived.artifacts[0].artifact_id) == archived.artifacts[0]
        first_page = provenance.timeline(job.task_id, limit=2)
        assert len(first_page.events) == 2 and first_page.has_more
        assert first_page.next_cursor is not None
        remaining = provenance.timeline(job.task_id, after_id=first_page.next_cursor)
        events = first_page.events + remaining.events
        assert {item.event_type for item in events} >= {
            "submitted",
            "attempt_claimed",
            "completed",
            "analysis_completed",
        }
        graph = provenance.graph(job.task_id).model_dump(mode="json", by_alias=True)
        assert graph["@context"] and graph["@graph"]
        package, filename = export_ro_crate(provenance, job.task_id)
        assert filename.endswith(".zip")
        with zipfile.ZipFile(io.BytesIO(package)) as archive:
            assert {
                "ro-crate-metadata.json",
                "provenance.jsonld",
                "task.json",
                "model-contract.json",
            } <= set(archive.namelist())
            assert all(
                not Path(name).is_absolute() and ".." not in Path(name).parts
                for name in archive.namelist()
            )
            metadata = json.loads(archive.read("ro-crate-metadata.json"))
            dataset = next(item for item in metadata["@graph"] if item["@id"] == "./")
            assert dataset["license"] == {"@id": "#data-license-unavailable"}
            assert any(
                entity.get("@id") == "#data-license-unavailable" for entity in metadata["@graph"]
            )
            assert not any(item.get("@type") == "Person" for item in metadata["@graph"])
            for record in [
                *archived.artifacts,
                *provenance.analysis(analysis["analysis_id"]).artifacts,
            ]:
                assert hashlib.sha256(archive.read(record.path)).hexdigest() == record.sha256
            unpacked = root / "roundtrip"
            archive.extractall(unpacked)
        assert (
            read_result_envelope(
                archived_result, artifacts_dir=unpacked, require_present=True
            ).integrity.status
            == "verified"
        )
        assert (
            read_result_envelope(
                archived_result,
                artifacts_dir=unpacked,
                analysis_id=analysis["analysis_id"],
                require_present=True,
            ).integrity.status
            == "verified"
        )
        verification_sources = [(job.task_id, False), (job.task_id, True)]
        for identifier, physical in (
            (
                "core_verify_langevin",
                LangevinParameters(
                    1.0, 1.0, 4, num_steps=4, time_step=0.1, sample_stride=2, seed=42
                ),
            ),
            ("core_verify_lj", parameters),
        ):
            store.submit(
                identifier, {"model": checkpoint_model_version(physical), **asdict(physical)}, {}
            )
            source_job = store.claim()
            assert source_job is not None and source_job.task_id == identifier
            source_result = run_simulation(
                physical, task_id=identifier, attempt_id=source_job.owner, artifacts_dir=root
            )
            source_envelope = read_result_envelope(
                source_result, artifacts_dir=root, require_present=True
            )
            assert store.finish(
                source_job, source_result, evidence=build_archive_evidence(source_envelope)
            )
            verification_sources.append((identifier, False))
        time.sleep(0.02)
        assert all(store.get(identifier) is None for identifier, _ in verification_sources)
        source_archive = store.get_task_archive(job.task_id)
        source_bytes = {
            name: (root / path).read_bytes() for name, path in archived_result["artifacts"].items()
        }
        verification_store = MemoryVerificationStore(ttl_seconds=0.01)
        telemetry = Telemetry(enabled=False)
        verification_manager = VerificationManager(
            verification_store,
            store,
            Settings(_env_file=None, artifacts_dir=root, redis_uri="", mlflow_tracking_uri=""),
            telemetry,
        )

        async def verify_archived_sources() -> None:
            await verification_manager.start()
            try:
                accepted = [
                    await verification_manager.submit(identifier, rerun=rerun)
                    for identifier, rerun in verification_sources
                ]

                async def completed(identifier: str) -> dict:
                    deadline = time.monotonic() + 15
                    while time.monotonic() < deadline:
                        record = verification_manager.get(identifier)
                        if record["status"] == "completed":
                            return record
                        assert record["status"] in {"pending", "running"}, record
                        await asyncio.sleep(0.01)
                    raise AssertionError("Core-only verification did not complete")

                for item in accepted:
                    record = await completed(item["verification_id"])
                    assert record["verdict"] == "passed", record
                    raw, digest = verification_manager.certificate(item["verification_id"])
                    assert hashlib.sha256(raw).hexdigest() == digest == record["certificate_sha256"]
                    certificate = validate_verification_certificate(raw, expected_sha256=digest)
                    assert certificate.authenticity == "unsigned"
                    assert certificate.scope == "recorded_artifacts_at_verification_time"
                    assert certificate.source.task_id == item["source_task_id"]
                    assert certificate.source.attempt_id == item["source_attempt_id"]
                    assert certificate.checks and certificate.limitations
                original_id = accepted[0]["verification_id"]
                original_certificate = verification_manager.certificate(original_id)
                state_path = root / archived_result["artifacts"]["final_state"]
                try:
                    state_path.write_bytes(source_bytes["final_state"] + b"corrupted")
                    bad = await verification_manager.submit(job.task_id)
                    assert (await completed(bad["verification_id"]))["verdict"] == "failed"
                    assert verification_manager.certificate(original_id) == original_certificate
                finally:
                    state_path.write_bytes(source_bytes["final_state"])
                assert len(verification_manager.list(job.task_id)["verifications"]) == 3
            finally:
                await verification_manager.stop()

        asyncio.run(verify_archived_sources())
        assert store.get_task_archive(job.task_id) == source_archive
        assert all(
            (root / archived_result["artifacts"][name]).read_bytes() == raw
            for name, raw in source_bytes.items()
        )
        assert export_ro_crate(provenance, job.task_id)[0] == package
        verification_store.close()

        research_source = "core_verify_lj"
        source_before_research = store.get_task_archive(research_source)
        assert source_before_research is not None
        research_result = source_before_research["record"]
        research_source_bytes = {
            name: (root / path).read_bytes() for name, path in research_result["artifacts"].items()
        }
        research_crate = export_ro_crate(provenance, research_source)[0]
        research_store = MemoryResearchStore(ttl_seconds=0.01)
        research_manager = ResearchManager(
            research_store,
            store,
            Settings(_env_file=None, artifacts_dir=root, redis_uri="", mlflow_tracking_uri=""),
            telemetry,
        )
        research_request = LJResearchRequest(
            task_ids=[research_source],
            observables=["pressure"],
            bootstrap_samples=200,
            bootstrap_seed=2**64 - 1,
        )

        async def analyze_saved_lj() -> None:
            await research_manager.start()
            try:

                async def finished(identifier: str) -> dict:
                    deadline = time.monotonic() + 15
                    while time.monotonic() < deadline:
                        record = research_manager.get(identifier)
                        if record["status"] in {"completed", "failed", "cancelled"}:
                            return record
                        await asyncio.sleep(0.01)
                    raise AssertionError("Core-only LJ research did not finish")

                reports = []
                accepted = []
                for _ in range(2):
                    item = await research_manager.submit(research_request)
                    accepted.append(item)
                    record = await finished(item["analysis_id"])
                    assert record["status"] == "completed", record
                    assert record["availability"] == "unavailable"
                    raw, digest = research_manager.report(item["analysis_id"])
                    assert hashlib.sha256(raw).hexdigest() == digest == record["report_sha256"]
                    report = validate_lj_research_report(raw, expected_sha256=digest)
                    assert report.options.bootstrap_seed == 2**64 - 1
                    assert report.authenticity == "unsigned"
                    assert report.source_integrity == "verified_against_committed_archive"
                    assert report.sources[0].identity.task_id == research_source
                    assert report.statistics["metrics"]["pressure"]["confidence_interval"] is None
                    assert report.statistics["sampling"]["retained_sample_count"] < 128
                    reports.append(report)
                assert accepted[0]["analysis_id"] != accepted[1]["analysis_id"]
                assert reports[0].statistics == reports[1].statistics
                original_report = research_manager.report(accepted[0]["analysis_id"])
                observed_path = root / research_result["artifacts"]["observables"]
                expected_failure_log = logging.getLogger("scientific_api.core.research_manager")
                previous_log_level = expected_failure_log.level
                try:
                    expected_failure_log.setLevel(logging.CRITICAL)
                    observed_path.write_bytes(research_source_bytes["observables"] + b" ")
                    bad = await research_manager.submit(research_request)
                    bad_record = await finished(bad["analysis_id"])
                    assert bad_record["status"] == "failed"
                    assert bad_record["error_code"] == "RESEARCH_SOURCE_INTEGRITY_FAILED"
                    assert research_manager.report(accepted[0]["analysis_id"]) == original_report
                finally:
                    expected_failure_log.setLevel(previous_log_level)
                    observed_path.write_bytes(research_source_bytes["observables"])
                assert len(research_manager.list(research_source)["analyses"]) == 3
            finally:
                await research_manager.stop()

        asyncio.run(analyze_saved_lj())
        assert store.get(research_source) is None
        assert store.get_task_archive(research_source) == source_before_research
        assert all(
            (root / research_result["artifacts"][name]).read_bytes() == raw
            for name, raw in research_source_bytes.items()
        )
        assert export_ro_crate(provenance, research_source)[0] == research_crate
        assert (
            read_result_envelope(research_result, artifacts_dir=root).uncertainty.availability
            == "unavailable"
        )
        research_store.close()
        telemetry.shutdown()
        store.close()

    for operation in (
        lambda: require_backend("numba"),
        lambda: create_simulator(
            SimulationParameters(initial_temperature=1.0, num_particles=4, backend="numba")
        ),
        lambda: create_langevin_simulator(LangevinParameters(1.0, 1.0, 4, backend="numba")),
        lambda: create_lj_simulator(LennardJonesParameters(backend="numba")),
    ):
        try:
            operation()
        except BackendUnavailable as error:
            assert "scientific-api[numba]" in str(error)
        else:
            raise AssertionError("Unavailable Numba backend silently executed or fell back")

    for unknown_backend in ("jax", "auto", "unknown"):
        try:
            require_backend(unknown_backend)
        except ValueError:
            pass
        else:
            raise AssertionError("An unknown backend was admitted")

    assert not any(name.split(".", 1)[0] in optional_packages for name in sys.modules), (
        "The default runtime imported an optional compiled backend"
    )
    assert "scientific_api.models.numba_engine" not in sys.modules
    assert "scientific_api.models.langevin_numba" not in sys.modules
    assert "scientific_api.models.lj_numba" not in sys.modules
    print("Core-only wheel: checkpoints, provenance, verification and immutable queued LJ research")


if __name__ == "__main__":
    main()
