"""Report coverage, source binding, budgets and ambiguous durable commits."""

import hashlib

import pytest
from test_lj_research_source import archived_source

from scientific_api.core import task_store
from scientific_api.core.config import Settings
from scientific_api.core.research_manager import ResearchManager
from scientific_api.core.research_store import MemoryResearchStore
from scientific_api.core.telemetry import Telemetry
from scientific_api.domain.simulation import SimulationCancelled
from scientific_api.schemas.lj_research import LJResearchRequest
from scientific_api.services import lj_research_service as service


def compute(root, request, **kwargs):
    return service.compute_lj_research(
        request,
        analysis_id="a" * 32,
        analysis_attempt_id="owner",
        artifacts_dir=root,
        check_cancelled=lambda: None,
        **kwargs,
    )


def prepared(root, **options):
    store, _, archive = archived_source(root)
    request = service.prepare_lj_research(store, LJResearchRequest(task_ids=["source"], **options))
    return store, archive, request


def test_report_is_reproducible_in_statistics_and_does_not_change_saved_results(tmp_path):
    store, archive, request = prepared(tmp_path, bootstrap_seed=2**64 - 1)
    original = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    first, second = compute(tmp_path, request), compute(tmp_path, request)
    assert first.statistics == second.statistics
    assert first.availability == "unavailable"
    assert first.options.bootstrap_seed == 2**64 - 1
    assert (
        first.sources[0].identity.parameters_sha256
        == request["sources"][0]["source"]["parameters_sha256"]
    )
    assert store.get_task_archive("source") == archive
    assert original == {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    raw = service.research_report_bytes(first)
    digest = hashlib.sha256(raw).hexdigest()
    assert service.validate_lj_research_report(raw, expected_sha256=digest) == first


def test_real_saved_lj_trajectory_can_supply_a_conditional_batch_means_interval(tmp_path):
    store, _, _ = archived_source(
        tmp_path,
        seed=7,
        box_size=(100.0, 100.0),
        initial_temperature=1.0,
        num_steps=10_000,
        sample_stride=10,
        equilibration_steps=1000,
        ensemble="nvt",
        bath_temperature=1.0,
        friction=10.0,
        time_step=0.01,
    )
    options = LJResearchRequest(
        task_ids=["source"],
        observables=["thermal_temperature"],
        assume_stationary=True,
        bootstrap_samples=200,
    )
    report = compute(tmp_path, service.prepare_lj_research(store, options))
    assert report.availability == "available", report.statistics
    metric = report.statistics["metrics"]["thermal_temperature"]
    assert "Student" in metric["confidence_interval"]["method"]
    assert metric["confidence_interval"]["coverage_guarantee"] is False
    # A coarse secondary bootstrap budget does not disable the primary interval.
    assert metric["exploratory_bootstrap_percentile"] is None
    service.validate_lj_research_report(service.research_report_bytes(report))


@pytest.mark.parametrize(
    "field,value", [("initial_temperature", 0.2), ("seed", 7), ("num_steps", 8)]
)
def test_visible_physics_parameters_must_match_the_pinned_digest(tmp_path, field, value):
    _, _, request = prepared(tmp_path)
    report = compute(tmp_path, request)
    report.sources[0].parameters[field] = value
    with pytest.raises(service.ResearchError, match="RESEARCH_REPORT_INVALID"):
        service.validate_lj_research_report(service.research_report_bytes(report))


@pytest.mark.parametrize(
    "mutation", ["missing_observable", "wrong_kind", "false_availability", "wrong_rng"]
)
def test_incomplete_or_contradictory_report_cannot_claim_available_statistics(tmp_path, mutation):
    _, _, request = prepared(tmp_path)
    report = compute(tmp_path, request)
    if mutation == "missing_observable":
        report.statistics["metrics"].pop("pressure")
    elif mutation == "wrong_kind":
        report.statistics["analysis_version"] = "different-policy"
    elif mutation == "false_availability":
        report.availability = "available"
    else:
        report.statistics["bootstrap"]["seed"] += 1
    with pytest.raises(service.ResearchError, match="RESEARCH_REPORT_INVALID"):
        service.validate_lj_research_report(service.research_report_bytes(report))


def test_worker_rechecks_collective_budget_before_opening_any_scientific_file(
    tmp_path, monkeypatch
):
    _, _, request = prepared(tmp_path)
    monkeypatch.setattr(service, "MAX_RESEARCH_SOURCE_BYTES", 1)

    def no_reads(*args, **kwargs):
        pytest.fail("The actor exceeded its cumulative budget before checking limits")

    monkeypatch.setattr(service, "load_lj_research_source", no_reads)
    with pytest.raises(service.ResearchError, match="RESEARCH_INPUT_LIMIT_EXCEEDED"):
        compute(tmp_path, request)


@pytest.mark.parametrize(
    "field,value", [("source_id", "other"), ("policy_version", "future"), ("sources", [])]
)
def test_research_job_remains_bound_to_exact_requested_sources(tmp_path, field, value):
    _, _, request = prepared(tmp_path)
    request[field] = value
    with pytest.raises(service.ResearchError, match="RESEARCH_REQUEST_INTEGRITY_FAILED"):
        compute(tmp_path, request)


def test_physical_burn_in_is_ulp_aware_and_never_reintroduces_excluded_samples():
    from scientific_api.domain.lennard_jones import LennardJonesParameters

    parameters = LennardJonesParameters(
        num_particles=16, num_steps=10, time_step=0.003, sample_stride=1, equilibration_steps=2
    )
    assert service._burn_in_step(parameters, 0.009) == 3
    assert service._burn_in_step(parameters, 3 * 0.003) == 3
    assert service._burn_in_step(parameters, 0.0) == 2
    with pytest.raises(service.ResearchError, match="RESEARCH_BURN_IN_OUTSIDE_RUN"):
        service._burn_in_step(parameters, 2.0)


def test_replica_rows_cannot_invent_more_independent_units_than_requested(tmp_path):
    store, _, _ = archived_source(tmp_path)
    archived_source(tmp_path, store=store, task_id="source-2", seed=7)
    options = LJResearchRequest(
        kind="replicas", task_ids=["source", "source-2"], observables=["pressure"]
    )
    report = compute(tmp_path, service.prepare_lj_research(store, options))
    row = report.statistics["statistics"]["pressure"]
    row["replicas"] = 8
    row["replica_means"] *= 4
    row["confidence_interval"] = {
        "lower": 0.0,
        "upper": 1.0,
        "confidence_level": 0.95,
        "degrees_of_freedom": 7,
        "method": "Student-t",
        "coverage_guarantee": False,
    }
    report.availability = "available"
    with pytest.raises(service.ResearchError, match="RESEARCH_REPORT_INVALID"):
        service.validate_lj_research_report(service.research_report_bytes(report))


def test_cancellation_is_not_a_completed_unavailable_report(tmp_path):
    _, _, request = prepared(tmp_path)

    def cancel():
        raise SimulationCancelled("worker lost its lease")

    with pytest.raises(SimulationCancelled):
        service.compute_lj_research(
            request,
            analysis_id="a" * 32,
            analysis_attempt_id="owner",
            artifacts_dir=tmp_path,
            check_cancelled=cancel,
        )


@pytest.mark.parametrize("bad", [True, 1.0, "1"])
def test_report_schema_version_is_a_literal_integer(tmp_path, bad):
    from scientific_api.schemas.lj_research import LJResearchReport

    _, _, request = prepared(tmp_path)
    data = compute(tmp_path, request).model_dump(mode="json")
    data["schema_version"] = bad
    with pytest.raises(ValueError):
        LJResearchReport.model_validate(data)


@pytest.mark.asyncio
@pytest.mark.parametrize("committed", [False, True])
async def test_ambiguous_commit_retains_the_fenced_outcome_without_fabricating_failure(
    tmp_path, monkeypatch, committed
):
    source, archive, _ = prepared(tmp_path)
    backend = MemoryResearchStore(lease_seconds=3)
    manager = ResearchManager(
        backend,
        source,
        Settings(_env_file=None, artifacts_dir=tmp_path, redis_uri="", mlflow_tracking_uri=""),
        Telemetry(),
    )
    accepted = await manager.submit(LJResearchRequest(task_ids=["source"]))
    job = backend.claim()
    assert job is not None
    finish = backend.finish

    def lost_ack(delivery, outcome):
        if committed:
            assert finish(delivery, outcome)
        raise TimeoutError("The commit acknowledgement was lost")

    monkeypatch.setattr(backend, "finish", lost_ack)
    try:
        await manager._execute(job)
        record = backend.get(accepted["analysis_id"])
        assert record["status"] == ("completed" if committed else "running")
        if committed:
            raw, digest = manager.report(job.task_id)
            service.validate_lj_research_report(raw, expected_sha256=digest)
        else:
            monkeypatch.setattr(task_store, "_now", lambda: job.submitted_at + 100)
            retry = backend.claim()
            assert retry is not None and retry.owner != job.owner
        assert source.get_task_archive("source") == archive
    finally:
        await manager.stop()
        backend.close()
        source.close()
