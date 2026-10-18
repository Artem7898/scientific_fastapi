from __future__ import annotations

import importlib.util
import io
import json
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/publish_release.py"
SPEC = importlib.util.spec_from_file_location("publish_release", SCRIPT)
assert SPEC and SPEC.loader
publisher = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = publisher
SPEC.loader.exec_module(publisher)

SHA = "a" * 40
OTHER_SHA = "b" * 40
REPOSITORY = "Artem7898/scientific_fastapi"


@pytest.fixture
def release(tmp_path: Path):
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "scientific-api"\nversion = "2.0.0"\n'
    )
    source = tmp_path / "src/scientific_api"
    source.mkdir(parents=True)
    (source / "__init__.py").write_text('__version__ = "2.0.0"\n')
    (tmp_path / "CITATION.cff").write_text("version: 2.0.0\n")
    notes = tmp_path / "docs/releases/v2.0.0.md"
    notes.parent.mkdir(parents=True)
    notes.write_text("# v2.0.0\nReproducible scientific workflows.\n")
    dist = tmp_path / "dist"
    dist.mkdir()
    package_metadata = b"Name: scientific-api\nVersion: 2.0.0\n"
    with zipfile.ZipFile(dist / "scientific_api-2.0.0-py3-none-any.whl", "w") as archive:
        archive.writestr("scientific_api/__init__.py", '__version__ = "2.0.0"\n')
        archive.writestr("scientific_api-2.0.0.dist-info/METADATA", package_metadata)
    with tarfile.open(dist / "scientific_api-2.0.0.tar.gz", "w:gz") as archive:
        entry = tarfile.TarInfo("scientific_api-2.0.0/PKG-INFO")
        entry.size = len(package_metadata)
        archive.addfile(entry, io.BytesIO(package_metadata))
    return publisher.metadata(tmp_path, REPOSITORY, SHA)


class GitHub:
    """Model REST metadata, immutable asset bytes and interrupted draft uploads."""

    def __init__(self, release):
        self.release = release
        self.head = SHA
        self.ci_sha = SHA
        self.ci_event = "push"
        self.ci_conclusion = "success"
        self.tag = None
        self.state = None
        self.other_releases = []
        self.assets = {}
        self.mutations = []
        self.operations = []
        self.fail_upload = False
        self.hide_draft_from_index = False

    def draft(self):
        self.state = {
            "id": 1,
            "tag_name": "v2.0.0",
            "draft": True,
            "prerelease": False,
            "target_commitish": SHA,
            "assets": [],
            "html_url": "https://github.com/example/releases/tag/v2.0.0",
        }

    def run(self, *args, binary=False, input_data=None):
        if args[:3] == ("git", "rev-parse", "HEAD"):
            return SHA + "\n"
        if args[:2] == ("gh", "api"):
            if "--method" in args:
                self.mutations.append(args)
                method = args[args.index("--method") + 1]
                if args[2].startswith("https://uploads.github.com/"):
                    self.operations.append("upload")
                    assert method == "POST" and "Content-Type: application/octet-stream" in args
                    assert "/releases/1/assets?name=" in args[2]
                    if self.fail_upload:
                        raise subprocess.CalledProcessError(1, args, stderr="Upload interrupted")
                    asset = Path(args[args.index("--input") + 1])
                    asset_id = len(self.assets) + 1
                    self.assets[asset_id] = asset.read_bytes()
                    result = {"id": asset_id, "name": asset.name}
                    self.state["assets"].append(result)
                    return json.dumps(result)
                payload = json.loads(input_data)
                if method == "POST" and args[2] == f"repos/{REPOSITORY}/releases":
                    self.operations.append("create")
                    assert payload["target_commitish"] == SHA and payload["draft"] is True
                    assert payload["tag_name"] == "v2.0.0"
                    self.draft()
                elif method == "PATCH" and args[2] == f"repos/{REPOSITORY}/releases/1":
                    self.operations.append("edit")
                    assert payload == {"draft": False, "make_latest": "true"}
                    self.state["draft"] = False
                    self.tag = SHA
                else:
                    raise AssertionError(f"Unexpected API write: {args}")
                return json.dumps(self.state)
            if "--paginate" in args:
                assert "--slurp" not in args
                visible = self.state and not self.hide_draft_from_index
                return json.dumps(self.other_releases + ([self.state] if visible else []))
            path = args[2].removeprefix(f"repos/{REPOSITORY}").removeprefix("/")
            if path == "":
                result = {"default_branch": "main", "full_name": REPOSITORY}
            elif path == "branches/main":
                result = {"commit": {"sha": self.head}}
            elif path.startswith("actions/workflows/ci.yml/runs?"):
                result = {
                    "workflow_runs": [
                        {
                            "id": 10,
                            "head_sha": self.ci_sha,
                            "head_branch": "main",
                            "event": self.ci_event,
                            "conclusion": self.ci_conclusion,
                        }
                    ]
                }
            elif path == "git/ref/tags/v2.0.0":
                if self.tag is None:
                    raise subprocess.CalledProcessError(1, args, stderr="gh: HTTP 404")
                result = {"object": {"type": "commit", "sha": self.tag}}
            elif path == "releases/1":
                result = self.state
            elif path.startswith("releases/assets/") and binary:
                return self.assets[int(path.rsplit("/", 1)[1])]
            else:
                raise AssertionError(f"Unexpected API read: {args}")
            return json.dumps(result)
        raise AssertionError(f"Unexpected command: {args}")


@pytest.fixture
def github(release, monkeypatch):
    fake = GitHub(release)
    monkeypatch.setattr(publisher, "run", fake.run)
    return fake


def test_preflight_is_read_only(release, github):
    publisher.publish(release, check_only=True)
    assert github.mutations == []


@pytest.mark.parametrize(
    "field,value",
    [
        ("head", OTHER_SHA),
        ("ci_sha", OTHER_SHA),
        ("ci_conclusion", "failure"),
        ("ci_conclusion", None),
        ("ci_event", "pull_request"),
        ("tag", OTHER_SHA),
    ],
)
def test_rejects_untested_or_mismatched_commit(release, github, field, value):
    setattr(github, field, value)
    with pytest.raises(ValueError):
        publisher.publish(release)
    assert github.mutations == []


def test_versions_must_agree(release):
    (release.root / "CITATION.cff").write_text("version: 1.0.0\n")
    with pytest.raises(ValueError, match="versions must agree"):
        publisher.metadata(release.root, REPOSITORY, SHA)


def test_rejects_backwards_release(release, github):
    github.other_releases = [{"tag_name": "v3.0.0", "draft": False, "prerelease": False}]
    with pytest.raises(ValueError, match="older version"):
        publisher.publish(release)
    assert github.mutations == []


def test_paginated_release_arrays_check_versions_on_later_pages(release, github, monkeypatch):
    original_run = github.run

    def run(*args, **kwargs):
        if "--paginate" in args:
            assert "--slurp" not in args
            first = [{"tag_name": "v1.0.0", "draft": False, "prerelease": False}]
            second = [{"tag_name": "v3.0.0", "draft": False, "prerelease": False}]
            return "\n " + json.dumps(first) + "\n\n" + json.dumps(second) + "\n "
        return original_run(*args, **kwargs)

    monkeypatch.setattr(publisher, "run", run)
    with pytest.raises(ValueError, match="older version"):
        publisher.publish(release)
    assert github.mutations == []


def test_paginated_release_arrays_find_existing_release_on_later_page(release, github, monkeypatch):
    github.draft()
    github.state["draft"] = False
    github.tag = OTHER_SHA
    original_run = github.run

    def run(*args, **kwargs):
        if "--paginate" in args:
            return "[]\n" + json.dumps([github.state])
        return original_run(*args, **kwargs)

    monkeypatch.setattr(publisher, "run", run)
    publisher.publish(release)
    assert github.mutations == []


def test_paginated_release_response_rejects_non_list_page(release, github, monkeypatch):
    original_run = github.run

    def run(*args, **kwargs):
        if "--paginate" in args:
            return '[]\n{"message": "unexpected response"}'
        return original_run(*args, **kwargs)

    monkeypatch.setattr(publisher, "run", run)
    with pytest.raises(ValueError, match="release list"):
        publisher.publish(release)
    assert github.mutations == []


def test_publishes_only_after_verifying_all_assets(release, github):
    publisher.publish(release)
    assert github.operations == ["create", "upload", "upload", "upload", "edit"]
    assert github.tag == SHA and not github.state["draft"]
    assert len(github.assets) == 3
    checksum = next(item for item in github.state["assets"] if item["name"] == "SHA256SUMS")
    assert github.assets[checksum["id"]].count(b"\n") == 2


def test_creation_response_survives_draft_missing_from_release_index(release, github):
    github.hide_draft_from_index = True
    publisher.publish(release)
    assert github.operations == ["create", "upload", "upload", "upload", "edit"]
    assert github.tag == SHA and not github.state["draft"]


def test_creation_response_must_match_the_requested_commit(release, github, monkeypatch):
    original_run = github.run

    def run(*args, **kwargs):
        result = original_run(*args, **kwargs)
        if args[2:3] == (f"repos/{REPOSITORY}/releases",) and "--method" in args:
            response = json.loads(result)
            response["target_commitish"] = OTHER_SHA
            return json.dumps(response)
        return result

    monkeypatch.setattr(publisher, "run", run)
    with pytest.raises(ValueError, match="requested draft and commit"):
        publisher.publish(release)
    assert github.operations == ["create"]
    assert github.tag is None and github.state["draft"]


def test_failed_upload_leaves_draft_then_resumes(release, github):
    github.fail_upload = True
    with pytest.raises(subprocess.CalledProcessError):
        publisher.publish(release)
    assert github.state["draft"] and github.tag is None
    github.fail_upload = False
    publisher.publish(release)
    assert not github.state["draft"]
    assert github.operations.count("create") == 1


def test_resume_verifies_existing_assets_and_refuses_to_overwrite(release, github):
    github.draft()
    github.state["assets"] = [{"id": 1, "name": "scientific_api-2.0.0.tar.gz"}]
    github.assets[1] = b"different artifact"
    with pytest.raises(ValueError, match="refusing to overwrite"):
        publisher.publish(release)
    assert github.mutations == []


def test_partial_upload_resumes_without_replacing_existing_asset(release, github):
    files = publisher.artifacts(release)
    github.draft()
    github.state["assets"] = [{"id": 1, "name": files[0].name}]
    github.assets[1] = files[0].read_bytes()
    publisher.publish(release)
    uploads = [command for command in github.mutations if command[2].startswith("https://")]
    assert all(str(files[0]) not in command for command in uploads)
    assert len(github.assets) == 3 and not github.state["draft"]


def test_main_moving_during_upload_leaves_unpublished_draft(release, github, monkeypatch):
    original_run = github.run

    def run(*args, **kwargs):
        result = original_run(*args, **kwargs)
        if args[:2] == ("gh", "api") and args[2].startswith("https://uploads.github.com/"):
            github.head = OTHER_SHA
        return result

    monkeypatch.setattr(publisher, "run", run)
    with pytest.raises(ValueError, match="no longer the current main"):
        publisher.publish(release)
    assert github.state["draft"] and github.tag is None
    assert "edit" not in github.operations


def test_rejects_wheel_with_incorrect_embedded_version(release, github):
    wheel = release.root / "dist/scientific_api-2.0.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("scientific_api/__init__.py", '__version__ = "1.0.0"\n')
        archive.writestr(
            "scientific_api-2.0.0.dist-info/METADATA", "Name: scientific-api\nVersion: 2.0.0\n"
        )
    with pytest.raises(ValueError, match="Wheel package version"):
        publisher.publish(release)
    assert github.mutations == []


def test_unchanged_version_on_new_main_preserves_published_release(release, github):
    github.draft()
    github.state["draft"] = False
    github.tag = OTHER_SHA
    publisher.publish(release)
    assert github.tag == OTHER_SHA and github.mutations == []


def test_draft_for_other_commit_is_not_reused(release, github):
    github.draft()
    github.state["target_commitish"] = OTHER_SHA
    with pytest.raises(ValueError, match="another commit"):
        publisher.publish(release)
    assert github.mutations == []


@pytest.mark.parametrize("binary", [False, True])
def test_cli_reports_github_error_reason_without_tokens(release, monkeypatch, binary):
    monkeypatch.setenv("GH_TOKEN", "gh-secret-do-not-log")
    monkeypatch.setenv("GITHUB_TOKEN", "github-secret-do-not-log")
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--sha", SHA])
    monkeypatch.setattr(publisher, "metadata", lambda *args: release)
    detail = (
        "gh: Resource not accessible by integration (HTTP 403); "
        "GH_TOKEN=gh-secret-do-not-log; GITHUB_TOKEN=github-secret-do-not-log"
    )

    def fail(*args, **kwargs):
        raise subprocess.CalledProcessError(
            1,
            ("gh", "release", "create", "v2.0.0"),
            stderr=detail.encode() if binary else detail,
        )

    monkeypatch.setattr(publisher, "publish", fail)
    with pytest.raises(SystemExit) as caught:
        publisher.main()
    message = str(caught.value)
    assert "Resource not accessible by integration (HTTP 403)" in message
    assert message.count("[REDACTED]") == 2
    assert "gh-secret-do-not-log" not in message
    assert "github-secret-do-not-log" not in message


def test_other_command_stderr_is_not_reported():
    error = subprocess.CalledProcessError(1, ("git", "rev-parse", "HEAD"), stderr="private data")
    assert "private data" not in publisher.failure_reason(error)
