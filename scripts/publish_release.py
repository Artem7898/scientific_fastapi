"""Release the current tested main commit using GitHub Actions' scoped GH_TOKEN.

Never move tags or replace published assets. Interrupted drafts resume only when
their commit and existing asset bytes agree with this build.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import subprocess
import tarfile
import tomllib
import zipfile
from dataclasses import dataclass
from email.parser import Parser
from pathlib import Path
from typing import Any

STABLE_VERSION = re.compile(r"(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)")


def run(*args: str, binary: bool = False) -> Any:
    return subprocess.run(args, check=True, capture_output=True, text=not binary).stdout


def api(repository: str, path: str, *, missing_ok: bool = False) -> Any:
    try:
        return json.loads(run("gh", "api", f"repos/{repository}/{path}".rstrip("/")))
    except subprocess.CalledProcessError as error:
        if missing_ok and "HTTP 404" in str(error.stderr):
            return None
        raise


def source_version(source: str) -> str:
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "__version__" for target in node.targets
        ):
            return str(ast.literal_eval(node.value))
    raise ValueError("Missing package __version__")


@dataclass(frozen=True)
class Release:
    root: Path
    repository: str
    sha: str
    version: str

    @property
    def tag(self) -> str:
        return f"v{self.version}"

    @property
    def notes(self) -> Path:
        return self.root / "docs" / "releases" / f"{self.tag}.md"


def metadata(root: Path, repository: str, sha: str) -> Release:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository):
        raise ValueError("GITHUB_REPOSITORY must identify owner/repository")
    if not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise ValueError("Release target must be a full commit SHA")
    project = tomllib.loads((root / "pyproject.toml").read_text())["project"]
    version = project["version"]
    if project["name"] != "scientific-api" or not STABLE_VERSION.fullmatch(version):
        raise ValueError("Expected scientific-api with a stable semantic version")
    package_version = source_version((root / "src/scientific_api/__init__.py").read_text())
    citation = re.search(
        r"^version:\s*[\"']?([^\s\"']+)",
        (root / "CITATION.cff").read_text(),
        re.M,
    )
    if citation is None or citation[1] != version or package_version != version:
        raise ValueError("Package, pyproject and citation versions must agree")
    release = Release(root, repository, sha, version)
    if not release.notes.is_file() or not release.notes.read_text().strip():
        raise ValueError(f"Missing release notes: {release.notes}")
    return release


def guard_main(release: Release) -> None:
    if run("git", "rev-parse", "HEAD").strip() != release.sha:
        raise ValueError("Checkout does not match the release target")
    info = api(release.repository, "")
    if info["default_branch"] != "main" or info["full_name"] != release.repository:
        raise ValueError("Release requires this repository's default main branch")
    if api(release.repository, "branches/main")["commit"]["sha"] != release.sha:
        raise ValueError("Release target is no longer the current main commit")
    runs = api(
        release.repository,
        f"actions/workflows/ci.yml/runs?head_sha={release.sha}&branch=main&per_page=100",
    )["workflow_runs"]
    matching = [
        item
        for item in runs
        if item["head_sha"] == release.sha
        and item["head_branch"] == "main"
        and item["event"] == "push"
    ]
    if not matching or max(matching, key=lambda item: item["id"])["conclusion"] != "success":
        raise ValueError("The latest main CI run for this exact commit must succeed")


def tag_sha(release: Release) -> str | None:
    ref = api(release.repository, f"git/ref/tags/{release.tag}", missing_ok=True)
    if ref is None:
        return None
    target = ref["object"]
    for _ in range(5):
        if target["type"] == "commit":
            return str(target["sha"])
        if target["type"] != "tag":
            break
        target = api(release.repository, f"git/tags/{target['sha']}")["object"]
    raise ValueError("Tag must resolve to a commit")


def release_state(release: Release) -> dict[str, Any] | None:
    pages = json.loads(
        run(
            "gh",
            "api",
            "--paginate",
            "--slurp",
            f"repos/{release.repository}/releases?per_page=100",
        )
    )
    releases = [item for page in pages for item in page]
    version_tuple = tuple(map(int, release.version.split(".")))
    for item in releases:
        previous = STABLE_VERSION.fullmatch(item["tag_name"].removeprefix("v"))
        if previous and not item["draft"] and not item["prerelease"]:
            if tuple(map(int, previous.groups())) > version_tuple:
                raise ValueError("Refusing to publish an older version than an existing release")
    existing = next((item for item in releases if item["tag_name"] == release.tag), None)
    target = tag_sha(release)
    if existing and not existing["draft"]:
        # Subsequent main commits without a version bump must not move this tag.
        return existing
    if target is not None and target != release.sha:
        raise ValueError("Existing tag points to another commit; refusing to move it")
    if existing and existing["target_commitish"] != release.sha:
        raise ValueError("Existing draft targets another commit")
    return existing


def artifacts(release: Release) -> list[Path]:
    dist = release.root / "dist"
    wheel = dist / f"scientific_api-{release.version}-py3-none-any.whl"
    sdist = dist / f"scientific_api-{release.version}.tar.gz"
    if set(dist.glob("*.whl")) | set(dist.glob("*.tar.gz")) != {wheel, sdist}:
        raise ValueError("Expected exactly the current version's wheel and source distribution")
    with zipfile.ZipFile(wheel) as archive:
        names = [name for name in archive.namelist() if name.endswith(".dist-info/METADATA")]
        if len(names) != 1:
            raise ValueError("Wheel must contain one package metadata record")
        wheel_metadata = Parser().parsestr(archive.read(names[0]).decode())
        if source_version(archive.read("scientific_api/__init__.py").decode()) != release.version:
            raise ValueError("Wheel package version disagrees with release")
    with tarfile.open(sdist, "r:gz") as archive:
        member = archive.extractfile(f"scientific_api-{release.version}/PKG-INFO")
        if member is None:
            raise ValueError("Source distribution has no package metadata")
        sdist_metadata = Parser().parsestr(member.read().decode())
    for item in (wheel_metadata, sdist_metadata):
        if item["Name"] != "scientific-api" or item["Version"] != release.version:
            raise ValueError("Distribution metadata disagrees with release")
    checksums = dist / "SHA256SUMS"
    checksums.write_text(
        "".join(
            f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n"
            for path in (wheel, sdist)
        )
    )
    return [wheel, sdist, checksums]


def verify_assets(release: Release, state: dict[str, Any], files: list[Path]) -> set[str]:
    expected = {path.name: hashlib.sha256(path.read_bytes()).digest() for path in files}
    seen: set[str] = set()
    for asset in state["assets"]:
        name = asset["name"]
        if name not in expected or name in seen:
            raise ValueError("Draft has unexpected or duplicate assets; refusing to publish")
        data = run(
            "gh",
            "api",
            f"repos/{release.repository}/releases/assets/{asset['id']}",
            "-H",
            "Accept: application/octet-stream",
            binary=True,
        )
        if hashlib.sha256(data).digest() != expected[name]:
            raise ValueError(f"Existing asset differs: {name}; refusing to overwrite")
        seen.add(name)
    return seen


def publish(release: Release, *, check_only: bool = False) -> None:
    guard_main(release)
    existing = release_state(release)
    if existing and not existing["draft"]:
        print(f"Version already published; preserved unchanged: {existing['html_url']}")
        return
    if check_only:
        print(f"Ready to build {release.tag} from {release.sha}")
        return
    files = artifacts(release)
    if existing is None:
        run(
            "gh",
            "release",
            "create",
            release.tag,
            "--repo",
            release.repository,
            "--target",
            release.sha,
            "--draft",
            "--title",
            f"Scientific API Gateway {release.tag}",
            "--notes-file",
            str(release.notes),
        )
        existing = release_state(release)
        if existing is None or not existing["draft"]:
            raise ValueError("Expected a draft before uploading assets")
    uploaded = verify_assets(release, existing, files)
    missing = [str(path) for path in files if path.name not in uploaded]
    if missing:
        run("gh", "release", "upload", release.tag, *missing, "--repo", release.repository)
    existing = api(release.repository, f"releases/{existing['id']}")
    if not existing["draft"] or verify_assets(release, existing, files) != {p.name for p in files}:
        raise ValueError("All draft assets must be verified before publication")
    guard_main(release)
    state = release_state(release)
    if state is None or not state["draft"] or state["id"] != existing["id"]:
        raise ValueError("Draft changed before publication")
    run(
        "gh",
        "release",
        "edit",
        release.tag,
        "--repo",
        release.repository,
        "--draft=false",
        "--latest",
    )
    if tag_sha(release) != release.sha:
        raise ValueError("Published tag does not resolve to the tested commit")
    print(f"Published https://github.com/{release.repository}/releases/tag/{release.tag}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sha", required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    try:
        publish(
            metadata(root, os.environ.get("GITHUB_REPOSITORY", ""), args.sha), check_only=args.check
        )
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        raise SystemExit(f"Release stopped: {error}") from error


if __name__ == "__main__":
    main()
