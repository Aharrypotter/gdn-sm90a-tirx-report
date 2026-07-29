#!/usr/bin/env python3
# Copyright 2026 Aharrypotter
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Audit a draft or public GitHub release against its immutable tagged assets."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlencode

import build_release_assets as builder
import verify_release_assets as verifier

AUDIT_SCHEMA = "gdn-sm90a.public-release-audit.v1"
DEFAULT_API_BASE = "https://api.github.com"
DEFAULT_WEB_BASE = "https://github.com"
API_VERSION = "2022-11-28"


class PublicReleaseAuditError(ValueError):
    """The GitHub release or one of its public source links is inconsistent."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PublicReleaseAuditError(message)


def _json_object(value: bytes, context: str) -> dict[str, Any]:
    try:
        result = json.loads(value)
    except json.JSONDecodeError as error:
        raise PublicReleaseAuditError(f"{context}: invalid JSON: {error}") from error
    if not isinstance(result, dict):
        raise PublicReleaseAuditError(f"{context}: expected a JSON object")
    return result


class GitHubClient:
    """Small retrying GitHub REST/HTTP client that never logs credentials."""

    def __init__(
        self,
        *,
        api_base_url: str = DEFAULT_API_BASE,
        token: str | None = None,
        timeout_seconds: float = 30.0,
        retries: int = 3,
    ) -> None:
        self.api_base_url = api_base_url.rstrip("/")
        self.token = token
        self.timeout_seconds = timeout_seconds
        self.retries = retries

    def _request(
        self,
        url: str,
        *,
        authenticated: bool,
        accept: str,
    ) -> bytes:
        headers = {
            "Accept": accept,
            "User-Agent": "gdn-sm90a-public-release-auditor/1",
            "X-GitHub-Api-Version": API_VERSION,
        }
        if authenticated:
            if not self.token:
                raise PublicReleaseAuditError(
                    "authenticated draft audit requires GH_TOKEN or GITHUB_TOKEN"
                )
            headers["Authorization"] = f"Bearer {self.token}"
        request = urllib.request.Request(url, headers=headers, method="GET")
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                with urllib.request.urlopen(
                    request,
                    timeout=self.timeout_seconds,
                ) as response:
                    status = getattr(response, "status", 200)
                    if status < 200 or status >= 300:
                        raise PublicReleaseAuditError(f"GET {url}: HTTP {status}")
                    return response.read()
            except urllib.error.HTTPError as error:
                last_error = error
                if error.code not in {429, 500, 502, 503, 504}:
                    break
            except (TimeoutError, urllib.error.URLError) as error:
                last_error = error
            if attempt < self.retries:
                time.sleep(0.25 * (2**attempt))
        if isinstance(last_error, urllib.error.HTTPError):
            raise PublicReleaseAuditError(f"GET {url}: HTTP {last_error.code}") from last_error
        raise PublicReleaseAuditError(f"GET {url} failed: {last_error}") from last_error

    def api_bytes(
        self,
        path_or_url: str,
        *,
        authenticated: bool,
        accept: str = "application/vnd.github+json",
    ) -> bytes:
        url = (
            path_or_url
            if path_or_url.startswith(("http://", "https://"))
            else f"{self.api_base_url}/{path_or_url.lstrip('/')}"
        )
        return self._request(url, authenticated=authenticated, accept=accept)

    def api_json(self, path: str, *, authenticated: bool) -> dict[str, Any]:
        return _json_object(
            self.api_bytes(path, authenticated=authenticated),
            f"GitHub API {path}",
        )

    def api_list(self, path: str, *, authenticated: bool) -> list[Any]:
        raw = self.api_bytes(path, authenticated=authenticated)
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as error:
            raise PublicReleaseAuditError(f"GitHub API {path}: invalid JSON: {error}") from error
        if not isinstance(value, list):
            raise PublicReleaseAuditError(f"GitHub API {path}: expected a JSON array")
        return value

    def web_bytes(self, url: str, *, authenticated: bool = False) -> bytes:
        return self._request(
            url,
            authenticated=authenticated,
            accept="text/html,application/octet-stream;q=0.9,*/*;q=0.8",
        )


def _release_by_tag(
    client: GitHubClient,
    *,
    repository: str,
    tag: str,
    authenticated: bool,
) -> dict[str, Any]:
    encoded_tag = quote(tag, safe="")
    path = f"repos/{repository}/releases/tags/{encoded_tag}"
    try:
        return client.api_json(path, authenticated=authenticated)
    except PublicReleaseAuditError:
        if not authenticated:
            raise
    releases_raw = client.api_bytes(
        f"repos/{repository}/releases?per_page=100",
        authenticated=True,
    )
    try:
        releases = json.loads(releases_raw)
    except json.JSONDecodeError as error:
        raise PublicReleaseAuditError("GitHub draft release list returned invalid JSON") from error
    if not isinstance(releases, list):
        raise PublicReleaseAuditError("GitHub draft release list is not an array")
    matches = [
        release
        for release in releases
        if isinstance(release, dict) and release.get("tag_name") == tag
    ]
    _require(len(matches) == 1, f"expected exactly one draft release for {tag}")
    return matches[0]


def _audit_tag(
    client: GitHubClient,
    *,
    repository: str,
    tag: str,
    expected_tag_object: str,
    expected_commit: str,
    expected_tree: str,
    authenticated: bool,
) -> dict[str, Any]:
    encoded_tag = quote(tag, safe="")
    reference = client.api_json(
        f"repos/{repository}/git/ref/tags/{encoded_tag}",
        authenticated=authenticated,
    )
    object_identity = reference.get("object", {})
    _require(object_identity.get("type") == "tag", f"{repository}:{tag}: not annotated")
    _require(
        object_identity.get("sha") == expected_tag_object,
        f"{repository}:{tag}: tag object mismatch",
    )
    tag_object = client.api_json(
        f"repos/{repository}/git/tags/{expected_tag_object}",
        authenticated=authenticated,
    )
    peeled = tag_object.get("object", {})
    _require(peeled.get("type") == "commit", f"{repository}:{tag}: does not peel to commit")
    _require(peeled.get("sha") == expected_commit, f"{repository}:{tag}: commit mismatch")
    commit = client.api_json(
        f"repos/{repository}/git/commits/{expected_commit}",
        authenticated=authenticated,
    )
    _require(
        commit.get("tree", {}).get("sha") == expected_tree,
        f"{repository}:{tag}: tree mismatch",
    )
    return {
        "repository": repository,
        "tag": tag,
        "tag_object": expected_tag_object,
        "commit": expected_commit,
        "tree": expected_tree,
        "status": "PASS",
    }


def _audit_commit(
    client: GitHubClient,
    *,
    repository: str,
    commit: str,
    expected_tree: str,
    authenticated: bool,
) -> dict[str, Any]:
    value = client.api_json(
        f"repos/{repository}/git/commits/{commit}",
        authenticated=authenticated,
    )
    _require(value.get("tree", {}).get("sha") == expected_tree, f"{repository}: tree mismatch")
    return {
        "repository": repository,
        "commit": commit,
        "tree": expected_tree,
        "status": "PASS",
    }


def _canonical_body(value: str) -> bytes:
    return (value.rstrip("\n") + "\n").encode()


def _download_release_assets(
    client: GitHubClient,
    *,
    release: dict[str, Any],
    local_assets: Path,
    mode: str,
    expected_names: set[str],
) -> list[dict[str, Any]]:
    remote_assets = release.get("assets")
    _require(isinstance(remote_assets, list), "GitHub release assets are missing")
    by_name = {
        asset.get("name"): asset
        for asset in remote_assets
        if isinstance(asset, dict) and isinstance(asset.get("name"), str)
    }
    _require(set(by_name) == expected_names, "GitHub release asset membership mismatch")
    results = []
    with tempfile.TemporaryDirectory(prefix="gdn-release-download-") as temporary:
        download_root = Path(temporary)
        for name in sorted(expected_names):
            asset = by_name[name]
            local_path = local_assets / name
            _require(asset.get("size") == local_path.stat().st_size, f"{name}: remote size drift")
            if mode == "public":
                url = asset.get("browser_download_url")
                _require(isinstance(url, str), f"{name}: missing browser download URL")
                payload = client.web_bytes(url, authenticated=False)
                anonymous = True
            else:
                url = asset.get("url")
                _require(isinstance(url, str), f"{name}: missing asset API URL")
                payload = client.api_bytes(
                    url,
                    authenticated=True,
                    accept="application/octet-stream",
                )
                anonymous = False
            downloaded = download_root / name
            downloaded.write_bytes(payload)
            local_sha256 = builder.sha256_file(local_path)
            _require(payload == local_path.read_bytes(), f"{name}: downloaded bytes drift")
            digest = asset.get("digest")
            if digest is not None:
                _require(
                    digest == f"sha256:{local_sha256}",
                    f"{name}: GitHub asset digest mismatch",
                )
            results.append(
                {
                    "name": name,
                    "size_bytes": len(payload),
                    "sha256": local_sha256,
                    "anonymous_download": anonymous,
                    "status": "PASS",
                }
            )
    return results


def _audit_superseded_release_lock(
    client: GitHubClient,
    *,
    record: dict[str, Any],
    expected_body: str,
) -> dict[str, Any]:
    """Anonymously prove that the superseded public release remains immutable."""

    _require(
        record.get("schema") == "gdn-sm90a.report-release-supersession.v1",
        "superseded release record schema mismatch",
    )
    _require(
        record.get("status") == "SUPERSEDED_FOR_PUBLICATION_GUIDANCE",
        "superseded release record status mismatch",
    )
    superseded = record.get("superseded_release")
    _require(isinstance(superseded, dict), "superseded release identity is missing")
    repository = superseded.get("repository")
    tag = superseded.get("tag")
    tag_object = superseded.get("tag_object")
    commit = superseded.get("commit")
    tree = superseded.get("tree")
    for name, value in {
        "repository": repository,
        "tag": tag,
        "tag_object": tag_object,
        "commit": commit,
        "tree": tree,
    }.items():
        _require(isinstance(value, str) and value, f"superseded release {name} is missing")
    _require(
        superseded.get("retention") == "PUBLIC_UNMODIFIED_AUDIT_RECORD",
        "superseded release retention policy mismatch",
    )
    expected_mutation_policy = {
        "edit_r1_release_body": False,
        "move_or_delete_r1_tag": False,
        "replace_or_delete_r1_assets": False,
    }
    _require(
        record.get("mutation_policy") == expected_mutation_policy,
        "superseded release mutation policy mismatch",
    )
    expected_validity = {
        "evidence_integrity": "VALID_UNCHANGED",
        "performance_facts": "VALID_UNCHANGED",
        "publication_guidance": "SUPERSEDED",
        "runtime_source": "VALID_UNCHANGED",
        "source_provenance": "VALID_UNCHANGED",
    }
    _require(
        record.get("validity") == expected_validity,
        "superseded release validity record mismatch",
    )

    tag_result = _audit_tag(
        client,
        repository=repository,
        tag=tag,
        expected_tag_object=tag_object,
        expected_commit=commit,
        expected_tree=tree,
        authenticated=False,
    )
    release = _release_by_tag(
        client,
        repository=repository,
        tag=tag,
        authenticated=False,
    )
    _require(release.get("tag_name") == tag, "superseded release tag mismatch")
    _require(
        release.get("id") == superseded.get("release_id"),
        "superseded release ID mismatch",
    )
    _require(release.get("draft") is False, "superseded release became draft")
    _require(release.get("prerelease") is False, "superseded release became prerelease")
    _require(
        release.get("published_at") == superseded.get("published_at_utc"),
        "superseded release publication time mismatch",
    )
    release_url = superseded.get("release_url")
    _require(isinstance(release_url, str) and release_url, "superseded release URL is missing")
    _require(release.get("html_url") == release_url, "superseded release URL mismatch")
    body = release.get("body")
    _require(isinstance(body, str), "superseded release body is missing")
    _require(
        _canonical_body(body) == _canonical_body(expected_body),
        "superseded release body drift",
    )

    asset_lock = record.get("asset_lock")
    _require(isinstance(asset_lock, list) and asset_lock, "superseded asset lock is missing")
    expected_assets = {}
    for entry in asset_lock:
        _require(isinstance(entry, dict), "superseded asset lock entry is invalid")
        name = entry.get("name")
        size_bytes = entry.get("size_bytes")
        sha256 = entry.get("sha256")
        _require(isinstance(name, str) and name, "superseded asset name is missing")
        _require(
            isinstance(size_bytes, int) and not isinstance(size_bytes, bool) and size_bytes >= 0,
            f"{name}: invalid locked size",
        )
        _require(
            isinstance(sha256, str)
            and len(sha256) == 64
            and all(character in "0123456789abcdef" for character in sha256),
            f"{name}: invalid locked SHA256",
        )
        _require(name not in expected_assets, f"{name}: duplicate superseded asset lock")
        expected_assets[name] = entry

    remote_assets = release.get("assets")
    _require(isinstance(remote_assets, list), "superseded release assets are missing")
    _require(
        len(remote_assets) == len(expected_assets),
        "superseded release asset count mismatch",
    )
    by_name = {
        asset.get("name"): asset
        for asset in remote_assets
        if isinstance(asset, dict) and isinstance(asset.get("name"), str)
    }
    _require(
        set(by_name) == set(expected_assets),
        "superseded release asset membership mismatch",
    )
    downloaded_assets = []
    for name in sorted(expected_assets):
        expected = expected_assets[name]
        asset = by_name[name]
        _require(asset.get("size") == expected["size_bytes"], f"{name}: locked size drift")
        digest = asset.get("digest")
        if digest is not None:
            _require(
                digest == f"sha256:{expected['sha256']}",
                f"{name}: GitHub asset digest mismatch",
            )
        download_url = asset.get("browser_download_url")
        _require(isinstance(download_url, str), f"{name}: missing browser download URL")
        payload = client.web_bytes(download_url, authenticated=False)
        _require(len(payload) == expected["size_bytes"], f"{name}: downloaded size drift")
        _require(
            builder.sha256_bytes(payload) == expected["sha256"],
            f"{name}: downloaded SHA256 drift",
        )
        downloaded_assets.append(
            {
                "name": name,
                "size_bytes": len(payload),
                "sha256": expected["sha256"],
                "anonymous_download": True,
                "status": "PASS",
            }
        )
    client.web_bytes(release_url, authenticated=False)

    return {
        "status": "PASS",
        "repository": repository,
        "tag": tag,
        "tag_object": tag_result["tag_object"],
        "commit": tag_result["commit"],
        "tree": tag_result["tree"],
        "release_id": release["id"],
        "release_url": release_url,
        "published_at_utc": release["published_at"],
        "draft": release["draft"],
        "prerelease": release["prerelease"],
        "body": "EXACT_MATCH",
        "downloaded_assets": downloaded_assets,
        "mutation_policy": expected_mutation_policy,
        "validity": expected_validity,
    }


def _audit_sources(
    client: GitHubClient,
    *,
    coordinates: dict[str, Any],
    report_repository_url: str,
    authenticated: bool,
    release_body: str,
) -> list[dict[str, Any]]:
    results = []
    for source_id in sorted(coordinates):
        source = coordinates[source_id]
        if source["kind"] == "annotated_tag_release":
            result = _audit_tag(
                client,
                repository=source["repository"],
                tag=source["tag"],
                expected_tag_object=source["tag_object"],
                expected_commit=source["commit"],
                expected_tree=source["tree"],
                authenticated=authenticated,
            )
            source_release = _release_by_tag(
                client,
                repository=source["repository"],
                tag=source["tag"],
                authenticated=authenticated,
            )
            _require(source_release.get("draft") is False, f"{source_id}: source release is draft")
            _require(
                source_release.get("prerelease") is False,
                f"{source_id}: source release is prerelease",
            )
            source_body = source_release.get("body")
            _require(isinstance(source_body, str), f"{source_id}: source release body missing")
            _require(
                report_repository_url in source_body,
                f"{source_id}: source release does not cross-link the report repository",
            )
            _require(
                source["release_url"] in release_body,
                f"{source_id}: report release does not cross-link the source release",
            )
            client.web_bytes(source["release_url"], authenticated=False)
            result["release_url"] = source["release_url"]
        else:
            result = _audit_commit(
                client,
                repository=source["repository"],
                commit=source["commit"],
                expected_tree=source["tree"],
                authenticated=authenticated,
            )
            _require(
                source["url"] in release_body,
                f"{source_id}: report release does not cross-link the source commit",
            )
        client.web_bytes(source["url"], authenticated=False)
        result["source_url"] = source["url"]
        results.append(result)
    return results


def _repository_from_url(url: str) -> str:
    prefix = "https://github.com/"
    _require(url.startswith(prefix), f"not a GitHub repository URL: {url}")
    repository = url.removeprefix(prefix).strip("/")
    _require(repository.count("/") == 1, f"invalid GitHub repository URL: {url}")
    return repository


def _audit_upstream_prs(
    client: GitHubClient,
    *,
    coordinates: dict[str, Any],
    authenticated: bool,
    keyword_search: bool,
) -> list[dict[str, Any]]:
    """Query live upstream state; never trust the source manifest's historical booleans."""

    results = []
    for source_id in sorted(coordinates):
        source = coordinates[source_id]
        if source["kind"] != "annotated_tag_release":
            continue
        upstream = _repository_from_url(source["upstream_repository"])
        fork_owner = source["repository"].split("/", 1)[0]
        branch = source["release_branch"]
        query = urlencode(
            {
                "state": "all",
                "head": f"{fork_owner}:{branch}",
                "per_page": 100,
            }
        )
        pulls = client.api_list(
            f"repos/{upstream}/pulls?{query}",
            authenticated=authenticated,
        )
        _require(
            not pulls,
            f"{source_id}: upstream has PRs from {fork_owner}:{branch}",
        )
        keyword_checks = []
        if keyword_search and source_id in {"tvm", "tirx_kernels"}:
            for keyword in ("gdn", "sm90"):
                search_query = f"repo:{upstream} is:pr author:{fork_owner} {keyword}"
                search = client.api_json(
                    f"search/issues?{urlencode({'q': search_query, 'per_page': 100})}",
                    authenticated=authenticated,
                )
                count = search.get("total_count")
                _require(
                    isinstance(count, int) and not isinstance(count, bool),
                    f"{source_id}/{keyword}: invalid GitHub search count",
                )
                _require(
                    count == 0,
                    f"{source_id}/{keyword}: author PR search returned {count} match(es)",
                )
                keyword_checks.append({"keyword": keyword, "match_count": count})
        results.append(
            {
                "source_id": source_id,
                "upstream_repository": upstream,
                "head": f"{fork_owner}:{branch}",
                "head_pr_count": len(pulls),
                "keyword_checks": keyword_checks,
                "status": "PASS",
            }
        )
    return results


def audit_public_release(
    *,
    root: Path,
    tag: str,
    contract_path: str,
    assets: Path,
    mode: str,
    client: GitHubClient,
    rebuild_count: int = 2,
    run_evidence_verifiers: bool = True,
) -> dict[str, Any]:
    """Verify local assets, GitHub identities, cross-links, and downloaded bytes."""

    _require(mode in {"draft", "public"}, "mode must be draft or public")
    root = builder.repository_root(root)
    assets = assets.expanduser().resolve()
    local_verification = verifier.verify_release_assets(
        root=root,
        tag=tag,
        contract_path=contract_path,
        assets=assets,
        require_contract_tag=True,
        rebuild_count=rebuild_count,
        run_evidence_verifiers=run_evidence_verifiers,
    )
    identity = builder.load_tag_identity(root, tag)
    blobs = builder.load_tree(root, identity)
    contract = builder.load_contract(blobs, contract_path)
    manifest_path = assets / contract["release_manifest_filename"]
    manifest = _json_object(manifest_path.read_bytes(), manifest_path.name)
    repository = contract["release"]["repository"]
    authenticated = mode == "draft"

    report_tag = _audit_tag(
        client,
        repository=repository,
        tag=tag,
        expected_tag_object=identity.tag_object,
        expected_commit=identity.commit,
        expected_tree=identity.tree,
        authenticated=authenticated,
    )
    release = _release_by_tag(
        client,
        repository=repository,
        tag=tag,
        authenticated=authenticated,
    )
    _require(release.get("tag_name") == tag, "GitHub release tag mismatch")
    _require(
        release.get("draft") is (mode == "draft"),
        f"GitHub release draft state differs from requested {mode} audit",
    )
    _require(release.get("prerelease") is False, "GitHub release is marked prerelease")
    _require(release.get("name") == contract["release"]["title"], "release title mismatch")
    body = release.get("body")
    _require(isinstance(body, str), "GitHub release body is missing")
    notes = builder.blob_bytes(blobs, contract["release"]["notes_path"])
    _require(_canonical_body(body) == _canonical_body(notes.decode()), "release body drift")

    expected_asset_names = {
        *{contract["packages"][package_key]["filename"] for package_key in builder.PACKAGE_KEYS},
        contract["release_manifest_filename"],
        contract["checksum_filename"],
    }
    downloaded_assets = _download_release_assets(
        client,
        release=release,
        local_assets=assets,
        mode=mode,
        expected_names=expected_asset_names,
    )
    report_repository_url = f"{DEFAULT_WEB_BASE}/{repository}"
    sources = _audit_sources(
        client,
        coordinates=manifest["source_lock"]["coordinates"],
        report_repository_url=report_repository_url,
        authenticated=authenticated,
        release_body=body,
    )
    upstream_pr_audit = _audit_upstream_prs(
        client,
        coordinates=manifest["source_lock"]["coordinates"],
        authenticated=authenticated,
        keyword_search=mode == "public",
    )
    anonymous_release_page = False
    latest_release = None
    superseded_release_lock = None
    if mode == "public":
        client.web_bytes(manifest["release"]["release_url"], authenticated=False)
        anonymous_release_page = True
        latest = client.api_json(
            f"repos/{repository}/releases/latest",
            authenticated=False,
        )
        _require(latest.get("tag_name") == tag, "public release is not GitHub latest")
        latest_release = {
            "tag": latest["tag_name"],
            "release_id": latest.get("id"),
            "status": "PASS",
        }
        supersession_path = "releases/gdn-sm90a-r1-supersession.json"
        if tag == "gdn-sm90a-r2":
            _require(
                supersession_path in blobs,
                "r2 public audit requires the r1 supersession record",
            )
        if supersession_path in blobs:
            supersession_record = _json_object(
                builder.blob_bytes(blobs, supersession_path),
                supersession_path,
            )
            superseding = supersession_record.get("superseding_release")
            _require(isinstance(superseding, dict), "superseding release identity is missing")
            _require(
                superseding.get("repository") == repository,
                "superseding release repository mismatch",
            )
            _require(superseding.get("tag") == tag, "superseding release tag mismatch")
            superseded_release_lock = _audit_superseded_release_lock(
                client,
                record=supersession_record,
                expected_body=builder.blob_bytes(
                    blobs,
                    "releases/gdn-sm90a-r1.md",
                ).decode(),
            )

    return {
        "schema": AUDIT_SCHEMA,
        "status": "PASS",
        "mode": mode,
        "repository": repository,
        "release_url": release.get("html_url", manifest["release"]["release_url"]),
        "release_id": release.get("id"),
        "tag": tag,
        "tag_object": identity.tag_object,
        "commit": identity.commit,
        "tree": identity.tree,
        "draft": release["draft"],
        "prerelease": release["prerelease"],
        "anonymous_release_page": anonymous_release_page,
        "latest_release": latest_release,
        "superseded_release_lock": superseded_release_lock,
        "report_tag": report_tag,
        "downloaded_assets": downloaded_assets,
        "source_links": sources,
        "upstream_pr_audit": upstream_pr_audit,
        "local_asset_verification": {
            "status": local_verification["status"],
            "byte_identical_rebuilds": local_verification["byte_identical_rebuilds"],
            "rebuild_count": local_verification["rebuild_count"],
            "fresh_evidence_class": local_verification["fresh_evidence_class"],
            "fresh_decision_status": local_verification["fresh_decision_status"],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit a draft or public GitHub release and all immutable source links."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Report repository root.",
    )
    parser.add_argument(
        "--tag",
        help="Public annotated tag. Defaults to release.expected_public_tag.",
    )
    parser.add_argument(
        "--contract",
        default="contracts/release-assets.json",
        help="Repository-relative release contract path.",
    )
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--mode", choices=("draft", "public"), required=True)
    parser.add_argument("--api-base-url", default=DEFAULT_API_BASE)
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--rebuild-count", type=int, default=2)
    args = parser.parse_args()
    root = builder.repository_root(args.root)
    tag = args.tag
    if tag is None:
        working_contract = _json_object(
            (root / args.contract).read_bytes(),
            args.contract,
        )
        tag = working_contract.get("release", {}).get("expected_public_tag")
        if not isinstance(tag, str):
            raise PublicReleaseAuditError("working-tree contract has no public tag")
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    client = GitHubClient(
        api_base_url=args.api_base_url,
        token=token,
        timeout_seconds=args.timeout_seconds,
        retries=args.retries,
    )
    try:
        result = audit_public_release(
            root=root,
            tag=tag,
            contract_path=args.contract,
            assets=args.assets,
            mode=args.mode,
            client=client,
            rebuild_count=args.rebuild_count,
        )
    except (
        OSError,
        PublicReleaseAuditError,
        builder.ReleaseAssetError,
        verifier.ReleaseVerificationError,
    ) as error:
        print(
            json.dumps(
                {
                    "schema": AUDIT_SCHEMA,
                    "status": "FAIL",
                    "mode": args.mode,
                    "error": str(error),
                },
                indent=2,
                sort_keys=True,
            )
        )
        raise SystemExit(1) from error
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
