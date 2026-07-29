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

"""Build byte-reproducible release assets exclusively from an annotated Git tag."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import subprocess
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import quote

import publication_state

SCHEMA = "gdn-sm90a.report-release-assets.v1"
MANIFEST_SCHEMA = "gdn-sm90a.report-release-manifest.v1"
PACKAGE_KEYS = ("source", "evidence", "content")
ALLOWED_GIT_MODES = {"100644", "100755"}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ReleaseAssetError(ValueError):
    """The tagged source cannot produce the declared public release."""


@dataclass(frozen=True)
class GitBlob:
    """One ordinary tracked file read from a Git object tree."""

    path: str
    mode: str
    oid: str
    data: bytes

    @property
    def archive_mode(self) -> int:
        return 0o755 if self.mode == "100755" else 0o644


@dataclass(frozen=True)
class TagIdentity:
    """Immutable annotated-tag identity used by every generated asset."""

    name: str
    tag_object: str
    commit: str
    tree: str
    message: bytes


def canonical_json_bytes(value: Any) -> bytes:
    """Return the repository's stable human-readable JSON representation."""

    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _run_git(root: Path, *arguments: str, check: bool = True) -> subprocess.CompletedProcess[bytes]:
    result = subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=False,
        capture_output=True,
    )
    if check and result.returncode != 0:
        detail = result.stderr.decode(errors="replace").strip()
        raise ReleaseAssetError(f"git {' '.join(arguments)} failed: {detail}")
    return result


def repository_root(path: Path) -> Path:
    result = _run_git(path.resolve(), "rev-parse", "--show-toplevel")
    return Path(result.stdout.decode().strip()).resolve()


def _safe_relative_path(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise ReleaseAssetError(f"{context}: expected a non-empty path")
    if "\x00" in value or "\\" in value:
        raise ReleaseAssetError(f"{context}: unsafe path {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ReleaseAssetError(f"{context}: unsafe path {value!r}")
    return path.as_posix()


def _safe_filename(value: Any, context: str) -> str:
    path = _safe_relative_path(value, context)
    if "/" in path:
        raise ReleaseAssetError(f"{context}: filename must not contain a directory")
    return path


def _safe_prefix(value: Any, context: str) -> str:
    return _safe_relative_path(value, context).rstrip("/")


def _parse_json(raw: bytes, context: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ReleaseAssetError(f"{context}: invalid JSON: {error}") from error
    if not isinstance(value, dict):
        raise ReleaseAssetError(f"{context}: expected a JSON object")
    return value


def load_tag_identity(root: Path, tag: str) -> TagIdentity:
    """Resolve and validate a real annotated tag, never a branch or lightweight tag."""

    if not tag or any(character.isspace() for character in tag):
        raise ReleaseAssetError("tag must be a non-empty ref name without whitespace")
    check_ref = _run_git(root, "check-ref-format", f"refs/tags/{tag}", check=False)
    if check_ref.returncode != 0:
        raise ReleaseAssetError(f"invalid tag name: {tag!r}")
    result = _run_git(root, "rev-parse", "--verify", f"refs/tags/{tag}", check=False)
    if result.returncode != 0:
        raise ReleaseAssetError(f"annotated tag is missing: {tag}")
    tag_object = result.stdout.decode().strip()
    object_type = _run_git(root, "cat-file", "-t", tag_object).stdout.decode().strip()
    if object_type != "tag":
        raise ReleaseAssetError(f"{tag}: expected an annotated tag, found {object_type}")
    raw_tag = _run_git(root, "cat-file", "tag", tag_object).stdout
    header, separator, message = raw_tag.partition(b"\n\n")
    if not separator:
        raise ReleaseAssetError(f"{tag}: malformed annotated-tag object")
    fields: dict[str, str] = {}
    for line in header.splitlines():
        key, space, value = line.partition(b" ")
        if space:
            fields[key.decode()] = value.decode()
    if fields.get("type") != "commit" or fields.get("tag") != tag:
        raise ReleaseAssetError(f"{tag}: annotated-tag header is inconsistent")
    commit = fields.get("object")
    if commit is None or not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ReleaseAssetError(f"{tag}: invalid peeled commit identity")
    if _run_git(root, "cat-file", "-t", commit).stdout.decode().strip() != "commit":
        raise ReleaseAssetError(f"{tag}: annotated tag does not reference a commit")
    peeled = _run_git(root, "rev-parse", f"{tag}^{{}}").stdout.decode().strip()
    if peeled != commit:
        raise ReleaseAssetError(f"{tag}: tag peel differs from its object header")
    tree = _run_git(root, "rev-parse", f"{commit}^{{tree}}").stdout.decode().strip()
    return TagIdentity(
        name=tag,
        tag_object=tag_object,
        commit=commit,
        tree=tree,
        message=message,
    )


def load_tree(root: Path, identity: TagIdentity) -> dict[str, GitBlob]:
    """Load all ordinary files from the tagged tree and reject special entries."""

    raw = _run_git(root, "ls-tree", "-rz", "--full-tree", identity.commit).stdout
    blobs: dict[str, GitBlob] = {}
    for record in raw.split(b"\x00"):
        if not record:
            continue
        metadata, separator, raw_path = record.partition(b"\t")
        if not separator:
            raise ReleaseAssetError("malformed git ls-tree record")
        try:
            mode, object_type, oid = metadata.decode().split(" ")
            path = raw_path.decode()
        except (UnicodeDecodeError, ValueError) as error:
            raise ReleaseAssetError("non-UTF-8 or malformed tracked path") from error
        path = _safe_relative_path(path, "tracked tree")
        if object_type != "blob" or mode not in ALLOWED_GIT_MODES:
            raise ReleaseAssetError(
                f"{path}: release archives forbid type={object_type!r}, mode={mode!r}"
            )
        if path in blobs:
            raise ReleaseAssetError(f"duplicate tracked path: {path}")
        data = _run_git(root, "cat-file", "blob", oid).stdout
        blobs[path] = GitBlob(path=path, mode=mode, oid=oid, data=data)
    if not blobs:
        raise ReleaseAssetError("tagged tree contains no ordinary files")
    return dict(sorted(blobs.items()))


def blob_bytes(blobs: dict[str, GitBlob], path: str) -> bytes:
    try:
        return blobs[path].data
    except KeyError as error:
        raise ReleaseAssetError(f"tagged source is missing required file: {path}") from error


def load_contract(blobs: dict[str, GitBlob], contract_path: str) -> dict[str, Any]:
    contract_path = _safe_relative_path(contract_path, "contract path")
    contract = _parse_json(blob_bytes(blobs, contract_path), contract_path)
    if contract.get("schema") != SCHEMA:
        raise ReleaseAssetError(f"{contract_path}: unexpected schema")
    required_top_level = {
        "schema",
        "release",
        "evidence",
        "packages",
        "source_links",
        "required_claims",
        "release_manifest_filename",
        "checksum_filename",
        "source_lock_path",
        "claim_registry_path",
    }
    if set(contract) != required_top_level:
        raise ReleaseAssetError(
            f"{contract_path}: keys differ from the frozen schema: "
            f"{sorted(set(contract) ^ required_top_level)}"
        )
    _safe_filename(contract["release_manifest_filename"], "release manifest filename")
    _safe_filename(contract["checksum_filename"], "checksum filename")
    release = contract["release"]
    if not isinstance(release, dict) or set(release) != {
        "repository",
        "expected_public_tag",
        "title",
        "notes_path",
        "tag_message_path",
    }:
        raise ReleaseAssetError("release contract shape is invalid")
    if not re.fullmatch(r"[^/]+/[^/]+", release["repository"]):
        raise ReleaseAssetError("release repository must be owner/name")
    _safe_relative_path(release["notes_path"], "release notes path")
    _safe_relative_path(release["tag_message_path"], "tag message path")
    _safe_relative_path(contract["source_lock_path"], "source lock path")
    _safe_relative_path(contract["claim_registry_path"], "claim registry path")
    packages = contract["packages"]
    if not isinstance(packages, dict) or set(packages) != set(PACKAGE_KEYS):
        raise ReleaseAssetError(f"package keys must be exactly {PACKAGE_KEYS}")
    filenames = []
    for key in PACKAGE_KEYS:
        package = packages[key]
        if not isinstance(package, dict) or set(package) != {
            "filename",
            "format",
            "include",
            "prefix",
        }:
            raise ReleaseAssetError(f"package {key}: invalid contract shape")
        filename = _safe_filename(package["filename"], f"package {key} filename")
        filenames.append(filename)
        expected_format = "zip" if key == "content" else "tar"
        if package["format"] != expected_format:
            raise ReleaseAssetError(f"package {key}: format must be {expected_format}")
        _safe_prefix(package["prefix"], f"package {key} prefix")
        includes = package["include"]
        if not isinstance(includes, list) or not includes:
            raise ReleaseAssetError(f"package {key}: include list is empty")
        if includes != ["**"]:
            for index, include in enumerate(includes):
                _safe_relative_path(include, f"package {key} include[{index}]")
    reserved = {
        contract["release_manifest_filename"],
        contract["checksum_filename"],
    }
    if len(filenames) != len(set(filenames)) or set(filenames) & reserved:
        raise ReleaseAssetError("release asset filenames are not unique")
    evidence = contract["evidence"]
    if not isinstance(evidence, dict) or set(evidence) != {"historical", "fresh"}:
        raise ReleaseAssetError("evidence contract must contain historical and fresh")
    for evidence_class, specification in evidence.items():
        required = {
            "path",
            "manifest_path",
            "seal_path",
            "publication_path",
            "evidence_class",
            "required_status",
        }
        if evidence_class == "fresh":
            required |= {"evidence_kind", "decision_status"}
        if not isinstance(specification, dict) or set(specification) != required:
            raise ReleaseAssetError(f"evidence {evidence_class}: invalid contract shape")
        root = _safe_relative_path(specification["path"], f"evidence {evidence_class} path")
        for key in ("manifest_path", "seal_path", "publication_path"):
            path = _safe_relative_path(specification[key], f"evidence {evidence_class} {key}")
            if path != root and not path.startswith(root + "/"):
                raise ReleaseAssetError(f"evidence {evidence_class} {key} escapes its root")
    if not isinstance(contract["source_links"], list) or not contract["source_links"]:
        raise ReleaseAssetError("source_links must be a non-empty list")
    source_ids = []
    for source in contract["source_links"]:
        if not isinstance(source, dict):
            raise ReleaseAssetError("source link must be an object")
        kind = source.get("kind")
        expected_keys = {"id", "kind", "repository", "url"}
        if kind == "annotated_tag_release":
            expected_keys.add("release_url")
        if set(source) != expected_keys or kind not in {"annotated_tag_release", "commit"}:
            raise ReleaseAssetError(f"source link has invalid shape: {source!r}")
        source_ids.append(source["id"])
        if not re.fullmatch(r"[^/]+/[^/]+", source["repository"]):
            raise ReleaseAssetError(f"source {source['id']}: repository must be owner/name")
        for key in ("url", "release_url"):
            if key in source and not source[key].startswith("https://github.com/"):
                raise ReleaseAssetError(f"source {source['id']}: {key} is not a GitHub URL")
    if len(source_ids) != len(set(source_ids)):
        raise ReleaseAssetError("source link IDs are not unique")
    required_claims = contract["required_claims"]
    if not isinstance(required_claims, dict) or not required_claims:
        raise ReleaseAssetError("required_claims must be a non-empty object")
    if any(not isinstance(value, bool) for value in required_claims.values()):
        raise ReleaseAssetError("required_claims values must be booleans")
    return contract


def select_paths(
    blobs: dict[str, GitBlob],
    includes: list[str],
    *,
    package_key: str,
) -> list[str]:
    if includes == ["**"]:
        return list(blobs)
    selected: set[str] = set()
    for include in includes:
        matches = [
            path for path in blobs if path == include or path.startswith(include.rstrip("/") + "/")
        ]
        if not matches:
            raise ReleaseAssetError(
                f"package {package_key}: include has no tagged files: {include}"
            )
        selected.update(matches)
    return sorted(selected)


def _tar_bytes(blobs: dict[str, GitBlob], paths: list[str], prefix: str) -> bytes:
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w", format=tarfile.GNU_FORMAT) as archive:
        for path in paths:
            blob = blobs[path]
            info = tarfile.TarInfo(name=f"{prefix}/{path}")
            info.size = len(blob.data)
            info.mode = blob.archive_mode
            info.mtime = 0
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            info.type = tarfile.REGTYPE
            archive.addfile(info, io.BytesIO(blob.data))
    return stream.getvalue()


def _zip_bytes(blobs: dict[str, GitBlob], paths: list[str], prefix: str) -> bytes:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, mode="w", compression=zipfile.ZIP_STORED) as archive:
        for path in paths:
            blob = blobs[path]
            info = zipfile.ZipInfo(f"{prefix}/{path}", date_time=(1980, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = blob.archive_mode << 16
            archive.writestr(info, blob.data)
    return stream.getvalue()


def build_package_bytes(
    blobs: dict[str, GitBlob],
    package_key: str,
    specification: dict[str, Any],
) -> tuple[bytes, list[str]]:
    paths = select_paths(blobs, specification["include"], package_key=package_key)
    prefix = specification["prefix"]
    if specification["format"] == "tar":
        payload = _tar_bytes(blobs, paths, prefix)
    elif specification["format"] == "zip":
        payload = _zip_bytes(blobs, paths, prefix)
    else:  # pragma: no cover - contract validation closes this branch
        raise ReleaseAssetError(f"package {package_key}: unsupported format")
    return payload, paths


def _validate_sealed_bundle(
    blobs: dict[str, GitBlob],
    specification: dict[str, Any],
) -> dict[str, Any]:
    root = specification["path"]
    manifest_path = specification["manifest_path"]
    seal_path = specification["seal_path"]
    manifest_raw = blob_bytes(blobs, manifest_path)
    manifest = _parse_json(manifest_raw, manifest_path)
    expected_seal = f"{sha256_bytes(manifest_raw)}  manifest.json\n".encode()
    if blob_bytes(blobs, seal_path) != expected_seal:
        raise ReleaseAssetError(f"{seal_path}: evidence manifest seal mismatch")
    files = manifest.get("files")
    if not isinstance(files, list) or manifest.get("file_count") != len(files):
        raise ReleaseAssetError(f"{manifest_path}: invalid evidence file inventory")
    relative_paths = []
    for entry in files:
        if not isinstance(entry, dict) or set(entry) != {"path", "sha256", "size_bytes"}:
            raise ReleaseAssetError(f"{manifest_path}: invalid evidence manifest entry")
        relative = _safe_relative_path(entry["path"], f"{manifest_path} payload")
        relative_paths.append(relative)
        tagged_path = f"{root}/{relative}"
        payload = blob_bytes(blobs, tagged_path)
        if entry["size_bytes"] != len(payload) or entry["sha256"] != sha256_bytes(payload):
            raise ReleaseAssetError(f"{tagged_path}: evidence payload identity mismatch")
    if len(relative_paths) != len(set(relative_paths)):
        raise ReleaseAssetError(f"{manifest_path}: evidence payload paths are not unique")
    expected_paths = {
        *{f"{root}/{path}" for path in relative_paths},
        manifest_path,
        seal_path,
    }
    actual_paths = {path for path in blobs if path.startswith(root + "/")}
    if actual_paths != expected_paths:
        difference = sorted(actual_paths ^ expected_paths)
        raise ReleaseAssetError(f"{root}: sealed evidence membership differs: {difference}")
    return {
        "path": root,
        "evidence_class": specification["evidence_class"],
        "manifest_path": manifest_path,
        "manifest_sha256": sha256_bytes(manifest_raw),
        "manifest_file_count": len(files),
        "seal_path": seal_path,
        "seal_sha256": sha256_bytes(blob_bytes(blobs, seal_path)),
    }


def _validate_evidence(
    blobs: dict[str, GitBlob],
    contract: dict[str, Any],
) -> dict[str, Any]:
    result = {}
    for evidence_class in ("historical", "fresh"):
        specification = contract["evidence"][evidence_class]
        summary = _validate_sealed_bundle(blobs, specification)
        publication = _parse_json(
            blob_bytes(blobs, specification["publication_path"]),
            specification["publication_path"],
        )
        if evidence_class == "historical":
            if publication.get("package_status") != specification["required_status"]:
                raise ReleaseAssetError("historical evidence status differs from contract")
            if publication.get("upstream_merge") is not False:
                raise ReleaseAssetError("historical publication claims an upstream merge")
            summary["publication_status"] = publication["package_status"]
            summary["publication_sha256"] = sha256_bytes(
                blob_bytes(blobs, specification["publication_path"])
            )
        else:
            if publication.get("status") != specification["required_status"]:
                raise ReleaseAssetError("fresh evidence status differs from contract")
            if publication.get("evidence_kind") != specification["evidence_kind"]:
                raise ReleaseAssetError("fresh evidence kind differs from contract")
            if publication.get("decision_status") != specification["decision_status"]:
                raise ReleaseAssetError("fresh evidence decision differs from contract")
            if publication.get("upstream_merge_claim") is not False:
                raise ReleaseAssetError("fresh publication claims an upstream merge")
            performance_path = f"{specification['path']}/performance.json"
            performance = _parse_json(blob_bytes(blobs, performance_path), performance_path)
            if (
                performance.get("status") != "PASS"
                or performance.get("decision_status") != specification["decision_status"]
            ):
                raise ReleaseAssetError("fresh performance decision is not publishable")
            receipt_count = publication.get("receipt_count")
            if receipt_count not in {54, 66} or performance.get("receipt_count") != receipt_count:
                raise ReleaseAssetError("fresh evidence receipt count is not the frozen 54/66 set")
            geomean = performance.get("primary_geomean")
            if not isinstance(geomean, dict) or set(geomean) != {
                "tirx_over_cutedsl",
                "tirx_over_fla",
            }:
                raise ReleaseAssetError("fresh evidence lacks the primary geometric means")
            if not all(
                isinstance(value, int | float) and not isinstance(value, bool) and value > 0
                for value in geomean.values()
            ):
                raise ReleaseAssetError("fresh primary geometric means are invalid")
            summary.update(
                {
                    "publication_status": publication["status"],
                    "publication_sha256": sha256_bytes(
                        blob_bytes(blobs, specification["publication_path"])
                    ),
                    "decision_status": publication["decision_status"],
                    "evidence_kind": publication["evidence_kind"],
                    "run_id": publication.get("run_id"),
                    "receipt_count": receipt_count,
                    "primary_geomean": geomean,
                    "packed_n10_escalation_required": performance.get(
                        "packed_n10_escalation_required"
                    ),
                }
            )
        result[evidence_class] = summary
    return result


def _validate_claims(
    blobs: dict[str, GitBlob],
    contract: dict[str, Any],
) -> dict[str, Any]:
    path = contract["claim_registry_path"]
    raw = blob_bytes(blobs, path)
    registry = _parse_json(raw, path)
    claims = registry.get("claims")
    if not isinstance(claims, list):
        raise ReleaseAssetError("claim registry has no claims list")
    by_id = {
        claim.get("id"): claim
        for claim in claims
        if isinstance(claim, dict) and isinstance(claim.get("id"), str)
    }
    states = {}
    for claim_id, required_state in contract["required_claims"].items():
        if claim_id not in by_id:
            raise ReleaseAssetError(f"required claim is missing: {claim_id}")
        observed = by_id[claim_id].get("enabled")
        if observed is not required_state:
            raise ReleaseAssetError(
                f"required claim {claim_id} enabled={observed!r}, expected {required_state!r}"
            )
        states[claim_id] = observed
    return {
        "path": path,
        "sha256": sha256_bytes(raw),
        "required_states": states,
    }


def _source_coordinates(
    blobs: dict[str, GitBlob],
    contract: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    path = contract["source_lock_path"]
    raw = blob_bytes(blobs, path)
    source_lock = _parse_json(raw, path)
    repositories = source_lock.get("repositories")
    if not isinstance(repositories, dict):
        raise ReleaseAssetError("public source lock has no repositories")
    coordinates = {}
    for source in contract["source_links"]:
        source_id = source["id"]
        if source_id not in repositories:
            raise ReleaseAssetError(f"source link has no source-lock entry: {source_id}")
        locked = repositories[source_id]
        coordinate = {
            "id": source_id,
            "kind": source["kind"],
            "repository": source["repository"],
            "url": source["url"],
        }
        if source["kind"] == "annotated_tag_release":
            release = locked.get("public_release", {})
            required = {"tag", "tag_object", "commit", "tree"}
            if not required <= set(release):
                raise ReleaseAssetError(f"source {source_id}: incomplete public release lock")
            upstream_repository = locked.get("upstream_repository")
            branch = release.get("branch")
            if (
                not isinstance(upstream_repository, str)
                or not upstream_repository.startswith("https://github.com/")
                or not isinstance(branch, str)
                or not branch
            ):
                raise ReleaseAssetError(
                    f"source {source_id}: upstream repository or release branch is missing"
                )
            coordinate.update(
                {
                    "release_url": source["release_url"],
                    "release_branch": branch,
                    "upstream_repository": upstream_repository,
                    **{key: release[key] for key in sorted(required)},
                }
            )
        else:
            if not {"commit", "tree"} <= set(locked):
                raise ReleaseAssetError(f"source {source_id}: incomplete commit lock")
            coordinate.update(commit=locked["commit"], tree=locked["tree"])
        coordinates[source_id] = coordinate
    return coordinates, sha256_bytes(raw)


def build_release_assets(
    *,
    root: Path,
    tag: str,
    contract_path: str,
    output: Path,
    require_contract_tag: bool = False,
) -> dict[str, Any]:
    """Build all release assets and return their immutable identity summary."""

    root = repository_root(root)
    output = output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to reuse release output: {output}")
    identity = load_tag_identity(root, tag)
    blobs = load_tree(root, identity)
    contract_path = _safe_relative_path(contract_path, "contract path")
    contract = load_contract(blobs, contract_path)
    expected_tag = contract["release"]["expected_public_tag"]
    if require_contract_tag and tag != expected_tag:
        raise ReleaseAssetError(f"tag {tag!r} differs from public tag {expected_tag!r}")
    canonical_tag_message = blob_bytes(blobs, contract["release"]["tag_message_path"])
    if identity.message != canonical_tag_message:
        raise ReleaseAssetError("annotated-tag message differs from its tagged canonical file")

    notes_path = contract["release"]["notes_path"]
    notes = blob_bytes(blobs, notes_path)
    contract_raw = blob_bytes(blobs, contract_path)
    evidence = _validate_evidence(blobs, contract)
    claims = _validate_claims(blobs, contract)
    source_coordinates, source_lock_sha256 = _source_coordinates(blobs, contract)
    content_paths = select_paths(
        blobs,
        contract["packages"]["content"]["include"],
        package_key="content",
    )
    state_findings = publication_state.scan_stale_fresh_state(
        {relative: blobs[relative].data for relative in content_paths}
    )
    if state_findings:
        raise ReleaseAssetError(
            "publication content contains obsolete pre-rerun guidance: "
            + json.dumps(state_findings, sort_keys=True)
        )

    output.mkdir(parents=True)
    package_summaries = []
    package_paths: dict[str, list[str]] = {}
    for package_key in PACKAGE_KEYS:
        specification = contract["packages"][package_key]
        payload, paths = build_package_bytes(blobs, package_key, specification)
        asset_path = output / specification["filename"]
        asset_path.write_bytes(payload)
        package_paths[package_key] = paths
        package_summaries.append(
            {
                "name": specification["filename"],
                "format": specification["format"],
                "role": package_key,
                "file_count": len(paths),
                "sha256": sha256_bytes(payload),
                "size_bytes": len(payload),
            }
        )

    repository = contract["release"]["repository"]
    encoded_tag = quote(tag, safe="")
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "release": {
            "repository": repository,
            "title": contract["release"]["title"],
            "expected_public_tag": expected_tag,
            "tag": tag,
            "tag_object": identity.tag_object,
            "commit": identity.commit,
            "tree": identity.tree,
            "tag_message_path": contract["release"]["tag_message_path"],
            "tag_message_sha256": sha256_bytes(canonical_tag_message),
            "notes_path": notes_path,
            "notes_sha256": sha256_bytes(notes),
            "notes_size_bytes": len(notes),
            "tag_url": f"https://github.com/{repository}/tree/{encoded_tag}",
            "release_url": f"https://github.com/{repository}/releases/tag/{encoded_tag}",
        },
        "contract": {
            "path": contract_path,
            "sha256": sha256_bytes(contract_raw),
        },
        "source_lock": {
            "path": contract["source_lock_path"],
            "sha256": source_lock_sha256,
            "coordinates": source_coordinates,
        },
        "evidence": evidence,
        "claims": claims,
        "boundaries": {
            "decision_status": evidence["fresh"]["decision_status"],
            "measurement_scope": (
                "one NVIDIA H20, six frozen BF16/head-dimension-128 "
                "GDN prefill operator-latency rows"
            ),
            "unofficial_personal_forks": True,
            "upstream_merge_claim": False,
            "upstream_pr_created_for_publication": False,
        },
        "assets": package_summaries,
    }
    manifest_name = contract["release_manifest_filename"]
    manifest_path = output / manifest_name
    manifest_payload = canonical_json_bytes(manifest)
    manifest_path.write_bytes(manifest_payload)

    checksum_entries = [(summary["name"], summary["sha256"]) for summary in package_summaries]
    checksum_entries.append((manifest_name, sha256_bytes(manifest_payload)))
    checksum_entries.sort()
    checksum_payload = "".join(f"{digest}  {name}\n" for name, digest in checksum_entries).encode()
    checksum_path = output / contract["checksum_filename"]
    checksum_path.write_bytes(checksum_payload)

    return {
        "schema": "gdn-sm90a.release-asset-build.v1",
        "status": "PASS",
        "tag": tag,
        "tag_object": identity.tag_object,
        "commit": identity.commit,
        "tree": identity.tree,
        "output": str(output),
        "asset_count": len(package_summaries) + 2,
        "assets": [
            {
                "name": path.name,
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for path in sorted(output.iterdir())
        ],
        "package_paths": package_paths,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build deterministic public release assets from an annotated Git tag."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Report repository root.",
    )
    parser.add_argument(
        "--tag",
        help="Annotated source tag. Defaults to release.expected_public_tag.",
    )
    parser.add_argument(
        "--contract",
        default="contracts/release-assets.json",
        help="Repository-relative contract path read from the tag.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--require-contract-tag",
        action="store_true",
        help="Reject a local test tag whose name differs from expected_public_tag.",
    )
    args = parser.parse_args()
    root = repository_root(args.root)
    tag = args.tag
    if tag is None:
        working_contract = _parse_json(
            (root / args.contract).read_bytes(),
            args.contract,
        )
        tag = working_contract.get("release", {}).get("expected_public_tag")
        if not isinstance(tag, str):
            raise ReleaseAssetError("working-tree contract does not declare a public tag")
    try:
        result = build_release_assets(
            root=root,
            tag=tag,
            contract_path=args.contract,
            output=args.output,
            require_contract_tag=args.require_contract_tag,
        )
    except (FileExistsError, OSError, ReleaseAssetError, subprocess.SubprocessError) as error:
        print(
            json.dumps(
                {
                    "schema": "gdn-sm90a.release-asset-build.v1",
                    "status": "FAIL",
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
