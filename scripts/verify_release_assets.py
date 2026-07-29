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

"""Independently inspect and reproducibly rebuild tagged release assets."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

import build_release_assets as builder
import publication_state

VERIFY_SCHEMA = "gdn-sm90a.release-asset-verification.v1"


class ReleaseVerificationError(ValueError):
    """A release asset differs from its tagged, normalized source."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ReleaseVerificationError(message)


def _read_json(path: Path, context: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ReleaseVerificationError(f"{context}: invalid JSON: {error}") from error
    if not isinstance(value, dict):
        raise ReleaseVerificationError(f"{context}: expected a JSON object")
    return value


def _expected_asset_names(contract: dict[str, Any]) -> set[str]:
    return {
        *{contract["packages"][package_key]["filename"] for package_key in builder.PACKAGE_KEYS},
        contract["release_manifest_filename"],
        contract["checksum_filename"],
    }


def _parse_checksums(path: Path) -> dict[str, str]:
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise ReleaseVerificationError(f"cannot read checksum file: {error}") from error
    _require(raw.endswith(b"\n"), "checksum file must end with LF")
    checksums: dict[str, str] = {}
    names_in_order = []
    for line_number, raw_line in enumerate(raw.splitlines(), start=1):
        try:
            line = raw_line.decode("ascii")
        except UnicodeDecodeError as error:
            raise ReleaseVerificationError(
                f"checksum line {line_number}: expected ASCII"
            ) from error
        digest, separator, name = line.partition("  ")
        _require(separator == "  ", f"checksum line {line_number}: invalid separator")
        _require(
            bool(builder.SHA256_RE.fullmatch(digest)),
            f"checksum line {line_number}: invalid SHA-256",
        )
        _require(
            name == PurePosixPath(name).name and name not in {"", ".", ".."},
            f"checksum line {line_number}: unsafe filename",
        )
        _require(name not in checksums, f"checksum line {line_number}: duplicate filename")
        checksums[name] = digest
        names_in_order.append(name)
    _require(names_in_order == sorted(names_in_order), "checksum entries are not sorted")
    return checksums


def _inspect_tar(
    *,
    path: Path,
    prefix: str,
    expected_paths: list[str],
    blobs: dict[str, builder.GitBlob],
) -> None:
    try:
        archive = tarfile.open(path, mode="r:")
    except (OSError, tarfile.TarError) as error:
        raise ReleaseVerificationError(f"{path.name}: invalid uncompressed tar: {error}") from error
    with archive:
        members = archive.getmembers()
        expected_names = [f"{prefix}/{relative}" for relative in expected_paths]
        actual_names = [member.name for member in members]
        _require(actual_names == expected_names, f"{path.name}: tar membership/order drift")
        for member, relative in zip(members, expected_paths):
            blob = blobs[relative]
            _require(member.isfile(), f"{path.name}:{member.name}: non-file member")
            _require(member.mode == blob.archive_mode, f"{path.name}:{member.name}: mode drift")
            _require(member.mtime == 0, f"{path.name}:{member.name}: mtime is not zero")
            _require(
                member.uid == 0 and member.gid == 0,
                f"{path.name}:{member.name}: uid/gid are not zero",
            )
            _require(
                member.uname == "" and member.gname == "",
                f"{path.name}:{member.name}: owner names are not empty",
            )
            _require(member.size == len(blob.data), f"{path.name}:{member.name}: size drift")
            stream = archive.extractfile(member)
            _require(stream is not None, f"{path.name}:{member.name}: cannot read payload")
            _require(stream.read() == blob.data, f"{path.name}:{member.name}: byte drift")


def _inspect_zip(
    *,
    path: Path,
    prefix: str,
    expected_paths: list[str],
    blobs: dict[str, builder.GitBlob],
) -> None:
    try:
        archive = zipfile.ZipFile(path, mode="r")
    except (OSError, zipfile.BadZipFile) as error:
        raise ReleaseVerificationError(f"{path.name}: invalid zip: {error}") from error
    with archive:
        _require(archive.comment == b"", f"{path.name}: zip comment must be empty")
        entries = archive.infolist()
        expected_names = [f"{prefix}/{relative}" for relative in expected_paths]
        _require(
            [entry.filename for entry in entries] == expected_names,
            f"{path.name}: zip membership/order drift",
        )
        for entry, relative in zip(entries, expected_paths):
            blob = blobs[relative]
            _require(not entry.is_dir(), f"{path.name}:{entry.filename}: directory entry")
            _require(
                entry.date_time == (1980, 1, 1, 0, 0, 0),
                f"{path.name}:{entry.filename}: timestamp drift",
            )
            _require(
                entry.compress_type == zipfile.ZIP_STORED,
                f"{path.name}:{entry.filename}: compression is not ZIP_STORED",
            )
            _require(
                entry.create_system == 3,
                f"{path.name}:{entry.filename}: creator is not Unix",
            )
            observed_mode = (entry.external_attr >> 16) & 0o777
            _require(
                observed_mode == blob.archive_mode,
                f"{path.name}:{entry.filename}: mode drift",
            )
            _require(
                archive.read(entry) == blob.data,
                f"{path.name}:{entry.filename}: byte drift",
            )


def _compare_directories(left: Path, right: Path, expected_names: set[str]) -> None:
    left_names = {path.name for path in left.iterdir() if path.is_file()}
    right_names = {path.name for path in right.iterdir() if path.is_file()}
    _require(left_names == expected_names, "supplied release asset set differs from contract")
    _require(right_names == expected_names, "rebuilt release asset set differs from contract")
    for name in sorted(expected_names):
        _require(
            (left / name).read_bytes() == (right / name).read_bytes(),
            f"{name}: rebuilt bytes differ",
        )


def _verify_publication_state(
    expected_paths: list[str],
    blobs: dict[str, builder.GitBlob],
) -> int:
    payloads = {relative: blobs[relative].data for relative in expected_paths}
    findings = publication_state.scan_stale_fresh_state(payloads)
    _require(
        not findings,
        "publication content contains obsolete pre-rerun guidance: "
        + json.dumps(findings, sort_keys=True),
    )
    return sum(
        PurePosixPath(relative).suffix.lower() in publication_state.TEXT_SUFFIXES
        for relative in expected_paths
    )


def _safe_materialize_tar(path: Path, destination: Path) -> Path:
    """Materialize an already-validated regular-file archive without extractall."""

    with tarfile.open(path, mode="r:") as archive:
        members = archive.getmembers()
        _require(bool(members), f"{path.name}: archive is empty")
        first = PurePosixPath(members[0].name)
        _require(len(first.parts) >= 2, f"{path.name}: missing top-level prefix")
        prefix = first.parts[0]
        for member in members:
            relative = PurePosixPath(member.name)
            _require(
                not relative.is_absolute()
                and ".." not in relative.parts
                and relative.parts[0] == prefix,
                f"{path.name}:{member.name}: unsafe extraction path",
            )
            _require(member.isfile(), f"{path.name}:{member.name}: non-file extraction")
            target = destination.joinpath(*relative.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            stream = archive.extractfile(member)
            _require(stream is not None, f"{path.name}:{member.name}: unreadable payload")
            target.write_bytes(stream.read())
            target.chmod(member.mode)
    return destination / prefix


def _run_evidence_verifiers(
    *,
    evidence_archive: Path,
    contract: dict[str, Any],
) -> list[dict[str, Any]]:
    with tempfile.TemporaryDirectory(prefix="gdn-release-evidence-") as temporary:
        root = _safe_materialize_tar(evidence_archive, Path(temporary))
        environment = os.environ.copy()
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        environment["PYTHONPATH"] = str(root)
        commands = [
            (
                "historical",
                [
                    sys.executable,
                    str(root / "scripts/verify_public_evidence.py"),
                    "--bundle",
                    str(root / contract["evidence"]["historical"]["path"]),
                ],
            ),
            (
                "fresh",
                [
                    sys.executable,
                    "-m",
                    "reproduce.fresh_evidence.verify",
                    "--bundle",
                    str(root / contract["evidence"]["fresh"]["path"]),
                ],
            ),
        ]
        results = []
        for kind, command in commands:
            completed = subprocess.run(
                command,
                cwd=root,
                env=environment,
                check=False,
                capture_output=True,
                text=True,
            )
            if completed.returncode != 0:
                detail = completed.stderr.strip() or completed.stdout.strip()
                raise ReleaseVerificationError(
                    f"bundled {kind} evidence verifier failed ({command[1]}): {detail}"
                )
            stdout = completed.stdout.strip()
            if kind == "historical":
                try:
                    payload = json.loads(stdout)
                except json.JSONDecodeError as error:
                    raise ReleaseVerificationError(
                        "bundled historical evidence verifier returned invalid JSON"
                    ) from error
                _require(
                    isinstance(payload, dict) and payload.get("status") == "PASS",
                    "bundled historical evidence verifier did not report PASS",
                )
                results.append({"kind": kind, **payload})
                continue
            marker = re.fullmatch(
                r"GDN_FRESH_PUBLIC_EVIDENCE_VERIFY_PASS "
                r"run_id=(\S+) receipts=(54|66) "
                r"tirx/cutedsl=([0-9]+\.[0-9]{6}) "
                r"tirx/fla=([0-9]+\.[0-9]{6})",
                stdout,
            )
            _require(marker is not None, "bundled fresh evidence verifier marker drift")
            publication_path = root / contract["evidence"]["fresh"]["publication_path"]
            publication = _read_json(publication_path, "bundled fresh publication")
            _require(
                marker.group(1) == publication.get("run_id"),
                "fresh verifier marker run ID differs from the bundle",
            )
            _require(
                int(marker.group(2)) == publication.get("receipt_count"),
                "fresh verifier marker receipt count differs from the bundle",
            )
            results.append(
                {
                    "kind": kind,
                    "status": "PASS",
                    "marker": marker.group(0),
                    "run_id": marker.group(1),
                    "receipt_count": int(marker.group(2)),
                    "tirx_over_cutedsl": float(marker.group(3)),
                    "tirx_over_fla": float(marker.group(4)),
                }
            )
        return results


def _validate_manifest(
    *,
    assets: Path,
    contract: dict[str, Any],
    identity: builder.TagIdentity,
    blobs: dict[str, builder.GitBlob],
    contract_path: str,
    require_contract_tag: bool,
) -> dict[str, Any]:
    manifest_name = contract["release_manifest_filename"]
    manifest = _read_json(assets / manifest_name, manifest_name)
    _require(
        manifest.get("schema") == builder.MANIFEST_SCHEMA,
        "release manifest schema mismatch",
    )
    _require(
        set(manifest)
        == {
            "schema",
            "release",
            "contract",
            "source_lock",
            "evidence",
            "claims",
            "boundaries",
            "assets",
        },
        "release manifest top-level shape drift",
    )
    release = manifest["release"]
    _require(release["tag"] == identity.name, "release manifest tag mismatch")
    _require(release["tag_object"] == identity.tag_object, "release tag object mismatch")
    _require(release["commit"] == identity.commit, "release commit mismatch")
    _require(release["tree"] == identity.tree, "release tree mismatch")
    if require_contract_tag:
        _require(
            release["tag"] == contract["release"]["expected_public_tag"],
            "release does not use the contract public tag",
        )
    contract_raw = builder.blob_bytes(blobs, contract_path)
    _require(
        manifest["contract"]
        == {"path": contract_path, "sha256": builder.sha256_bytes(contract_raw)},
        "release contract binding mismatch",
    )
    _require(
        manifest["evidence"]["fresh"]["evidence_class"] == "fresh-public-tag-h20-characterization",
        "fresh evidence class overstates the characterization",
    )
    _require(
        manifest["evidence"]["fresh"]["decision_status"] == "CHARACTERIZATION",
        "fresh evidence decision is not CHARACTERIZATION",
    )
    _require(
        manifest["boundaries"]["upstream_merge_claim"] is False
        and manifest["boundaries"]["upstream_pr_created_for_publication"] is False,
        "release manifest claims upstream publication",
    )
    package_assets = manifest["assets"]
    _require(
        isinstance(package_assets, list) and len(package_assets) == len(builder.PACKAGE_KEYS),
        "release manifest package inventory mismatch",
    )
    by_name = {entry.get("name"): entry for entry in package_assets if isinstance(entry, dict)}
    expected_package_names = {
        contract["packages"][package_key]["filename"] for package_key in builder.PACKAGE_KEYS
    }
    _require(set(by_name) == expected_package_names, "release package names differ")
    for name, entry in by_name.items():
        path = assets / name
        _require(entry["sha256"] == builder.sha256_file(path), f"{name}: manifest hash drift")
        _require(entry["size_bytes"] == path.stat().st_size, f"{name}: manifest size drift")
    return manifest


def verify_release_assets(
    *,
    root: Path,
    tag: str,
    contract_path: str,
    assets: Path,
    require_contract_tag: bool = False,
    rebuild_count: int = 2,
    run_evidence_verifiers: bool = True,
) -> dict[str, Any]:
    """Verify archive normalization, identities, hashes, and byte reproducibility."""

    root = builder.repository_root(root)
    assets = assets.expanduser().resolve()
    _require(assets.is_dir(), f"release asset directory is missing: {assets}")
    _require(rebuild_count >= 1, "rebuild_count must be positive")
    identity = builder.load_tag_identity(root, tag)
    blobs = builder.load_tree(root, identity)
    contract_path = builder._safe_relative_path(contract_path, "contract path")
    contract = builder.load_contract(blobs, contract_path)
    expected_names = _expected_asset_names(contract)
    actual_names = {path.name for path in assets.iterdir() if path.is_file()}
    actual_nonfiles = [path.name for path in assets.iterdir() if not path.is_file()]
    _require(not actual_nonfiles, f"release asset directory has non-files: {actual_nonfiles}")
    _require(actual_names == expected_names, "release asset set differs from contract")

    checksum_name = contract["checksum_filename"]
    checksums = _parse_checksums(assets / checksum_name)
    expected_checksum_names = expected_names - {checksum_name}
    _require(
        set(checksums) == expected_checksum_names,
        "checksum membership differs from release assets",
    )
    for name, expected_digest in checksums.items():
        _require(
            builder.sha256_file(assets / name) == expected_digest,
            f"{name}: SHA256SUMS mismatch",
        )

    manifest = _validate_manifest(
        assets=assets,
        contract=contract,
        identity=identity,
        blobs=blobs,
        contract_path=contract_path,
        require_contract_tag=require_contract_tag,
    )
    package_paths = {}
    publication_state_file_count = 0
    for package_key in builder.PACKAGE_KEYS:
        specification = contract["packages"][package_key]
        expected_paths = builder.select_paths(
            blobs,
            specification["include"],
            package_key=package_key,
        )
        package_paths[package_key] = expected_paths
        package_path = assets / specification["filename"]
        if specification["format"] == "tar":
            _inspect_tar(
                path=package_path,
                prefix=specification["prefix"],
                expected_paths=expected_paths,
                blobs=blobs,
            )
        else:
            _inspect_zip(
                path=package_path,
                prefix=specification["prefix"],
                expected_paths=expected_paths,
                blobs=blobs,
            )
        if package_key == "content":
            publication_state_file_count = _verify_publication_state(expected_paths, blobs)

    rebuild_digests = []
    with tempfile.TemporaryDirectory(prefix="gdn-release-rebuilds-") as temporary:
        temporary_root = Path(temporary)
        previous = assets
        for index in range(rebuild_count):
            rebuilt = temporary_root / f"build-{index}"
            builder.build_release_assets(
                root=root,
                tag=tag,
                contract_path=contract_path,
                output=rebuilt,
                require_contract_tag=require_contract_tag,
            )
            _compare_directories(previous, rebuilt, expected_names)
            rebuild_digests.append(
                {name: builder.sha256_file(rebuilt / name) for name in sorted(expected_names)}
            )
            previous = rebuilt

    evidence_results = []
    if run_evidence_verifiers:
        evidence_results = _run_evidence_verifiers(
            evidence_archive=assets / contract["packages"]["evidence"]["filename"],
            contract=contract,
        )

    return {
        "schema": VERIFY_SCHEMA,
        "status": "PASS",
        "tag": identity.name,
        "tag_object": identity.tag_object,
        "commit": identity.commit,
        "tree": identity.tree,
        "asset_count": len(expected_names),
        "rebuild_count": rebuild_count,
        "byte_identical_rebuilds": True,
        "fresh_evidence_class": manifest["evidence"]["fresh"]["evidence_class"],
        "fresh_decision_status": manifest["evidence"]["fresh"]["decision_status"],
        "publication_state_scan": {
            "status": "PASS",
            "text_file_count": publication_state_file_count,
        },
        "package_file_counts": {key: len(value) for key, value in sorted(package_paths.items())},
        "evidence_verifier_results": evidence_results,
        "asset_sha256": {
            name: builder.sha256_file(assets / name) for name in sorted(expected_names)
        },
        "rebuild_sha256": rebuild_digests,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify deterministic public release assets against an annotated Git tag."
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
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--require-contract-tag", action="store_true")
    parser.add_argument("--rebuild-count", type=int, default=2)
    args = parser.parse_args()
    root = builder.repository_root(args.root)
    tag = args.tag
    if tag is None:
        working_contract = _read_json(root / args.contract, args.contract)
        tag = working_contract.get("release", {}).get("expected_public_tag")
        if not isinstance(tag, str):
            raise ReleaseVerificationError("working-tree contract has no public tag")
    try:
        result = verify_release_assets(
            root=root,
            tag=tag,
            contract_path=args.contract,
            assets=args.assets,
            require_contract_tag=args.require_contract_tag,
            rebuild_count=args.rebuild_count,
        )
    except (
        FileExistsError,
        OSError,
        ReleaseVerificationError,
        builder.ReleaseAssetError,
        subprocess.SubprocessError,
    ) as error:
        print(
            json.dumps(
                {
                    "schema": VERIFY_SCHEMA,
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
