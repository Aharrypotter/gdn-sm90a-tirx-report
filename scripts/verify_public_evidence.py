#!/usr/bin/env python3
# Copyright 2026 Hongyi Wu
# Licensed under the Apache License, Version 2.0.
"""Verify a derived public evidence bundle without access to private inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any

ROW_ORDER = (
    "single-t512-h8-mha-zero",
    "single-t1024-h8-mha-state",
    "single-t1024-h8-hv16-gva-state",
    "single-t4096-h16-mha-zero",
    "packed-n10-t4096-h8-mha-state",
    "packed-n20-t8192-h8-hv16-gva-state",
)
IMPLEMENTATION_ORDER = ("tirx", "cutedsl", "fla")
FORBIDDEN_JSON_KEYS = {
    "cuda_cache_env",
    "gpu_state_before",
    "gpu_state_after",
    "gpu_uuid",
    "python_executable",
    "run_root",
    "container",
    "container_id",
    "host",
    "hostname",
    "pid",
    "process_id",
    "owned_processes",
    "trace",
    "tvm_path",
}
FORBIDDEN_SUFFIXES = {
    ".cubin",
    ".sass",
    ".ptx",
    ".log",
    ".pid",
    ".rc",
    ".done",
}
SENSITIVE_TEXT_PATTERNS = {
    "macOS user path": re.compile(rb"/Users/[^/\s]+/"),
    "Linux home path": re.compile(rb"/home/[^/\s]+/"),
    "remote workspace path": re.compile(rb"/workspace(?:/|\b)"),
    "GPU UUID": re.compile(rb"GPU-[0-9a-fA-F-]{12,}"),
    "GitHub token": re.compile(rb"(?:github_" + rb"pat_|gh" + rb"p_)[A-Za-z0-9_]+"),
    "private key": re.compile(rb"BEGIN (?:RSA |OPENSSH )?PRIVATE KEY"),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def walk_keys(value: Any, path: str = "$") -> list[str]:
    errors = []
    if isinstance(value, dict):
        for key, child in value.items():
            if key in FORBIDDEN_JSON_KEYS:
                errors.append(f"{path}.{key}: forbidden private-field key")
            errors.extend(walk_keys(child, f"{path}.{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            errors.extend(walk_keys(child, f"{path}[{index}]"))
    return errors


def verify_manifest(root: Path) -> list[str]:
    errors = []
    manifest_path = root / "manifest.json"
    seal_path = root / "MANIFEST.sha256"
    expected_line = seal_path.read_text()
    expected_hash, expected_name = expected_line.rstrip("\n").split("  ", 1)
    if expected_name != "manifest.json":
        errors.append("MANIFEST.sha256 must seal manifest.json")
    if sha256_file(manifest_path) != expected_hash:
        errors.append("manifest.json digest does not match MANIFEST.sha256")
    manifest = load_json(manifest_path)
    if manifest.get("schema") != "gdn-sm90a.public-evidence-manifest.v1":
        errors.append("unexpected manifest schema")
    rows = manifest.get("files", [])
    if manifest.get("file_count") != len(rows):
        errors.append("manifest file_count mismatch")
    declared_paths = [row["path"] for row in rows]
    if declared_paths != sorted(declared_paths):
        errors.append("manifest paths are not sorted")
    if len(declared_paths) != len(set(declared_paths)):
        errors.append("manifest has duplicate paths")
    actual_paths = sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
        and path.relative_to(root).as_posix() not in {"manifest.json", "MANIFEST.sha256"}
    )
    if declared_paths != actual_paths:
        errors.append("manifest membership does not match payload files")
    for row in rows:
        path = root / row["path"]
        if not path.is_file():
            errors.append(f"{row['path']}: missing")
            continue
        if sha256_file(path) != row["sha256"]:
            errors.append(f"{row['path']}: digest mismatch")
        if path.stat().st_size != row["size_bytes"]:
            errors.append(f"{row['path']}: size mismatch")
    return errors


def parse_receipts(path: Path) -> list[dict[str, Any]]:
    receipts = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        value = json.loads(line)
        if not isinstance(value, dict):
            raise TypeError(f"timing receipt line {line_number}: expected object")
        receipts.append(value)
    return receipts


def verify_receipts_and_performance(root: Path) -> list[str]:
    errors = []
    receipts = parse_receipts(root / "receipts/timing.jsonl")
    if len(receipts) != 66:
        errors.append(f"expected 66 timing receipts, found {len(receipts)}")
        return errors
    seen: set[tuple[str, str, int]] = set()
    for receipt in receipts:
        key = (
            receipt.get("row_id"),
            receipt.get("implementation"),
            receipt.get("process_index"),
        )
        if key in seen:
            errors.append(f"duplicate timing identity {key}")
        seen.add(key)
        if receipt.get("status") != "timing_ok":
            errors.append(f"{key}: non-passing timing status")
        if receipt.get("correctness_passed") is not True:
            errors.append(f"{key}: correctness did not pass")
        samples = receipt.get("raw_per_iter_ms", [])
        if len(samples) != 100:
            errors.append(f"{key}: expected 100 raw timing samples")
            continue
        if not all(
            isinstance(value, (int, float)) and math.isfinite(value) and value >= 0
            for value in samples
        ):
            errors.append(f"{key}: invalid raw timing sample")
        if not math.isclose(
            statistics.fmean(samples),
            receipt["summary_ms"]["average_ms"],
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            errors.append(f"{key}: average does not match raw timing samples")

    expected_counts = {
        (row, impl): 7 if row == "packed-n10-t4096-h8-mha-state" else 3
        for row in ROW_ORDER
        for impl in IMPLEMENTATION_ORDER
    }
    observed_counts = {
        key: sum(
            receipt["row_id"] == key[0] and receipt["implementation"] == key[1]
            for receipt in receipts
        )
        for key in expected_counts
    }
    if observed_counts != expected_counts:
        errors.append("per-row/per-implementation receipt counts mismatch")

    performance = load_json(root / "results/performance.json")
    ratios_cutedsl = []
    ratios_fla = []
    for row in ROW_ORDER:
        summary = performance["rows"][row]
        for impl in IMPLEMENTATION_ORDER:
            averages = [
                receipt["summary_ms"]["average_ms"]
                for receipt in receipts
                if receipt["row_id"] == row and receipt["implementation"] == impl
            ]
            if averages != summary["process_averages_ms"][impl]:
                errors.append(f"{row}/{impl}: process averages mismatch")
            if not math.isclose(
                statistics.median(averages),
                summary["median_ms"][impl],
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                errors.append(f"{row}/{impl}: median mismatch")
        ratio_cutedsl = summary["median_ms"]["tirx"] / summary["median_ms"]["cutedsl"]
        ratio_fla = summary["median_ms"]["tirx"] / summary["median_ms"]["fla"]
        if not math.isclose(ratio_cutedsl, summary["tirx_over_cutedsl"], rel_tol=1e-12):
            errors.append(f"{row}: TIRx/CuTeDSL ratio mismatch")
        if not math.isclose(ratio_fla, summary["tirx_over_fla"], rel_tol=1e-12):
            errors.append(f"{row}: TIRx/FLA ratio mismatch")
        ratios_cutedsl.append(summary["tirx_over_cutedsl"])
        ratios_fla.append(summary["tirx_over_fla"])
    geomean_cutedsl = math.exp(statistics.fmean(math.log(value) for value in ratios_cutedsl))
    geomean_fla = math.exp(statistics.fmean(math.log(value) for value in ratios_fla))
    if not math.isclose(
        geomean_cutedsl,
        performance["primary_geomean"]["tirx_over_cutedsl"],
        rel_tol=1e-12,
    ):
        errors.append("TIRx/CuTeDSL geomean mismatch")
    if not math.isclose(
        geomean_fla,
        performance["primary_geomean"]["tirx_over_fla"],
        rel_tol=1e-12,
    ):
        errors.append("TIRx/FLA geomean mismatch")
    return errors


def verify_disclosure(root: Path) -> list[str]:
    errors = []
    for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        relative = path.relative_to(root).as_posix()
        if path.suffix in FORBIDDEN_SUFFIXES:
            errors.append(f"{relative}: forbidden raw-artifact suffix")
        payload = path.read_bytes()
        for label, pattern in SENSITIVE_TEXT_PATTERNS.items():
            if pattern.search(payload):
                errors.append(f"{relative}: contains {label}")
        if path.suffix == ".json":
            errors.extend(f"{relative}:{error}" for error in walk_keys(load_json(path)))
        elif path.suffix == ".jsonl":
            for line_number, line in enumerate(path.read_text().splitlines(), 1):
                errors.extend(
                    f"{relative}:{line_number}:{error}" for error in walk_keys(json.loads(line))
                )
    return errors


def verify_status(root: Path) -> list[str]:
    errors = []
    publication = load_json(root / "PUBLICATION.json")
    decision = load_json(root / "results/release-decision.json")
    source_lock = load_json(root / "metadata/source-lock.json")
    if publication.get("package_status") != "HISTORICAL_EVIDENCE_BOUND":
        errors.append("publication package status was promoted beyond historical evidence")
    if publication.get("fresh_public_tag_rerun_status") != "REQUIRED":
        errors.append("publication does not require a fresh public-tag rerun")
    if decision.get("fresh_public_tag_rerun_required") is not True:
        errors.append("release decision does not require a fresh public-tag rerun")
    if decision.get("upstream_merge") is not False:
        errors.append("release decision incorrectly claims an upstream merge")
    if source_lock.get("evidence_class") != "HISTORICAL_EVIDENCE_BOUND":
        errors.append("source lock evidence class mismatch")
    if source_lock["implementation_mapping"]["cutedsl"]["mapping_status"] != (
        "COMMIT_ENTRYPOINT_BACKEND_AND_DSL_VERSION_BOUND"
    ):
        errors.append("CuTeDSL source qualification does not match the receipt audit")
    superseded = source_lock.get("superseded_artifacts", {}).get("gdn2-sm90a-comparator-r0", {})
    if superseded.get("status") != "NOT_USED_BY_HISTORICAL_GDN_RECEIPTS":
        errors.append("superseded GDN2 comparator artifact is not explicitly excluded")
    return errors


def verify(root: Path) -> dict[str, Any]:
    errors = []
    errors.extend(verify_manifest(root))
    errors.extend(verify_receipts_and_performance(root))
    errors.extend(verify_disclosure(root))
    errors.extend(verify_status(root))
    return {
        "schema": "gdn-sm90a.public-evidence-verification.v1",
        "status": "PASS" if not errors else "FAIL",
        "error_count": len(errors),
        "errors": errors,
        "manifest_sha256": sha256_file(root / "manifest.json"),
        "receipt_count": len(parse_receipts(root / "receipts/timing.jsonl")),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True, type=Path)
    args = parser.parse_args()
    result = verify(args.bundle.resolve())
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
