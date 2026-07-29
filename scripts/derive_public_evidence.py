#!/usr/bin/env python3
# Copyright 2026 Hongyi Wu
# Licensed under the Apache License, Version 2.0.
"""Derive a deterministic, allowlisted public bundle from the sealed A46-S3 evidence.

This program intentionally reconstructs every public object field by field.  It
does not copy raw logs and it does not apply regex-based redaction to private
artifacts.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
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
ORIGINAL_ROOTS = ("contract", "logs", "results", "tools")
ORIGINAL_MANIFEST_EXCLUSIONS = {
    "results/artifact-manifest.sha256",
    "results/artifact-verification-remote.json",
    "results/artifact-verification-local.json",
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"{path}: expected a JSON object")
    return value


def require_keys(value: dict[str, Any], keys: tuple[str, ...], context: str) -> None:
    missing = [key for key in keys if key not in value]
    if missing:
        raise KeyError(f"{context}: missing required fields {missing}")


def selected(value: dict[str, Any], keys: tuple[str, ...], context: str) -> dict[str, Any]:
    require_keys(value, keys, context)
    return {key: copy.deepcopy(value[key]) for key in keys}


def write_json(root: Path, relative: str, value: Any) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.write_text(payload)


def write_text(root: Path, relative: str, value: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value)


def original_eligible(root: Path) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for directory in ORIGINAL_ROOTS:
        base = root / directory
        if not base.is_dir():
            continue
        for path in base.rglob("*"):
            if not path.is_file():
                continue
            relative = path.relative_to(root).as_posix()
            if (
                relative in ORIGINAL_MANIFEST_EXCLUSIONS
                or "__pycache__" in path.parts
                or path.suffix == ".pyc"
                or path.name.endswith(".tmp")
            ):
                continue
            paths[relative] = path
    return paths


def verify_original_seal(root: Path, expected: dict[str, Any]) -> dict[str, Any]:
    manifest = root / "results/artifact-manifest.sha256"
    actual_manifest_sha256 = sha256_file(manifest)
    if actual_manifest_sha256 != expected["artifact_manifest_sha256"]:
        raise ValueError(
            "original artifact manifest digest mismatch: "
            f"{actual_manifest_sha256} != {expected['artifact_manifest_sha256']}"
        )

    declared: dict[str, str] = {}
    for line_number, line in enumerate(manifest.read_text().splitlines(), 1):
        try:
            digest, relative = line.split("  ", 1)
        except ValueError as err:
            raise ValueError(f"malformed original manifest line {line_number}") from err
        declared[relative] = digest

    current = original_eligible(root)
    if set(declared) != set(current):
        missing = sorted(set(declared) - set(current))
        extra = sorted(set(current) - set(declared))
        raise ValueError(f"original seal membership mismatch: missing={missing}, extra={extra}")
    mismatches = [
        relative
        for relative, digest in declared.items()
        if sha256_file(current[relative]) != digest
    ]
    if mismatches:
        raise ValueError(f"original seal content mismatch: {mismatches}")
    if len(declared) != expected["artifact_file_count"]:
        raise ValueError(
            f"original seal count mismatch: {len(declared)} != {expected['artifact_file_count']}"
        )

    source_bundle = root / "contract/source-bundle.json"
    decision = root / "results/release-reseal-decision.json"
    if sha256_file(source_bundle) != expected["source_bundle_sha256"]:
        raise ValueError("original source-bundle digest mismatch")
    if sha256_file(decision) != expected["release_decision_sha256"]:
        raise ValueError("original release-decision digest mismatch")
    return {
        "status": "PASS",
        "file_count": len(declared),
        "artifact_manifest_sha256": actual_manifest_sha256,
        "source_bundle_sha256": sha256_file(source_bundle),
        "release_decision_sha256": sha256_file(decision),
    }


def derive_source_lock(
    config: dict[str, Any],
    benchmark_contract: dict[str, Any],
) -> dict[str, Any]:
    implementations = benchmark_contract["implementations"]
    return {
        "schema": "gdn-sm90a.public-evidence-source-lock.v1",
        "evidence_class": "HISTORICAL_EVIDENCE_BOUND",
        "historical_seal": copy.deepcopy(config["historical_seal"]),
        "public_sources": copy.deepcopy(config["repositories"]),
        "licenses": copy.deepcopy(config["licenses"]),
        "signature_status": copy.deepcopy(config["signature_status"]),
        "upstream_pr_audit": copy.deepcopy(config["upstream_pr_audit"]),
        "aggregate_definition": copy.deepcopy(config["aggregate_definition"]),
        "implementation_mapping": {
            "tirx": {
                "historical_declared_source_identity": implementations["tirx"]["source_identity"],
                "historical_backend_identity": implementations["tirx"]["backend_identity"],
                "public_source_key": "tvm + tirx_kernels",
                "mapping_status": "BYTE_MAPPED_TO_PUBLIC_RUNTIME_DELTAS",
            },
            "cutedsl": {
                "historical_declared_source_identity": implementations["cutedsl"][
                    "source_identity"
                ],
                "historical_backend_identity": implementations["cutedsl"]["backend_identity"],
                "public_source_key": "cutedsl_comparator",
                "mapping_status": "COMMIT_ENTRYPOINT_BACKEND_AND_DSL_VERSION_BOUND",
            },
            "fla": {
                "historical_declared_source_identity": implementations["fla"]["source_identity"],
                "historical_backend_identity": implementations["fla"]["backend_identity"],
                "public_source_key": "fla",
                "mapping_status": "COMMIT_DECLARED",
            },
        },
        "required_reseal": "FRESH_PUBLIC_TAG_66_RECEIPT_RERUN",
        "superseded_artifacts": copy.deepcopy(config["superseded_artifacts"]),
        "disclaimers": copy.deepcopy(config["disclaimers"]),
    }


def derive_benchmark_contract(source: dict[str, Any]) -> dict[str, Any]:
    timing = selected(
        source["timing"],
        (
            "max_gpu_util_pct",
            "quiet_timeout_s",
            "quiet_poll_interval_s",
            "warmup_iters",
            "timed_iters",
            "base_processes",
            "escalated_processes",
            "noise_band_pct",
        ),
        "benchmark timing",
    )
    implementations: dict[str, Any] = {}
    for name in IMPLEMENTATION_ORDER:
        implementation = source["implementations"][name]
        implementations[name] = {
            "historical_declared_source_identity": implementation["source_identity"],
            "backend_identity": implementation["backend_identity"],
        }
    rows = []
    for row in source["rows"]:
        rows.append(
            selected(
                row,
                (
                    "row_id",
                    "sequence_lengths",
                    "q_heads",
                    "v_heads",
                    "scale",
                    "seed",
                    "stateful",
                    "primary",
                    "critical",
                    "expected_backends",
                ),
                f"benchmark row {row.get('row_id')}",
            )
        )
    if tuple(row["row_id"] for row in rows) != ROW_ORDER:
        raise ValueError("benchmark row order or membership does not match the frozen six-row set")
    return {
        "schema": "gdn-sm90a.public-benchmark-contract.v1",
        "historical_schema": source["schema"],
        "historical_run_id": source["run_id"],
        "historical_source_manifest_sha256": source["source_manifest_sha256"],
        "implementations": implementations,
        "correctness": copy.deepcopy(source["correctness"]),
        "timing": timing,
        "rows": rows,
        "device_binding": {
            "accelerator": "NVIDIA H20",
            "physical_device_binding_verified": True,
            "private_device_identifiers_published": False,
        },
    }


def derive_receipts(
    source_root: Path,
    benchmark: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    paths = list((source_root / "results/benchmark-full/timing").glob("*.json"))
    if len(paths) != 66:
        raise ValueError(f"expected exactly 66 timing receipts, found {len(paths)}")

    row_rank = {row: index for index, row in enumerate(ROW_ORDER)}
    impl_rank = {impl: index for index, impl in enumerate(IMPLEMENTATION_ORDER)}
    source_records: list[tuple[dict[str, Any], Path]] = []
    for path in paths:
        value = load_json(path)
        source_records.append((value, path))
    source_records.sort(
        key=lambda item: (
            row_rank[item[0]["row_id"]],
            impl_rank[item[0]["impl"]],
            item[0]["process_index"],
        )
    )

    expected_run_id = benchmark["historical_run_id"]
    expected_contract = None
    seen: set[tuple[str, str, int]] = set()
    counts = {impl: 0 for impl in IMPLEMENTATION_ORDER}
    package_versions: dict[str, Any] = {}
    torch_versions: set[tuple[str, str]] = set()
    public_receipts = []
    for source, path in source_records:
        require_keys(
            source,
            (
                "schema",
                "run_id",
                "contract_sha256",
                "row_id",
                "impl",
                "process_index",
                "execution_order",
                "input_seed",
                "source_identity",
                "resolved_backend_identity",
                "wrapped_callable_chain",
                "oracle_sha256",
                "output_hash",
                "correctness_passed",
                "output_metrics",
                "state_hash",
                "state_metrics",
                "status",
                "warmup_iters",
                "timed_iters",
                "timer",
                "raw_per_iter_ms",
                "summary",
                "compile_first_call_ms",
                "gpu_quiet_gate",
                "package_versions",
                "torch_version",
                "torch_cuda_version",
            ),
            path.name,
        )
        if source["row_id"] not in row_rank or source["impl"] not in impl_rank:
            raise ValueError(f"{path.name}: unknown row or implementation")
        key = (source["row_id"], source["impl"], source["process_index"])
        if key in seen:
            raise ValueError(f"duplicate receipt identity: {key}")
        seen.add(key)
        if source["run_id"] != expected_run_id:
            raise ValueError(f"{path.name}: run id mismatch")
        expected_contract = expected_contract or source["contract_sha256"]
        if source["contract_sha256"] != expected_contract:
            raise ValueError(f"{path.name}: benchmark contract digest mismatch")
        if source["status"] != "timing_ok" or not source["correctness_passed"]:
            raise ValueError(f"{path.name}: receipt is not a passing timing receipt")
        if source["warmup_iters"] != 20 or source["timed_iters"] != 100:
            raise ValueError(f"{path.name}: unexpected iteration protocol")
        if source["timer"] != "cuda_event":
            raise ValueError(f"{path.name}: unexpected timer")
        if len(source["raw_per_iter_ms"]) != source["timed_iters"]:
            raise ValueError(f"{path.name}: timing sample count mismatch")
        if not all(
            isinstance(value, (int, float)) and math.isfinite(value) and value >= 0
            for value in source["raw_per_iter_ms"]
        ):
            raise ValueError(f"{path.name}: invalid timing sample")
        recomputed_average = statistics.fmean(source["raw_per_iter_ms"])
        if not math.isclose(
            recomputed_average,
            source["summary"]["average_ms"],
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError(f"{path.name}: receipt average is not reproducible")

        counts[source["impl"]] += 1
        package_versions.setdefault(source["impl"], source["package_versions"])
        if package_versions[source["impl"]] != source["package_versions"]:
            raise ValueError(f"{path.name}: package versions drift within implementation")
        torch_versions.add((source["torch_version"], source["torch_cuda_version"]))

        source_lock_key = {
            "tirx": "tvm + tirx_kernels",
            "cutedsl": "cutedsl_comparator",
            "fla": "fla",
        }[source["impl"]]
        public_receipts.append(
            {
                "schema": "gdn-sm90a.public-timing-receipt.v1",
                "historical_schema": source["schema"],
                "historical_run_id": source["run_id"],
                "contract_sha256": source["contract_sha256"],
                "row_id": source["row_id"],
                "implementation": source["impl"],
                "process_index": source["process_index"],
                "execution_order": source["execution_order"],
                "input_seed": source["input_seed"],
                "historical_declared_source_identity": source["source_identity"],
                "public_source_lock_key": source_lock_key,
                "resolved_backend_identity": source["resolved_backend_identity"],
                "wrapped_callable_chain": copy.deepcopy(source["wrapped_callable_chain"]),
                "oracle_sha256": source["oracle_sha256"],
                "output_sha256": source["output_hash"],
                "state_sha256": source["state_hash"],
                "correctness_passed": source["correctness_passed"],
                "output_metrics": selected(
                    source["output_metrics"],
                    ("allclose", "max_abs", "present_match", "relative_rms"),
                    f"{path.name} output metrics",
                ),
                "state_metrics": selected(
                    source["state_metrics"],
                    ("allclose", "max_abs", "present_match", "relative_rms"),
                    f"{path.name} state metrics",
                ),
                "status": source["status"],
                "warmup_iters": source["warmup_iters"],
                "timed_iters": source["timed_iters"],
                "timer": source["timer"],
                "raw_per_iter_ms": copy.deepcopy(source["raw_per_iter_ms"]),
                "summary_ms": selected(
                    source["summary"],
                    ("average_ms", "median_ms", "min_ms", "max_ms"),
                    f"{path.name} summary",
                ),
                "compile_first_call_ms": source["compile_first_call_ms"],
                "compile_first_call_role": "diagnostic_only",
                "gpu_quiet_gate": selected(
                    source["gpu_quiet_gate"],
                    (
                        "elapsed_s",
                        "max_observed_util_pct",
                        "poll_interval_s",
                        "polls",
                        "timeout_s",
                    ),
                    f"{path.name} quiet gate",
                ),
                "source_receipt_sha256": sha256_file(path),
            }
        )

    if counts != {"tirx": 22, "cutedsl": 22, "fla": 22}:
        raise ValueError(f"unexpected implementation receipt counts: {counts}")
    for row in ROW_ORDER:
        expected = 7 if row == "packed-n10-t4096-h8-mha-state" else 3
        for impl in IMPLEMENTATION_ORDER:
            observed = sum(
                receipt["row_id"] == row and receipt["implementation"] == impl
                for receipt in public_receipts
            )
            if observed != expected:
                raise ValueError(f"{row}/{impl}: expected {expected} receipts, found {observed}")
    if len(torch_versions) != 1:
        raise ValueError(f"PyTorch/CUDA version drift across receipts: {torch_versions}")

    software = {
        "torch_version": next(iter(torch_versions))[0],
        "torch_cuda_version": next(iter(torch_versions))[1],
        "package_versions_by_implementation": package_versions,
    }
    return public_receipts, software


def derive_performance(
    report: dict[str, Any],
    gates: dict[str, Any],
    receipts: list[dict[str, Any]],
) -> dict[str, Any]:
    report_rows: dict[str, Any] = {}
    ratios_cutedsl = []
    ratios_fla = []
    for row_id in ROW_ORDER:
        row = report["rows"][row_id]
        public_row = selected(
            row,
            (
                "critical",
                "primary",
                "n_processes",
                "median_ms",
                "process_averages_ms",
                "tirx_over_cutedsl",
                "tirx_over_fla",
                "oracle_sha256",
            ),
            f"performance row {row_id}",
        )
        for impl in IMPLEMENTATION_ORDER:
            receipt_averages = [
                receipt["summary_ms"]["average_ms"]
                for receipt in receipts
                if receipt["row_id"] == row_id and receipt["implementation"] == impl
            ]
            if public_row["process_averages_ms"][impl] != receipt_averages:
                raise ValueError(f"{row_id}/{impl}: report process averages do not match receipts")
            if not math.isclose(
                statistics.median(receipt_averages),
                public_row["median_ms"][impl],
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise ValueError(f"{row_id}/{impl}: report row median is not reproducible")
        expected_cutedsl = public_row["median_ms"]["tirx"] / public_row["median_ms"]["cutedsl"]
        expected_fla = public_row["median_ms"]["tirx"] / public_row["median_ms"]["fla"]
        if not math.isclose(expected_cutedsl, public_row["tirx_over_cutedsl"], rel_tol=1e-12):
            raise ValueError(f"{row_id}: CuTeDSL ratio mismatch")
        if not math.isclose(expected_fla, public_row["tirx_over_fla"], rel_tol=1e-12):
            raise ValueError(f"{row_id}: FLA ratio mismatch")
        ratios_cutedsl.append(public_row["tirx_over_cutedsl"])
        ratios_fla.append(public_row["tirx_over_fla"])
        report_rows[row_id] = public_row

    geomean_cutedsl = math.exp(statistics.fmean(math.log(value) for value in ratios_cutedsl))
    geomean_fla = math.exp(statistics.fmean(math.log(value) for value in ratios_fla))
    if not math.isclose(
        geomean_cutedsl,
        report["primary_geomean"]["tirx_over_cutedsl"],
        rel_tol=1e-12,
    ):
        raise ValueError("CuTeDSL geomean mismatch")
    if not math.isclose(
        geomean_fla,
        report["primary_geomean"]["tirx_over_fla"],
        rel_tol=1e-12,
    ):
        raise ValueError("FLA geomean mismatch")

    accepted = selected(
        gates["accepted_incumbent_evidence"],
        (
            "decision",
            "report_sha256",
            "run_id",
            "candidate_over_incumbent_every_row_max",
            "candidate_over_incumbent_geomean",
            "claim_boundary",
        ),
        "accepted incumbent evidence",
    )
    return {
        "schema": "gdn-sm90a.public-performance-summary.v1",
        "historical_schema": report["schema"],
        "evidence_class": "HISTORICAL_EVIDENCE_BOUND",
        "historical_run_id": report["run_id"],
        "status": report["status"],
        "decision_status": report["decision_status"],
        "errors": copy.deepcopy(report["errors"]),
        "escalation_needed": copy.deepcopy(report["escalation_needed"]),
        "ratio_direction": report["ratio_direction"],
        "receipt_count": report["receipt_count"],
        "unique_cache_count": report["unique_cache_count"],
        "primary_geomean": copy.deepcopy(report["primary_geomean"]),
        "rows": report_rows,
        "gate_evaluation": {
            "schema": gates["schema"],
            "stage": gates["stage"],
            "status": gates["status"],
            "decision": gates["decision"],
            "contract_sha256": gates["contract_sha256"],
            "report_sha256": gates["report_sha256"],
            "receipt_count": gates["receipt_count"],
            "gates": copy.deepcopy(gates["gates"]),
            "primary_geomean": copy.deepcopy(gates["primary_geomean"]),
            "ratios": copy.deepcopy(gates["ratios"]),
            "accepted_incumbent_evidence": accepted,
        },
        "packed_n10_interpretation": {
            "tirx_over_cutedsl": report_rows["packed-n10-t4096-h8-mha-state"]["tirx_over_cutedsl"],
            "noise_band_pct": 2.0,
            "status": "WITHIN_PREREGISTERED_NOISE_BAND_NOT_A_SPEED_WIN",
        },
    }


def derive_correctness(
    benchmark_contract: dict[str, Any],
    receipts: list[dict[str, Any]],
    release_summary: dict[str, Any],
    formal_gates: dict[str, Any],
) -> dict[str, Any]:
    def max_optional(values: list[Any]) -> Any:
        present = [value for value in values if value is not None]
        return max(present) if present else None

    def all_optional(values: list[Any]) -> Any:
        present = [value for value in values if value is not None]
        return all(present) if present else None

    per_row: dict[str, Any] = {}
    for row in ROW_ORDER:
        per_row[row] = {}
        for impl in IMPLEMENTATION_ORDER:
            selected_receipts = [
                receipt
                for receipt in receipts
                if receipt["row_id"] == row and receipt["implementation"] == impl
            ]
            per_row[row][impl] = {
                "receipt_count": len(selected_receipts),
                "all_correctness_passed": all(
                    receipt["correctness_passed"] for receipt in selected_receipts
                ),
                "output_allclose": all(
                    receipt["output_metrics"]["allclose"] for receipt in selected_receipts
                ),
                "output_max_abs_max": max(
                    receipt["output_metrics"]["max_abs"] for receipt in selected_receipts
                ),
                "output_relative_rms_max": max(
                    receipt["output_metrics"]["relative_rms"] for receipt in selected_receipts
                ),
                "state_allclose": all_optional(
                    [receipt["state_metrics"]["allclose"] for receipt in selected_receipts]
                ),
                "state_max_abs_max": max_optional(
                    [receipt["state_metrics"]["max_abs"] for receipt in selected_receipts]
                ),
                "state_relative_rms_max": max_optional(
                    [receipt["state_metrics"]["relative_rms"] for receipt in selected_receipts]
                ),
            }
    receipt_keys = ("compiler-tests-full", "public-route-tests", "public-gpu-semantics")
    formal_receipts = {}
    for key in receipt_keys:
        value = formal_gates["rc_receipts"][key]
        formal_receipts[key] = selected(value, ("rc", "sha256"), f"formal receipt {key}")
    return {
        "schema": "gdn-sm90a.public-correctness-summary.v1",
        "status": "PASS",
        "policy": copy.deepcopy(benchmark_contract["correctness"]),
        "receipt_count": len(receipts),
        "all_receipts_correctness_passed": all(
            receipt["correctness_passed"] for receipt in receipts
        ),
        "receipt_count_by_implementation": {
            impl: sum(receipt["implementation"] == impl for receipt in receipts)
            for impl in IMPLEMENTATION_ORDER
        },
        "per_row": per_row,
        "fresh_gate_status": selected(
            release_summary["fresh_gates"],
            (
                "formal_gates_status",
                "compiler_tests",
                "public_route_tests",
                "public_gpu_semantics",
            ),
            "release correctness gates",
        ),
        "formal_gate_receipts": formal_receipts,
    }


def derive_safety(
    release_summary: dict[str, Any],
    host_sync: dict[str, Any],
    redzone: dict[str, Any],
    formal_gates: dict[str, Any],
) -> dict[str, Any]:
    host_sync_public = selected(
        host_sync,
        (
            "schema",
            "status",
            "claim_boundary",
            "contract",
            "evidence_completeness",
            "errors",
            "evidence_gaps",
            "findings",
            "observed",
            "violations",
        ),
        "host sync analysis",
    )
    redzone_public = selected(
        redzone,
        (
            "schema",
            "status",
            "route",
            "adjacent_sequence_boundary_count",
            "guard_elements",
            "guards",
            "inputs_immutable",
        ),
        "redzone",
    )
    receipt_keys = (
        "memcheck-pipeline-boundary-r1",
        "memcheck-short-packed-r1",
        "memcheck-tail-single-r1",
        "memcheck-tail-packed-r1",
        "racecheck-tail-packed-r1",
        "synccheck-tail-packed-r1",
        "safety-pipeline-boundary",
        "safety-short-packed",
        "safety-tail-single",
        "safety-tail-packed",
        "sanitizer-r1",
        "host-sync-analysis",
        "host-sync-trace",
        "redzone",
    )
    receipts = {
        key: selected(formal_gates["rc_receipts"][key], ("rc", "sha256"), key)
        for key in receipt_keys
    }
    fresh_gates = release_summary["fresh_gates"]
    return {
        "schema": "gdn-sm90a.public-safety-summary.v1",
        "status": "PASS",
        "host_sync": host_sync_public,
        "redzone": redzone_public,
        "sanitizer_summary": copy.deepcopy(fresh_gates["sanitizer"]),
        "release_gate_summaries": {
            "host_sync": copy.deepcopy(fresh_gates["host_sync"]),
            "redzone": copy.deepcopy(fresh_gates["redzone"]),
            "sanitizer": copy.deepcopy(fresh_gates["sanitizer"]),
        },
        "formal_receipts": receipts,
        "claim_boundary": (
            "Passing checks apply to the frozen six-row product routes and do not constitute "
            "a formal proof for arbitrary inputs."
        ),
    }


def derive_codegen(source_root: Path, release_summary: dict[str, Any]) -> dict[str, Any]:
    resource_keys = (
        "schema",
        "status",
        "target",
        "stage",
        "registers_max",
        "shared_memory_bytes_max",
        "spill_load_bytes_max",
        "spill_store_bytes_max",
        "stack_frame_bytes_max",
        "sass_ldl_instructions",
        "sass_stl_instructions",
        "barriers_max",
        "setmaxnreg_resource_warning",
        "wgmma_serialization_warning",
        "cuda_source_sha256",
        "cubin_sha256",
        "source_manifest_digest",
    )
    resources = []
    base = source_root / "results/codegen"
    for path in sorted(base.glob("**/resource-inventory.json")):
        value = load_json(path)
        relative = path.relative_to(base)
        resources.append(
            {
                "case": relative.parts[0],
                **selected(value, resource_keys, relative.as_posix()),
            }
        )
    if len(resources) != 11:
        raise ValueError(f"expected 11 codegen stage inventories, found {len(resources)}")
    comparison = load_json(source_root / "results/codegen-comparison.json")
    comparison_public = selected(
        comparison,
        (
            "schema",
            "status",
            "source_bundle_sha256",
            "formal_audit_sha256",
            "formal_case_count",
            "formal_stage_count",
            "formal_resource_gate",
            "accepted_comparisons",
            "timing_inheritance",
        ),
        "codegen comparison",
    )
    return {
        "schema": "gdn-sm90a.public-codegen-resources.v1",
        "status": release_summary["fresh_gates"]["codegen"]["status"],
        "summary": copy.deepcopy(release_summary["fresh_gates"]["codegen"]),
        "stage_count": len(resources),
        "resources": resources,
        "comparison": comparison_public,
    }


def derive_controls(
    source_root: Path,
    seal: dict[str, Any],
    report: dict[str, Any],
    gates: dict[str, Any],
    decision: dict[str, Any],
) -> dict[str, Any]:
    verification_summaries = {}
    for name in ("local", "remote"):
        value = load_json(source_root / f"results/artifact-verification-{name}.json")
        verification_summaries[name] = selected(
            value,
            ("schema", "status", "file_count", "manifest_sha256", "errors", "unexpected_files"),
            f"artifact verification {name}",
        )
    sanitizer = load_json(source_root / "results/sanitizer-r0-supersession.json")
    sanitizer_public = selected(
        sanitizer,
        (
            "schema",
            "status",
            "attempt",
            "elapsed_seconds_before_termination",
            "completed_checks",
            "failed_check",
            "superseding_attempt",
            "product_source_changed_between_attempts",
        ),
        "sanitizer supersession",
    )
    diagnosis = selected(
        sanitizer["diagnosis"],
        (
            "classification",
            "compute_sanitizer_target_processes",
            "descendant_state",
            "kernel_error_observed",
        ),
        "sanitizer diagnosis",
    )
    sanitizer_public["diagnosis"] = diagnosis
    return {
        "schema": "gdn-sm90a.public-controls.v1",
        "status": "PASS",
        "original_seal_verification": seal,
        "artifact_verification": verification_summaries,
        "timing_controls": {
            "errors": copy.deepcopy(report["errors"]),
            "escalation_needed": copy.deepcopy(report["escalation_needed"]),
            "receipt_count": report["receipt_count"],
            "unique_cache_count": report["unique_cache_count"],
            "gates": copy.deepcopy(gates["gates"]),
        },
        "sanitizer_control_failure_supersession": sanitizer_public,
        "release_controls": selected(
            decision,
            (
                "full_canonical_rerun_passed",
                "codegen_inheritance_eligible",
                "sentinel_inheritance_eligible",
            ),
            "release controls",
        ),
    }


def derive_release_decision(
    decision: dict[str, Any],
    release_summary: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema": "gdn-sm90a.public-release-decision.v1",
        "historical_schema": decision["schema"],
        "historical_release_ready": decision["release_ready"],
        "historical_release_decision": decision["decision"],
        "historical_claim_boundary": decision["claim_boundary"],
        "historical_errors": copy.deepcopy(decision["errors"]),
        "historical_fresh_gate_failures": copy.deepcopy(decision["fresh_gate_failures"]),
        "historical_full_canonical_rerun_passed": decision["full_canonical_rerun_passed"],
        "historical_codegen_inheritance_eligible": decision["codegen_inheritance_eligible"],
        "historical_sentinel_inheritance_eligible": decision["sentinel_inheritance_eligible"],
        "historical_timing_decision": "USE_FRESH_CANONICAL_TIMING",
        "public_package_status": "HISTORICAL_EVIDENCE_BOUND",
        "fresh_public_tag_rerun_required": True,
        "fresh_public_tag_rerun_status": "NOT_YET_PUBLISHED",
        "upstream_merge": False,
    }


def schema_documents() -> dict[str, dict[str, Any]]:
    sha_pattern = "^[0-9a-f]{64}$"
    return {
        "schemas/publication.schema.json": {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$id": "https://github.com/Aharrypotter/gdn-sm90a-tirx-report/schemas/publication-v1",
            "type": "object",
            "required": [
                "schema",
                "package_id",
                "package_status",
                "historical_release_status",
                "fresh_public_tag_rerun_status",
                "original_seal",
            ],
            "properties": {
                "schema": {"const": "gdn-sm90a.publication.v1"},
                "package_id": {"type": "string"},
                "package_status": {"const": "HISTORICAL_EVIDENCE_BOUND"},
                "historical_release_status": {"const": "RELEASE_READY"},
                "fresh_public_tag_rerun_status": {"const": "REQUIRED"},
                "original_seal": {
                    "type": "object",
                    "required": [
                        "artifact_manifest_sha256",
                        "artifact_file_count",
                        "source_bundle_sha256",
                        "release_decision_sha256",
                    ],
                    "properties": {
                        "artifact_manifest_sha256": {
                            "type": "string",
                            "pattern": sha_pattern,
                        },
                        "artifact_file_count": {"const": 380},
                        "source_bundle_sha256": {"type": "string", "pattern": sha_pattern},
                        "release_decision_sha256": {
                            "type": "string",
                            "pattern": sha_pattern,
                        },
                    },
                    "additionalProperties": False,
                },
            },
            "additionalProperties": True,
        },
        "schemas/timing-receipt.schema.json": {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$id": "https://github.com/Aharrypotter/gdn-sm90a-tirx-report/schemas/timing-receipt-v1",
            "type": "object",
            "required": [
                "schema",
                "row_id",
                "implementation",
                "process_index",
                "raw_per_iter_ms",
                "summary_ms",
                "source_receipt_sha256",
            ],
            "properties": {
                "schema": {"const": "gdn-sm90a.public-timing-receipt.v1"},
                "row_id": {"enum": list(ROW_ORDER)},
                "implementation": {"enum": list(IMPLEMENTATION_ORDER)},
                "process_index": {"type": "integer", "minimum": 0},
                "raw_per_iter_ms": {
                    "type": "array",
                    "minItems": 100,
                    "maxItems": 100,
                    "items": {"type": "number", "minimum": 0},
                },
                "summary_ms": {"type": "object"},
                "source_receipt_sha256": {"type": "string", "pattern": sha_pattern},
            },
            "additionalProperties": True,
        },
        "schemas/manifest.schema.json": {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$id": "https://github.com/Aharrypotter/gdn-sm90a-tirx-report/schemas/manifest-v1",
            "type": "object",
            "required": ["schema", "file_count", "files"],
            "properties": {
                "schema": {"const": "gdn-sm90a.public-evidence-manifest.v1"},
                "file_count": {"type": "integer", "minimum": 1},
                "files": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["path", "sha256", "size_bytes"],
                        "properties": {
                            "path": {"type": "string"},
                            "sha256": {"type": "string", "pattern": sha_pattern},
                            "size_bytes": {"type": "integer", "minimum": 0},
                        },
                        "additionalProperties": False,
                    },
                },
            },
            "additionalProperties": False,
        },
    }


def field_map() -> dict[str, Any]:
    return {
        "schema": "gdn-sm90a.public-field-map.v1",
        "method": "explicit_field_allowlist",
        "outputs": {
            "PUBLICATION.json": [
                {
                    "source": "results/release-reseal-decision.json",
                    "json_paths": ["$.release_ready", "$.decision"],
                },
                {
                    "source": "results/artifact-manifest.sha256",
                    "json_paths": [],
                    "role": "verified immutable seal",
                },
            ],
            "metadata/source-lock.json": [
                {
                    "source": "contract/benchmark-full.json",
                    "json_paths": ["$.implementations.*"],
                },
                {
                    "source": "config/public-source-lock.json",
                    "json_paths": ["$.*"],
                    "role": "live public Git identities audited separately",
                },
            ],
            "metadata/environment.json": [
                {
                    "source": "results/formal-identity-r1.json",
                    "json_paths": [
                        "$.status",
                        "$.target_kind",
                        "$.target_arch",
                        "$.torch_version",
                        "$.torch_cuda_version",
                        "$.gates.*",
                    ],
                },
                {
                    "source": "results/benchmark-full/timing/*.json",
                    "json_paths": ["$.package_versions", "$.torch_version", "$.torch_cuda_version"],
                },
            ],
            "contracts/product.json": [
                {
                    "source": "results/release-summary.json",
                    "json_paths": ["$.product_contract"],
                }
            ],
            "contracts/benchmark.json": [
                {
                    "source": "contract/benchmark-full.json",
                    "json_paths": [
                        "$.schema",
                        "$.run_id",
                        "$.source_manifest_sha256",
                        "$.implementations.*",
                        "$.correctness.*",
                        "$.timing except physical_gpu_index and gpu_uuid",
                        "$.rows[*]",
                    ],
                }
            ],
            "receipts/timing.jsonl": [
                {
                    "source": "results/benchmark-full/timing/*.json",
                    "json_paths": [
                        "$.identity fields",
                        "$.correctness fields",
                        "$.raw_per_iter_ms",
                        "$.summary",
                        "$.compile_first_call_ms",
                        "$.gpu_quiet_gate",
                    ],
                }
            ],
            "results/performance.json": [
                {
                    "source": "results/benchmark-full-report.json",
                    "json_paths": ["$.* except no excluded fields are present"],
                },
                {
                    "source": "results/benchmark-full-gates.json",
                    "json_paths": ["$.*"],
                },
            ],
            "results/correctness.json": [
                {
                    "source": "results/benchmark-full/timing/*.json",
                    "json_paths": ["$.correctness_passed", "$.output_metrics", "$.state_metrics"],
                }
            ],
            "results/safety.json": [
                {
                    "source": "results/host-sync-analysis.json",
                    "json_paths": ["$.* except $.trace"],
                },
                {"source": "results/redzone.json", "json_paths": ["$.*"]},
                {
                    "source": "results/release-summary.json",
                    "json_paths": [
                        "$.fresh_gates.host_sync",
                        "$.fresh_gates.redzone",
                        "$.fresh_gates.sanitizer",
                    ],
                },
            ],
            "results/codegen-resources.json": [
                {
                    "source": "results/codegen/**/resource-inventory.json",
                    "json_paths": ["explicit resource fields"],
                },
                {
                    "source": "results/codegen-comparison.json",
                    "json_paths": ["$.*"],
                },
            ],
            "results/controls.json": [
                {
                    "source": "results/artifact-verification-{local,remote}.json",
                    "json_paths": [
                        "$.schema",
                        "$.status",
                        "$.file_count",
                        "$.manifest_sha256",
                        "$.errors",
                        "$.unexpected_files",
                    ],
                }
            ],
            "results/release-decision.json": [
                {
                    "source": "results/release-reseal-decision.json",
                    "json_paths": ["$.* except $.identities"],
                }
            ],
        },
        "globally_excluded": [
            "raw logs and job scripts",
            "process identifiers and return-code marker files",
            "filesystem and cache roots",
            "host, container, SSH alias, and GPU UUID values",
            "raw profiler traces",
            "generated CUDA, cubin, SASS, and ptxas logs",
            "raw reviewer report",
        ],
    }


def seal_public_bundle(root: Path) -> None:
    payload_paths = sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.relative_to(root).as_posix() not in {"manifest.json", "MANIFEST.sha256"}
    )
    rows = [
        {
            "path": path.relative_to(root).as_posix(),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in payload_paths
    ]
    manifest = {
        "schema": "gdn-sm90a.public-evidence-manifest.v1",
        "file_count": len(rows),
        "files": rows,
    }
    write_json(root, "manifest.json", manifest)
    manifest_hash = sha256_file(root / "manifest.json")
    write_text(root, "MANIFEST.sha256", f"{manifest_hash}  manifest.json\n")


def derive(source_root: Path, output_root: Path, config_path: Path) -> None:
    if output_root.exists():
        raise FileExistsError(f"output root already exists: {output_root}")
    output_root.mkdir(parents=True)

    config = load_json(config_path)
    require_keys(
        config,
        (
            "schema",
            "claim_status",
            "historical_seal",
            "repositories",
            "aggregate_definition",
            "upstream_pr_audit",
            "signature_status",
            "licenses",
            "disclaimers",
        ),
        "public source lock",
    )
    if config["claim_status"] != "HISTORICAL_EVIDENCE_BOUND":
        raise ValueError("historical public derivation must remain HISTORICAL_EVIDENCE_BOUND")

    seal = verify_original_seal(source_root, config["historical_seal"])
    benchmark_source = load_json(source_root / "contract/benchmark-full.json")
    benchmark_contract = derive_benchmark_contract(benchmark_source)
    receipts, software = derive_receipts(source_root, benchmark_contract)
    report = load_json(source_root / "results/benchmark-full-report.json")
    gates = load_json(source_root / "results/benchmark-full-gates.json")
    release_summary = load_json(source_root / "results/release-summary.json")
    formal_gates = load_json(source_root / "results/formal-gates-r1.json")
    decision = load_json(source_root / "results/release-reseal-decision.json")
    formal_identity = load_json(source_root / "results/formal-identity-r1.json")
    host_sync = load_json(source_root / "results/host-sync-analysis.json")
    redzone = load_json(source_root / "results/redzone.json")

    publication = {
        "schema": "gdn-sm90a.publication.v1",
        "package_id": "gdn-sm90a-h20-20260728-historical-v1",
        "derivation_date": "2026-07-29",
        "package_status": "HISTORICAL_EVIDENCE_BOUND",
        "historical_release_status": "RELEASE_READY",
        "fresh_public_tag_rerun_status": "REQUIRED",
        "original_seal": copy.deepcopy(config["historical_seal"]),
        "derivation": {
            "method": "explicit_field_allowlist",
            "raw_logs_included": False,
            "private_device_or_host_identifiers_included": False,
            "deterministic_manifest": True,
        },
        "claim_scope": (
            "Historical six-row public-call latency, correctness, safety, and codegen "
            "evidence on one NVIDIA H20 execution environment."
        ),
        "unofficial_personal_forks": True,
        "upstream_merge": False,
    }
    environment = {
        "schema": "gdn-sm90a.public-environment.v1",
        "status": formal_identity["status"],
        "accelerator": "NVIDIA H20",
        "physical_device_binding_verified": formal_identity["gates"]["physical_gpu0"],
        "private_device_identifiers_published": False,
        "target_kind": formal_identity["target_kind"],
        "target_arch": formal_identity["target_arch"],
        "torch_version": formal_identity["torch_version"],
        "torch_cuda_version": formal_identity["torch_cuda_version"],
        "formal_identity_gates": selected(
            formal_identity["gates"],
            ("fresh_native_first", "sm90a_target", "source_bundle", "tvm_source"),
            "formal identity gates",
        ),
        "software_from_all_66_receipts": software,
        "timing_protocol": copy.deepcopy(benchmark_contract["timing"]),
    }
    product_contract = {
        "schema": "gdn-sm90a.public-product-contract.v1",
        **copy.deepcopy(release_summary["product_contract"]),
        "public_tag_source_contract": {
            "input_dtype": "BF16",
            "resident_state_dtype": "FP32",
            "head_dimension": 128,
            "source_repository": "https://github.com/Aharrypotter/tirx-kernels",
            "source_tag": "gdn-sm90a-kernel-r0",
            "source_commit": "12ce3721f7c62c5fbd911103ae373de689e58385",
            "source_document": "docs/gdn_sm90.md",
        },
        "dtype_and_head_dimension_provenance": {
            "kind": "PUBLIC_TAG_SOURCE_CONTRACT",
            "historical_benchmark_json_contains_explicit_fields": False,
            "note": (
                "BF16 and head-dimension 128 are documented by the public tagged source; "
                "they are not synthesized into the historical benchmark contract."
            ),
        },
    }
    performance = derive_performance(report, gates, receipts)
    correctness = derive_correctness(benchmark_source, receipts, release_summary, formal_gates)
    safety = derive_safety(release_summary, host_sync, redzone, formal_gates)
    codegen = derive_codegen(source_root, release_summary)
    controls = derive_controls(source_root, seal, report, gates, decision)
    release_decision = derive_release_decision(decision, release_summary)

    write_json(output_root, "PUBLICATION.json", publication)
    write_json(
        output_root,
        "metadata/source-lock.json",
        derive_source_lock(config, benchmark_source),
    )
    write_json(output_root, "metadata/environment.json", environment)
    write_json(output_root, "contracts/product.json", product_contract)
    write_json(output_root, "contracts/benchmark.json", benchmark_contract)
    timing_payload = "".join(
        json.dumps(receipt, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
        for receipt in receipts
    )
    write_text(output_root, "receipts/timing.jsonl", timing_payload)
    write_json(output_root, "results/performance.json", performance)
    write_json(output_root, "results/correctness.json", correctness)
    write_json(output_root, "results/safety.json", safety)
    write_json(output_root, "results/codegen-resources.json", codegen)
    write_json(output_root, "results/controls.json", controls)
    write_json(output_root, "results/release-decision.json", release_decision)
    write_json(output_root, "provenance/field-map.json", field_map())
    for relative, value in schema_documents().items():
        write_json(output_root, relative, value)

    write_text(
        output_root,
        "REDACTION.md",
        """# Redaction and derivation policy

This package was reconstructed field by field from an immutable historical
release seal.  It is not a regex-scrubbed copy of the private evidence tree.

Included data is limited to benchmark contracts, numerical timing samples,
correctness metrics, bounded safety summaries, codegen resource inventories,
release controls, and immutable source identities.

Excluded data includes raw logs and job scripts, local or remote filesystem
roots, caches and Python executable paths, host and container identifiers,
SSH aliases, GPU UUIDs, process identifiers, raw profiler traces, generated
CUDA/cubin/SASS artifacts, and the private reviewer report.

`provenance/field-map.json` records the source file and allowlisted JSON paths
for every public output.  `manifest.json` seals the resulting public payload.
""",
    )
    write_text(
        output_root,
        "LIMITATIONS.md",
        """# Limitations

1. This is historical evidence bound to the original immutable release seal.
   It is not an independent benchmark rerun from the newly published tags.
2. All 22 historical CuTeDSL receipts identify commit `88737e9`, CuTe DSL
   4.5.1, backend `sm90_cutedsl_gdn`, and callable
   `cula.gdn.prefill.chunk_gated_delta_rule`.  The corrected public comparator
   tag points directly to that commit.  The earlier GDN2 tag is not evidence
   for this GDN report.
3. BF16 and head dimension 128 are properties of the public tagged source
   contract, not explicit fields in the historical benchmark JSON.
4. The scope is six GDN prefill rows on one NVIDIA H20 environment.  It is not
   a claim about every Hopper GPU, arbitrary shapes, or end-to-end model speed.
5. Five rows have a lower TIRx/CuTeDSL latency ratio.  Packed-10 is 1.46% higher
   and is classified only as inside the preregistered ±2% noise band.
6. All branches and tags are unofficial personal-fork artifacts.  No upstream
   pull request, merge, endorsement, or full TVM SM90 support claim is implied.
""",
    )
    seal_public_bundle(output_root)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--source-lock", required=True, type=Path)
    args = parser.parse_args()
    derive(
        args.source_root.resolve(),
        args.output_root.resolve(),
        args.source_lock.resolve(),
    )
    print(f"PUBLIC_EVIDENCE_DERIVED output={args.output_root}")


if __name__ == "__main__":
    main()
