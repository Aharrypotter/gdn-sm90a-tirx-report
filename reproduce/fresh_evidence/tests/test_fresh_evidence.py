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
"""CPU-only derivation, determinism, disclosure, and tamper tests."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import shutil
import statistics
import tempfile
import unittest
from pathlib import Path
from typing import Any

from reproduce.benchmark.contract import (
    CORRECTNESS,
    CUTEDSL_DEPENDENCY_EXPECTED_AGGREGATE_SHA256,
    IMPLEMENTATIONS,
    REPORT_REPOSITORY,
    SOURCE_LOCKS,
    canonical_json_bytes,
    frozen_rows,
)
from reproduce.fresh_evidence.derive import EvidenceError, derive_bundle, seal_bundle
from reproduce.fresh_evidence.verify import VerificationError, verify_bundle

GPU_UUID = "GPU-deadbeef-dead-beef-dead-beefdeadbeef"
PACKAGE_VERSIONS = {
    "apache-tvm-ffi": "0.1.8",
    "flash-linear-attention": "0.3.4",
    "nvidia-cutlass-dsl": "4.5.1",
    "nvidia-cutlass-dsl-libs-base": "4.5.1",
    "nvidia-cutlass-dsl-libs-cu13": "4.5.1",
    "triton": "3.4.0",
    "tvm-ffi": None,
}
CACHE_KEYS = (
    "CUDA_CACHE_PATH",
    "CUTE_DSL_CACHE_DIR",
    "CUTLASS_DSL_CACHE_DIR",
    "TRITON_CACHE_DIR",
    "TVM_FFI_CACHE_DIR",
    "TMPDIR",
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _source_attestations(roots: dict[str, str]) -> dict[str, dict[str, Any]]:
    result = {}
    for name, lock in SOURCE_LOCKS.items():
        result[name] = {
            "source": name,
            "root": roots[name],
            "head": lock["commit"],
            "tree": lock["tree"],
            "tracked_clean": True,
            "clean_checkout": True,
            "tag": lock.get("tag"),
            "tag_object": lock.get("tag_object"),
            "peeled_commit": lock["commit"] if lock.get("tag") is not None else None,
            "runtime_commit": lock.get("runtime_commit"),
            "required_path": lock["required_path"],
        }
    return result


def _build_attestation() -> dict[str, Any]:
    root = "/private/build/tvm"
    libraries = {}
    names = {
        "compiler": "libtvm_compiler.so",
        "runtime": "libtvm_runtime.so",
        "ffi": "libtvm_ffi.so",
    }
    for index, (logical_name, basename) in enumerate(names.items(), start=1):
        path = f"{root}/lib/{basename}"
        libraries[logical_name] = {
            "basename": basename,
            "path": path,
            "resolved_path": path,
            "size_bytes": 1000 + index,
            "mtime_ns": 1_700_000_000_000_000_000 + index,
            "sha256": _digest(f"build-{logical_name}"),
        }
    return {"root": root, "lib_dir": f"{root}/lib", "libraries": libraries}


def _private_contract() -> dict[str, Any]:
    roots = {
        "tvm": "/private/source/tvm",
        "tirx": "/private/source/tirx-kernels",
        "cutedsl": "/private/source/cuLA",
        "fla": "/private/source/flash-linear-attention",
    }
    sources = _source_attestations(roots)
    build = _build_attestation()
    report_attestation = {
        "root": "/private/report",
        "repository": REPORT_REPOSITORY,
        "head": _digest("report-head")[:40],
        "tree": _digest("report-tree")[:40],
        "clean_checkout": True,
    }
    dependency = {
        "schema": "gdn-sm90a.transferred-source-tree.v1",
        "root_label": "nvidia-cutlass-dsl-4.5.1",
        "aggregate_sha256": CUTEDSL_DEPENDENCY_EXPECTED_AGGREGATE_SHA256,
        "entry_count": 1,
        "file_count": 1,
        "symlink_count": 0,
        "total_file_bytes": 123,
        "entries": [
            {
                "executable": False,
                "path": "nvidia_cutlass_dsl/python_packages/cutlass/__init__.py",
                "sha256": _digest("cutedsl-init"),
                "size_bytes": 123,
                "type": "file",
            }
        ],
    }
    dependency_root = "/private/dependencies/cutedsl"
    ffi_file = {
        "path": "tvm_ffi/__init__.py",
        "size_bytes": 321,
        "sha256": _digest("tvm-ffi-init"),
    }
    ffi_aggregate = hashlib.sha256(canonical_json_bytes(ffi_file) + b"\n").hexdigest()
    runtime_identity = {
        "sys_executable": "/private/runtime/bin/python",
        "python_version": "3.13.5",
        "python_implementation": "CPython",
        "python_full_version": "3.13.5 synthetic CPython",
        "torch_module_version": "2.11.0a0+eb65b36914.nv26.02",
        "torch_cuda_build": "13.0",
        "distributions": {
            "torch": {
                "distribution": "torch",
                "version": "2.11.0a0+eb65b36914.nv26.2",
                "metadata_root": "/private/runtime/site-packages",
            },
            "triton": {
                "distribution": "triton",
                "version": "3.4.0",
                "metadata_root": "/private/runtime/site-packages",
                "module_file": None,
            },
            "tvm_ffi": {
                "distribution": "apache-tvm-ffi",
                "version": "0.1.8",
                "metadata_root": "/private/runtime/site-packages",
                "installed_files": {
                    "schema": "gdn-sm90a.installed-distribution-files.v1",
                    "aggregate_sha256": ffi_aggregate,
                    "entry_count": 1,
                    "total_file_bytes": 321,
                    "entries": [ffi_file],
                },
            },
            "nvidia_cutlass_dsl": {
                "distribution": "nvidia-cutlass-dsl",
                "version": "4.5.1",
                "metadata_root": dependency_root,
            },
            "nvidia_cutlass_dsl_libs_base": {
                "distribution": "nvidia-cutlass-dsl-libs-base",
                "version": "4.5.1",
                "metadata_root": dependency_root,
            },
            "nvidia_cutlass_dsl_libs_cu13": {
                "distribution": "nvidia-cutlass-dsl-libs-cu13",
                "version": "4.5.1",
                "metadata_root": dependency_root,
            },
        },
    }
    return {
        "schema": "gdn-sm90a.public-fresh-benchmark-contract.v1",
        "run_id": "synthetic-fresh-h20-rerun",
        "claim_scope": "fresh public-tag H20 six-row characterization",
        "report_root": report_attestation["root"],
        "report_repository": REPORT_REPOSITORY,
        "report_attestation": report_attestation,
        "report_attestation_sha256": _digest_json(report_attestation),
        "source_roots": roots,
        "source_locks": copy.deepcopy(SOURCE_LOCKS),
        "source_attestations": sources,
        "source_attestation_sha256": _digest_json(sources),
        "tvm_build_root": build["root"],
        "tvm_build_attestation": build,
        "tvm_build_attestation_sha256": _digest_json(build),
        "cutedsl_dependency_root": dependency_root,
        "cutedsl_dependency_attestation": dependency,
        "cutedsl_dependency_attestation_sha256": _digest_json(dependency),
        "runtime_identity": runtime_identity,
        "runtime_identity_sha256": _digest_json(runtime_identity),
        "oracle_root": "/private/run/oracles",
        "implementations": copy.deepcopy(IMPLEMENTATIONS),
        "correctness": copy.deepcopy(CORRECTNESS),
        "timing": {
            "physical_gpu_index": 2,
            "expected_gpu_uuid": GPU_UUID,
            "cuda_logical_device": 0,
            "cuda_binding": "CUDA_VISIBLE_DEVICES=<expected_gpu_uuid>",
            "max_gpu_util_pct": 5.0,
            "quiet_timeout_s": 120.0,
            "quiet_poll_interval_s": 1.0,
            "warmup_iters": 20,
            "timed_iters": 100,
            "base_processes": 3,
            "escalated_processes": 7,
            "noise_band_pct": 2.0,
            "escalation_row_id": "packed-n10-t4096-h8-mha-state",
            "escalation_ratios": ["tirx_over_cutedsl", "tirx_over_fla"],
            "timer": "cuda_event",
            "statistic": "median_of_process_averages",
            "post_util_policy": "record_only",
            "require_rotating_three_way_order": True,
            "require_unique_cache": True,
        },
        "rows": frozen_rows(),
        "ratio": {"numerator": "tirx", "denominator": "cutedsl"},
        "secondary_ratios": [{"name": "tirx_over_fla", "numerator": "tirx", "denominator": "fla"}],
    }


def _digest_json(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _environment() -> dict[str, Any]:
    return {
        "schema": "gdn-sm90a.h20-environment-check.v1",
        "accelerator_required": "NVIDIA H20",
        "compute_capability_required": "9.0",
        "target_arch": "sm_90a",
        "private_device_identifiers_emitted": False,
        "accelerator": "NVIDIA H20",
        "compute_capability": "9.0",
        "driver_version": "580.95.05",
        "memory_total_mib": 97871,
        "binding_mode": "full_gpu_uuid",
        "cuda_compiler_release": "13.0",
        "logical_cuda_device_count": 1,
        "torch_version": "2.11.0a0+eb65b36914.nv26.02",
        "torch_cuda_version": "13.0",
        "torch_logical_device_name": "NVIDIA H20",
        "torch_logical_compute_capability": "9.0",
        "status": "PASS",
        "physical_device_binding_verified": True,
        "error_count": 0,
        "errors": [],
    }


def _gpu_state(util: float) -> dict[str, Any]:
    return {
        "physical_index": 2,
        "uuid": GPU_UUID,
        "name": "NVIDIA H20",
        "util_pct": util,
        "memory_used_mib": 1024.0,
        "pstate": "P0",
        "sm_clock_mhz": 1980.0,
        "memory_clock_mhz": 2600.0,
        "temperature_c": 42.0,
        "power_draw_w": 180.0,
    }


def _metric(present: bool) -> dict[str, Any]:
    return {
        "present_match": True,
        "max_abs": 0.001 if present else None,
        "relative_rms": 0.01 if present else None,
        "allclose": True,
    }


def _runner(implementation: str, contract: dict[str, Any]) -> dict[str, Any]:
    if implementation == "tirx":
        return {
            "fallback": False,
            "entrypoint": IMPLEMENTATIONS["tirx"]["entrypoint"],
            "module_file": "/private/source/tirx-kernels/tirx_kernels/attention/gdn_sm90.py",
            "tvm_module_file": "/private/source/tvm/python/tvm/__init__.py",
            "tvm_ffi_module_file": "/private/runtime/site-packages/tvm_ffi/__init__.py",
            "tvm_ffi_distribution": {
                "distribution": "apache-tvm-ffi",
                "version": "0.1.8",
            },
            "tvm_runtime": {
                "runtime_only": False,
                "build_root": contract["tvm_build_root"],
                "build_lib_dir": contract["tvm_build_attestation"]["lib_dir"],
                "loaded_core_libraries": {
                    "tvm_runtime": contract["tvm_build_attestation"]["libraries"]["runtime"][
                        "resolved_path"
                    ],
                    "tvm_compiler": contract["tvm_build_attestation"]["libraries"]["compiler"][
                        "resolved_path"
                    ],
                },
            },
        }
    if implementation == "cutedsl":
        return {
            "fallback": False,
            "entrypoint": IMPLEMENTATIONS["cutedsl"]["entrypoint"],
            "forbidden_gdn2_imported": False,
            "cutedsl_version": "4.5.1",
            "module_file": "/private/source/cuLA/cula/gdn/prefill.py",
        }
    return {
        "fallback": False,
        "entrypoint": IMPLEMENTATIONS["fla"]["entrypoint"],
        "public_module_file": (
            "/private/source/flash-linear-attention/fla/ops/gated_delta_rule/__init__.py"
        ),
        "chunk_module_file": (
            "/private/source/flash-linear-attention/fla/ops/gated_delta_rule/chunk.py"
        ),
        "backend_dispatch_disabled": True,
    }


def _cutedsl_dependency_modules(contract: dict[str, Any]) -> dict[str, Any]:
    root = contract["cutedsl_dependency_root"]
    return {
        "root": root,
        "cutlass_module_file": (f"{root}/nvidia_cutlass_dsl/python_packages/cutlass/__init__.py"),
        "cutlass_cute_module_file": (
            f"{root}/nvidia_cutlass_dsl/python_packages/cutlass/cute/__init__.py"
        ),
        "distributions": {
            name: {"version": "4.5.1", "metadata_root": root}
            for name in (
                "nvidia-cutlass-dsl",
                "nvidia-cutlass-dsl-libs-base",
                "nvidia-cutlass-dsl-libs-cu13",
            )
        },
    }


def _rotation(process_index: int) -> tuple[str, str, str]:
    return (
        ("tirx", "cutedsl", "fla"),
        ("cutedsl", "fla", "tirx"),
        ("fla", "tirx", "cutedsl"),
    )[process_index % 3]


def _launch_token(
    contract_sha256: str, row_id: str, implementation: str, process_index: int
) -> str:
    return _digest_json(
        {
            "contract_sha256": contract_sha256,
            "row_id": row_id,
            "implementation": implementation,
            "process_index": process_index,
        }
    )


def _latency(row_index: int, implementation: str, process_index: int) -> float:
    implementation_factor = {"tirx": 1.0, "cutedsl": 1.2, "fla": 0.9}[implementation]
    return (0.5 + row_index * 0.2) * implementation_factor * (1.0 + process_index * 0.01)


def _receipt(
    contract: dict[str, Any],
    contract_sha256: str,
    row: dict[str, Any],
    row_index: int,
    implementation: str,
    process_index: int,
) -> dict[str, Any]:
    order = _rotation(process_index)
    worker_pid = 10_000 + row_index * 100 + process_index * 3 + order.index(implementation)
    latency = _latency(row_index, implementation, process_index)
    samples = [latency for _ in range(100)]
    summary = {
        "average_ms": statistics.fmean(samples),
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
    }
    cache_identity = f"{row['row_id']}-{implementation}-p{process_index}"
    return {
        "schema": "gdn-sm90a.public-fresh-timing-receipt.v1",
        "status": "timing_ok",
        "invalid_reasons": [],
        "correctness_passed": True,
        "correctness_policy": copy.deepcopy(CORRECTNESS),
        "run_id": contract["run_id"],
        "contract_sha256": contract_sha256,
        "report_attestation_sha256": contract["report_attestation_sha256"],
        "report_attestation_before": contract["report_attestation"],
        "report_attestation_after": contract["report_attestation"],
        "source_attestation_sha256": contract["source_attestation_sha256"],
        "source_attestation_before": contract["source_attestations"],
        "source_attestation_after": contract["source_attestations"],
        "tvm_build_attestation_sha256": contract["tvm_build_attestation_sha256"],
        "tvm_build_attestation_before": contract["tvm_build_attestation"],
        "tvm_build_attestation_after": contract["tvm_build_attestation"],
        "cutedsl_dependency_attestation_sha256": contract["cutedsl_dependency_attestation_sha256"],
        "cutedsl_dependency_attestation_before": contract["cutedsl_dependency_attestation"],
        "cutedsl_dependency_attestation_after": contract["cutedsl_dependency_attestation"],
        "runtime_identity_sha256": contract["runtime_identity_sha256"],
        "runtime_identity_before": contract["runtime_identity"],
        "runtime_identity_after": contract["runtime_identity"],
        "cutedsl_dependency_modules": _cutedsl_dependency_modules(contract),
        "tvm_build_verification_mode": "contract-sha256 plus per-worker stat and loaded-handle",
        "row_id": row["row_id"],
        "impl": implementation,
        "process_index": process_index,
        "execution_order": order.index(implementation),
        "execution_sequence": list(order),
        "launch_token": _launch_token(
            contract_sha256, row["row_id"], implementation, process_index
        ),
        "worker_pid": worker_pid,
        "worker_parent_pid": 9000,
        "source_identity": IMPLEMENTATIONS[implementation]["source_identity"],
        "resolved_backend_identity": row["expected_backends"][implementation],
        "resolved_public_entrypoint": IMPLEMENTATIONS[implementation]["entrypoint"],
        "fallback": False,
        "runner_attestation": _runner(implementation, contract),
        "wrapped_callable_chain": [IMPLEMENTATIONS[implementation]["entrypoint"]],
        "compile_first_call_ms": latency * 100.0,
        "warmup_iters": 20,
        "timed_iters": 100,
        "timer": "cuda_event",
        "raw_per_iter_ms": samples,
        "summary": summary,
        "gpu_binding": {
            "binding_method": "private UUID plus process mapping",
            "cuda_visible_devices": GPU_UUID,
            "logical_index": 0,
            "resolved_physical_index": 2,
            "resolved_gpu_uuid": GPU_UUID,
            "process_binding": {
                "gpu_uuid": GPU_UUID,
                "pid": worker_pid,
                "process_name": "python",
                "used_memory_mib": 1024.0,
            },
            "device_name": "NVIDIA H20",
            "compute_capability": "9.0",
            "device_count": 1,
        },
        "gpu_state_before": _gpu_state(0.0),
        "gpu_state_after": _gpu_state(3.0),
        "gpu_processes_before": [],
        "gpu_processes_after": [
            {
                "gpu_uuid": GPU_UUID,
                "pid": worker_pid,
                "process_name": "python",
                "used_memory_mib": 1024.0,
            }
        ],
        "foreign_gpu_processes_after": [],
        "gpu_quiet_gate": {
            "elapsed_s": 0.2,
            "polls": 1,
            "max_observed_util_pct": 0.0,
            "timeout_s": 120.0,
            "poll_interval_s": 1.0,
        },
        "cuda_cache_env": {
            key: f"/private/cache/{cache_identity}/{key.lower()}" for key in CACHE_KEYS
        },
        "output_metrics": _metric(True),
        "state_metrics": _metric(row["stateful"]),
        "output_hash": _digest(f"output-{cache_identity}"),
        "state_hash": _digest(f"state-{cache_identity}") if row["stateful"] else None,
        "oracle_sha256": _digest(f"oracle-{row['row_id']}"),
        "input_seed": row["seed"],
        "row": row,
        "torch_version": "2.11.0a0+eb65b36914.nv26.02",
        "torch_cuda_version": "13.0",
        "package_versions": PACKAGE_VERSIONS,
        "python_executable": "/private/venv/bin/python",
        "python_version": "3.13.5",
    }


def _aggregate_private(
    contract: dict[str, Any], receipts: list[dict[str, Any]]
) -> tuple[dict[str, Any], dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in contract["rows"]:
        for implementation in ("tirx", "cutedsl", "fla"):
            grouped[(row["row_id"], implementation)] = [
                receipt
                for receipt in receipts
                if receipt["row_id"] == row["row_id"] and receipt["impl"] == implementation
            ]
    packed_medians = {
        implementation: statistics.median(
            receipt["summary"]["average_ms"]
            for receipt in grouped[("packed-n10-t4096-h8-mha-state", implementation)]
        )
        for implementation in ("tirx", "cutedsl", "fla")
    }
    packed_ratios = {
        "tirx_over_cutedsl": packed_medians["tirx"] / packed_medians["cutedsl"],
        "tirx_over_fla": packed_medians["tirx"] / packed_medians["fla"],
    }
    rows = {}
    ratios_cutedsl = []
    ratios_fla = []
    for row in contract["rows"]:
        row_id = row["row_id"]
        averages = {}
        medians = {}
        observed = {}
        for implementation in ("tirx", "cutedsl", "fla"):
            items = sorted(
                grouped[(row_id, implementation)], key=lambda receipt: receipt["process_index"]
            )
            averages[implementation] = [receipt["summary"]["average_ms"] for receipt in items]
            medians[implementation] = statistics.median(averages[implementation])
            observed[implementation] = len(items)
        cutedsl = medians["tirx"] / medians["cutedsl"]
        fla = medians["tirx"] / medians["fla"]
        rows[row_id] = {
            "primary": row["primary"],
            "critical": row["critical"],
            "expected_processes": 3,
            "observed_processes": observed,
            "process_averages_ms": averages,
            "median_ms": medians,
            "oracle_sha256": _digest(f"oracle-{row_id}"),
            "tirx_over_cutedsl": cutedsl,
            "tirx_over_fla": fla,
        }
        ratios_cutedsl.append(cutedsl)
        ratios_fla.append(fla)

    def geomean(values: list[float]) -> float:
        return math.exp(statistics.fmean(math.log(value) for value in values))

    run_summary = {
        "schema": "gdn-sm90a.public-fresh-run-summary.v1",
        "status": "PASS",
        "run_id": contract["run_id"],
        "contract_sha256": None,
        "report_attestation_before": contract["report_attestation"],
        "report_attestation_after": contract["report_attestation"],
        "report_attestation_sha256": contract["report_attestation_sha256"],
        "physical_gpu_index": 2,
        "expected_gpu_uuid": GPU_UUID,
        "cuda_binding": "CUDA_VISIBLE_DEVICES exact GPU UUID",
        "base_processes": 3,
        "packed_n10_base_ratios": packed_ratios,
        "noise_band_pct": 2.0,
        "packed_n10_escalated": False,
        "packed_n10_final_processes": 3,
        "receipt_count": len(receipts),
        "source_attestation_before": contract["source_attestations"],
        "source_attestation_after": contract["source_attestations"],
        "tvm_build_attestation_before": contract["tvm_build_attestation"],
        "tvm_build_attestation_after": contract["tvm_build_attestation"],
        "tvm_build_attestation_sha256": contract["tvm_build_attestation_sha256"],
        "cutedsl_dependency_attestation_before": contract["cutedsl_dependency_attestation"],
        "cutedsl_dependency_attestation_after": contract["cutedsl_dependency_attestation"],
        "cutedsl_dependency_attestation_sha256": contract["cutedsl_dependency_attestation_sha256"],
        "runtime_identity_before": contract["runtime_identity"],
        "runtime_identity_after": contract["runtime_identity"],
        "runtime_identity_sha256": contract["runtime_identity_sha256"],
        "completed_unix_s": 1_700_000_000.0,
    }
    report = {
        "schema": "gdn-sm90a.public-fresh-three-way-report.v1",
        "run_id": contract["run_id"],
        "contract_sha256": None,
        "report_attestation_sha256": contract["report_attestation_sha256"],
        "report_attestation_before": contract["report_attestation"],
        "report_attestation_after": contract["report_attestation"],
        "source_attestation_sha256": contract["source_attestation_sha256"],
        "tvm_build_attestation_sha256": contract["tvm_build_attestation_sha256"],
        "cutedsl_dependency_attestation_sha256": contract["cutedsl_dependency_attestation_sha256"],
        "cutedsl_dependency_attestation_before": contract["cutedsl_dependency_attestation"],
        "cutedsl_dependency_attestation_after": contract["cutedsl_dependency_attestation"],
        "runtime_identity_sha256": contract["runtime_identity_sha256"],
        "runtime_identity_before": contract["runtime_identity"],
        "runtime_identity_after": contract["runtime_identity"],
        "ratio_direction": "TIRx latency / comparator latency; lower is faster",
        "timer": "CUDA events around the public call",
        "statistic": "median of per-process averages",
        "warmup_iters": 20,
        "timed_iters": 100,
        "base_processes": 3,
        "packed_n10_escalated_processes": 7,
        "packed_n10_noise_band_pct": 2.0,
        "packed_n10_base_ratios": packed_ratios,
        "packed_n10_escalation_required": False,
        "rows": rows,
        "primary_geomean": {
            "tirx_over_cutedsl": geomean(ratios_cutedsl),
            "tirx_over_fla": geomean(ratios_fla),
        },
        "receipt_count": len(receipts),
        "fresh_process_launch_count": len(receipts),
        "unique_cache_path_count": len(receipts) * len(CACHE_KEYS),
        "run_summary": run_summary,
        "errors": [],
        "incomplete_reasons": [],
        "status": "PASS",
        "decision_status": "CHARACTERIZATION",
    }
    return run_summary, report


def _make_inputs(root: Path) -> dict[str, Path]:
    contract = _private_contract()
    contract_path = root / "contract.json"
    _write_json(contract_path, contract)
    contract_sha256 = hashlib.sha256(contract_path.read_bytes()).hexdigest()
    environment_path = root / "environment.json"
    _write_json(environment_path, _environment())
    oracle_path = root / "oracles" / "manifest.json"
    _write_json(
        oracle_path,
        {
            "schema": "gdn-sm90a.cutedsl-oracle-manifest.v1",
            "status": "PASS",
            "created_unix_s": 1_700_000_000.0,
            "contract_sha256": contract_sha256,
            "report_attestation_before": contract["report_attestation"],
            "report_attestation_after": contract["report_attestation"],
            "report_attestation_sha256": contract["report_attestation_sha256"],
            "source_attestation_before": contract["source_attestations"],
            "source_attestation_after": contract["source_attestations"],
            "tvm_build_attestation_before": contract["tvm_build_attestation"],
            "tvm_build_attestation_after": contract["tvm_build_attestation"],
            "tvm_build_attestation_sha256": contract["tvm_build_attestation_sha256"],
            "cutedsl_dependency_attestation_before": contract["cutedsl_dependency_attestation"],
            "cutedsl_dependency_attestation_after": contract["cutedsl_dependency_attestation"],
            "cutedsl_dependency_attestation_sha256": contract[
                "cutedsl_dependency_attestation_sha256"
            ],
            "runtime_identity_before": contract["runtime_identity"],
            "runtime_identity_after": contract["runtime_identity"],
            "runtime_identity_sha256": contract["runtime_identity_sha256"],
            "entrypoint": IMPLEMENTATIONS["cutedsl"]["entrypoint"],
            "backend_identity": IMPLEMENTATIONS["cutedsl"]["backend_identity"],
            "gpu_binding": {
                "resolved_gpu_uuid": GPU_UUID,
                "compute_capability": "9.0",
            },
            "gpu_state_before": _gpu_state(0.0),
            "gpu_state_after": _gpu_state(1.0),
            "gpu_processes_before": [],
            "gpu_processes_after": [],
            "foreign_gpu_processes_after": [],
            "gpu_quiet_gate": {"polls": 1},
            "rows": [
                {
                    "row_id": row["row_id"],
                    "path": f"{row['row_id']}.pt",
                    "oracle_sha256": _digest(f"oracle-{row['row_id']}"),
                    "state_present": row["stateful"],
                }
                for row in contract["rows"]
            ],
        },
    )
    receipts = []
    launches = []
    receipts_path = root / "run" / "timing"
    for row_index, row in enumerate(contract["rows"]):
        for process_index in range(3):
            for implementation in _rotation(process_index):
                receipt = _receipt(
                    contract,
                    contract_sha256,
                    row,
                    row_index,
                    implementation,
                    process_index,
                )
                receipts.append(receipt)
                launches.append(
                    {
                        "row_id": row["row_id"],
                        "impl": implementation,
                        "process_index": process_index,
                        "execution_order": receipt["execution_order"],
                        "execution_sequence": receipt["execution_sequence"],
                        "launch_token": receipt["launch_token"],
                        "child_pid": receipt["worker_pid"],
                        "started_unix_s": 1_700_000_000.0 + len(launches),
                    }
                )
                _write_json(
                    receipts_path / f"{row['row_id']}__{implementation}__p{process_index}.json",
                    receipt,
                )
    launches_path = root / "run" / "launches.jsonl"
    launches_path.write_text(
        "".join(json.dumps(launch, sort_keys=True) + "\n" for launch in launches)
    )
    run_summary, report = _aggregate_private(contract, receipts)
    run_summary["contract_sha256"] = contract_sha256
    report["contract_sha256"] = contract_sha256
    report["run_summary"]["contract_sha256"] = contract_sha256
    run_summary_path = root / "run" / "run-summary.json"
    report_path = root / "run" / "benchmark-report.json"
    _write_json(run_summary_path, run_summary)
    _write_json(report_path, report)
    return {
        "contract_path": contract_path,
        "environment_path": environment_path,
        "oracle_manifest_path": oracle_path,
        "receipts_path": receipts_path,
        "launches_path": launches_path,
        "run_summary_path": run_summary_path,
        "report_path": report_path,
    }


class FreshEvidenceTest(unittest.TestCase):
    """Exercise the complete producer and independent verifier on synthetic data."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.inputs = _make_inputs(cls.root / "inputs")
        cls.bundle = cls.root / "bundle"
        derive_bundle(output=cls.bundle, **cls.inputs)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def copy_bundle(self, name: str) -> Path:
        destination = self.root / name
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(self.bundle, destination)
        return destination

    def test_happy_path_is_deterministic_and_private_free(self) -> None:
        result = verify_bundle(self.bundle)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["receipt_count"], 54)
        contract = json.loads((self.bundle / "contract.json").read_text())
        self.assertEqual(
            contract["runtime_identity"]["torch_module_version"],
            "2.11.0a0+eb65b36914.nv26.02",
        )
        self.assertEqual(
            contract["runtime_identity"]["distributions"]["torch"]["version"],
            "2.11.0a0+eb65b36914.nv26.2",
        )
        second = self.root / "bundle-second"
        if second.exists():
            shutil.rmtree(second)
        derive_bundle(output=second, **self.inputs)
        for path in sorted(self.bundle.iterdir()):
            self.assertEqual(path.read_bytes(), (second / path.name).read_bytes())
        payload = b"\n".join(path.read_bytes() for path in sorted(self.bundle.iterdir()))
        for forbidden in (
            b"/private/",
            b"GPU-deadbeef",
            b"worker_pid",
            b"cuda_cache_env",
            b"hostname",
            b"container_id",
        ):
            self.assertNotIn(forbidden, payload)

    def test_missing_private_launch_ledger_is_rejected(self) -> None:
        inputs = dict(self.inputs)
        inputs["launches_path"] = self.root / "missing-launches.jsonl"
        with self.assertRaisesRegex(EvidenceError, "launch ledger is missing"):
            derive_bundle(output=self.root / "missing-launch-bundle", **inputs)

    def test_environment_torch_module_version_drift_is_rejected(self) -> None:
        inputs = _make_inputs(self.root / "environment-torch-drift-inputs")
        environment = json.loads(inputs["environment_path"].read_text())
        environment["torch_version"] = "2.11.0a0+eb65b36914.nv26.03"
        _write_json(inputs["environment_path"], environment)
        with self.assertRaisesRegex(
            EvidenceError,
            "torch/CUDA versions differ from the contract runtime identity",
        ):
            derive_bundle(output=self.root / "environment-torch-drift-bundle", **inputs)

    def test_truncated_private_runner_attestation_is_rejected(self) -> None:
        inputs = _make_inputs(self.root / "truncated-runner-inputs")
        path = next(
            path
            for path in inputs["receipts_path"].glob("*.json")
            if json.loads(path.read_text())["impl"] == "tirx"
        )
        receipt = json.loads(path.read_text())
        receipt["runner_attestation"]["tvm_runtime"] = {"runtime_only": False}
        _write_json(path, receipt)
        with self.assertRaisesRegex(EvidenceError, "TVM runtime attestation fields drifted"):
            derive_bundle(output=self.root / "truncated-runner-bundle", **inputs)

    def test_private_run_summary_source_before_drift_is_rejected(self) -> None:
        inputs = _make_inputs(self.root / "source-before-inputs")
        run_summary = json.loads(inputs["run_summary_path"].read_text())
        run_summary["source_attestation_before"]["tirx"]["head"] = "0" * 40
        _write_json(inputs["run_summary_path"], run_summary)
        report = json.loads(inputs["report_path"].read_text())
        report["run_summary"] = run_summary
        _write_json(inputs["report_path"], report)
        with self.assertRaisesRegex(EvidenceError, "source_attestation_before mismatch"):
            derive_bundle(output=self.root / "source-before-bundle", **inputs)

    def test_public_launch_tamper_after_forged_reseal_is_rejected(self) -> None:
        bundle = self.copy_bundle("tamper-launch-resealed")
        path = bundle / "launches.jsonl"
        launches = [json.loads(line) for line in path.read_text().splitlines()]
        launches[0]["execution_order"] = (launches[0]["execution_order"] + 1) % 3
        path.write_bytes(b"".join(canonical_json_bytes(launch) + b"\n" for launch in launches))
        seal_bundle(bundle, replace=True)
        with self.assertRaisesRegex(VerificationError, "launch rotation drift"):
            verify_bundle(bundle)

    def test_manifest_tamper_is_rejected(self) -> None:
        bundle = self.copy_bundle("tamper-manifest")
        performance = json.loads((bundle / "performance.json").read_text())
        performance["receipt_count"] += 1
        _write_json(bundle / "performance.json", performance)
        with self.assertRaisesRegex(VerificationError, "manifest payload identity mismatch"):
            verify_bundle(bundle)

    def test_sample_tamper_after_forged_reseal_is_rejected(self) -> None:
        bundle = self.copy_bundle("tamper-sample-resealed")
        path = bundle / "timing-receipts.jsonl"
        receipts = [json.loads(line) for line in path.read_text().splitlines()]
        receipts[0]["raw_per_iter_ms"][0] *= 2.0
        path.write_bytes(b"".join(canonical_json_bytes(receipt) + b"\n" for receipt in receipts))
        seal_bundle(bundle, replace=True)
        with self.assertRaisesRegex(VerificationError, "summary does not match samples"):
            verify_bundle(bundle)

    def test_torch_module_version_tamper_after_forged_reseal_is_rejected(self) -> None:
        bundle = self.copy_bundle("tamper-torch-version-resealed")
        path = bundle / "timing-receipts.jsonl"
        receipts = [json.loads(line) for line in path.read_text().splitlines()]
        receipts[0]["software"]["torch_version"] = "2.11.0a0+eb65b36914.nv26.03"
        path.write_bytes(b"".join(canonical_json_bytes(receipt) + b"\n" for receipt in receipts))
        seal_bundle(bundle, replace=True)
        with self.assertRaisesRegex(VerificationError, "torch drift"):
            verify_bundle(bundle)

    def test_forbidden_private_field_after_forged_reseal_is_rejected(self) -> None:
        bundle = self.copy_bundle("tamper-private-resealed")
        publication = json.loads((bundle / "publication.json").read_text())
        publication["host"] = "private-h20-host"
        _write_json(bundle / "publication.json", publication)
        seal_bundle(bundle, replace=True)
        with self.assertRaisesRegex(VerificationError, "forbidden private key"):
            verify_bundle(bundle)

    def test_duplicate_receipt_after_forged_reseal_is_rejected(self) -> None:
        bundle = self.copy_bundle("tamper-duplicate-resealed")
        path = bundle / "timing-receipts.jsonl"
        lines = path.read_bytes().splitlines()
        lines[-1] = lines[0]
        path.write_bytes(b"\n".join(lines) + b"\n")
        seal_bundle(bundle, replace=True)
        with self.assertRaisesRegex(VerificationError, "duplicate source digest"):
            verify_bundle(bundle)


if __name__ == "__main__":
    unittest.main()
