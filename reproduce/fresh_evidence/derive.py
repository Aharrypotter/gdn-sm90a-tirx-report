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
"""Derive a deterministic, allowlist-only public fresh H20 evidence bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from reproduce.benchmark.contract import CUTEDSL_DISTRIBUTIONS, load_contract
from reproduce.benchmark.worker import (
    CACHE_ENV_KEYS,
    expected_launch_token,
    rotation_for_process,
)

IMPLEMENTATIONS = ("tirx", "cutedsl", "fla")
PACKED_ROW = "packed-n10-t4096-h8-mha-state"
PAYLOAD_FILES = (
    "publication.json",
    "contract.json",
    "source-lock.json",
    "build-lock.json",
    "environment.json",
    "oracle-manifest.json",
    "launches.jsonl",
    "timing-receipts.jsonl",
    "run-summary.json",
    "performance.json",
)
HEX = frozenset("0123456789abcdef")
TVM_FFI_CONSOLE_SCRIPT_PATHS = frozenset(
    {
        "../../../bin/tvm-ffi-config",
        "../../../bin/tvm-ffi-stubgen",
    }
)


class EvidenceError(ValueError):
    """Raised when private inputs cannot support a public evidence claim."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise EvidenceError(message)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> tuple[dict[str, Any], bytes]:
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise EvidenceError(f"{path.name}: invalid JSON: {error}") from error
    _require(isinstance(value, dict), f"{path.name}: expected a JSON object")
    return value, raw


def _is_hex(value: Any) -> bool:
    return (
        isinstance(value, str) and len(value) == 64 and all(character in HEX for character in value)
    )


def _is_safe_tvm_ffi_installed_path(value: Any) -> bool:
    if not isinstance(value, str) or not value or "\\" in value or value.startswith("/"):
        return False
    parts = value.split("/")
    if any(part in {"", "."} for part in parts):
        return False
    return ".." not in parts or value in TVM_FFI_CONSOLE_SCRIPT_PATHS


def _is_number(value: Any, *, positive: bool = False) -> bool:
    valid = (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )
    return valid and (not positive or float(value) > 0.0)


def _close(left: Any, right: Any) -> bool:
    return (
        _is_number(left)
        and _is_number(right)
        and math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1.0e-12)
    )


def _same(left: Any, right: Any) -> bool:
    if (
        isinstance(left, int | float)
        and not isinstance(left, bool)
        and isinstance(right, int | float)
        and not isinstance(right, bool)
    ):
        return _close(left, right)
    if isinstance(left, dict) and isinstance(right, dict):
        return set(left) == set(right) and all(_same(left[key], right[key]) for key in left)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            _same(left_item, right_item) for left_item, right_item in zip(left, right)
        )
    return left == right


def _metric_passes(metric: Any, max_abs: float, relative_rms: float) -> bool:
    if not isinstance(metric, dict):
        return False
    if metric.get("present_match") is not True or metric.get("allclose") is not True:
        return False
    observed_abs = metric.get("max_abs")
    observed_rms = metric.get("relative_rms")
    return bool(
        (observed_abs is None or (_is_number(observed_abs) and observed_abs <= max_abs))
        and (observed_rms is None or (_is_number(observed_rms) and observed_rms <= relative_rms))
    )


def _safe_metric(metric: dict[str, Any]) -> dict[str, Any]:
    return {
        "present_match": metric["present_match"],
        "max_abs": metric["max_abs"],
        "relative_rms": metric["relative_rms"],
        "allclose": metric["allclose"],
    }


def _safe_gpu_state(state: dict[str, Any]) -> dict[str, Any]:
    return {
        "accelerator": state["name"],
        "util_pct": state["util_pct"],
        "memory_used_mib": state["memory_used_mib"],
        "pstate": state["pstate"],
        "sm_clock_mhz": state["sm_clock_mhz"],
        "memory_clock_mhz": state["memory_clock_mhz"],
        "temperature_c": state["temperature_c"],
        "power_draw_w": state["power_draw_w"],
    }


def _validate_gpu_state(
    state: Any,
    *,
    expected_index: int,
    expected_uuid: str,
    expected_name: str,
    label: str,
) -> dict[str, Any]:
    _require(isinstance(state, dict), f"{label}: GPU telemetry is missing")
    _require(state.get("physical_index") == expected_index, f"{label}: GPU index drift")
    _require(state.get("uuid") == expected_uuid, f"{label}: GPU UUID drift")
    _require(state.get("name") == expected_name, f"{label}: GPU model drift")
    for key in ("util_pct", "memory_used_mib"):
        _require(_is_number(state.get(key)), f"{label}: invalid {key}")
    _require(isinstance(state.get("pstate"), str), f"{label}: invalid pstate")
    for key in (
        "sm_clock_mhz",
        "memory_clock_mhz",
        "temperature_c",
        "power_draw_w",
    ):
        value = state.get(key)
        _require(value is None or _is_number(value), f"{label}: invalid {key}")
    return state


def _environment_public(environment: dict[str, Any]) -> dict[str, Any]:
    _require(
        environment.get("schema") == "gdn-sm90a.h20-environment-check.v1",
        "environment: schema mismatch",
    )
    _require(environment.get("status") == "PASS", "environment: gate did not pass")
    _require(environment.get("errors") == [], "environment: successful gate contains errors")
    _require(
        environment.get("physical_device_binding_verified") is True,
        "environment: physical device binding was not verified",
    )
    accelerator = environment.get("accelerator")
    _require(
        isinstance(accelerator, str) and accelerator.startswith("NVIDIA H20"),
        "environment: accelerator is not NVIDIA H20",
    )
    _require(
        environment.get("compute_capability") == "9.0",
        "environment: compute capability is not 9.0",
    )
    _require(environment.get("target_arch") == "sm_90a", "environment: target is not sm_90a")
    for key in (
        "driver_version",
        "cuda_compiler_release",
        "torch_version",
        "torch_cuda_version",
    ):
        _require(
            isinstance(environment.get(key), str) and bool(environment[key]),
            f"environment: missing {key}",
        )
    _require(
        environment.get("logical_cuda_device_count") == 1,
        "environment: exactly one logical CUDA device is required",
    )
    _require(
        environment.get("torch_logical_device_name") == accelerator,
        "environment: physical and logical GPU models differ",
    )
    _require(
        environment.get("torch_logical_compute_capability") == "9.0",
        "environment: logical compute capability drift",
    )
    memory = environment.get("memory_total_mib")
    _require(isinstance(memory, int) and memory > 0, "environment: invalid GPU memory size")
    return {
        "schema": "gdn-sm90a.public-environment.v1",
        "accelerator": accelerator,
        "compute_capability": "9.0",
        "target_arch": "sm_90a",
        "memory_total_mib": memory,
        "driver_version": environment["driver_version"],
        "cuda_compiler_release": environment["cuda_compiler_release"],
        "torch_version": environment["torch_version"],
        "torch_cuda_version": environment["torch_cuda_version"],
        "logical_cuda_device_count": 1,
        "physical_device_binding_verified": True,
        "receipt_identity_consistent": True,
    }


def _safe_source_attestations(contract: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result = {}
    for name in sorted(contract["source_attestations"]):
        attestation = contract["source_attestations"][name]
        result[name] = {
            key: attestation.get(key)
            for key in (
                "source",
                "head",
                "tree",
                "tracked_clean",
                "clean_checkout",
                "tag",
                "tag_object",
                "peeled_commit",
                "runtime_commit",
                "required_path",
            )
        }
    return result


def _safe_report_attestation(contract: dict[str, Any]) -> dict[str, Any]:
    attestation = contract["report_attestation"]
    return {
        "repository": attestation["repository"],
        "head": attestation["head"],
        "tree": attestation["tree"],
        "clean_checkout": attestation["clean_checkout"],
    }


def _safe_dependency_attestation(contract: dict[str, Any]) -> dict[str, Any]:
    attestation = contract["cutedsl_dependency_attestation"]
    entries = []
    for index, entry in enumerate(attestation["entries"]):
        _require(isinstance(entry, dict), f"dependency entry {index}: expected object")
        relative = entry.get("path")
        _require(
            isinstance(relative, str)
            and relative
            and not Path(relative).is_absolute()
            and ".." not in Path(relative).parts,
            f"dependency entry {index}: unsafe relative path",
        )
        entry_type = entry.get("type")
        if entry_type == "file":
            _require(
                set(entry) == {"executable", "path", "sha256", "size_bytes", "type"}
                and isinstance(entry["executable"], bool)
                and _is_hex(entry["sha256"])
                and isinstance(entry["size_bytes"], int)
                and entry["size_bytes"] >= 0,
                f"dependency entry {index}: malformed file",
            )
        elif entry_type == "symlink":
            target = entry.get("target")
            _require(
                set(entry) == {"path", "target", "type"}
                and isinstance(target, str)
                and target
                and not Path(target).is_absolute(),
                f"dependency entry {index}: unsafe symlink",
            )
        else:
            raise EvidenceError(f"dependency entry {index}: unsupported type")
        entries.append(entry)
    _require(
        attestation["entry_count"] == len(entries),
        "dependency attestation: entry count mismatch",
    )
    _require(
        attestation["file_count"] == sum(entry["type"] == "file" for entry in entries)
        and attestation["symlink_count"] == sum(entry["type"] == "symlink" for entry in entries)
        and attestation["total_file_bytes"]
        == sum(entry["size_bytes"] for entry in entries if entry["type"] == "file"),
        "dependency attestation: summary counts mismatch",
    )
    return {
        "schema": attestation["schema"],
        "root_label": attestation["root_label"],
        "aggregate_sha256": attestation["aggregate_sha256"],
        "entry_count": attestation["entry_count"],
        "file_count": attestation["file_count"],
        "symlink_count": attestation["symlink_count"],
        "total_file_bytes": attestation["total_file_bytes"],
        "entries": entries,
    }


def _safe_runtime_identity(contract: dict[str, Any]) -> dict[str, Any]:
    identity = contract["runtime_identity"]
    distributions = {}
    for logical_name in sorted(identity["distributions"]):
        distribution = identity["distributions"][logical_name]
        public_distribution = {
            "distribution": distribution["distribution"],
            "version": distribution["version"],
            "identity_source": (
                "distribution" if distribution.get("distribution") is not None else "module"
            ),
        }
        if logical_name == "tvm_ffi":
            files = distribution.get("installed_files")
            _require(
                isinstance(files, dict)
                and files.get("schema") == "gdn-sm90a.installed-distribution-files.v1"
                and isinstance(files.get("entries"), list),
                "runtime identity: tvm-ffi installed-files fingerprint is missing",
            )
            total = 0
            aggregate = hashlib.sha256()
            for index, entry in enumerate(files["entries"]):
                _require(
                    isinstance(entry, dict)
                    and set(entry) == {"path", "size_bytes", "sha256"}
                    and _is_safe_tvm_ffi_installed_path(entry["path"])
                    and isinstance(entry["size_bytes"], int)
                    and entry["size_bytes"] >= 0
                    and _is_hex(entry["sha256"]),
                    f"runtime identity: invalid tvm-ffi file entry {index}",
                )
                total += entry["size_bytes"]
                aggregate.update(_canonical_bytes(entry))
                aggregate.update(b"\n")
            _require(
                files.get("entry_count") == len(files["entries"])
                and files.get("total_file_bytes") == total
                and files.get("aggregate_sha256") == aggregate.hexdigest(),
                "runtime identity: tvm-ffi installed-files aggregate mismatch",
            )
            public_distribution["installed_files"] = files
        distributions[logical_name] = public_distribution
    return {
        "python_version": identity["python_version"],
        "python_implementation": identity["python_implementation"],
        "python_full_version": identity["python_full_version"],
        "torch_module_version": identity["torch_module_version"],
        "torch_cuda_build": identity["torch_cuda_build"],
        "distributions": distributions,
    }


def _safe_timing(timing: dict[str, Any]) -> dict[str, Any]:
    return {
        key: timing[key]
        for key in (
            "max_gpu_util_pct",
            "quiet_timeout_s",
            "quiet_poll_interval_s",
            "warmup_iters",
            "timed_iters",
            "base_processes",
            "escalated_processes",
            "noise_band_pct",
            "escalation_row_id",
            "escalation_ratios",
            "timer",
            "statistic",
            "post_util_policy",
            "require_rotating_three_way_order",
        )
    }


def _relative_path_under(value: Any, root: Path, label: str) -> str:
    _require(isinstance(value, str) and Path(value).is_absolute(), f"{label}: path is not absolute")
    resolved = Path(value).resolve()
    resolved_root = root.resolve()
    _require(
        resolved == resolved_root or resolved.is_relative_to(resolved_root),
        f"{label}: path escaped its attested root",
    )
    return resolved.relative_to(resolved_root).as_posix()


def _safe_cutedsl_runtime(receipt: dict[str, Any], contract: dict[str, Any]) -> dict[str, Any]:
    modules = receipt.get("cutedsl_dependency_modules")
    _require(isinstance(modules, dict), "receipt: CuTeDSL runtime module identity is missing")
    _require(
        set(modules)
        == {
            "root",
            "cutlass_module_file",
            "cutlass_cute_module_file",
            "distributions",
        },
        "receipt: CuTeDSL runtime module identity fields drifted",
    )
    dependency_root = Path(contract["cutedsl_dependency_root"])
    _require(
        modules["root"] == str(dependency_root.resolve()),
        "receipt: CuTeDSL runtime root drifted",
    )
    cutlass_relative = _relative_path_under(
        modules["cutlass_module_file"],
        dependency_root,
        "receipt: cutlass module",
    )
    cute_relative = _relative_path_under(
        modules["cutlass_cute_module_file"],
        dependency_root,
        "receipt: cutlass.cute module",
    )
    _require(
        cutlass_relative.endswith("cutlass/__init__.py"),
        "receipt: cutlass module identity drifted",
    )
    _require(
        cute_relative.endswith("cutlass/cute/__init__.py"),
        "receipt: cutlass.cute module identity drifted",
    )
    distributions = modules["distributions"]
    _require(
        isinstance(distributions, dict) and set(distributions) == set(CUTEDSL_DISTRIBUTIONS),
        "receipt: CuTeDSL runtime distribution set drifted",
    )
    public_versions = {}
    for name in CUTEDSL_DISTRIBUTIONS:
        identity = distributions[name]
        _require(
            isinstance(identity, dict)
            and set(identity) == {"version", "metadata_root"}
            and identity["version"] == "4.5.1",
            f"receipt: CuTeDSL runtime distribution {name} drifted",
        )
        _relative_path_under(
            identity["metadata_root"],
            dependency_root,
            f"receipt: CuTeDSL distribution {name}",
        )
        public_versions[name] = "4.5.1"
    return {
        "dependency_attestation_sha256": contract["cutedsl_dependency_attestation_sha256"],
        "cutlass_module_relative_path": cutlass_relative,
        "cutlass_cute_module_relative_path": cute_relative,
        "distribution_versions": public_versions,
        "module_binding_verified": True,
    }


def _safe_runner_attestation(
    receipt: dict[str, Any], implementation: str, contract: dict[str, Any]
) -> dict[str, Any]:
    runner = receipt.get("runner_attestation")
    _require(isinstance(runner, dict), "receipt: runner attestation is missing")
    _require(runner.get("fallback") is False, "receipt: runner fallback gate failed")
    cutedsl_runtime = _safe_cutedsl_runtime(receipt, contract)

    if implementation == "tirx":
        expected_keys = {
            "entrypoint",
            "module_file",
            "tvm_module_file",
            "tvm_ffi_module_file",
            "tvm_ffi_distribution",
            "tvm_runtime",
            "fallback",
        }
        _require(set(runner) == expected_keys, "receipt: TIRx runner attestation fields drifted")
        _require(
            runner["entrypoint"] == "tirx_kernels.attention.gdn_sm90.chunk_gated_delta_rule",
            "receipt: TIRx public entrypoint mismatch",
        )
        tirx_relative = _relative_path_under(
            runner["module_file"],
            Path(contract["source_roots"]["tirx"]),
            "receipt: TIRx public module",
        )
        tvm_relative = _relative_path_under(
            runner["tvm_module_file"],
            Path(contract["source_roots"]["tvm"]),
            "receipt: TVM Python module",
        )
        _require(
            tirx_relative == contract["source_locks"]["tirx"]["required_path"],
            "receipt: TIRx public module path drifted",
        )
        _require(
            tvm_relative == contract["source_locks"]["tvm"]["required_path"],
            "receipt: TVM Python module path drifted",
        )

        runtime = runner["tvm_runtime"]
        _require(
            isinstance(runtime, dict)
            and set(runtime)
            == {
                "runtime_only",
                "build_root",
                "build_lib_dir",
                "loaded_core_libraries",
            },
            "receipt: TIRx TVM runtime attestation fields drifted",
        )
        build = contract["tvm_build_attestation"]
        _require(
            runtime["runtime_only"] is False
            and runtime["build_root"] == contract["tvm_build_root"]
            and runtime["build_lib_dir"] == build["lib_dir"],
            "receipt: TIRx TVM build binding drifted",
        )
        expected_loaded = {
            "tvm_runtime": build["libraries"]["runtime"]["resolved_path"],
            "tvm_compiler": build["libraries"]["compiler"]["resolved_path"],
        }
        _require(
            runtime["loaded_core_libraries"] == expected_loaded,
            "receipt: TIRx loaded core libraries drifted",
        )

        ffi_distribution = runner["tvm_ffi_distribution"]
        expected_ffi = contract["runtime_identity"]["distributions"]["tvm_ffi"]
        _require(
            isinstance(ffi_distribution, dict)
            and set(ffi_distribution) == {"distribution", "version"}
            and ffi_distribution["version"] == expected_ffi["version"]
            and ffi_distribution["distribution"].lower().replace("_", "-")
            == expected_ffi["distribution"].lower().replace("_", "-"),
            "receipt: tvm-ffi distribution identity drifted",
        )
        ffi_path = Path(runner["tvm_ffi_module_file"])
        source_roots = [Path(root).resolve() for root in contract["source_roots"].values()]
        _require(
            ffi_path.is_absolute()
            and not any(
                ffi_path.resolve() == root or ffi_path.resolve().is_relative_to(root)
                for root in source_roots
            ),
            "receipt: tvm-ffi was not a noneditable external installation",
        )
        _relative_path_under(
            runner["tvm_ffi_module_file"],
            Path(expected_ffi["metadata_root"]),
            "receipt: tvm-ffi module",
        )

        def library_summary(logical_name: str) -> dict[str, Any]:
            library = build["libraries"][logical_name]
            return {
                "basename": library["basename"],
                "size_bytes": library["size_bytes"],
                "sha256": library["sha256"],
            }

        return {
            "kind": "tirx",
            "entrypoint": runner["entrypoint"],
            "fallback": False,
            "module_relative_path": tirx_relative,
            "tvm_module_relative_path": tvm_relative,
            "tvm_build_attestation_sha256": contract["tvm_build_attestation_sha256"],
            "loaded_core_libraries": {
                "tvm_runtime": library_summary("runtime"),
                "tvm_compiler": library_summary("compiler"),
            },
            "tvm_ffi": {
                "distribution": expected_ffi["distribution"],
                "version": expected_ffi["version"],
                "installed_files_aggregate_sha256": expected_ffi["installed_files"][
                    "aggregate_sha256"
                ],
                "noneditable_installation_verified": True,
            },
            "cutedsl_runtime": cutedsl_runtime,
        }

    if implementation == "cutedsl":
        _require(
            set(runner)
            == {
                "entrypoint",
                "module_file",
                "cutedsl_version",
                "forbidden_gdn2_imported",
                "fallback",
            },
            "receipt: CuTe runner attestation fields drifted",
        )
        relative = _relative_path_under(
            runner["module_file"],
            Path(contract["source_roots"]["cutedsl"]),
            "receipt: CuTe GDN module",
        )
        _require(
            runner["entrypoint"] == "cula.gdn.prefill.chunk_gated_delta_rule"
            and runner["cutedsl_version"] == "4.5.1"
            and runner["forbidden_gdn2_imported"] is False
            and relative == contract["source_locks"]["cutedsl"]["required_path"],
            "receipt: CuTe GDN runner identity drifted",
        )
        return {
            "kind": "cutedsl",
            "entrypoint": runner["entrypoint"],
            "fallback": False,
            "module_relative_path": relative,
            "cutedsl_version": "4.5.1",
            "forbidden_gdn2_imported": False,
            "cutedsl_runtime": cutedsl_runtime,
        }

    _require(
        set(runner)
        == {
            "entrypoint",
            "public_module_file",
            "chunk_module_file",
            "backend_dispatch_disabled",
            "fallback",
        },
        "receipt: FLA runner attestation fields drifted",
    )
    public_relative = _relative_path_under(
        runner["public_module_file"],
        Path(contract["source_roots"]["fla"]),
        "receipt: FLA public module",
    )
    chunk_relative = _relative_path_under(
        runner["chunk_module_file"],
        Path(contract["source_roots"]["fla"]),
        "receipt: FLA chunk module",
    )
    _require(
        runner["entrypoint"] == "fla.ops.gated_delta_rule.chunk.chunk_gated_delta_rule"
        and runner["backend_dispatch_disabled"] is True
        and public_relative == "fla/ops/gated_delta_rule/__init__.py"
        and chunk_relative == contract["source_locks"]["fla"]["required_path"],
        "receipt: FLA runner identity drifted",
    )
    return {
        "kind": "fla",
        "entrypoint": runner["entrypoint"],
        "fallback": False,
        "public_module_relative_path": public_relative,
        "chunk_module_relative_path": chunk_relative,
        "backend_dispatch_disabled": True,
        "cutedsl_runtime": cutedsl_runtime,
    }


def _validate_extended_attestations(
    document: dict[str, Any], contract: dict[str, Any], label: str
) -> None:
    for stem, expected, digest_key in (
        (
            "report_attestation",
            contract["report_attestation"],
            "report_attestation_sha256",
        ),
        (
            "cutedsl_dependency_attestation",
            contract["cutedsl_dependency_attestation"],
            "cutedsl_dependency_attestation_sha256",
        ),
        (
            "runtime_identity",
            contract["runtime_identity"],
            "runtime_identity_sha256",
        ),
    ):
        _require(
            document.get(f"{stem}_before") == expected,
            f"{label}: {stem} before mismatch",
        )
        _require(
            document.get(f"{stem}_after") == expected,
            f"{label}: {stem} after mismatch",
        )
        _require(
            document.get(digest_key) == contract[digest_key],
            f"{label}: {digest_key} mismatch",
        )


def _public_contract(contract: dict[str, Any], contract_sha256: str) -> dict[str, Any]:
    implementations = {}
    for name, spec in contract["implementations"].items():
        implementations[name] = {
            key: value for key, value in spec.items() if key != "require_unique_cache"
        }
    return {
        "schema": "gdn-sm90a.public-evidence-contract.v1",
        "original_schema": contract["schema"],
        "run_id": contract["run_id"],
        "claim_scope": contract["claim_scope"],
        "contract_sha256": contract_sha256,
        "report_repository": contract["report_repository"],
        "report_attestation": _safe_report_attestation(contract),
        "report_attestation_sha256": contract["report_attestation_sha256"],
        "source_attestation_sha256": contract["source_attestation_sha256"],
        "tvm_build_attestation_sha256": contract["tvm_build_attestation_sha256"],
        "cutedsl_dependency_attestation": _safe_dependency_attestation(contract),
        "cutedsl_dependency_attestation_sha256": contract["cutedsl_dependency_attestation_sha256"],
        "runtime_identity": _safe_runtime_identity(contract),
        "runtime_identity_sha256": contract["runtime_identity_sha256"],
        "source_locks": contract["source_locks"],
        "source_attestations": _safe_source_attestations(contract),
        "implementations": implementations,
        "correctness": contract["correctness"],
        "timing": _safe_timing(contract["timing"]),
        "rows": contract["rows"],
        "ratio": contract["ratio"],
        "secondary_ratios": contract["secondary_ratios"],
    }


def _source_lock(contract: dict[str, Any], contract_sha256: str) -> dict[str, Any]:
    return {
        "schema": "gdn-sm90a.public-source-lock.v1",
        "contract_sha256": contract_sha256,
        "report_repository": contract["report_repository"],
        "report_attestation_sha256": contract["report_attestation_sha256"],
        "report": _safe_report_attestation(contract),
        "source_attestation_sha256": contract["source_attestation_sha256"],
        "locks": contract["source_locks"],
        "observed": _safe_source_attestations(contract),
    }


def _build_lock(contract: dict[str, Any], contract_sha256: str) -> dict[str, Any]:
    libraries = {}
    for logical_name in ("compiler", "runtime", "ffi"):
        library = contract["tvm_build_attestation"]["libraries"][logical_name]
        libraries[logical_name] = {
            "basename": library["basename"],
            "size_bytes": library["size_bytes"],
            "sha256": library["sha256"],
        }
    return {
        "schema": "gdn-sm90a.public-build-lock.v1",
        "contract_sha256": contract_sha256,
        "tvm_build_attestation_sha256": contract["tvm_build_attestation_sha256"],
        "cutedsl_dependency_attestation_sha256": contract["cutedsl_dependency_attestation_sha256"],
        "runtime_identity_sha256": contract["runtime_identity_sha256"],
        "libraries": libraries,
        "cutedsl_dependency": _safe_dependency_attestation(contract),
        "runtime_identity": _safe_runtime_identity(contract),
        "contract_digest_matched": True,
        "before_after_matched": True,
    }


def _oracle_public(
    oracle: dict[str, Any], contract: dict[str, Any], contract_sha256: str
) -> dict[str, Any]:
    _require(
        oracle.get("schema") == "gdn-sm90a.cutedsl-oracle-manifest.v1",
        "oracle manifest: schema mismatch",
    )
    _require(oracle.get("status") == "PASS", "oracle manifest: status is not PASS")
    _require(
        oracle.get("contract_sha256") == contract_sha256,
        "oracle manifest: contract digest mismatch",
    )
    _validate_extended_attestations(oracle, contract, "oracle manifest")
    for key in ("source_attestation_before", "source_attestation_after"):
        _require(
            oracle.get(key) == contract["source_attestations"],
            f"oracle manifest: {key} mismatch",
        )
    for key in ("tvm_build_attestation_before", "tvm_build_attestation_after"):
        _require(
            oracle.get(key) == contract["tvm_build_attestation"],
            f"oracle manifest: {key} mismatch",
        )
    _require(
        oracle.get("tvm_build_attestation_sha256") == contract["tvm_build_attestation_sha256"],
        "oracle manifest: TVM build digest mismatch",
    )
    comparator = contract["implementations"]["cutedsl"]
    _require(
        oracle.get("entrypoint") == comparator["entrypoint"],
        "oracle manifest: entrypoint mismatch",
    )
    _require(
        oracle.get("backend_identity") == comparator["backend_identity"],
        "oracle manifest: backend mismatch",
    )
    timing = contract["timing"]
    binding = oracle.get("gpu_binding")
    _require(isinstance(binding, dict), "oracle manifest: GPU binding is missing")
    _require(
        binding.get("resolved_gpu_uuid") == timing["expected_gpu_uuid"],
        "oracle manifest: GPU UUID mismatch",
    )
    _require(
        binding.get("compute_capability") == "9.0",
        "oracle manifest: compute capability mismatch",
    )
    _require(oracle.get("gpu_processes_before") == [], "oracle manifest: GPU was not quiet")
    _require(
        oracle.get("foreign_gpu_processes_after") == [],
        "oracle manifest: foreign process observed",
    )
    expected_rows = {row["row_id"]: row for row in contract["rows"]}
    rows = oracle.get("rows")
    _require(isinstance(rows, list), "oracle manifest: rows are missing")
    by_id = {row.get("row_id"): row for row in rows if isinstance(row, dict)}
    _require(set(by_id) == set(expected_rows), "oracle manifest: frozen row coverage mismatch")
    public_rows = []
    for row in contract["rows"]:
        item = by_id[row["row_id"]]
        _require(_is_hex(item.get("oracle_sha256")), "oracle manifest: invalid semantic hash")
        _require(
            item.get("state_present") is bool(row["stateful"]),
            f"oracle manifest: state presence mismatch for {row['row_id']}",
        )
        public_rows.append(
            {
                "row_id": row["row_id"],
                "oracle_sha256": item["oracle_sha256"],
                "state_present": item["state_present"],
            }
        )
    return {
        "schema": "gdn-sm90a.public-oracle-manifest.v1",
        "original_schema": oracle["schema"],
        "status": "PASS",
        "contract_sha256": contract_sha256,
        "report_attestation_sha256": contract["report_attestation_sha256"],
        "source_attestation_sha256": contract["source_attestation_sha256"],
        "tvm_build_attestation_sha256": contract["tvm_build_attestation_sha256"],
        "cutedsl_dependency_attestation_sha256": contract["cutedsl_dependency_attestation_sha256"],
        "runtime_identity_sha256": contract["runtime_identity_sha256"],
        "entrypoint": comparator["entrypoint"],
        "backend_identity": comparator["backend_identity"],
        "row_count": len(public_rows),
        "rows": public_rows,
        "physical_device_binding_verified": True,
        "process_isolation_verified": True,
    }


def _receipt_public(
    *,
    receipt: dict[str, Any],
    source_raw: bytes,
    contract: dict[str, Any],
    contract_sha256: str,
    environment: dict[str, Any],
    environment_sha256: str,
    row_map: dict[str, dict[str, Any]],
    private_locations: set[str],
) -> dict[str, Any]:
    prefix = f"{receipt.get('row_id')}/{receipt.get('impl')}/p{receipt.get('process_index')}"
    _require(
        receipt.get("schema") == "gdn-sm90a.public-fresh-timing-receipt.v1",
        f"{prefix}: receipt schema mismatch",
    )
    _require(receipt.get("status") == "timing_ok", f"{prefix}: timing did not pass")
    _require(receipt.get("invalid_reasons") == [], f"{prefix}: invalid reasons are present")
    _require(receipt.get("correctness_passed") is True, f"{prefix}: correctness failed")
    _require(
        receipt.get("correctness_policy") == contract["correctness"],
        f"{prefix}: correctness policy drift",
    )
    for key, expected in (
        ("run_id", contract["run_id"]),
        ("contract_sha256", contract_sha256),
        ("report_attestation_sha256", contract["report_attestation_sha256"]),
        ("source_attestation_sha256", contract["source_attestation_sha256"]),
        ("tvm_build_attestation_sha256", contract["tvm_build_attestation_sha256"]),
        (
            "cutedsl_dependency_attestation_sha256",
            contract["cutedsl_dependency_attestation_sha256"],
        ),
        ("runtime_identity_sha256", contract["runtime_identity_sha256"]),
    ):
        _require(receipt.get(key) == expected, f"{prefix}: {key} mismatch")
    _validate_extended_attestations(receipt, contract, prefix)
    for key in ("source_attestation_before", "source_attestation_after"):
        _require(
            receipt.get(key) == contract["source_attestations"],
            f"{prefix}: {key} mismatch",
        )
    for key in ("tvm_build_attestation_before", "tvm_build_attestation_after"):
        _require(
            receipt.get(key) == contract["tvm_build_attestation"],
            f"{prefix}: {key} mismatch",
        )

    row_id = receipt.get("row_id")
    implementation = receipt.get("impl")
    process_index = receipt.get("process_index")
    _require(row_id in row_map, f"{prefix}: unknown row")
    _require(implementation in IMPLEMENTATIONS, f"{prefix}: unknown implementation")
    _require(
        isinstance(process_index, int) and not isinstance(process_index, bool),
        f"{prefix}: invalid process index",
    )
    timing = contract["timing"]
    _require(
        0 <= process_index < timing["escalated_processes"],
        f"{prefix}: process index outside protocol",
    )
    _require(
        process_index < timing["base_processes"] or row_id == PACKED_ROW,
        f"{prefix}: unexpected escalation process",
    )
    order = rotation_for_process(process_index)
    _require(
        receipt.get("execution_sequence") == list(order),
        f"{prefix}: execution sequence drift",
    )
    _require(
        receipt.get("execution_order") == order.index(implementation),
        f"{prefix}: execution position drift",
    )
    token = expected_launch_token(contract_sha256, row_id, implementation, process_index)
    _require(receipt.get("launch_token") == token, f"{prefix}: launch token mismatch")
    implementation_contract = contract["implementations"][implementation]
    _require(
        receipt.get("source_identity") == implementation_contract["source_identity"],
        f"{prefix}: source identity mismatch",
    )
    _require(
        receipt.get("resolved_backend_identity")
        == row_map[row_id]["expected_backends"][implementation],
        f"{prefix}: backend identity mismatch",
    )
    _require(
        receipt.get("resolved_public_entrypoint") == implementation_contract["entrypoint"],
        f"{prefix}: public entrypoint mismatch",
    )
    _require(receipt.get("fallback") is False, f"{prefix}: fallback is not false")
    _require(
        receipt.get("tvm_build_verification_mode")
        == "contract-sha256 plus per-worker stat and loaded-handle",
        f"{prefix}: TVM build verification mode drift",
    )
    runner_attestation = _safe_runner_attestation(receipt, implementation, contract)

    _require(
        receipt.get("warmup_iters") == timing["warmup_iters"],
        f"{prefix}: warmup count mismatch",
    )
    _require(
        receipt.get("timed_iters") == timing["timed_iters"],
        f"{prefix}: sample count mismatch",
    )
    _require(receipt.get("timer") == "cuda_event", f"{prefix}: timer mismatch")
    samples = receipt.get("raw_per_iter_ms")
    _require(
        isinstance(samples, list)
        and len(samples) == timing["timed_iters"]
        and all(_is_number(value, positive=True) for value in samples),
        f"{prefix}: invalid timing samples",
    )
    summary = receipt.get("summary")
    _require(isinstance(summary, dict), f"{prefix}: timing summary missing")
    expected_summary = {
        "average_ms": statistics.fmean(samples),
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
    }
    _require(_same(summary, expected_summary), f"{prefix}: timing summary mismatch")
    _require(
        _is_number(receipt.get("compile_first_call_ms"), positive=True),
        f"{prefix}: invalid compile/first-call latency",
    )

    policy = contract["correctness"]
    output_metrics = receipt.get("output_metrics")
    state_metrics = receipt.get("state_metrics")
    _require(
        _metric_passes(output_metrics, policy["output_max_abs"], policy["output_relative_rms"]),
        f"{prefix}: output correctness metrics failed",
    )
    _require(
        _metric_passes(state_metrics, policy["state_max_abs"], policy["state_relative_rms"]),
        f"{prefix}: state correctness metrics failed",
    )
    _require(receipt.get("row") == row_map[row_id], f"{prefix}: row drift")
    _require(receipt.get("input_seed") == row_map[row_id]["seed"], f"{prefix}: seed drift")
    for key in ("output_hash", "oracle_sha256"):
        _require(_is_hex(receipt.get(key)), f"{prefix}: invalid {key}")
    state_hash = receipt.get("state_hash")
    _require(
        _is_hex(state_hash) if row_map[row_id]["stateful"] else state_hash is None,
        f"{prefix}: state hash mismatch",
    )

    expected_uuid = timing["expected_gpu_uuid"]
    expected_index = timing["physical_gpu_index"]
    binding = receipt.get("gpu_binding")
    _require(isinstance(binding, dict), f"{prefix}: GPU binding missing")
    _require(
        binding.get("resolved_gpu_uuid") == expected_uuid
        and binding.get("cuda_visible_devices") == expected_uuid,
        f"{prefix}: GPU UUID binding mismatch",
    )
    _require(
        binding.get("resolved_physical_index") == expected_index,
        f"{prefix}: physical GPU mismatch",
    )
    _require(binding.get("logical_index") == 0, f"{prefix}: logical GPU mismatch")
    _require(binding.get("device_count") == 1, f"{prefix}: visible GPU count mismatch")
    _require(
        binding.get("device_name") == environment["accelerator"],
        f"{prefix}: GPU model mismatch",
    )
    _require(
        binding.get("compute_capability") == environment["compute_capability"],
        f"{prefix}: compute capability mismatch",
    )
    before = _validate_gpu_state(
        receipt.get("gpu_state_before"),
        expected_index=expected_index,
        expected_uuid=expected_uuid,
        expected_name=environment["accelerator"],
        label=f"{prefix}: before",
    )
    after = _validate_gpu_state(
        receipt.get("gpu_state_after"),
        expected_index=expected_index,
        expected_uuid=expected_uuid,
        expected_name=environment["accelerator"],
        label=f"{prefix}: after",
    )
    _require(
        before["util_pct"] <= timing["max_gpu_util_pct"],
        f"{prefix}: pre-timing GPU was not quiet",
    )
    _require(receipt.get("gpu_processes_before") == [], f"{prefix}: preexisting process")
    _require(
        receipt.get("foreign_gpu_processes_after") == [],
        f"{prefix}: foreign process observed",
    )
    worker_pid = receipt.get("worker_pid")
    _require(
        isinstance(worker_pid, int) and not isinstance(worker_pid, bool) and worker_pid > 0,
        f"{prefix}: worker PID missing",
    )
    processes_after = receipt.get("gpu_processes_after")
    _require(
        isinstance(processes_after, list)
        and all(
            isinstance(process, dict) and process.get("pid") == worker_pid
            for process in processes_after
        ),
        f"{prefix}: post-timing process isolation failed",
    )
    quiet = receipt.get("gpu_quiet_gate")
    _require(isinstance(quiet, dict), f"{prefix}: quiet gate missing")
    _require(
        quiet.get("timeout_s") == timing["quiet_timeout_s"]
        and quiet.get("poll_interval_s") == timing["quiet_poll_interval_s"]
        and isinstance(quiet.get("polls"), int)
        and quiet["polls"] >= 1
        and _is_number(quiet.get("elapsed_s"))
        and _is_number(quiet.get("max_observed_util_pct")),
        f"{prefix}: quiet gate mismatch",
    )
    locations = receipt.get("cuda_cache_env")
    _require(
        isinstance(locations, dict) and set(locations) == set(CACHE_ENV_KEYS),
        f"{prefix}: isolated workspace set is incomplete",
    )
    values = list(locations.values())
    _require(
        all(isinstance(value, str) and Path(value).is_absolute() for value in values),
        f"{prefix}: isolated workspace location is invalid",
    )
    _require(len(values) == len(set(values)), f"{prefix}: workspace location reused locally")
    _require(
        not private_locations.intersection(values),
        f"{prefix}: workspace location reused across processes",
    )
    private_locations.update(values)

    _require(
        receipt.get("torch_version") == environment["torch_version"],
        f"{prefix}: torch version mismatch",
    )
    _require(
        receipt.get("torch_cuda_version") == environment["torch_cuda_version"],
        f"{prefix}: CUDA runtime version mismatch",
    )
    package_versions = receipt.get("package_versions")
    _require(isinstance(package_versions, dict), f"{prefix}: package versions missing")
    runtime_distributions = contract["runtime_identity"]["distributions"]
    _require(
        receipt["torch_version"] == contract["runtime_identity"]["torch_module_version"]
        and package_versions.get("triton") == runtime_distributions["triton"]["version"]
        and package_versions.get("nvidia-cutlass-dsl")
        == runtime_distributions["nvidia_cutlass_dsl"]["version"]
        and package_versions.get("nvidia-cutlass-dsl-libs-base")
        == runtime_distributions["nvidia_cutlass_dsl_libs_base"]["version"]
        and package_versions.get("nvidia-cutlass-dsl-libs-cu13")
        == runtime_distributions["nvidia_cutlass_dsl_libs_cu13"]["version"],
        f"{prefix}: package versions differ from runtime identity",
    )
    ffi = runtime_distributions["tvm_ffi"]
    ffi_name = ffi["distribution"].lower().replace("_", "-")
    ffi_key = "apache-tvm-ffi" if ffi_name == "apache-tvm-ffi" else "tvm-ffi"
    _require(
        package_versions.get(ffi_key) == ffi["version"],
        f"{prefix}: tvm-ffi version differs from runtime identity",
    )
    wrapped = receipt.get("wrapped_callable_chain")
    _require(
        isinstance(wrapped, list)
        and wrapped
        and all(
            isinstance(item, str) and item and "/" not in item and "\\" not in item
            for item in wrapped
        ),
        f"{prefix}: callable-chain identity is unsafe",
    )

    return {
        "schema": "gdn-sm90a.public-evidence-timing-receipt.v1",
        "source_receipt_sha256": _sha256_bytes(source_raw),
        "status": "timing_ok",
        "correctness_passed": True,
        "correctness_policy": contract["correctness"],
        "run_id": contract["run_id"],
        "contract_sha256": contract_sha256,
        "report_attestation_sha256": contract["report_attestation_sha256"],
        "source_attestation_sha256": contract["source_attestation_sha256"],
        "tvm_build_attestation_sha256": contract["tvm_build_attestation_sha256"],
        "cutedsl_dependency_attestation_sha256": contract["cutedsl_dependency_attestation_sha256"],
        "runtime_identity_sha256": contract["runtime_identity_sha256"],
        "environment_sha256": environment_sha256,
        "row_id": row_id,
        "implementation": implementation,
        "process_index": process_index,
        "execution_order": receipt["execution_order"],
        "execution_sequence": receipt["execution_sequence"],
        "launch_token": token,
        "source_identity": receipt["source_identity"],
        "backend_identity": receipt["resolved_backend_identity"],
        "public_entrypoint": receipt["resolved_public_entrypoint"],
        "fallback": False,
        "runner_attestation": runner_attestation,
        "wrapped_callable_chain": wrapped,
        "compile_first_call_ms": receipt["compile_first_call_ms"],
        "warmup_iters": receipt["warmup_iters"],
        "timed_iters": receipt["timed_iters"],
        "timer": "cuda_event",
        "raw_per_iter_ms": samples,
        "summary": expected_summary,
        "gpu": {
            "accelerator": environment["accelerator"],
            "compute_capability": "9.0",
            "target_arch": "sm_90a",
            "logical_device_count": 1,
            "binding_verified": True,
            "before": _safe_gpu_state(before),
            "after": _safe_gpu_state(after),
            "quiet_gate": {
                key: quiet[key]
                for key in (
                    "elapsed_s",
                    "polls",
                    "max_observed_util_pct",
                    "timeout_s",
                    "poll_interval_s",
                )
            },
        },
        "process_isolation": {
            "fresh_process_verified": True,
            "no_preexisting_compute_processes": True,
            "no_foreign_compute_processes": True,
        },
        "output_metrics": _safe_metric(output_metrics),
        "state_metrics": _safe_metric(state_metrics),
        "output_hash": receipt["output_hash"],
        "state_hash": state_hash,
        "oracle_sha256": receipt["oracle_sha256"],
        "input_seed": receipt["input_seed"],
        "row": row_map[row_id],
        "software": {
            "python_version": receipt["python_version"],
            "torch_version": receipt["torch_version"],
            "torch_cuda_version": receipt["torch_cuda_version"],
            "package_versions": package_versions,
        },
    }


def _launches_public(
    *,
    path: Path,
    contract: dict[str, Any],
    contract_sha256: str,
    public_receipts: list[dict[str, Any]],
    private_receipts: dict[tuple[str, str, int], dict[str, Any]],
) -> tuple[list[dict[str, Any]], bytes]:
    _require(path.is_file(), "launch ledger is missing")
    raw = path.read_bytes()
    lines = raw.splitlines()
    _require(bool(lines), "launch ledger is empty")
    receipt_map = {
        (receipt["row_id"], receipt["implementation"], receipt["process_index"]): receipt
        for receipt in public_receipts
    }
    launches = []
    identities = set()
    tokens = set()
    source_hashes = set()
    expected_fields = {
        "row_id",
        "impl",
        "process_index",
        "execution_order",
        "execution_sequence",
        "launch_token",
        "child_pid",
        "started_unix_s",
    }
    for line_number, line in enumerate(lines, start=1):
        try:
            launch = json.loads(line)
        except json.JSONDecodeError as error:
            raise EvidenceError(
                f"launch ledger line {line_number}: invalid JSON: {error}"
            ) from error
        _require(
            isinstance(launch, dict) and set(launch) == expected_fields,
            f"launch ledger line {line_number}: field set mismatch",
        )
        row_id = launch["row_id"]
        implementation = launch["impl"]
        process_index = launch["process_index"]
        identity = (row_id, implementation, process_index)
        prefix = f"{row_id}/{implementation}/p{process_index}"
        _require(identity in receipt_map, f"{prefix}: launch has no timing receipt")
        _require(identity not in identities, f"{prefix}: duplicate launch identity")
        identities.add(identity)
        _require(
            isinstance(process_index, int) and not isinstance(process_index, bool),
            f"{prefix}: invalid launch process index",
        )
        order = rotation_for_process(process_index)
        _require(
            launch["execution_sequence"] == list(order)
            and launch["execution_order"] == order.index(implementation),
            f"{prefix}: launch rotation drift",
        )
        token = expected_launch_token(contract_sha256, row_id, implementation, process_index)
        _require(launch["launch_token"] == token, f"{prefix}: launch token mismatch")
        _require(token not in tokens, f"{prefix}: duplicate launch token")
        tokens.add(token)
        child_pid = launch["child_pid"]
        private_receipt = private_receipts[identity]
        _require(
            isinstance(child_pid, int)
            and not isinstance(child_pid, bool)
            and child_pid > 0
            and private_receipt.get("worker_pid") == child_pid,
            f"{prefix}: launch child/worker binding mismatch",
        )
        _require(
            _is_number(launch["started_unix_s"], positive=True),
            f"{prefix}: invalid launch timestamp",
        )
        source_hash = _sha256_bytes(line)
        _require(source_hash not in source_hashes, f"{prefix}: duplicate source launch digest")
        source_hashes.add(source_hash)
        receipt = receipt_map[identity]
        launches.append(
            {
                "schema": "gdn-sm90a.public-evidence-launch.v1",
                "source_launch_sha256": source_hash,
                "source_receipt_sha256": receipt["source_receipt_sha256"],
                "run_id": contract["run_id"],
                "contract_sha256": contract_sha256,
                "row_id": row_id,
                "implementation": implementation,
                "process_index": process_index,
                "execution_order": launch["execution_order"],
                "execution_sequence": launch["execution_sequence"],
                "launch_token": token,
                "parent_child_binding_verified": True,
                "fresh_process_launch_verified": True,
            }
        )
    _require(set(receipt_map) == identities, "launch ledger coverage differs from timing receipts")
    row_order = {row["row_id"]: index for index, row in enumerate(contract["rows"])}
    impl_order = {implementation: index for index, implementation in enumerate(IMPLEMENTATIONS)}
    launches.sort(
        key=lambda launch: (
            row_order[launch["row_id"]],
            launch["process_index"],
            impl_order[launch["implementation"]],
        )
    )
    return launches, raw


def _geomean(values: Iterable[float]) -> float:
    values = list(values)
    _require(bool(values) and all(value > 0 for value in values), "invalid geomean input")
    return math.exp(statistics.fmean(math.log(value) for value in values))


def _escalates(ratio: float, band_pct: float) -> bool:
    distance = abs(ratio - 1.0)
    limit = band_pct / 100.0
    return distance <= limit or math.isclose(distance, limit, rel_tol=0.0, abs_tol=1e-12)


def _aggregate(
    contract: dict[str, Any], receipts: list[dict[str, Any]]
) -> tuple[dict[str, Any], dict[str, Any], bool]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    identities = set()
    tokens = set()
    source_hashes = set()
    for receipt in receipts:
        identity = (
            receipt["row_id"],
            receipt["implementation"],
            receipt["process_index"],
        )
        _require(identity not in identities, f"duplicate receipt identity: {identity}")
        identities.add(identity)
        _require(receipt["launch_token"] not in tokens, "duplicate launch token")
        tokens.add(receipt["launch_token"])
        _require(
            receipt["source_receipt_sha256"] not in source_hashes,
            "duplicate source receipt digest",
        )
        source_hashes.add(receipt["source_receipt_sha256"])
        grouped[(identity[0], identity[1])].append(receipt)

    timing = contract["timing"]
    base = timing["base_processes"]
    packed_base_medians = {}
    for implementation in IMPLEMENTATIONS:
        items = [
            item for item in grouped[(PACKED_ROW, implementation)] if item["process_index"] < base
        ]
        _require(len(items) == base, "packed-n10 base receipts are incomplete")
        packed_base_medians[implementation] = statistics.median(
            item["summary"]["average_ms"] for item in items
        )
    packed_base_ratios = {
        "tirx_over_cutedsl": (packed_base_medians["tirx"] / packed_base_medians["cutedsl"]),
        "tirx_over_fla": packed_base_medians["tirx"] / packed_base_medians["fla"],
    }
    escalated = any(
        _escalates(packed_base_ratios[name], timing["noise_band_pct"])
        for name in timing["escalation_ratios"]
    )
    expected_by_row = {
        row["row_id"]: (
            timing["escalated_processes"] if escalated and row["row_id"] == PACKED_ROW else base
        )
        for row in contract["rows"]
    }

    rows = {}
    primary_cutedsl = []
    primary_fla = []
    for row in contract["rows"]:
        row_id = row["row_id"]
        expected_count = expected_by_row[row_id]
        medians = {}
        process_averages = {}
        observed = {}
        oracle_hashes = set()
        for implementation in IMPLEMENTATIONS:
            items = sorted(
                grouped[(row_id, implementation)], key=lambda item: item["process_index"]
            )
            observed[implementation] = len(items)
            _require(
                [item["process_index"] for item in items] == list(range(expected_count)),
                f"{row_id}/{implementation}: process coverage mismatch",
            )
            process_averages[implementation] = [item["summary"]["average_ms"] for item in items]
            medians[implementation] = statistics.median(process_averages[implementation])
            oracle_hashes.update(item["oracle_sha256"] for item in items)
        _require(len(oracle_hashes) == 1, f"{row_id}: oracle hash drift")
        tirx_over_cutedsl = medians["tirx"] / medians["cutedsl"]
        tirx_over_fla = medians["tirx"] / medians["fla"]
        row_report = {
            "primary": bool(row["primary"]),
            "critical": bool(row["critical"]),
            "expected_processes": expected_count,
            "observed_processes": observed,
            "process_averages_ms": process_averages,
            "median_ms": medians,
            "oracle_sha256": next(iter(oracle_hashes)),
            "tirx_over_cutedsl": tirx_over_cutedsl,
            "tirx_over_fla": tirx_over_fla,
        }
        rows[row_id] = row_report
        if row["primary"]:
            primary_cutedsl.append(tirx_over_cutedsl)
            primary_fla.append(tirx_over_fla)

    performance = {
        "schema": "gdn-sm90a.public-performance.v1",
        "run_id": contract["run_id"],
        "contract_sha256": None,
        "report_attestation_sha256": contract["report_attestation_sha256"],
        "source_attestation_sha256": contract["source_attestation_sha256"],
        "tvm_build_attestation_sha256": contract["tvm_build_attestation_sha256"],
        "cutedsl_dependency_attestation_sha256": contract["cutedsl_dependency_attestation_sha256"],
        "runtime_identity_sha256": contract["runtime_identity_sha256"],
        "ratio_direction": "TIRx latency / comparator latency; lower is faster",
        "timer": "CUDA events around the public call",
        "statistic": "median of per-process averages",
        "warmup_iters": timing["warmup_iters"],
        "timed_iters": timing["timed_iters"],
        "base_processes": base,
        "packed_n10_escalated_processes": timing["escalated_processes"],
        "packed_n10_noise_band_pct": timing["noise_band_pct"],
        "packed_n10_base_ratios": packed_base_ratios,
        "packed_n10_escalation_required": escalated,
        "rows": rows,
        "primary_geomean": {
            "tirx_over_cutedsl": _geomean(primary_cutedsl),
            "tirx_over_fla": _geomean(primary_fla),
        },
        "receipt_count": len(receipts),
        "fresh_process_launch_count": len(receipts),
        "process_isolation_verified": True,
        "status": "PASS",
        "decision_status": "CHARACTERIZATION",
    }
    summary = {
        "schema": "gdn-sm90a.public-run-summary.v1",
        "status": "PASS",
        "run_id": contract["run_id"],
        "contract_sha256": None,
        "report_attestation_sha256": contract["report_attestation_sha256"],
        "source_attestation_sha256": contract["source_attestation_sha256"],
        "tvm_build_attestation_sha256": contract["tvm_build_attestation_sha256"],
        "cutedsl_dependency_attestation_sha256": contract["cutedsl_dependency_attestation_sha256"],
        "runtime_identity_sha256": contract["runtime_identity_sha256"],
        "base_processes": base,
        "packed_n10_base_ratios": packed_base_ratios,
        "noise_band_pct": timing["noise_band_pct"],
        "packed_n10_escalated": escalated,
        "packed_n10_final_processes": expected_by_row[PACKED_ROW],
        "receipt_count": len(receipts),
        "fresh_process_launch_count": len(receipts),
        "physical_device_binding_verified": True,
        "process_isolation_verified": True,
    }
    return performance, summary, escalated


def _check_private_aggregates(
    *,
    report: dict[str, Any],
    run_summary: dict[str, Any],
    contract: dict[str, Any],
    contract_sha256: str,
    performance: dict[str, Any],
    summary: dict[str, Any],
    private_location_count: int,
) -> None:
    _require(
        report.get("schema") == "gdn-sm90a.public-fresh-three-way-report.v1",
        "report: schema mismatch",
    )
    _require(report.get("status") == "PASS", "report: status is not PASS")
    _require(report.get("decision_status") == "CHARACTERIZATION", "report: decision drift")
    _require(report.get("errors") == [], "report: errors are present")
    _require(report.get("incomplete_reasons") == [], "report: incomplete reasons are present")
    _require(report.get("run_summary") == run_summary, "report: embedded run summary drift")
    _validate_extended_attestations(report, contract, "report")
    _require(
        run_summary.get("schema") == "gdn-sm90a.public-fresh-run-summary.v1",
        "run summary: schema mismatch",
    )
    _require(run_summary.get("status") == "PASS", "run summary: status is not PASS")
    _validate_extended_attestations(run_summary, contract, "run summary")
    for document_name, document in (("report", report), ("run summary", run_summary)):
        _require(document.get("run_id") == contract["run_id"], f"{document_name}: run ID drift")
        _require(
            document.get("contract_sha256") == contract_sha256,
            f"{document_name}: contract digest drift",
        )
        _require(
            document.get("tvm_build_attestation_sha256")
            == contract["tvm_build_attestation_sha256"],
            f"{document_name}: TVM build digest drift",
        )
        for digest_key in (
            "report_attestation_sha256",
            "cutedsl_dependency_attestation_sha256",
            "runtime_identity_sha256",
        ):
            _require(
                document.get(digest_key) == contract[digest_key],
                f"{document_name}: {digest_key} drift",
            )
    _require(
        run_summary.get("physical_gpu_index") == contract["timing"]["physical_gpu_index"]
        and run_summary.get("expected_gpu_uuid") == contract["timing"]["expected_gpu_uuid"],
        "run summary: private GPU identity mismatch",
    )
    _require(
        run_summary.get("source_attestation_before") == contract["source_attestations"],
        "run summary: source_attestation_before mismatch",
    )
    _require(
        run_summary.get("source_attestation_after") == contract["source_attestations"],
        "run summary: source_attestation_after mismatch",
    )
    for key in ("tvm_build_attestation_before", "tvm_build_attestation_after"):
        _require(
            run_summary.get(key) == contract["tvm_build_attestation"],
            f"run summary: {key} mismatch",
        )
    report_projection = {
        key: report.get(key)
        for key in (
            "ratio_direction",
            "timer",
            "statistic",
            "warmup_iters",
            "timed_iters",
            "base_processes",
            "packed_n10_escalated_processes",
            "packed_n10_noise_band_pct",
            "packed_n10_base_ratios",
            "packed_n10_escalation_required",
            "rows",
            "primary_geomean",
            "receipt_count",
            "fresh_process_launch_count",
            "status",
            "decision_status",
        )
    }
    public_projection = {key: performance[key] for key in report_projection}
    _require(_same(report_projection, public_projection), "report: aggregate values drift")
    summary_projection = {
        key: run_summary.get(key)
        for key in (
            "base_processes",
            "packed_n10_base_ratios",
            "noise_band_pct",
            "packed_n10_escalated",
            "packed_n10_final_processes",
            "receipt_count",
        )
    }
    _require(
        _same(
            summary_projection,
            {key: summary[key] for key in summary_projection},
        ),
        "run summary: aggregate values drift",
    )
    _require(
        report.get("unique_cache_path_count") == private_location_count,
        "report: isolated workspace count mismatch",
    )
    _require(
        private_location_count == len(CACHE_ENV_KEYS) * performance["receipt_count"],
        "private isolated-workspace cardinality mismatch",
    )


def _write(path: Path, value: Any) -> None:
    path.write_bytes(_json_bytes(value))


def seal_bundle(root: Path, *, replace: bool = False) -> None:
    """Write the deterministic payload manifest and its SHA-256 seal."""

    manifest_path = root / "manifest.json"
    seal_path = root / "MANIFEST.sha256"
    if not replace and (manifest_path.exists() or seal_path.exists()):
        raise FileExistsError("refusing to overwrite an existing evidence manifest")
    files = []
    for relative in PAYLOAD_FILES:
        path = root / relative
        if not path.is_file():
            raise EvidenceError(f"cannot seal missing payload {relative}")
        files.append(
            {
                "path": relative,
                "sha256": _sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    manifest = {
        "schema": "gdn-sm90a.public-evidence-manifest.v1",
        "file_count": len(files),
        "files": files,
    }
    manifest_path.write_bytes(_json_bytes(manifest))
    seal_path.write_text(f"{_sha256_file(manifest_path)}  manifest.json\n")


def derive_bundle(
    *,
    contract_path: Path,
    environment_path: Path,
    oracle_manifest_path: Path,
    receipts_path: Path,
    launches_path: Path,
    run_summary_path: Path,
    report_path: Path,
    output: Path,
) -> Path:
    """Audit private inputs, reconstruct allowlisted documents, and seal the bundle."""

    output = output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to reuse public evidence output: {output}")

    contract_path = contract_path.expanduser().resolve()
    environment_path = environment_path.expanduser().resolve()
    oracle_manifest_path = oracle_manifest_path.expanduser().resolve()
    receipts_path = receipts_path.expanduser().resolve()
    launches_path = launches_path.expanduser().resolve()
    run_summary_path = run_summary_path.expanduser().resolve()
    report_path = report_path.expanduser().resolve()
    contract, contract_sha256 = load_contract(contract_path)
    contract_raw = contract_path.read_bytes()
    environment_private, environment_raw = _read_json(environment_path)
    environment = _environment_public(environment_private)
    _require(
        environment["torch_version"] == contract["runtime_identity"]["torch_module_version"]
        and environment["torch_cuda_version"] == contract["runtime_identity"]["torch_cuda_build"],
        "environment: torch/CUDA versions differ from the contract runtime identity",
    )
    environment_sha256 = _sha256_bytes(_canonical_bytes(environment))
    oracle_private, oracle_raw = _read_json(oracle_manifest_path)
    oracle = _oracle_public(oracle_private, contract, contract_sha256)
    run_summary_private, run_summary_raw = _read_json(run_summary_path)
    report_private, report_raw = _read_json(report_path)

    paths = sorted(receipts_path.glob("*.json"))
    _require(bool(paths), "no timing receipts were found")
    row_map = {row["row_id"]: row for row in contract["rows"]}
    private_locations: set[str] = set()
    receipts = []
    private_receipts: dict[tuple[str, str, int], dict[str, Any]] = {}
    for path in paths:
        raw = path.read_bytes()
        try:
            private_receipt = json.loads(raw)
        except json.JSONDecodeError as error:
            raise EvidenceError(f"{path.name}: invalid JSON: {error}") from error
        _require(isinstance(private_receipt, dict), f"{path.name}: expected JSON object")
        public_receipt = _receipt_public(
            receipt=private_receipt,
            source_raw=raw,
            contract=contract,
            contract_sha256=contract_sha256,
            environment=environment,
            environment_sha256=environment_sha256,
            row_map=row_map,
            private_locations=private_locations,
        )
        identity = (
            public_receipt["row_id"],
            public_receipt["implementation"],
            public_receipt["process_index"],
        )
        _require(
            identity not in private_receipts,
            f"duplicate private receipt identity: {identity}",
        )
        private_receipts[identity] = private_receipt
        receipts.append(public_receipt)
    row_order = {row["row_id"]: index for index, row in enumerate(contract["rows"])}
    impl_order = {implementation: index for index, implementation in enumerate(IMPLEMENTATIONS)}
    receipts.sort(
        key=lambda receipt: (
            row_order[receipt["row_id"]],
            receipt["process_index"],
            impl_order[receipt["implementation"]],
        )
    )
    launches, launches_raw = _launches_public(
        path=launches_path,
        contract=contract,
        contract_sha256=contract_sha256,
        public_receipts=receipts,
        private_receipts=private_receipts,
    )
    performance, summary, _ = _aggregate(contract, receipts)
    performance["contract_sha256"] = contract_sha256
    summary["contract_sha256"] = contract_sha256
    _check_private_aggregates(
        report=report_private,
        run_summary=run_summary_private,
        contract=contract,
        contract_sha256=contract_sha256,
        performance=performance,
        summary=summary,
        private_location_count=len(private_locations),
    )
    oracle_hashes = {row["row_id"]: row["oracle_sha256"] for row in oracle["rows"]}
    for receipt in receipts:
        _require(
            receipt["oracle_sha256"] == oracle_hashes[receipt["row_id"]],
            f"{receipt['row_id']}: receipt/oracle manifest hash mismatch",
        )

    source = _source_lock(contract, contract_sha256)
    build = _build_lock(contract, contract_sha256)
    public_contract = _public_contract(contract, contract_sha256)
    receipt_set_sha256 = _sha256_bytes(
        _canonical_bytes(sorted(receipt["source_receipt_sha256"] for receipt in receipts))
    )
    launch_set_sha256 = _sha256_bytes(
        _canonical_bytes(sorted(launch["source_launch_sha256"] for launch in launches))
    )
    publication = {
        "schema": "gdn-sm90a.public-fresh-evidence.v1",
        "status": "PASS",
        "evidence_kind": "fresh-public-tag-h20-rerun",
        "decision_status": "CHARACTERIZATION",
        "upstream_merge_claim": False,
        "run_id": contract["run_id"],
        "claim_scope": contract["claim_scope"],
        "contract_sha256": contract_sha256,
        "report_attestation_sha256": contract["report_attestation_sha256"],
        "source_attestation_sha256": contract["source_attestation_sha256"],
        "tvm_build_attestation_sha256": contract["tvm_build_attestation_sha256"],
        "cutedsl_dependency_attestation_sha256": contract["cutedsl_dependency_attestation_sha256"],
        "runtime_identity_sha256": contract["runtime_identity_sha256"],
        "environment_sha256": environment_sha256,
        "receipt_count": len(receipts),
        "fresh_process_launch_count": len(receipts),
        "physical_device_binding_verified": True,
        "process_isolation_verified": True,
        "input_artifact_sha256": {
            "benchmark_contract": _sha256_bytes(contract_raw),
            "environment_check": _sha256_bytes(environment_raw),
            "oracle_manifest": _sha256_bytes(oracle_raw),
            "run_summary": _sha256_bytes(run_summary_raw),
            "benchmark_report": _sha256_bytes(report_raw),
            "launch_ledger": _sha256_bytes(launches_raw),
            "launch_ledger_set": launch_set_sha256,
            "timing_receipt_set": receipt_set_sha256,
        },
    }

    output.mkdir(parents=True)
    _write(output / "publication.json", publication)
    _write(output / "contract.json", public_contract)
    _write(output / "source-lock.json", source)
    _write(output / "build-lock.json", build)
    _write(output / "environment.json", environment)
    _write(output / "oracle-manifest.json", oracle)
    with (output / "launches.jsonl").open("xb") as stream:
        for launch in launches:
            stream.write(_canonical_bytes(launch) + b"\n")
    with (output / "timing-receipts.jsonl").open("xb") as stream:
        for receipt in receipts:
            stream.write(_canonical_bytes(receipt) + b"\n")
    _write(output / "run-summary.json", summary)
    _write(output / "performance.json", performance)
    seal_bundle(output)

    from .verify import verify_bundle

    verify_bundle(output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Derive a deterministic allowlist-only public fresh H20 evidence bundle."
    )
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--environment", type=Path, required=True)
    parser.add_argument("--oracle-manifest", type=Path, required=True)
    parser.add_argument("--receipts", type=Path, required=True)
    parser.add_argument("--launches", type=Path, required=True)
    parser.add_argument("--run-summary", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = derive_bundle(
        contract_path=args.contract,
        environment_path=args.environment,
        oracle_manifest_path=args.oracle_manifest,
        receipts_path=args.receipts,
        launches_path=args.launches,
        run_summary_path=args.run_summary,
        report_path=args.report,
        output=args.output,
    )
    print(f"GDN_FRESH_PUBLIC_EVIDENCE_OK bundle={output}")


if __name__ == "__main__":
    main()
