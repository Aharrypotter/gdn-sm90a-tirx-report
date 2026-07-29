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
"""Audit and aggregate the public fresh-process three-way GDN receipts."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from .contract import (
    CUTEDSL_DISTRIBUTIONS,
    IMPLEMENTATION_ORDER,
    PACKED_N10_ROW_ID,
    activate_cutedsl_dependency_root,
    load_contract,
    verify_contract_cutedsl_dependency_root,
    verify_contract_report_root,
    verify_contract_runtime_identity,
    verify_contract_runtime_root,
    verify_contract_source_roots,
)
from .run import escalation_required
from .worker import CACHE_ENV_KEYS, expected_launch_token, rotation_for_process


def geometric_mean(values: list[float]) -> float | None:
    """Compute a geometric mean without pooling process-level samples."""

    if not values:
        return None
    if any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("geometric mean requires positive finite values")
    return math.exp(statistics.fmean(math.log(value) for value in values))


def median_process_average(receipts: list[dict[str, Any]]) -> float:
    """Apply the frozen median-of-process-averages statistic."""

    if not receipts:
        raise ValueError("at least one receipt is required")
    return statistics.median(float(receipt["summary"]["average_ms"]) for receipt in receipts)


def _is_positive_number(value: Any) -> bool:
    return (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value > 0
    )


def _load_launches(path: Path, errors: list[str]) -> dict[str, dict[str, Any]]:
    result = {}
    if not path.is_file():
        errors.append(f"{path}: launch ledger is missing")
        return result
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        try:
            launch = json.loads(line)
        except json.JSONDecodeError as error:
            errors.append(f"{path}:{line_number}: invalid JSON: {error}")
            continue
        token = launch.get("launch_token")
        if not isinstance(token, str) or len(token) != 64:
            errors.append(f"{path}:{line_number}: invalid launch token")
            continue
        if token in result:
            errors.append(f"{path}:{line_number}: duplicate launch token {token}")
        result[token] = launch
    return result


def _audit_receipt(
    *,
    path: Path,
    receipt: dict[str, Any],
    contract: dict[str, Any],
    contract_sha256: str,
    row_map: dict[str, dict[str, Any]],
    cache_owners: dict[str, tuple[str, str, int, str]],
    launches: dict[str, dict[str, Any]],
) -> list[str]:
    local_errors = []
    row_id = receipt.get("row_id")
    implementation = receipt.get("impl")
    process_index = receipt.get("process_index")
    if row_id not in row_map or implementation not in IMPLEMENTATION_ORDER:
        return ["unexpected row or implementation"]
    if receipt.get("schema") != "gdn-sm90a.public-fresh-timing-receipt.v1":
        local_errors.append("receipt schema mismatch")
    if receipt.get("status") != "timing_ok":
        local_errors.append(f"status is not timing_ok: {receipt.get('invalid_reasons')}")
    if receipt.get("invalid_reasons") != []:
        local_errors.append("successful receipt contains invalid reasons")
    if receipt.get("correctness_passed") is not True:
        local_errors.append("correctness did not pass")
    if receipt.get("correctness_policy") != contract["correctness"]:
        local_errors.append("correctness policy mismatch")
    if receipt.get("run_id") != contract["run_id"]:
        local_errors.append("run ID mismatch")
    if receipt.get("contract_sha256") != contract_sha256:
        local_errors.append("contract digest mismatch")
    if receipt.get("report_attestation_sha256") != contract["report_attestation_sha256"]:
        local_errors.append("report harness attestation digest mismatch")
    if receipt.get("report_attestation_before") != contract["report_attestation"]:
        local_errors.append("pre-timing report harness attestation mismatch")
    if receipt.get("report_attestation_after") != contract["report_attestation"]:
        local_errors.append("post-timing report harness attestation mismatch")
    if (
        receipt.get("cutedsl_dependency_attestation_sha256")
        != contract["cutedsl_dependency_attestation_sha256"]
    ):
        local_errors.append("CuTeDSL dependency attestation digest mismatch")
    if (
        receipt.get("cutedsl_dependency_attestation_before")
        != contract["cutedsl_dependency_attestation"]
    ):
        local_errors.append("pre-timing CuTeDSL dependency attestation mismatch")
    if (
        receipt.get("cutedsl_dependency_attestation_after")
        != contract["cutedsl_dependency_attestation"]
    ):
        local_errors.append("post-timing CuTeDSL dependency attestation mismatch")
    if receipt.get("runtime_identity_sha256") != contract["runtime_identity_sha256"]:
        local_errors.append("runtime identity digest mismatch")
    if receipt.get("runtime_identity_before") != contract["runtime_identity"]:
        local_errors.append("pre-timing runtime identity mismatch")
    if receipt.get("runtime_identity_after") != contract["runtime_identity"]:
        local_errors.append("post-timing runtime identity mismatch")
    dependency_modules = receipt.get("cutedsl_dependency_modules")
    dependency_root = Path(contract["cutedsl_dependency_root"]).resolve()
    if not isinstance(dependency_modules, dict):
        local_errors.append("CuTeDSL dependency module identity is missing")
    else:
        if dependency_modules.get("root") != str(dependency_root):
            local_errors.append("CuTeDSL dependency module root mismatch")
        for key in ("cutlass_module_file", "cutlass_cute_module_file"):
            module_file = dependency_modules.get(key)
            if not isinstance(module_file, str) or not Path(module_file).resolve().is_relative_to(
                dependency_root
            ):
                local_errors.append(f"{key} escaped the CuTeDSL dependency root")
        distribution_modules = dependency_modules.get("distributions")
        if not isinstance(distribution_modules, dict) or set(distribution_modules) != set(
            CUTEDSL_DISTRIBUTIONS
        ):
            local_errors.append("CuTeDSL runtime distribution identity set is incomplete")
        else:
            for name, identity in distribution_modules.items():
                if (
                    not isinstance(identity, dict)
                    or identity.get("version") != "4.5.1"
                    or not isinstance(identity.get("metadata_root"), str)
                    or not Path(identity["metadata_root"]).resolve().is_relative_to(dependency_root)
                ):
                    local_errors.append(f"CuTeDSL runtime distribution {name} drifted")
    if receipt.get("source_attestation_sha256") != contract["source_attestation_sha256"]:
        local_errors.append("source attestation digest mismatch")
    if receipt.get("source_attestation_before") != contract["source_attestations"]:
        local_errors.append("pre-timing source attestation mismatch")
    if receipt.get("source_attestation_after") != contract["source_attestations"]:
        local_errors.append("post-timing source attestation mismatch")
    if receipt.get("tvm_build_attestation_sha256") != contract["tvm_build_attestation_sha256"]:
        local_errors.append("TVM build attestation digest mismatch")
    if receipt.get("tvm_build_attestation_before") != contract["tvm_build_attestation"]:
        local_errors.append("pre-timing TVM build attestation mismatch")
    if receipt.get("tvm_build_attestation_after") != contract["tvm_build_attestation"]:
        local_errors.append("post-timing TVM build attestation mismatch")
    implementation_contract = contract["implementations"][implementation]
    if receipt.get("source_identity") != implementation_contract["source_identity"]:
        local_errors.append("implementation source identity mismatch")
    if (
        receipt.get("resolved_backend_identity")
        != row_map[row_id]["expected_backends"][implementation]
    ):
        local_errors.append("backend identity mismatch")
    if receipt.get("resolved_public_entrypoint") != implementation_contract["entrypoint"]:
        local_errors.append("public entrypoint mismatch")
    if receipt.get("fallback") is not False:
        local_errors.append("fallback is not false")
    runner = receipt.get("runner_attestation")
    if not isinstance(runner, dict) or runner.get("fallback") is not False:
        local_errors.append("runner did not attest fallback=false")
    elif implementation == "cutedsl":
        if runner.get("entrypoint") != "cula.gdn.prefill.chunk_gated_delta_rule":
            local_errors.append("CuTe runner did not use cula.gdn.prefill")
        if runner.get("forbidden_gdn2_imported") is not False:
            local_errors.append("CuTe runner imported the forbidden cula.gdn2 namespace")
        module_file = str(runner.get("module_file", ""))
        if "/cula/gdn/prefill.py" not in module_file or "/cula/gdn2/" in module_file:
            local_errors.append("CuTe comparator module path is not cula/gdn/prefill.py")
        if runner.get("cutedsl_version") != "4.5.1":
            local_errors.append("CuTe DSL version mismatch")
    elif implementation == "fla":
        if runner.get("backend_dispatch_disabled") is not True:
            local_errors.append("FLA external backend dispatch was not disabled")
    elif implementation == "tirx":
        if runner.get("entrypoint") != ("tirx_kernels.attention.gdn_sm90.chunk_gated_delta_rule"):
            local_errors.append("TIRx did not use its public GDN module")
        tvm_runtime = runner.get("tvm_runtime", {})
        if tvm_runtime.get("runtime_only") is not False:
            local_errors.append("TIRx did not attest the TVM compiler runtime")
        if set(tvm_runtime.get("loaded_core_libraries", {})) != {
            "tvm_runtime",
            "tvm_compiler",
        }:
            local_errors.append("TIRx TVM core-library attestation is incomplete")
        else:
            loaded = tvm_runtime["loaded_core_libraries"]
            expected_loaded = {
                "tvm_runtime": contract["tvm_build_attestation"]["libraries"]["runtime"][
                    "resolved_path"
                ],
                "tvm_compiler": contract["tvm_build_attestation"]["libraries"]["compiler"][
                    "resolved_path"
                ],
            }
            if {
                name: str(Path(path).resolve()) for name, path in loaded.items()
            } != expected_loaded:
                local_errors.append("TIRx loaded core-library paths differ from the contract")
        if tvm_runtime.get("build_root") != contract["tvm_build_root"]:
            local_errors.append("TIRx loaded TVM from the wrong build root")
        if tvm_runtime.get("build_lib_dir") != contract["tvm_build_attestation"]["lib_dir"]:
            local_errors.append("TIRx loaded TVM from the wrong build/lib directory")
        tvm_ffi_file = runner.get("tvm_ffi_module_file")
        if not isinstance(tvm_ffi_file, str) or any(
            Path(tvm_ffi_file).resolve() == Path(root).resolve()
            or Path(tvm_ffi_file).resolve().is_relative_to(Path(root).resolve())
            for root in contract["source_roots"].values()
        ):
            local_errors.append("tvm-ffi was not a noneditable external installation")
        distribution = runner.get("tvm_ffi_distribution")
        if (
            not isinstance(distribution, dict)
            or distribution.get("distribution") not in {"apache-tvm-ffi", "tvm-ffi"}
            or not isinstance(distribution.get("version"), str)
            or not distribution["version"]
        ):
            local_errors.append("tvm-ffi distribution version is missing")

    timing = contract["timing"]
    if receipt.get("warmup_iters") != timing["warmup_iters"]:
        local_errors.append("warmup count mismatch")
    if receipt.get("timed_iters") != timing["timed_iters"]:
        local_errors.append("timed iteration count mismatch")
    if receipt.get("timer") != "cuda_event":
        local_errors.append("timer is not cuda_event")
    samples = receipt.get("raw_per_iter_ms")
    if (
        not isinstance(samples, list)
        or len(samples) != timing["timed_iters"]
        or not all(_is_positive_number(value) for value in samples)
    ):
        local_errors.append("raw CUDA-event samples are invalid")
    else:
        average = statistics.fmean(samples)
        summary = receipt.get("summary")
        reported = summary.get("average_ms") if isinstance(summary, dict) else None
        if not _is_positive_number(reported) or not math.isclose(
            average, float(reported), rel_tol=0.0, abs_tol=1.0e-12
        ):
            local_errors.append("reported average does not match raw samples")
    if not _is_positive_number(receipt.get("compile_first_call_ms")):
        local_errors.append("compile/first-call latency is invalid")

    if not isinstance(process_index, int) or isinstance(process_index, bool):
        local_errors.append("process index is invalid")
        process_index = -1
    elif not 0 <= process_index < timing["escalated_processes"]:
        local_errors.append("process index is outside the frozen 0..6 range")
    elif process_index >= timing["base_processes"] and row_id != PACKED_N10_ROW_ID:
        local_errors.append("non-packed-n10 row contains escalation process")
    expected_order = rotation_for_process(max(process_index, 0))
    if receipt.get("execution_sequence") != list(expected_order):
        local_errors.append("three-way execution sequence drifted")
    expected_position = (
        expected_order.index(implementation) if implementation in expected_order else None
    )
    if receipt.get("execution_order") != expected_position:
        local_errors.append("three-way execution position drifted")
    token = receipt.get("launch_token")
    expected_token = expected_launch_token(
        contract_sha256, row_id, implementation, max(process_index, 0)
    )
    if token != expected_token:
        local_errors.append("launch token mismatch")
    launch = launches.get(str(token))
    if launch is None:
        local_errors.append("receipt has no fresh-process launch-ledger entry")
    else:
        expected_launch_identity = (
            row_id,
            implementation,
            process_index,
            receipt.get("execution_order"),
        )
        observed_launch_identity = (
            launch.get("row_id"),
            launch.get("impl"),
            launch.get("process_index"),
            launch.get("execution_order"),
        )
        if observed_launch_identity != expected_launch_identity:
            local_errors.append("launch-ledger process identity mismatch")
        if launch.get("child_pid") != receipt.get("worker_pid"):
            local_errors.append("launch-ledger child PID differs from worker PID")

    binding = receipt.get("gpu_binding")
    expected_uuid = timing["expected_gpu_uuid"]
    if not isinstance(binding, dict):
        local_errors.append("CUDA logical-to-physical binding is missing")
    else:
        if binding.get("cuda_visible_devices") != expected_uuid:
            local_errors.append("CUDA_VISIBLE_DEVICES was not bound by exact UUID")
        if binding.get("logical_index") != timing["cuda_logical_device"]:
            local_errors.append("logical CUDA index mismatch")
        if binding.get("resolved_physical_index") != timing["physical_gpu_index"]:
            local_errors.append("resolved physical GPU index mismatch")
        if binding.get("resolved_gpu_uuid") != expected_uuid:
            local_errors.append("logical CUDA process mapped to the wrong physical UUID")
        if binding.get("device_count") != 1:
            local_errors.append("UUID binding did not expose exactly one CUDA device")
        if binding.get("compute_capability") != "9.0":
            local_errors.append("benchmark did not run on compute capability 9.0")
    before = receipt.get("gpu_state_before")
    after = receipt.get("gpu_state_after")
    for label, state in (("before", before), ("after", after)):
        if not isinstance(state, dict):
            local_errors.append(f"GPU state {label} is missing")
            continue
        if state.get("physical_index") != timing["physical_gpu_index"]:
            local_errors.append(f"GPU physical index {label} mismatch")
        if state.get("uuid") != expected_uuid:
            local_errors.append(f"GPU UUID {label} mismatch")
    if isinstance(before, dict):
        util = before.get("util_pct")
        if not isinstance(util, int | float) or util > timing["max_gpu_util_pct"]:
            local_errors.append("pre-timing GPU utilization exceeds policy")
    if receipt.get("gpu_processes_before") != []:
        local_errors.append("GPU had compute processes before timing")
    if receipt.get("foreign_gpu_processes_after") != []:
        local_errors.append("foreign compute process was observed after timing")
    worker_pid = receipt.get("worker_pid")
    processes_after = receipt.get("gpu_processes_after")
    if (
        not isinstance(processes_after, list)
        or any(not isinstance(item, dict) for item in processes_after)
        or any(item.get("pid") != worker_pid for item in processes_after)
    ):
        local_errors.append("post-timing compute-process list contains a foreign PID")
    quiet = receipt.get("gpu_quiet_gate")
    if (
        not isinstance(quiet, dict)
        or quiet.get("timeout_s") != timing["quiet_timeout_s"]
        or quiet.get("poll_interval_s") != timing["quiet_poll_interval_s"]
        or not isinstance(quiet.get("polls"), int)
        or quiet["polls"] < 1
    ):
        local_errors.append("quiet-GPU gate receipt mismatch")

    caches = receipt.get("cuda_cache_env")
    if not isinstance(caches, dict) or set(caches) != set(CACHE_ENV_KEYS):
        local_errors.append("cache identity set is incomplete")
    else:
        local_paths = list(caches.values())
        if not all(isinstance(cache, str) for cache in local_paths):
            local_errors.append("cache identity contains a non-string path")
        elif len(local_paths) != len(set(local_paths)):
            local_errors.append("cache paths are not unique within the receipt")
        for key, cache in caches.items():
            if not isinstance(cache, str) or not Path(cache).is_absolute():
                local_errors.append(f"cache path {key} is not absolute")
                continue
            owner = (row_id, implementation, process_index, key)
            if cache in cache_owners:
                local_errors.append(f"cache path reused by {cache_owners[cache]}")
            else:
                cache_owners[cache] = owner

    if receipt.get("row") != row_map[row_id]:
        local_errors.append("embedded row contract drifted")
    if receipt.get("input_seed") != row_map[row_id]["seed"]:
        local_errors.append("input seed drifted")
    oracle_sha256 = receipt.get("oracle_sha256")
    if (
        not isinstance(oracle_sha256, str)
        or len(oracle_sha256) != 64
        or any(character not in "0123456789abcdef" for character in oracle_sha256)
    ):
        local_errors.append("oracle semantic hash is invalid")
    return local_errors


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit raw fresh-process GDN receipts and write an aggregate JSON report."
    )
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--receipts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite report: {args.output}")
    contract, contract_sha256 = load_contract(args.contract)
    errors: list[str] = []
    incomplete: list[str] = []
    activate_cutedsl_dependency_root(Path(contract["cutedsl_dependency_root"]))
    try:
        report_before = verify_contract_report_root(contract, executing_file=Path(__file__))
    except Exception as error:
        report_before = None
        errors.append(f"pre-report harness verification failed: {error}")
    try:
        dependency_before = verify_contract_cutedsl_dependency_root(contract)
    except Exception as error:
        dependency_before = None
        errors.append(f"pre-report CuTeDSL dependency verification failed: {error}")
    try:
        runtime_before = verify_contract_runtime_identity(contract)
    except Exception as error:
        runtime_before = None
        errors.append(f"pre-report runtime verification failed: {error}")
    try:
        verify_contract_source_roots(contract)
    except Exception as error:
        errors.append(f"live source-lock verification failed: {error}")
    try:
        verify_contract_runtime_root(contract, verify_sha256=True)
    except Exception as error:
        errors.append(f"live TVM build verification failed: {error}")
    row_map = {row["row_id"]: row for row in contract["rows"]}
    launches = _load_launches(args.receipts.parent / "launches.jsonl", errors)
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    receipt_identities: set[tuple[str, str, Any]] = set()
    cache_owners: dict[str, tuple[str, str, int, str]] = {}
    oracle_hashes: dict[str, set[str]] = defaultdict(set)

    paths = sorted(args.receipts.glob("*.json"))
    if not paths:
        incomplete.append("no timing receipts were found")
    for path in paths:
        try:
            receipt = json.loads(path.read_text())
        except json.JSONDecodeError as error:
            errors.append(f"{path}: invalid JSON: {error}")
            continue
        row_id = receipt.get("row_id")
        implementation = receipt.get("impl")
        process_index = receipt.get("process_index")
        local_errors = _audit_receipt(
            path=path,
            receipt=receipt,
            contract=contract,
            contract_sha256=contract_sha256,
            row_map=row_map,
            cache_owners=cache_owners,
            launches=launches,
        )
        errors.extend(f"{path}: {message}" for message in local_errors)
        if row_id in row_map and implementation in IMPLEMENTATION_ORDER:
            identity_index: Any = process_index
            if not isinstance(process_index, int) or isinstance(process_index, bool):
                identity_index = f"INVALID:{process_index!r}"
            identity = (row_id, implementation, identity_index)
            if identity in receipt_identities:
                errors.append(f"{path}: duplicate receipt identity {identity}")
            receipt_identities.add(identity)
            grouped[(row_id, implementation)].append(receipt)
            oracle_sha256 = receipt.get("oracle_sha256")
            if isinstance(oracle_sha256, str) and len(oracle_sha256) == 64:
                oracle_hashes[row_id].add(oracle_sha256)

    packed_base_receipts = {
        implementation: [
            receipt
            for receipt in grouped[(PACKED_N10_ROW_ID, implementation)]
            if isinstance(receipt.get("process_index"), int)
            and receipt["process_index"] < contract["timing"]["base_processes"]
            and receipt.get("status") == "timing_ok"
            and _is_positive_number(
                receipt.get("summary", {}).get("average_ms")
                if isinstance(receipt.get("summary"), dict)
                else None
            )
        ]
        for implementation in IMPLEMENTATION_ORDER
    }
    packed_base_ratios = None
    packed_should_escalate = None
    if all(
        len(receipts) == contract["timing"]["base_processes"]
        for receipts in packed_base_receipts.values()
    ):
        packed_medians = {
            implementation: median_process_average(receipts)
            for implementation, receipts in packed_base_receipts.items()
        }
        packed_base_ratios = {
            "tirx_over_cutedsl": packed_medians["tirx"] / packed_medians["cutedsl"],
            "tirx_over_fla": packed_medians["tirx"] / packed_medians["fla"],
        }
        packed_should_escalate = any(
            escalation_required(packed_base_ratios[name], contract["timing"]["noise_band_pct"])
            for name in contract["timing"]["escalation_ratios"]
        )
    else:
        incomplete.append("packed-n10 base receipts are incomplete; escalation cannot be decided")

    expected_counts: dict[str, int] = {}
    for row_id in row_map:
        expected_counts[row_id] = contract["timing"]["base_processes"]
    if packed_should_escalate:
        expected_counts[PACKED_N10_ROW_ID] = contract["timing"]["escalated_processes"]

    rows: dict[str, Any] = {}
    primary_cutedsl = []
    primary_fla = []
    for row_id, row in row_map.items():
        expected_count = expected_counts[row_id]
        medians = {}
        process_averages = {}
        observed_counts = {}
        for implementation in IMPLEMENTATION_ORDER:
            receipts = grouped[(row_id, implementation)]
            indices = [receipt.get("process_index") for receipt in receipts]
            observed_counts[implementation] = len(receipts)
            valid_indices = [
                index for index in indices if isinstance(index, int) and not isinstance(index, bool)
            ]
            if len(valid_indices) != len(set(valid_indices)):
                errors.append(f"{row_id}/{implementation}: duplicate process indices")
            expected_indices = list(range(expected_count))
            if sorted(valid_indices) != expected_indices:
                if any(index >= expected_count for index in valid_indices):
                    errors.append(
                        f"{row_id}/{implementation}: unexpected process indices {indices!r}"
                    )
                else:
                    incomplete.append(
                        f"{row_id}/{implementation}: expected processes {expected_indices}, "
                        f"observed {indices!r}"
                    )
            accepted = [
                receipt
                for receipt in receipts
                if receipt.get("status") == "timing_ok"
                and _is_positive_number(
                    receipt.get("summary", {}).get("average_ms")
                    if isinstance(receipt.get("summary"), dict)
                    else None
                )
            ]
            if len(accepted) == expected_count:
                process_averages[implementation] = [
                    float(receipt["summary"]["average_ms"])
                    for receipt in sorted(accepted, key=lambda item: item["process_index"])
                ]
                medians[implementation] = median_process_average(accepted)
        if len(oracle_hashes[row_id]) != 1:
            errors.append(f"{row_id}: oracle semantic hashes are not unique")
        row_report: dict[str, Any] = {
            "primary": bool(row["primary"]),
            "critical": bool(row["critical"]),
            "expected_processes": expected_count,
            "observed_processes": observed_counts,
            "process_averages_ms": process_averages,
            "median_ms": medians,
            "oracle_sha256": next(iter(oracle_hashes[row_id]), None),
        }
        if set(medians) == set(IMPLEMENTATION_ORDER):
            tirx_over_cutedsl = medians["tirx"] / medians["cutedsl"]
            tirx_over_fla = medians["tirx"] / medians["fla"]
            row_report["tirx_over_cutedsl"] = tirx_over_cutedsl
            row_report["tirx_over_fla"] = tirx_over_fla
            if row["primary"]:
                primary_cutedsl.append(tirx_over_cutedsl)
                primary_fla.append(tirx_over_fla)
        rows[row_id] = row_report

    if len(launches) != len(paths):
        errors.append(
            f"fresh-process launch count {len(launches)} differs from receipt count {len(paths)}"
        )
    run_summary_path = args.receipts.parent / "run-summary.json"
    run_summary = None
    if run_summary_path.is_file():
        run_summary = json.loads(run_summary_path.read_text())
        if run_summary.get("status") != "PASS":
            errors.append("run summary did not pass")
        if run_summary.get("contract_sha256") != contract_sha256:
            errors.append("run summary contract digest mismatch")
        if run_summary.get("receipt_count") != len(paths):
            errors.append("run summary receipt count mismatch")
        if run_summary.get("packed_n10_escalated") != packed_should_escalate:
            errors.append("run summary escalation decision mismatch")
        if run_summary.get("packed_n10_base_ratios") != packed_base_ratios:
            errors.append("run summary packed-n10 base ratios mismatch")
        if (
            run_summary.get("physical_gpu_index") != contract["timing"]["physical_gpu_index"]
            or run_summary.get("expected_gpu_uuid") != contract["timing"]["expected_gpu_uuid"]
        ):
            errors.append("run summary GPU identity mismatch")
        if run_summary.get("source_attestation_after") != contract["source_attestations"]:
            errors.append("run summary source attestation mismatch")
        if (
            run_summary.get("report_attestation_before") != contract["report_attestation"]
            or run_summary.get("report_attestation_after") != contract["report_attestation"]
            or run_summary.get("report_attestation_sha256") != contract["report_attestation_sha256"]
        ):
            errors.append("run summary report harness attestation mismatch")
        if (
            run_summary.get("cutedsl_dependency_attestation_before")
            != contract["cutedsl_dependency_attestation"]
            or run_summary.get("cutedsl_dependency_attestation_after")
            != contract["cutedsl_dependency_attestation"]
            or run_summary.get("cutedsl_dependency_attestation_sha256")
            != contract["cutedsl_dependency_attestation_sha256"]
        ):
            errors.append("run summary CuTeDSL dependency attestation mismatch")
        if (
            run_summary.get("runtime_identity_before") != contract["runtime_identity"]
            or run_summary.get("runtime_identity_after") != contract["runtime_identity"]
            or run_summary.get("runtime_identity_sha256") != contract["runtime_identity_sha256"]
        ):
            errors.append("run summary runtime identity mismatch")
        if (
            run_summary.get("tvm_build_attestation_before") != contract["tvm_build_attestation"]
            or run_summary.get("tvm_build_attestation_after") != contract["tvm_build_attestation"]
            or run_summary.get("tvm_build_attestation_sha256")
            != contract["tvm_build_attestation_sha256"]
        ):
            errors.append("run summary TVM build attestation mismatch")
    else:
        incomplete.append("run summary is missing")

    try:
        report_after = verify_contract_report_root(contract, executing_file=Path(__file__))
    except Exception as error:
        report_after = None
        errors.append(f"post-report harness verification failed: {error}")
    try:
        dependency_after = verify_contract_cutedsl_dependency_root(contract)
    except Exception as error:
        dependency_after = None
        errors.append(f"post-report CuTeDSL dependency verification failed: {error}")
    try:
        runtime_after = verify_contract_runtime_identity(contract)
    except Exception as error:
        runtime_after = None
        errors.append(f"post-report runtime verification failed: {error}")

    status = "FAIL" if errors else ("INCOMPLETE" if incomplete else "PASS")
    report = {
        "schema": "gdn-sm90a.public-fresh-three-way-report.v1",
        "run_id": contract["run_id"],
        "contract_sha256": contract_sha256,
        "report_attestation_sha256": contract["report_attestation_sha256"],
        "report_attestation_before": report_before,
        "report_attestation_after": report_after,
        "cutedsl_dependency_attestation_sha256": contract["cutedsl_dependency_attestation_sha256"],
        "cutedsl_dependency_attestation_before": dependency_before,
        "cutedsl_dependency_attestation_after": dependency_after,
        "runtime_identity_sha256": contract["runtime_identity_sha256"],
        "runtime_identity_before": runtime_before,
        "runtime_identity_after": runtime_after,
        "source_attestation_sha256": contract["source_attestation_sha256"],
        "tvm_build_attestation_sha256": contract["tvm_build_attestation_sha256"],
        "ratio_direction": "TIRx latency / comparator latency; lower is faster",
        "timer": "CUDA events around the public call",
        "statistic": "median of per-process averages",
        "warmup_iters": 20,
        "timed_iters": 100,
        "base_processes": 3,
        "packed_n10_escalated_processes": 7,
        "packed_n10_noise_band_pct": 2.0,
        "packed_n10_base_ratios": packed_base_ratios,
        "packed_n10_escalation_required": packed_should_escalate,
        "rows": rows,
        "primary_geomean": {
            "tirx_over_cutedsl": geometric_mean(primary_cutedsl),
            "tirx_over_fla": geometric_mean(primary_fla),
        },
        "receipt_count": len(paths),
        "fresh_process_launch_count": len(launches),
        "unique_cache_path_count": len(cache_owners),
        "run_summary": run_summary,
        "errors": errors,
        "incomplete_reasons": incomplete,
        "status": status,
        "decision_status": "CHARACTERIZATION" if status == "PASS" else status,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if status != "PASS":
        raise SystemExit(f"GDN_REPORT_{status}")
    print(
        f"GDN_REPORT_PASS tirx/cutedsl="
        f"{report['primary_geomean']['tirx_over_cutedsl']:.6f} "
        f"tirx/fla={report['primary_geomean']['tirx_over_fla']:.6f} "
        f"receipts={len(paths)}"
    )


if __name__ == "__main__":
    main()
