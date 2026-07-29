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
"""Run the frozen public-tag three-way GDN benchmark in fresh processes."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .contract import (
    PACKED_N10_ROW_ID,
    activate_cutedsl_dependency_root,
    load_contract,
    verify_contract_cutedsl_dependency_root,
    verify_contract_report_root,
    verify_contract_runtime_identity,
    verify_contract_runtime_root,
    verify_contract_source_roots,
)
from .worker import (
    CACHE_ENV_KEYS,
    _gpu_state,
    expected_launch_token,
    rotation_for_process,
)


def escalation_required(ratio: float, noise_band_pct: float = 2.0) -> bool:
    """Return whether a base TIRx/CuTe ratio requires the frozen 3->7 escalation."""

    if not isinstance(ratio, int | float) or not 0 < ratio < float("inf"):
        raise ValueError(f"ratio must be a positive finite number, got {ratio!r}")
    if noise_band_pct < 0:
        raise ValueError("noise_band_pct must be non-negative")
    distance = abs(float(ratio) - 1.0)
    limit = float(noise_band_pct) / 100.0
    return distance <= limit or math.isclose(distance, limit, rel_tol=0.0, abs_tol=1.0e-12)


def _prepare_cache_environment(root: Path) -> dict[str, str]:
    if root.exists():
        raise FileExistsError(f"refusing to reuse cache identity: {root}")
    names = {
        "CUDA_CACHE_PATH": "cuda",
        "CUTE_DSL_CACHE_DIR": "cute",
        "CUTLASS_DSL_CACHE_DIR": "cutlass",
        "TRITON_CACHE_DIR": "triton",
        "TVM_FFI_CACHE_DIR": "tvm-ffi",
        "TMPDIR": "tmp",
    }
    environment = {}
    for key in CACHE_ENV_KEYS:
        path = (root / names[key]).resolve()
        path.mkdir(parents=True, exist_ok=False)
        environment[key] = str(path)
    return environment


def compose_pythonpath(contract: dict[str, Any], existing_pythonpath: str | None) -> str:
    """Prepend locked sources while preserving an explicit noneditable dependency path."""

    roots = contract["source_roots"]
    tvm_root = Path(roots["tvm"])
    dependency_root = Path(contract["cutedsl_dependency_root"])
    candidates = (
        Path(contract["report_root"]),
        Path(roots["tirx"]),
        tvm_root / "python",
        Path(roots["cutedsl"]),
        Path(roots["fla"]),
        dependency_root,
        dependency_root / "nvidia_cutlass_dsl" / "python_packages",
    )
    paths = [str(path.expanduser().resolve()) for path in candidates]
    if existing_pythonpath:
        paths.extend(
            str(Path(item).expanduser().resolve())
            for item in existing_pythonpath.split(os.pathsep)
            if item
        )
    return os.pathsep.join(dict.fromkeys(paths))


def _receipt_path(output: Path, row_id: str, implementation: str, process_index: int) -> Path:
    return output / "timing" / f"{row_id}__{implementation}__p{process_index}.json"


def _base_ratios(output: Path, contract: dict[str, Any], contract_sha256: str) -> dict[str, float]:
    medians = {}
    base_processes = int(contract["timing"]["base_processes"])
    for implementation in ("tirx", "cutedsl", "fla"):
        averages = []
        for process_index in range(base_processes):
            path = _receipt_path(output, PACKED_N10_ROW_ID, implementation, process_index)
            receipt = json.loads(path.read_text())
            if receipt.get("status") != "timing_ok":
                raise RuntimeError(f"cannot decide escalation from invalid receipt {path}")
            if receipt.get("contract_sha256") != contract_sha256:
                raise RuntimeError(f"cannot decide escalation from contract-drifted receipt {path}")
            if (
                receipt.get("row_id"),
                receipt.get("impl"),
                receipt.get("process_index"),
            ) != (PACKED_N10_ROW_ID, implementation, process_index):
                raise RuntimeError(f"cannot decide escalation from misidentified receipt {path}")
            averages.append(float(receipt["summary"]["average_ms"]))
        medians[implementation] = statistics.median(averages)
    return {
        "tirx_over_cutedsl": medians["tirx"] / medians["cutedsl"],
        "tirx_over_fla": medians["tirx"] / medians["fla"],
    }


def _verify_oracles(contract: dict[str, Any], contract_sha256: str) -> None:
    oracle_root = Path(contract["oracle_root"])
    manifest_path = oracle_root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"missing oracle manifest; run python -m reproduce.benchmark.oracle first: "
            f"{manifest_path}"
        )
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "PASS":
        raise RuntimeError("oracle manifest did not pass")
    if manifest.get("contract_sha256") != contract_sha256:
        raise RuntimeError("oracle manifest contract digest drifted")
    if manifest.get("entrypoint") != "cula.gdn.prefill.chunk_gated_delta_rule":
        raise RuntimeError("oracle manifest does not use the corrected cula.gdn comparator")
    if manifest.get("backend_identity") != "sm90_cutedsl_gdn":
        raise RuntimeError("oracle manifest backend identity drifted")
    if (
        manifest.get("report_attestation_before") != contract["report_attestation"]
        or manifest.get("report_attestation_after") != contract["report_attestation"]
        or manifest.get("report_attestation_sha256") != contract["report_attestation_sha256"]
    ):
        raise RuntimeError("oracle manifest report harness attestation drifted")
    if (
        manifest.get("cutedsl_dependency_attestation_before")
        != contract["cutedsl_dependency_attestation"]
        or manifest.get("cutedsl_dependency_attestation_after")
        != contract["cutedsl_dependency_attestation"]
        or manifest.get("cutedsl_dependency_attestation_sha256")
        != contract["cutedsl_dependency_attestation_sha256"]
    ):
        raise RuntimeError("oracle manifest CuTeDSL dependency attestation drifted")
    if (
        manifest.get("runtime_identity_before") != contract["runtime_identity"]
        or manifest.get("runtime_identity_after") != contract["runtime_identity"]
        or manifest.get("runtime_identity_sha256") != contract["runtime_identity_sha256"]
    ):
        raise RuntimeError("oracle manifest runtime identity drifted")
    expected_rows = {row["row_id"] for row in contract["rows"]}
    observed_rows = {row.get("row_id") for row in manifest.get("rows", [])}
    if observed_rows != expected_rows:
        raise RuntimeError("oracle manifest does not cover the frozen six rows")
    for row_id in expected_rows:
        if not (oracle_root / f"{row_id}.pt").is_file():
            raise FileNotFoundError(f"missing CuTe correctness oracle for {row_id}")


def _append_jsonl(path: Path, value: dict[str, Any]) -> None:
    with path.open("a") as stream:
        stream.write(json.dumps(value, sort_keys=True) + "\n")


def _run_one(
    *,
    contract_path: Path,
    contract: dict[str, Any],
    contract_sha256: str,
    output: Path,
    cache_root: Path,
    row_id: str,
    implementation: str,
    process_index: int,
    execution_order: int,
) -> None:
    receipt = _receipt_path(output, row_id, implementation, process_index)
    log = output / "logs" / f"{row_id}__{implementation}__p{process_index}.log"
    cache = cache_root / contract["run_id"] / row_id / implementation / f"p{process_index}"
    if receipt.exists():
        raise FileExistsError(f"refusing to overwrite receipt: {receipt}")
    environment = os.environ.copy()
    environment.update(_prepare_cache_environment(cache))
    expected_uuid = contract["timing"]["expected_gpu_uuid"]
    environment["CUDA_VISIBLE_DEVICES"] = expected_uuid
    environment["FLA_DISABLE_BACKEND_DISPATCH"] = "1"
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PYTHONPATH"] = compose_pythonpath(contract, environment.get("PYTHONPATH"))
    tvm_library_path = Path(contract["tvm_build_attestation"]["lib_dir"])
    environment["TVM_LIBRARY_PATH"] = str(tvm_library_path.resolve())
    existing_ld = environment.get("LD_LIBRARY_PATH")
    environment["LD_LIBRARY_PATH"] = (
        str(tvm_library_path.resolve())
        if not existing_ld
        else f"{tvm_library_path.resolve()}{os.pathsep}{existing_ld}"
    )

    launch_token = expected_launch_token(contract_sha256, row_id, implementation, process_index)
    command = [
        sys.executable,
        "-m",
        "reproduce.benchmark.worker",
        "--contract",
        str(contract_path),
        "--row-id",
        row_id,
        "--implementation",
        implementation,
        "--process-index",
        str(process_index),
        "--execution-order",
        str(execution_order),
        "--launch-token",
        launch_token,
        "--receipt",
        str(receipt),
    ]
    receipt.parent.mkdir(parents=True, exist_ok=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    print(
        f"START row={row_id} impl={implementation} process={process_index} order={execution_order}",
        flush=True,
    )
    with log.open("x") as stream:
        process = subprocess.Popen(
            command,
            cwd=Path(contract["report_root"]),
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
        _append_jsonl(
            output / "launches.jsonl",
            {
                "row_id": row_id,
                "impl": implementation,
                "process_index": process_index,
                "execution_order": execution_order,
                "execution_sequence": list(rotation_for_process(process_index)),
                "launch_token": launch_token,
                "child_pid": process.pid,
                "started_unix_s": time.time(),
            },
        )
        returncode = process.wait()
    parsed_receipt = json.loads(receipt.read_text()) if receipt.is_file() else None
    if (
        returncode != 0
        or not isinstance(parsed_receipt, dict)
        or parsed_receipt.get("status") != "timing_ok"
    ):
        tail = "\n".join(log.read_text(errors="replace").splitlines()[-80:])
        raise RuntimeError(
            f"{row_id}/{implementation}/p{process_index} failed with status {returncode}\n{tail}"
        )
    print(f"PASS row={row_id} impl={implementation} process={process_index}", flush=True)


def _run_process_range(
    *,
    rows: list[dict[str, Any]],
    process_start: int,
    process_stop: int,
    contract_path: Path,
    contract: dict[str, Any],
    contract_sha256: str,
    output: Path,
    cache_root: Path,
) -> None:
    for row in rows:
        row_id = row["row_id"]
        for process_index in range(process_start, process_stop):
            order = rotation_for_process(process_index)
            for execution_order, implementation in enumerate(order):
                _run_one(
                    contract_path=contract_path,
                    contract=contract,
                    contract_sha256=contract_sha256,
                    output=output,
                    cache_root=cache_root,
                    row_id=row_id,
                    implementation=implementation,
                    process_index=process_index,
                    execution_order=execution_order,
                )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the exact six-row, three-way benchmark. This command never installs "
            "software or kills external GPU processes."
        )
    )
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    args = parser.parse_args()
    contract_path = args.contract.expanduser().resolve()
    contract, contract_sha256 = load_contract(contract_path)
    output = args.output.expanduser().resolve()
    cache_root = args.cache_root.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to reuse benchmark output root: {output}")
    if cache_root.exists():
        raise FileExistsError(f"refusing to reuse benchmark cache root: {cache_root}")
    activate_cutedsl_dependency_root(Path(contract["cutedsl_dependency_root"]))
    report_before = verify_contract_report_root(contract, executing_file=Path(__file__))
    dependency_before = verify_contract_cutedsl_dependency_root(contract)
    runtime_before = verify_contract_runtime_identity(contract)
    source_before = verify_contract_source_roots(contract)
    tvm_build_before = verify_contract_runtime_root(contract, verify_sha256=True)
    output.mkdir(parents=True)
    cache_root.mkdir(parents=True)
    (output / "contract.json").write_bytes(contract_path.read_bytes())
    (output / "contract.sha256").write_text(f"{contract_sha256}  contract.json\n")
    _verify_oracles(contract, contract_sha256)
    gpu = _gpu_state(int(contract["timing"]["physical_gpu_index"]))
    if gpu["uuid"] != contract["timing"]["expected_gpu_uuid"]:
        raise RuntimeError(
            f"physical GPU identity mismatch: expected "
            f"{contract['timing']['expected_gpu_uuid']}, observed {gpu['uuid']}"
        )

    base_processes = int(contract["timing"]["base_processes"])
    _run_process_range(
        rows=contract["rows"],
        process_start=0,
        process_stop=base_processes,
        contract_path=contract_path,
        contract=contract,
        contract_sha256=contract_sha256,
        output=output,
        cache_root=cache_root,
    )
    packed_ratios = _base_ratios(output, contract, contract_sha256)
    escalate = any(
        escalation_required(packed_ratios[name], contract["timing"]["noise_band_pct"])
        for name in contract["timing"]["escalation_ratios"]
    )
    if escalate:
        packed_row = [row for row in contract["rows"] if row["row_id"] == PACKED_N10_ROW_ID]
        _run_process_range(
            rows=packed_row,
            process_start=base_processes,
            process_stop=int(contract["timing"]["escalated_processes"]),
            contract_path=contract_path,
            contract=contract,
            contract_sha256=contract_sha256,
            output=output,
            cache_root=cache_root,
        )
    report_after = verify_contract_report_root(contract, executing_file=Path(__file__))
    dependency_after = verify_contract_cutedsl_dependency_root(contract)
    runtime_after = verify_contract_runtime_identity(contract)
    source_after = verify_contract_source_roots(contract)
    tvm_build_after = verify_contract_runtime_root(contract, verify_sha256=True)
    receipt_count = len(list((output / "timing").glob("*.json")))
    summary = {
        "schema": "gdn-sm90a.public-fresh-run-summary.v1",
        "status": "PASS",
        "run_id": contract["run_id"],
        "contract_sha256": contract_sha256,
        "physical_gpu_index": contract["timing"]["physical_gpu_index"],
        "expected_gpu_uuid": contract["timing"]["expected_gpu_uuid"],
        "cuda_binding": "CUDA_VISIBLE_DEVICES exact GPU UUID",
        "base_processes": base_processes,
        "packed_n10_base_ratios": packed_ratios,
        "noise_band_pct": contract["timing"]["noise_band_pct"],
        "packed_n10_escalated": escalate,
        "packed_n10_final_processes": (
            int(contract["timing"]["escalated_processes"]) if escalate else base_processes
        ),
        "receipt_count": receipt_count,
        "report_attestation_before": report_before,
        "report_attestation_after": report_after,
        "report_attestation_sha256": contract["report_attestation_sha256"],
        "cutedsl_dependency_attestation_before": dependency_before,
        "cutedsl_dependency_attestation_after": dependency_after,
        "cutedsl_dependency_attestation_sha256": contract["cutedsl_dependency_attestation_sha256"],
        "runtime_identity_before": runtime_before,
        "runtime_identity_after": runtime_after,
        "runtime_identity_sha256": contract["runtime_identity_sha256"],
        "source_attestation_before": source_before,
        "source_attestation_after": source_after,
        "tvm_build_attestation_before": tvm_build_before,
        "tvm_build_attestation_after": tvm_build_after,
        "tvm_build_attestation_sha256": contract["tvm_build_attestation_sha256"],
        "completed_unix_s": time.time(),
    }
    (output / "run-summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(
        f"GDN_BENCHMARK_RUN_OK receipts={receipt_count} "
        f"packed_n10_tirx/cutedsl={packed_ratios['tirx_over_cutedsl']:.6f} "
        f"packed_n10_tirx/fla={packed_ratios['tirx_over_fla']:.6f} escalated={escalate}"
    )


if __name__ == "__main__":
    main()
