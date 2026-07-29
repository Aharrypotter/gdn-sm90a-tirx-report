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
"""Generate frozen correctness tensors from the corrected CuTe GDN comparator."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import time
from pathlib import Path

from .contract import (
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
    _cutedsl_runner,
    _gpu_processes,
    _gpu_state,
    _inputs,
    _verify_cuda_binding,
    _wait_for_quiet_gpu,
    semantic_hash,
)


def _prepare_cache_root(root: Path) -> dict[str, str]:
    if root.exists():
        raise FileExistsError(f"refusing to reuse oracle cache root: {root}")
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate six CuTe GDN correctness oracles from the locked public tag."
    )
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    args = parser.parse_args()
    contract, contract_sha256 = load_contract(args.contract)
    output = args.output.expanduser().resolve()
    if output != Path(contract["oracle_root"]).resolve():
        raise ValueError(
            f"--output must equal contract oracle_root: {output} != {contract['oracle_root']}"
        )
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to reuse non-empty oracle output: {output}")

    activate_cutedsl_dependency_root(Path(contract["cutedsl_dependency_root"]))
    report_before = verify_contract_report_root(contract, executing_file=Path(__file__))
    dependency_before = verify_contract_cutedsl_dependency_root(contract)
    runtime_before = verify_contract_runtime_identity(contract)
    source_before = verify_contract_source_roots(contract)
    tvm_build_before = verify_contract_runtime_root(contract, verify_sha256=False)
    output.mkdir(parents=True, exist_ok=True)
    os.environ.update(_prepare_cache_root(args.cache_root.expanduser().resolve()))
    timing = contract["timing"]
    expected_uuid = str(timing["expected_gpu_uuid"])
    os.environ["CUDA_VISIBLE_DEVICES"] = expected_uuid
    os.environ["FLA_DISABLE_BACKEND_DISPATCH"] = "1"
    before, processes_before, quiet_gate = _wait_for_quiet_gpu(
        int(timing["physical_gpu_index"]),
        expected_uuid,
        max_util_pct=float(timing["max_gpu_util_pct"]),
        timeout_s=float(timing["quiet_timeout_s"]),
        poll_interval_s=float(timing["quiet_poll_interval_s"]),
    )
    torch = importlib.import_module("torch")
    gpu_binding = _verify_cuda_binding(torch, timing)

    rows = []
    for row in contract["rows"]:
        path = output / f"{row['row_id']}.pt"
        if path.exists():
            raise FileExistsError(f"refusing to overwrite oracle: {path}")
        inputs = _inputs(torch, row)
        call, backend, wrapped_chain, runner_attestation = _cutedsl_runner(row, inputs, contract)
        expected_backend = row["expected_backends"]["cutedsl"]
        if backend != expected_backend:
            raise RuntimeError(
                f"{row['row_id']}: expected backend {expected_backend!r}, resolved {backend!r}"
            )
        output_tensor, state_tensor = call()
        torch.cuda.synchronize()
        output_cpu = output_tensor.detach().cpu()
        state_cpu = None if state_tensor is None else state_tensor.detach().cpu()
        oracle_sha256 = semantic_hash(output_cpu, state_cpu, row)
        payload = {
            "schema": "gdn-sm90a.cutedsl-oracle.v1",
            "row_id": row["row_id"],
            "contract_sha256": contract_sha256,
            "source_identity": contract["implementations"]["cutedsl"]["source_identity"],
            "backend_identity": backend,
            "entrypoint": "cula.gdn.prefill.chunk_gated_delta_rule",
            "runner_attestation": runner_attestation,
            "wrapped_callable_chain": wrapped_chain,
            "row": row,
            "output": output_cpu,
            "state": state_cpu,
            "oracle_sha256": oracle_sha256,
        }
        temporary = path.with_suffix(".pt.tmp")
        torch.save(payload, temporary)
        temporary.replace(path)
        rows.append(
            {
                "row_id": row["row_id"],
                "path": path.name,
                "oracle_sha256": oracle_sha256,
                "state_present": state_cpu is not None,
            }
        )
        print(f"GDN_ORACLE_OK row={row['row_id']} sha256={oracle_sha256} path={path}")

    after = _gpu_state(int(timing["physical_gpu_index"]))
    processes_after = _gpu_processes(expected_uuid)
    foreign_processes = [item for item in processes_after if item["pid"] != os.getpid()]
    if after["uuid"] != expected_uuid or foreign_processes:
        raise RuntimeError(
            "oracle GPU contention/identity gate failed: "
            f"after={after}, foreign={foreign_processes}"
        )
    report_after = verify_contract_report_root(contract, executing_file=Path(__file__))
    dependency_after = verify_contract_cutedsl_dependency_root(contract)
    runtime_after = verify_contract_runtime_identity(
        contract,
        require_cuda_uninitialized=False,
    )
    source_after = verify_contract_source_roots(contract)
    tvm_build_after = verify_contract_runtime_root(contract, verify_sha256=False)
    manifest = {
        "schema": "gdn-sm90a.cutedsl-oracle-manifest.v1",
        "status": "PASS",
        "created_unix_s": time.time(),
        "contract_sha256": contract_sha256,
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
        "entrypoint": "cula.gdn.prefill.chunk_gated_delta_rule",
        "backend_identity": contract["implementations"]["cutedsl"]["backend_identity"],
        "gpu_binding": gpu_binding,
        "gpu_state_before": before,
        "gpu_state_after": after,
        "gpu_processes_before": processes_before,
        "gpu_processes_after": processes_after,
        "foreign_gpu_processes_after": foreign_processes,
        "gpu_quiet_gate": quiet_gate,
        "rows": rows,
    }
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite oracle manifest: {manifest_path}")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"GDN_ORACLE_SET_OK rows={len(rows)} manifest={manifest_path}")


if __name__ == "__main__":
    main()
