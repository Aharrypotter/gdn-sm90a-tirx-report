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
"""One fresh-process, source-bound SM90a GDN timing worker."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from .contract import (
    CUTEDSL_DISTRIBUTIONS,
    IMPLEMENTATION_ORDER,
    PACKED_N10_ROW_ID,
    ContractError,
    activate_cutedsl_dependency_root,
    canonical_json_bytes,
    load_contract,
    verify_contract_cutedsl_dependency_root,
    verify_contract_report_root,
    verify_contract_runtime_identity,
    verify_contract_runtime_root,
    verify_contract_source_roots,
)

CACHE_ENV_KEYS = (
    "CUDA_CACHE_PATH",
    "CUTE_DSL_CACHE_DIR",
    "CUTLASS_DSL_CACHE_DIR",
    "TRITON_CACHE_DIR",
    "TVM_FFI_CACHE_DIR",
    "TMPDIR",
)
ROTATING_ORDERS = (
    ("tirx", "cutedsl", "fla"),
    ("cutedsl", "fla", "tirx"),
    ("fla", "tirx", "cutedsl"),
)


def rotation_for_process(process_index: int) -> tuple[str, str, str]:
    """Return the frozen three-way order for one independent process index."""

    if process_index < 0:
        raise ValueError("process_index must be non-negative")
    return ROTATING_ORDERS[process_index % len(ROTATING_ORDERS)]


def expected_launch_token(
    contract_sha256: str, row_id: str, implementation: str, process_index: int
) -> str:
    """Create a deterministic, globally unique launch identity."""

    payload = {
        "contract_sha256": contract_sha256,
        "row_id": row_id,
        "implementation": implementation,
        "process_index": process_index,
    }
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _optional_float(value: str) -> float | None:
    value = value.strip()
    if not value or value.upper() in {"N/A", "[N/A]", "NOT SUPPORTED"}:
        return None
    return float(value)


def parse_gpu_state_row(row: Sequence[str]) -> dict[str, Any]:
    """Parse one nvidia-smi telemetry row without importing CUDA libraries."""

    if len(row) != 10:
        raise RuntimeError(f"expected 10 nvidia-smi GPU fields, got {len(row)}")
    return {
        "physical_index": int(row[0].strip()),
        "uuid": row[1].strip(),
        "name": row[2].strip(),
        "util_pct": float(row[3].strip()),
        "memory_used_mib": float(row[4].strip()),
        "pstate": row[5].strip(),
        "sm_clock_mhz": _optional_float(row[6]),
        "memory_clock_mhz": _optional_float(row[7]),
        "temperature_c": _optional_float(row[8]),
        "power_draw_w": _optional_float(row[9]),
    }


def _gpu_state(physical_gpu: int) -> dict[str, Any]:
    output = subprocess.check_output(
        (
            "nvidia-smi",
            "--query-gpu=index,uuid,name,utilization.gpu,memory.used,pstate,"
            "clocks.sm,clocks.mem,temperature.gpu,power.draw",
            "--format=csv,noheader,nounits",
        ),
        text=True,
    )
    for row in csv.reader(output.splitlines()):
        parsed = parse_gpu_state_row(row)
        if parsed["physical_index"] == physical_gpu:
            return parsed
    raise RuntimeError(f"physical GPU {physical_gpu} was not reported by nvidia-smi")


def _gpu_processes(gpu_uuid: str) -> list[dict[str, Any]]:
    completed = subprocess.run(
        (
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ),
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or "nvidia-smi process query failed")
    result = []
    for row in csv.reader(completed.stdout.splitlines()):
        fields = [field.strip() for field in row]
        if len(fields) != 4 or fields[0] != gpu_uuid:
            continue
        result.append(
            {
                "gpu_uuid": fields[0],
                "pid": int(fields[1]),
                "process_name": fields[2],
                "used_memory_mib": _optional_float(fields[3]),
            }
        )
    return result


def _wait_for_quiet_gpu(
    physical_gpu: int,
    expected_uuid: str,
    *,
    max_util_pct: float,
    timeout_s: float,
    poll_interval_s: float,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    start = time.monotonic()
    polls = 0
    max_observed = 0.0
    while True:
        state = _gpu_state(physical_gpu)
        if state["uuid"] != expected_uuid:
            raise RuntimeError(
                f"physical GPU UUID mismatch: expected {expected_uuid}, observed {state['uuid']}"
            )
        processes = _gpu_processes(expected_uuid)
        polls += 1
        max_observed = max(max_observed, float(state["util_pct"]))
        elapsed = time.monotonic() - start
        if not processes and state["util_pct"] <= max_util_pct:
            return (
                state,
                processes,
                {
                    "elapsed_s": elapsed,
                    "polls": polls,
                    "max_observed_util_pct": max_observed,
                    "timeout_s": timeout_s,
                    "poll_interval_s": poll_interval_s,
                },
            )
        if elapsed >= timeout_s:
            raise RuntimeError(
                "GPU did not become quiet; this harness never kills external processes: "
                f"state={state}, processes={processes}"
            )
        time.sleep(poll_interval_s)


def _wait_for_pid_binding(gpu_uuid: str, pid: int, timeout_s: float = 5.0) -> dict[str, Any]:
    start = time.monotonic()
    while True:
        matches = [item for item in _gpu_processes(gpu_uuid) if item["pid"] == pid]
        if len(matches) == 1:
            return matches[0]
        if time.monotonic() - start >= timeout_s:
            raise RuntimeError(
                f"CUDA process {pid} was not observed on expected physical UUID {gpu_uuid}"
            )
        time.sleep(0.1)


def _require_path_under(path: Path, root: Path, label: str) -> str:
    resolved_path = path.resolve()
    resolved_root = root.resolve()
    if not resolved_path.is_relative_to(resolved_root):
        raise RuntimeError(
            f"{label} resolved outside locked root: {resolved_path} vs {resolved_root}"
        )
    return str(resolved_path)


def _require_module_under(module: Any, root: Path, label: str) -> str:
    module_file = getattr(module, "__file__", None)
    if not module_file:
        raise RuntimeError(f"{label} module has no source file")
    return _require_path_under(Path(module_file), root, label)


def path_is_within_any_root(path: Path, roots: Sequence[Path]) -> bool:
    """Return whether a resolved dependency path belongs to any source checkout."""

    resolved_path = path.resolve()
    return any(
        resolved_path == root.resolve() or resolved_path.is_relative_to(root.resolve())
        for root in roots
    )


def _require_module_outside_roots(module: Any, roots: Sequence[Path], label: str) -> str:
    module_file = getattr(module, "__file__", None)
    if not module_file:
        raise RuntimeError(f"{label} module has no source file")
    resolved = Path(module_file).resolve()
    if path_is_within_any_root(resolved, roots):
        raise RuntimeError(f"{label} must be a noneditable installation: {resolved}")
    return str(resolved)


def _tvm_ffi_distribution_identity() -> dict[str, str]:
    for distribution_name in ("apache-tvm-ffi", "tvm-ffi"):
        try:
            return {
                "distribution": distribution_name,
                "version": importlib.metadata.version(distribution_name),
            }
        except importlib.metadata.PackageNotFoundError:
            continue
    raise RuntimeError("tvm-ffi distribution metadata is unavailable")


def _cutedsl_dependency_module_identity(contract: dict[str, Any]) -> dict[str, Any]:
    """Prove that the CuTe DSL modules and distributions resolve under the attested root."""

    root = Path(contract["cutedsl_dependency_root"])
    activate_cutedsl_dependency_root(root)
    cutlass = importlib.import_module("cutlass")
    cute = importlib.import_module("cutlass.cute")
    distributions = {}
    for distribution_name in CUTEDSL_DISTRIBUTIONS:
        distribution = importlib.metadata.distribution(distribution_name)
        metadata_root = Path(distribution.locate_file("")).resolve()
        if not (metadata_root == root.resolve() or metadata_root.is_relative_to(root.resolve())):
            raise RuntimeError(
                f"{distribution_name} metadata escaped the attested dependency root: "
                f"{metadata_root}"
            )
        if distribution.version != "4.5.1":
            raise RuntimeError(
                f"{distribution_name} version drift: expected 4.5.1, "
                f"observed {distribution.version}"
            )
        distributions[distribution_name] = {
            "version": distribution.version,
            "metadata_root": str(metadata_root),
        }
    return {
        "root": str(root.resolve()),
        "cutlass_module_file": _require_module_under(cutlass, root, "cutlass"),
        "cutlass_cute_module_file": _require_module_under(cute, root, "cutlass.cute"),
        "distributions": distributions,
    }


def _prepend_sys_paths(paths: Sequence[Path]) -> None:
    resolved = [str(path.resolve()) for path in paths if path.exists()]
    sys.path[:] = resolved + [item for item in sys.path if item not in resolved]


def _wrapped_chain(function: Callable[..., Any]) -> list[str]:
    result = []
    current = function
    seen: set[int] = set()
    while callable(current) and id(current) not in seen:
        seen.add(id(current))
        name = getattr(
            current, "__qualname__", getattr(current, "__name__", type(current).__name__)
        )
        result.append(f"{current.__module__}.{name}")
        current = getattr(current, "__wrapped__", None)
    return result


def _versions() -> dict[str, str | None]:
    result = {}
    for name in (
        "apache-tvm-ffi",
        "flash-linear-attention",
        "nvidia-cutlass-dsl",
        "nvidia-cutlass-dsl-libs-base",
        "nvidia-cutlass-dsl-libs-cu13",
        "triton",
        "tvm-ffi",
    ):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            if name != "triton":
                result[name] = None
                continue
            module = importlib.import_module("triton")
            module_version = getattr(module, "__version__", None)
            if not isinstance(module_version, str) or not module_version:
                raise RuntimeError(
                    "triton has neither distribution metadata nor a module version"
                ) from None
            result[name] = module_version
    return result


def _cache_environment() -> dict[str, str]:
    values: dict[str, str] = {}
    seen: set[Path] = set()
    for key in CACHE_ENV_KEYS:
        value = os.environ.get(key)
        if not value:
            raise RuntimeError(f"isolated cache environment variable {key} is missing")
        path = Path(value).resolve()
        if not path.is_absolute() or not path.is_dir():
            raise RuntimeError(
                f"isolated cache path for {key} is not an absolute directory: {path}"
            )
        if path in seen:
            raise RuntimeError(f"two cache environment variables resolve to the same path: {path}")
        seen.add(path)
        values[key] = str(path)
    return values


def _verify_cuda_binding(torch: Any, timing: dict[str, Any]) -> dict[str, Any]:
    expected_uuid = str(timing["expected_gpu_uuid"])
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible != expected_uuid:
        raise RuntimeError(
            "CUDA logical device must be bound by exact UUID: "
            f"expected CUDA_VISIBLE_DEVICES={expected_uuid}, observed {visible!r}"
        )
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            "UUID binding must expose exactly one logical CUDA device, got "
            f"{torch.cuda.device_count()}"
        )
    logical_device = int(timing["cuda_logical_device"])
    torch.cuda.set_device(logical_device)
    probe = torch.empty(1, device=f"cuda:{logical_device}")
    probe.add_(1)
    torch.cuda.synchronize()
    properties = torch.cuda.get_device_properties(logical_device)
    if (int(properties.major), int(properties.minor)) != (9, 0):
        raise RuntimeError(
            f"fresh benchmark requires compute capability 9.0, got "
            f"{properties.major}.{properties.minor}"
        )
    process_binding = _wait_for_pid_binding(expected_uuid, os.getpid())
    return {
        "binding_method": "CUDA_VISIBLE_DEVICES exact GPU UUID plus nvidia-smi PID mapping",
        "cuda_visible_devices": visible,
        "logical_index": logical_device,
        "resolved_physical_index": int(timing["physical_gpu_index"]),
        "resolved_gpu_uuid": process_binding["gpu_uuid"],
        "process_binding": process_binding,
        "device_name": properties.name,
        "compute_capability": f"{properties.major}.{properties.minor}",
        "device_count": torch.cuda.device_count(),
    }


def _inputs(torch: Any, row: dict[str, Any]) -> dict[str, Any]:
    generator = torch.Generator(device="cuda").manual_seed(int(row["seed"]))
    lengths = tuple(int(value) for value in row["sequence_lengths"])
    total = sum(lengths)
    q_heads = int(row["q_heads"])
    v_heads = int(row["v_heads"])
    output_heads = max(q_heads, v_heads)
    q = (torch.randn(total, q_heads, 128, device="cuda", generator=generator) * 0.03).to(
        torch.bfloat16
    )
    k = (torch.randn(total, q_heads, 128, device="cuda", generator=generator) * 0.01).to(
        torch.bfloat16
    )
    v = (torch.randn(total, v_heads, 128, device="cuda", generator=generator) * 0.1).to(
        torch.bfloat16
    )
    alpha = torch.rand(total, output_heads, device="cuda", generator=generator) * 0.08 + 0.90
    beta = torch.rand(total, output_heads, device="cuda", generator=generator) * 0.5
    boundaries = [0]
    for length in lengths:
        boundaries.append(boundaries[-1] + length)
    cu_seqlens = torch.tensor(boundaries, dtype=torch.int32, device="cuda")
    initial_state = (
        torch.randn(len(lengths), output_heads, 128, 128, device="cuda", generator=generator) * 0.01
        if row["stateful"]
        else None
    )
    return {
        "q": q,
        "k": k,
        "v": v,
        "alpha": alpha,
        "beta": beta,
        "cu_seqlens": cu_seqlens,
        "cu_seqlens_long": cu_seqlens.to(torch.long),
        "initial_state": initial_state,
    }


def _tvm_runtime_attestation(tvm: Any, contract: dict[str, Any]) -> dict[str, Any]:
    if bool(tvm.base._RUNTIME_ONLY):
        raise RuntimeError("locked TIRx benchmark requires the TVM compiler library")
    build_attestation = contract["tvm_build_attestation"]
    build_lib_root = Path(build_attestation["lib_dir"])
    expected_libraries = {
        "tvm_runtime": build_attestation["libraries"]["runtime"]["resolved_path"],
        "tvm_compiler": build_attestation["libraries"]["compiler"]["resolved_path"],
    }
    loaded = {}
    for name in ("tvm_runtime", "tvm_compiler"):
        handle = tvm.base._LOADED_LIBS.get(name)
        if handle is None:
            raise RuntimeError(f"TVM did not load required core library {name}")
        library_name = getattr(handle, "_name", None)
        if not library_name:
            raise RuntimeError(f"TVM core library {name} has no resolved path")
        loaded_path = _require_path_under(Path(library_name), build_lib_root, f"TVM {name}")
        if Path(loaded_path).resolve() != Path(expected_libraries[name]).resolve():
            raise RuntimeError(
                f"TVM {name} loaded from {loaded_path}, expected {expected_libraries[name]}"
            )
        loaded[name] = loaded_path
    return {
        "runtime_only": False,
        "build_root": contract["tvm_build_root"],
        "build_lib_dir": build_attestation["lib_dir"],
        "loaded_core_libraries": loaded,
    }


def _tirx_runner(
    row: dict[str, Any], inputs: dict[str, Any], contract: dict[str, Any]
) -> tuple[Callable[[], tuple[Any, Any]], str, list[str], dict[str, Any]]:
    tirx_root = Path(contract["source_roots"]["tirx"])
    tvm_root = Path(contract["source_roots"]["tvm"])
    _prepend_sys_paths(
        (
            tirx_root,
            tvm_root / "python",
        )
    )
    public_module = importlib.import_module("tirx_kernels.attention.gdn_sm90")
    tvm = importlib.import_module("tvm")
    tvm_ffi = importlib.import_module("tvm_ffi")
    public_file = _require_module_under(public_module, tirx_root, "TIRx public API")
    tvm_file = _require_module_under(tvm, tvm_root, "TVM Python")
    source_roots = [Path(root) for root in contract["source_roots"].values()]
    tvm_ffi_file = _require_module_outside_roots(tvm_ffi, source_roots, "tvm-ffi Python")
    tvm_ffi_distribution = _tvm_ffi_distribution_identity()
    meta = public_module.KERNEL_META
    expected_backend = contract["implementations"]["tirx"]["backend_identity"]
    if meta.get("backend") != expected_backend:
        raise RuntimeError(
            f"TIRx backend identity drift: expected {expected_backend!r}, "
            f"got {meta.get('backend')!r}"
        )
    if meta.get("fallback") is not False:
        raise RuntimeError("TIRx public API reports a fallback path")
    if meta.get("target_arch") != "sm_90a":
        raise RuntimeError(f"TIRx public target drifted: {meta.get('target_arch')!r}")
    function = public_module.chunk_gated_delta_rule

    def call() -> tuple[Any, Any]:
        result = function(
            inputs["q"],
            inputs["k"],
            inputs["v"],
            inputs["alpha"],
            inputs["beta"],
            scale=float(row["scale"]),
            initial_state=inputs["initial_state"],
            output_final_state=bool(row["stateful"]),
            cu_seqlens=inputs["cu_seqlens"],
        )
        return result if row["stateful"] else (result, None)

    runner_attestation = {
        "entrypoint": "tirx_kernels.attention.gdn_sm90.chunk_gated_delta_rule",
        "module_file": public_file,
        "tvm_module_file": tvm_file,
        "tvm_ffi_module_file": tvm_ffi_file,
        "tvm_ffi_distribution": tvm_ffi_distribution,
        "tvm_runtime": _tvm_runtime_attestation(tvm, contract),
        "fallback": False,
    }
    return call, expected_backend, _wrapped_chain(function), runner_attestation


def _cutedsl_runner(
    row: dict[str, Any], inputs: dict[str, Any], contract: dict[str, Any]
) -> tuple[Callable[[], tuple[Any, Any]], str, list[str], dict[str, Any]]:
    root = Path(contract["source_roots"]["cutedsl"])
    _prepend_sys_paths((root,))
    module = importlib.import_module("cula.gdn.prefill")
    module_file = _require_module_under(module, root, "CuTeDSL cula.gdn.prefill")
    if any(name == "cula.gdn2" or name.startswith("cula.gdn2.") for name in sys.modules):
        raise RuntimeError("forbidden cula.gdn2 namespace was imported")
    required_version = contract["implementations"]["cutedsl"]["required_dsl_version"]
    observed_version = importlib.metadata.version("nvidia-cutlass-dsl")
    if observed_version != required_version:
        raise RuntimeError(
            f"CuTe DSL version drift: expected {required_version}, observed {observed_version}"
        )
    function = module.chunk_gated_delta_rule
    backend = module.get_sm90_gdn_prefill_backend_identity()

    def call() -> tuple[Any, Any]:
        result = function(
            inputs["q"],
            inputs["k"],
            inputs["v"],
            g=inputs["alpha"],
            beta=inputs["beta"],
            scale=float(row["scale"]),
            initial_state=inputs["initial_state"],
            output_final_state=bool(row["stateful"]),
            cu_seqlens=inputs["cu_seqlens"],
        )
        return result if row["stateful"] else (result, None)

    runner_attestation = {
        "entrypoint": "cula.gdn.prefill.chunk_gated_delta_rule",
        "module_file": module_file,
        "cutedsl_version": observed_version,
        "forbidden_gdn2_imported": False,
        "fallback": False,
    }
    return call, backend, _wrapped_chain(function), runner_attestation


def _fla_runner(
    row: dict[str, Any], inputs: dict[str, Any], contract: dict[str, Any]
) -> tuple[Callable[[], tuple[Any, Any]], str, list[str], dict[str, Any]]:
    root = Path(contract["source_roots"]["fla"])
    _prepend_sys_paths((root,))
    if os.environ.get("FLA_DISABLE_BACKEND_DISPATCH") != "1":
        raise RuntimeError("FLA_DISABLE_BACKEND_DISPATCH=1 is required before importing FLA")
    public_module = importlib.import_module("fla.ops.gated_delta_rule")
    chunk_module = importlib.import_module("fla.ops.gated_delta_rule.chunk")
    public_file = _require_module_under(public_module, root, "FLA public API")
    chunk_file = _require_module_under(chunk_module, root, "FLA chunk implementation")
    function = public_module.chunk_gated_delta_rule
    if function is not chunk_module.chunk_gated_delta_rule:
        raise RuntimeError(
            "FLA public API did not resolve to its locked in-tree chunk implementation"
        )
    backend = contract["implementations"]["fla"]["backend_identity"]

    def call() -> tuple[Any, Any]:
        output, state = function(
            inputs["q"].unsqueeze(0),
            inputs["k"].unsqueeze(0),
            inputs["v"].unsqueeze(0),
            torch_log(inputs["alpha"]).unsqueeze(0),
            inputs["beta"].unsqueeze(0),
            scale=float(row["scale"]),
            initial_state=inputs["initial_state"],
            output_final_state=bool(row["stateful"]),
            state_v_first=True,
            cu_seqlens=inputs["cu_seqlens_long"],
            chunk_size=64,
        )
        return output.squeeze(0), state

    runner_attestation = {
        "entrypoint": "fla.ops.gated_delta_rule.chunk.chunk_gated_delta_rule",
        "public_module_file": public_file,
        "chunk_module_file": chunk_file,
        "backend_dispatch_disabled": True,
        "fallback": False,
    }
    return call, backend, _wrapped_chain(function), runner_attestation


def torch_log(value: Any) -> Any:
    """Delay the torch dependency until the GPU worker has bound its device."""

    torch = importlib.import_module("torch")
    return torch.log(value)


def _runner(
    implementation: str,
    row: dict[str, Any],
    inputs: dict[str, Any],
    contract: dict[str, Any],
) -> tuple[Callable[[], tuple[Any, Any]], str, list[str], dict[str, Any]]:
    if implementation == "tirx":
        return _tirx_runner(row, inputs, contract)
    if implementation == "cutedsl":
        return _cutedsl_runner(row, inputs, contract)
    if implementation == "fla":
        return _fla_runner(row, inputs, contract)
    raise ValueError(f"unknown implementation {implementation!r}")


def _relative_rms_error(torch: Any, expected: Any, actual: Any) -> float:
    expected = expected.float()
    actual = actual.float()
    numerator = torch.sqrt(torch.mean((expected - actual) ** 2))
    denominator = torch.sqrt(torch.mean(expected**2)).clamp_min(1.0e-12)
    return float((numerator / denominator).item())


def _metrics(
    torch: Any,
    expected: Any | None,
    actual: Any | None,
    *,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    if expected is None or actual is None:
        matched = expected is None and actual is None
        return {
            "present_match": matched,
            "max_abs": None,
            "relative_rms": None,
            "allclose": matched,
        }
    error = (expected.float() - actual.float()).abs()
    return {
        "present_match": True,
        "max_abs": float(error.max().item()),
        "relative_rms": _relative_rms_error(torch, expected, actual),
        "allclose": bool(torch.allclose(expected.float(), actual.float(), atol=atol, rtol=rtol)),
    }


def _metrics_pass(metrics: dict[str, Any], *, max_abs: float, relative_rms: float) -> bool:
    return bool(
        metrics["present_match"]
        and metrics["allclose"]
        and (metrics["max_abs"] is None or metrics["max_abs"] <= max_abs)
        and (metrics["relative_rms"] is None or metrics["relative_rms"] <= relative_rms)
    )


def _hash_tensor(value: Any | None) -> str | None:
    if value is None:
        return None
    data = value.detach().float().contiguous().cpu().numpy().tobytes()
    return hashlib.sha256(data).hexdigest()


def semantic_hash(output: Any, state: Any | None, row: dict[str, Any]) -> str:
    """Hash one correctness oracle or result using the frozen row plus FP32 bytes."""

    digest = hashlib.sha256(canonical_json_bytes(row))
    digest.update(output.detach().float().contiguous().cpu().numpy().tobytes())
    if state is not None:
        digest.update(state.detach().float().contiguous().cpu().numpy().tobytes())
    return digest.hexdigest()


def _load_oracle(torch: Any, contract: dict[str, Any], contract_sha256: str, row: dict[str, Any]):
    path = Path(contract["oracle_root"]) / f"{row['row_id']}.pt"
    if not path.is_file():
        raise FileNotFoundError(f"missing CuTe correctness oracle: {path}")
    oracle = torch.load(path, map_location="cuda", weights_only=True)
    if oracle.get("schema") != "gdn-sm90a.cutedsl-oracle.v1":
        raise RuntimeError(f"unexpected oracle schema in {path}")
    if oracle.get("row") != row or oracle.get("row_id") != row["row_id"]:
        raise RuntimeError(f"oracle row drift in {path}")
    if oracle.get("contract_sha256") != contract_sha256:
        raise RuntimeError(f"oracle contract digest drift in {path}")
    comparator = contract["implementations"]["cutedsl"]
    if oracle.get("source_identity") != comparator["source_identity"]:
        raise RuntimeError(f"oracle source identity drift in {path}")
    if oracle.get("backend_identity") != comparator["backend_identity"]:
        raise RuntimeError(f"oracle backend identity drift in {path}")
    observed_hash = semantic_hash(oracle["output"], oracle.get("state"), row)
    if oracle.get("oracle_sha256") != observed_hash:
        raise RuntimeError(f"oracle semantic hash mismatch in {path}")
    return oracle


def _atomic_json_write(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite receipt: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _execute(args: argparse.Namespace) -> dict[str, Any]:
    contract, contract_sha256 = load_contract(args.contract)
    row = next((item for item in contract["rows"] if item["row_id"] == args.row_id), None)
    if row is None:
        raise ContractError(f"unknown frozen row {args.row_id!r}")
    if args.implementation not in IMPLEMENTATION_ORDER:
        raise ContractError(f"unknown implementation {args.implementation!r}")
    timing = contract["timing"]
    if not 0 <= args.process_index < int(timing["escalated_processes"]):
        raise ContractError("process_index is outside the frozen 0..6 range")
    if args.process_index >= int(timing["base_processes"]) and args.row_id != PACKED_N10_ROW_ID:
        raise ContractError("only packed-n10 may receive escalation processes 3..6")
    expected_order = rotation_for_process(args.process_index)
    expected_position = expected_order.index(args.implementation)
    if args.execution_order != expected_position:
        raise ContractError(
            f"execution order drift: expected position {expected_position}, "
            f"observed {args.execution_order}"
        )
    expected_token = expected_launch_token(
        contract_sha256, args.row_id, args.implementation, args.process_index
    )
    if args.launch_token != expected_token:
        raise ContractError("launch token does not match the frozen process identity")
    if args.receipt.exists():
        raise FileExistsError(f"refusing to overwrite receipt: {args.receipt}")

    activate_cutedsl_dependency_root(Path(contract["cutedsl_dependency_root"]))
    report_before = verify_contract_report_root(contract, executing_file=Path(__file__))
    dependency_before = verify_contract_cutedsl_dependency_root(contract)
    runtime_before = verify_contract_runtime_identity(contract)
    cutedsl_dependency_modules = _cutedsl_dependency_module_identity(contract)
    source_before = verify_contract_source_roots(contract)
    tvm_build_before = verify_contract_runtime_root(contract, verify_sha256=False)
    cache_environment = _cache_environment()
    physical_index = int(timing["physical_gpu_index"])
    expected_uuid = str(timing["expected_gpu_uuid"])
    before, processes_before, quiet_gate = _wait_for_quiet_gpu(
        physical_index,
        expected_uuid,
        max_util_pct=float(timing["max_gpu_util_pct"]),
        timeout_s=float(timing["quiet_timeout_s"]),
        poll_interval_s=float(timing["quiet_poll_interval_s"]),
    )

    torch = importlib.import_module("torch")
    gpu_binding = _verify_cuda_binding(torch, timing)
    inputs = _inputs(torch, row)
    oracle = _load_oracle(torch, contract, contract_sha256, row)
    call, backend, wrapped_chain, runner_attestation = _runner(
        args.implementation, row, inputs, contract
    )
    expected_backend = row["expected_backends"][args.implementation]
    if backend != expected_backend:
        raise RuntimeError(f"backend mismatch: expected {expected_backend!r}, resolved {backend!r}")
    if runner_attestation.get("fallback") is not False:
        raise RuntimeError("runner attestation did not explicitly reject fallback")

    compile_start = time.perf_counter()
    actual_output, actual_state = call()
    torch.cuda.synchronize()
    compile_first_call_ms = (time.perf_counter() - compile_start) * 1000.0

    correctness = contract["correctness"]
    output_metrics = _metrics(
        torch,
        oracle["output"],
        actual_output,
        atol=float(correctness["output_atol"]),
        rtol=float(correctness["output_rtol"]),
    )
    state_metrics = _metrics(
        torch,
        oracle.get("state"),
        actual_state,
        atol=float(correctness["state_atol"]),
        rtol=float(correctness["state_rtol"]),
    )
    correctness_passed = bool(
        _metrics_pass(
            output_metrics,
            max_abs=float(correctness["output_max_abs"]),
            relative_rms=float(correctness["output_relative_rms"]),
        )
        and _metrics_pass(
            state_metrics,
            max_abs=float(correctness["state_max_abs"]),
            relative_rms=float(correctness["state_relative_rms"]),
        )
    )
    if not correctness_passed:
        raise RuntimeError(f"correctness failed: output={output_metrics}, state={state_metrics}")

    for _ in range(int(timing["warmup_iters"])):
        call()
    torch.cuda.synchronize()
    samples = []
    for _ in range(int(timing["timed_iters"])):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        actual_output, actual_state = call()
        end.record()
        end.synchronize()
        elapsed = float(start.elapsed_time(end))
        if not math.isfinite(elapsed) or elapsed <= 0:
            raise RuntimeError(f"invalid CUDA event sample: {elapsed}")
        samples.append(elapsed)
    torch.cuda.synchronize()

    after = _gpu_state(physical_index)
    processes_after = _gpu_processes(expected_uuid)
    foreign_processes = [item for item in processes_after if item["pid"] != os.getpid()]
    invalid_reasons = []
    if after["uuid"] != expected_uuid:
        invalid_reasons.append(
            f"post-timing UUID mismatch: expected {expected_uuid}, observed {after['uuid']}"
        )
    if foreign_processes:
        invalid_reasons.append(f"foreign compute processes observed: {foreign_processes}")
    try:
        report_after = verify_contract_report_root(contract, executing_file=Path(__file__))
    except (ContractError, OSError, subprocess.SubprocessError) as error:
        report_after = None
        invalid_reasons.append(f"post-timing report harness drift: {error}")
    try:
        dependency_after = verify_contract_cutedsl_dependency_root(contract)
    except (ContractError, OSError) as error:
        dependency_after = None
        invalid_reasons.append(f"post-timing CuTeDSL dependency drift: {error}")
    try:
        runtime_after = verify_contract_runtime_identity(
            contract,
            require_cuda_uninitialized=False,
        )
    except (ContractError, OSError, ImportError) as error:
        runtime_after = None
        invalid_reasons.append(f"post-timing runtime identity drift: {error}")
    try:
        source_after = verify_contract_source_roots(contract)
    except (ContractError, OSError, subprocess.SubprocessError) as error:
        source_after = None
        invalid_reasons.append(f"post-timing source drift: {error}")
    try:
        tvm_build_after = verify_contract_runtime_root(contract, verify_sha256=False)
    except (ContractError, OSError) as error:
        tvm_build_after = None
        invalid_reasons.append(f"post-timing TVM build drift: {error}")

    implementation_contract = contract["implementations"][args.implementation]
    receipt = {
        "schema": "gdn-sm90a.public-fresh-timing-receipt.v1",
        "status": "invalid" if invalid_reasons else "timing_ok",
        "invalid_reasons": invalid_reasons,
        "correctness_passed": correctness_passed,
        "correctness_policy": correctness,
        "run_id": contract["run_id"],
        "contract_sha256": contract_sha256,
        "report_attestation_sha256": contract["report_attestation_sha256"],
        "report_attestation_before": report_before,
        "report_attestation_after": report_after,
        "cutedsl_dependency_attestation_sha256": contract["cutedsl_dependency_attestation_sha256"],
        "cutedsl_dependency_attestation_before": dependency_before,
        "cutedsl_dependency_attestation_after": dependency_after,
        "cutedsl_dependency_modules": cutedsl_dependency_modules,
        "runtime_identity_sha256": contract["runtime_identity_sha256"],
        "runtime_identity_before": runtime_before,
        "runtime_identity_after": runtime_after,
        "source_attestation_sha256": contract["source_attestation_sha256"],
        "source_attestation_before": source_before,
        "source_attestation_after": source_after,
        "tvm_build_attestation_sha256": contract["tvm_build_attestation_sha256"],
        "tvm_build_attestation_before": tvm_build_before,
        "tvm_build_attestation_after": tvm_build_after,
        "tvm_build_verification_mode": "contract-sha256 plus per-worker stat and loaded-handle",
        "row_id": args.row_id,
        "impl": args.implementation,
        "process_index": args.process_index,
        "execution_order": args.execution_order,
        "execution_sequence": list(expected_order),
        "launch_token": args.launch_token,
        "worker_pid": os.getpid(),
        "worker_parent_pid": os.getppid(),
        "source_identity": implementation_contract["source_identity"],
        "resolved_backend_identity": backend,
        "resolved_public_entrypoint": runner_attestation["entrypoint"],
        "fallback": False,
        "runner_attestation": runner_attestation,
        "wrapped_callable_chain": wrapped_chain,
        "compile_first_call_ms": compile_first_call_ms,
        "warmup_iters": int(timing["warmup_iters"]),
        "timed_iters": int(timing["timed_iters"]),
        "timer": "cuda_event",
        "raw_per_iter_ms": samples,
        "summary": {
            "average_ms": statistics.fmean(samples),
            "median_ms": statistics.median(samples),
            "min_ms": min(samples),
            "max_ms": max(samples),
        },
        "gpu_binding": gpu_binding,
        "gpu_state_before": before,
        "gpu_state_after": after,
        "gpu_processes_before": processes_before,
        "gpu_processes_after": processes_after,
        "foreign_gpu_processes_after": foreign_processes,
        "gpu_quiet_gate": quiet_gate,
        "cuda_cache_env": cache_environment,
        "output_metrics": output_metrics,
        "state_metrics": state_metrics,
        "output_hash": _hash_tensor(actual_output),
        "state_hash": _hash_tensor(actual_state),
        "oracle_sha256": oracle["oracle_sha256"],
        "input_seed": int(row["seed"]),
        "row": row,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "package_versions": _versions(),
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
    }
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--row-id", required=True)
    parser.add_argument("--implementation", choices=IMPLEMENTATION_ORDER, required=True)
    parser.add_argument("--process-index", type=int, required=True)
    parser.add_argument("--execution-order", type=int, required=True)
    parser.add_argument("--launch-token", required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    started = time.time()
    try:
        receipt = _execute(args)
    except Exception as error:
        if not args.receipt.exists():
            failure = {
                "schema": "gdn-sm90a.public-fresh-timing-receipt.v1",
                "status": "invalid",
                "invalid_reasons": [f"{type(error).__name__}: {error}"],
                "row_id": args.row_id,
                "impl": args.implementation,
                "process_index": args.process_index,
                "execution_order": args.execution_order,
                "launch_token": args.launch_token,
                "worker_pid": os.getpid(),
                "started_unix_s": started,
                "failed_unix_s": time.time(),
            }
            _atomic_json_write(args.receipt, failure)
        raise
    _atomic_json_write(args.receipt, receipt)
    if receipt["status"] != "timing_ok":
        raise SystemExit(f"GDN_TIMING_INVALID reasons={receipt['invalid_reasons']}")
    print(
        f"GDN_TIMING_RECEIPT_OK row={args.row_id} impl={args.implementation} "
        f"process={args.process_index} average_ms={receipt['summary']['average_ms']:.6f}"
    )


if __name__ == "__main__":
    main()
