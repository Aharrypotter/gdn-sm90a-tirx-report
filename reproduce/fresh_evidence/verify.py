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
"""Independently verify a public fresh H20 evidence bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

IMPLEMENTATIONS = ("tirx", "cutedsl", "fla")
PACKED_ROW = "packed-n10-t4096-h8-mha-state"
REPORT_REPOSITORY = "https://github.com/Aharrypotter/gdn-sm90a-tirx-report"
CUTEDSL_DEPENDENCY_LABEL = "nvidia-cutlass-dsl-4.5.1"
CUTEDSL_DEPENDENCY_AGGREGATE = "41bc70784cde0774308db6883d52e61cdeefe90bedd95631f9da64cee32c5506"
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
WINDOWS_ABSOLUTE = re.compile(r"^[A-Za-z]:[\\/]")

SOURCE_LOCKS: dict[str, dict[str, Any]] = {
    "tvm": {
        "repository": "https://github.com/Aharrypotter/tvm",
        "commit": "acb1312de80b39340e09b0aaad818ff029e745d6",
        "tree": "d7989ef4a8621448755da21bd1dec8b5d6c18a1b",
        "tag": "gdn-sm90a-compiler-r0",
        "tag_object": "18e2172e54aefcba7e11f3e62fa8bfa137b480d4",
        "required_path": "python/tvm/__init__.py",
    },
    "tirx": {
        "repository": "https://github.com/Aharrypotter/tirx-kernels",
        "commit": "12ce3721f7c62c5fbd911103ae373de689e58385",
        "tree": "cc04daa65ff52014348c8e078721e9afb017467a",
        "tag": "gdn-sm90a-kernel-r0",
        "tag_object": "f233dcbfc314415b9af496e3fd855554d81d662c",
        "runtime_commit": "90c9c62c84ecc452dd86602f0ea49a625845045c",
        "required_path": "tirx_kernels/attention/gdn_sm90.py",
    },
    "cutedsl": {
        "repository": "https://github.com/Aharrypotter/cuLA",
        "commit": "88737e9d906cf313995a092624656a89d74dd65e",
        "tree": "aa01d1a169b72dedd582f86fc9b257e9e2776344",
        "tag": "gdn-sm90a-comparator-r1",
        "tag_object": "0e2c50a4f39b58811e234466682a62f8926998c4",
        "required_path": "cula/gdn/prefill.py",
    },
    "fla": {
        "repository": "https://github.com/fla-org/flash-linear-attention",
        "commit": "d1ce07369d581813553f30a750af3b6b5f9af6a9",
        "tree": "e5ea97e3041c3e4dd0bf6974c2259f7ed104ddc2",
        "required_path": "fla/ops/gated_delta_rule/chunk.py",
    },
}

EXPECTED_IMPLEMENTATIONS: dict[str, dict[str, Any]] = {
    "tirx": {
        "entrypoint": "tirx_kernels.attention.gdn_sm90.chunk_gated_delta_rule",
        "backend_identity": "tirx.gdn.sm90a.wgmma.product-dispatch.packed.v3",
        "source_identity": (
            "tvm-git:acb1312de80b39340e09b0aaad818ff029e745d6;"
            "tirx-git:12ce3721f7c62c5fbd911103ae373de689e58385;"
            "tirx-runtime:90c9c62c84ecc452dd86602f0ea49a625845045c"
        ),
        "require_fallback_false": True,
    },
    "cutedsl": {
        "entrypoint": "cula.gdn.prefill.chunk_gated_delta_rule",
        "backend_identity": "sm90_cutedsl_gdn",
        "source_identity": (
            "git:88737e9d906cf313995a092624656a89d74dd65e;tag:gdn-sm90a-comparator-r1;dsl:4.5.1"
        ),
        "required_dsl_version": "4.5.1",
        "require_fallback_false": True,
    },
    "fla": {
        "entrypoint": "fla.ops.gated_delta_rule.chunk.chunk_gated_delta_rule",
        "backend_identity": "fla.gdn.chunk.triton.d1ce07369d58",
        "source_identity": (
            "git:d1ce07369d581813553f30a750af3b6b5f9af6a9;op:fla.ops.gated_delta_rule.chunk"
        ),
        "backend_dispatch_policy": "FLA_DISABLE_BACKEND_DISPATCH=1",
        "require_fallback_false": True,
    },
}

CORRECTNESS = {
    "oracle": "CuTeDSL tensors generated from the same frozen seeded inputs",
    "output_atol": 0.01,
    "output_rtol": 0.01,
    "output_max_abs": 0.075,
    "output_relative_rms": 0.15,
    "state_atol": 0.005,
    "state_rtol": 0.001,
    "state_max_abs": 0.075,
    "state_relative_rms": 0.15,
}

EXPECTED_BACKENDS = {
    implementation: spec["backend_identity"]
    for implementation, spec in EXPECTED_IMPLEMENTATIONS.items()
}

ROWS: tuple[dict[str, Any], ...] = (
    {
        "row_id": "single-t512-h8-mha-zero",
        "sequence_lengths": [512],
        "q_heads": 8,
        "v_heads": 8,
        "scale": 0.73,
        "seed": 240801,
        "stateful": False,
        "primary": True,
        "critical": False,
        "expected_backends": EXPECTED_BACKENDS,
    },
    {
        "row_id": "single-t1024-h8-mha-state",
        "sequence_lengths": [1024],
        "q_heads": 8,
        "v_heads": 8,
        "scale": 0.73,
        "seed": 240802,
        "stateful": True,
        "primary": True,
        "critical": False,
        "expected_backends": EXPECTED_BACKENDS,
    },
    {
        "row_id": "single-t1024-h8-hv16-gva-state",
        "sequence_lengths": [1024],
        "q_heads": 8,
        "v_heads": 16,
        "scale": 0.73,
        "seed": 240803,
        "stateful": True,
        "primary": True,
        "critical": False,
        "expected_backends": EXPECTED_BACKENDS,
    },
    {
        "row_id": "single-t4096-h16-mha-zero",
        "sequence_lengths": [4096],
        "q_heads": 16,
        "v_heads": 16,
        "scale": 0.73,
        "seed": 240804,
        "stateful": False,
        "primary": True,
        "critical": True,
        "expected_backends": EXPECTED_BACKENDS,
    },
    {
        "row_id": PACKED_ROW,
        "sequence_lengths": [410, 410, 410, 410, 410, 410, 409, 409, 409, 409],
        "q_heads": 8,
        "v_heads": 8,
        "scale": 0.73,
        "seed": 240805,
        "stateful": True,
        "primary": True,
        "critical": False,
        "expected_backends": EXPECTED_BACKENDS,
    },
    {
        "row_id": "packed-n20-t8192-h8-hv16-gva-state",
        "sequence_lengths": [
            5000,
            170,
            170,
            170,
            170,
            170,
            170,
            170,
            170,
            170,
            170,
            170,
            170,
            170,
            170,
            170,
            170,
            170,
            170,
            132,
        ],
        "q_heads": 8,
        "v_heads": 16,
        "scale": 0.73,
        "seed": 240806,
        "stateful": True,
        "primary": True,
        "critical": True,
        "expected_backends": EXPECTED_BACKENDS,
    },
)

TIMING = {
    "max_gpu_util_pct": 5.0,
    "quiet_timeout_s": 120.0,
    "quiet_poll_interval_s": 1.0,
    "warmup_iters": 20,
    "timed_iters": 100,
    "base_processes": 3,
    "escalated_processes": 7,
    "noise_band_pct": 2.0,
    "escalation_row_id": PACKED_ROW,
    "escalation_ratios": ["tirx_over_cutedsl", "tirx_over_fla"],
    "timer": "cuda_event",
    "statistic": "median_of_process_averages",
    "post_util_policy": "record_only",
    "require_rotating_three_way_order": True,
}

PACKAGE_KEYS = {
    "apache-tvm-ffi",
    "flash-linear-attention",
    "nvidia-cutlass-dsl",
    "nvidia-cutlass-dsl-libs-base",
    "nvidia-cutlass-dsl-libs-cu13",
    "triton",
    "tvm-ffi",
}


class VerificationError(ValueError):
    """Raised when a public evidence bundle is incomplete, unsafe, or inconsistent."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def _keys(value: Any, expected: set[str], label: str) -> None:
    _require(isinstance(value, dict), f"{label}: expected an object")
    _require(set(value) == expected, f"{label}: field set mismatch")


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _is_hex(value: Any) -> bool:
    return (
        isinstance(value, str) and len(value) == 64 and all(character in HEX for character in value)
    )


def _is_number(value: Any, *, positive: bool = False) -> bool:
    valid = (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )
    return valid and (not positive or float(value) > 0.0)


def _same_number(left: Any, right: Any) -> bool:
    return (
        _is_number(left)
        and _is_number(right)
        and math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1.0e-12)
    )


def _deep_same(left: Any, right: Any) -> bool:
    if (
        isinstance(left, int | float)
        and not isinstance(left, bool)
        and isinstance(right, int | float)
        and not isinstance(right, bool)
    ):
        return _same_number(left, right)
    if isinstance(left, dict) and isinstance(right, dict):
        return set(left) == set(right) and all(_deep_same(left[key], right[key]) for key in left)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            _deep_same(left_item, right_item) for left_item, right_item in zip(left, right)
        )
    return left == right


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except json.JSONDecodeError as error:
        raise VerificationError(f"{path.name}: invalid JSON: {error}") from error
    _require(isinstance(value, dict), f"{path.name}: expected a JSON object")
    return value


def _rotation(process_index: int) -> tuple[str, str, str]:
    orders = (
        ("tirx", "cutedsl", "fla"),
        ("cutedsl", "fla", "tirx"),
        ("fla", "tirx", "cutedsl"),
    )
    return orders[process_index % len(orders)]


def _launch_token(
    contract_sha256: str, row_id: str, implementation: str, process_index: int
) -> str:
    return _sha256_bytes(
        _canonical_bytes(
            {
                "contract_sha256": contract_sha256,
                "row_id": row_id,
                "implementation": implementation,
                "process_index": process_index,
            }
        )
    )


def _scan_disclosure(value: Any, location: str) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            lowered = key.lower()
            forbidden = (
                "uuid" in lowered
                or "cache" in lowered
                or "container" in lowered
                or lowered
                in {
                    "host",
                    "hostname",
                    "pid",
                    "worker_pid",
                    "worker_parent_pid",
                    "child_pid",
                    "source_roots",
                    "tvm_build_root",
                    "oracle_root",
                    "root",
                    "lib_dir",
                    "resolved_path",
                    "module_file",
                    "python_executable",
                    "process_binding",
                }
                or lowered.endswith("_pid")
            )
            _require(not forbidden, f"{location}: forbidden private key {key!r}")
            _scan_disclosure(child, f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _scan_disclosure(child, f"{location}[{index}]")
    elif isinstance(value, str):
        stripped = value.strip()
        _require("GPU-" not in value, f"{location}: GPU UUID-like value disclosed")
        _require(not stripped.startswith("/"), f"{location}: absolute POSIX path disclosed")
        _require(not stripped.startswith("~"), f"{location}: home-relative path disclosed")
        _require(
            WINDOWS_ABSOLUTE.match(stripped) is None,
            f"{location}: absolute Windows path disclosed",
        )
        _require(not stripped.startswith("file://"), f"{location}: file URI disclosed")


def _verify_manifest(bundle: Path) -> None:
    expected_files = set(PAYLOAD_FILES) | {"manifest.json", "MANIFEST.sha256"}
    actual_files = {
        str(path.relative_to(bundle))
        for path in bundle.rglob("*")
        if path.is_file() or path.is_symlink()
    }
    _require(actual_files == expected_files, "bundle file set differs from the sealed layout")
    for relative in actual_files:
        _require(not (bundle / relative).is_symlink(), f"{relative}: symlinks are forbidden")
    manifest_path = bundle / "manifest.json"
    seal_path = bundle / "MANIFEST.sha256"
    expected_seal = f"{_sha256_file(manifest_path)}  manifest.json\n"
    _require(seal_path.read_text() == expected_seal, "manifest seal mismatch")
    manifest = _read_json(manifest_path)
    _keys(manifest, {"schema", "file_count", "files"}, "manifest")
    _require(
        manifest["schema"] == "gdn-sm90a.public-evidence-manifest.v1",
        "manifest schema mismatch",
    )
    _require(manifest["file_count"] == len(PAYLOAD_FILES), "manifest file count mismatch")
    _require(isinstance(manifest["files"], list), "manifest files are missing")
    expected_entries = []
    for relative in PAYLOAD_FILES:
        path = bundle / relative
        expected_entries.append(
            {
                "path": relative,
                "sha256": _sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    _require(manifest["files"] == expected_entries, "manifest payload identity mismatch")


def _expected_observed_sources() -> dict[str, dict[str, Any]]:
    result = {}
    for name, lock in SOURCE_LOCKS.items():
        result[name] = {
            "source": name,
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


def _verify_contract(contract: dict[str, Any]) -> None:
    _keys(
        contract,
        {
            "schema",
            "original_schema",
            "run_id",
            "claim_scope",
            "contract_sha256",
            "report_repository",
            "report_attestation",
            "report_attestation_sha256",
            "source_attestation_sha256",
            "tvm_build_attestation_sha256",
            "cutedsl_dependency_attestation",
            "cutedsl_dependency_attestation_sha256",
            "runtime_identity",
            "runtime_identity_sha256",
            "source_locks",
            "source_attestations",
            "implementations",
            "correctness",
            "timing",
            "rows",
            "ratio",
            "secondary_ratios",
        },
        "contract",
    )
    _require(
        contract["schema"] == "gdn-sm90a.public-evidence-contract.v1",
        "contract schema mismatch",
    )
    _require(
        contract["original_schema"] == "gdn-sm90a.public-fresh-benchmark-contract.v1",
        "original contract schema mismatch",
    )
    _require(
        contract["claim_scope"] == "fresh public-tag H20 six-row characterization",
        "claim scope drift",
    )
    _require(
        isinstance(contract["run_id"], str)
        and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", contract["run_id"]),
        "run ID is invalid",
    )
    for key in (
        "contract_sha256",
        "report_attestation_sha256",
        "source_attestation_sha256",
        "tvm_build_attestation_sha256",
        "cutedsl_dependency_attestation_sha256",
        "runtime_identity_sha256",
    ):
        _require(_is_hex(contract[key]), f"contract: invalid {key}")
    _require(
        contract["report_repository"] == REPORT_REPOSITORY,
        "report harness repository drifted",
    )
    _require(
        isinstance(contract["report_attestation"], dict)
        and contract["report_attestation"]
        == {
            "repository": REPORT_REPOSITORY,
            "head": contract["report_attestation"]["head"],
            "tree": contract["report_attestation"]["tree"],
            "clean_checkout": True,
        }
        and len(contract["report_attestation"]["head"]) == 40
        and len(contract["report_attestation"]["tree"]) == 40
        and all(
            character in HEX
            for key in ("head", "tree")
            for character in contract["report_attestation"][key]
        ),
        "report harness attestation drifted",
    )
    _require(contract["source_locks"] == SOURCE_LOCKS, "immutable source locks drifted")
    _require(
        contract["source_attestations"] == _expected_observed_sources(),
        "observed source identities drifted",
    )
    _require(
        contract["implementations"] == EXPECTED_IMPLEMENTATIONS,
        "implementation identities drifted",
    )
    _require(contract["correctness"] == CORRECTNESS, "correctness policy drifted")
    _require(contract["timing"] == TIMING, "timing protocol drifted")
    _require(contract["rows"] == list(ROWS), "frozen six-row matrix drifted")
    _require(
        contract["ratio"] == {"numerator": "tirx", "denominator": "cutedsl"},
        "primary ratio drifted",
    )
    _require(
        contract["secondary_ratios"]
        == [{"name": "tirx_over_fla", "numerator": "tirx", "denominator": "fla"}],
        "secondary ratio drifted",
    )
    _verify_dependency_attestation(
        contract["cutedsl_dependency_attestation"],
        contract["cutedsl_dependency_attestation_sha256"],
        "contract dependency",
    )
    _verify_runtime_identity(contract["runtime_identity"], "contract runtime identity")


def _verify_dependency_attestation(dependency: Any, expected_digest: str, label: str) -> None:
    _keys(
        dependency,
        {
            "schema",
            "root_label",
            "aggregate_sha256",
            "entry_count",
            "file_count",
            "symlink_count",
            "total_file_bytes",
            "entries",
        },
        label,
    )
    _require(
        dependency["schema"] == "gdn-sm90a.transferred-source-tree.v1"
        and dependency["root_label"] == CUTEDSL_DEPENDENCY_LABEL
        and dependency["aggregate_sha256"] == CUTEDSL_DEPENDENCY_AGGREGATE,
        f"{label}: identity drift",
    )
    _require(isinstance(dependency["entries"], list), f"{label}: entries are missing")
    file_count = 0
    symlink_count = 0
    total_bytes = 0
    previous_path = None
    for index, entry in enumerate(dependency["entries"]):
        _require(isinstance(entry, dict), f"{label} entry {index}: expected object")
        relative = entry.get("path")
        _require(
            isinstance(relative, str)
            and relative
            and not Path(relative).is_absolute()
            and ".." not in Path(relative).parts,
            f"{label} entry {index}: unsafe path",
        )
        _require(
            previous_path is None or previous_path < relative,
            f"{label}: entry order is not deterministic",
        )
        previous_path = relative
        if entry.get("type") == "file":
            _keys(
                entry,
                {"executable", "path", "sha256", "size_bytes", "type"},
                f"{label} entry {index}",
            )
            _require(
                isinstance(entry["executable"], bool)
                and _is_hex(entry["sha256"])
                and isinstance(entry["size_bytes"], int)
                and not isinstance(entry["size_bytes"], bool)
                and entry["size_bytes"] >= 0,
                f"{label} entry {index}: malformed file",
            )
            file_count += 1
            total_bytes += entry["size_bytes"]
        elif entry.get("type") == "symlink":
            _keys(entry, {"path", "target", "type"}, f"{label} entry {index}")
            _require(
                isinstance(entry["target"], str)
                and entry["target"]
                and not Path(entry["target"]).is_absolute(),
                f"{label} entry {index}: unsafe symlink",
            )
            symlink_count += 1
        else:
            raise VerificationError(f"{label} entry {index}: unsupported type")
    _require(
        dependency["entry_count"] == len(dependency["entries"])
        and dependency["file_count"] == file_count
        and dependency["symlink_count"] == symlink_count
        and dependency["total_file_bytes"] == total_bytes,
        f"{label}: summary counts drifted",
    )
    _require(
        _sha256_bytes(_canonical_bytes(dependency)) == expected_digest,
        f"{label}: attestation digest mismatch",
    )


def _verify_runtime_identity(identity: Any, label: str) -> None:
    _keys(
        identity,
        {
            "python_version",
            "python_implementation",
            "python_full_version",
            "torch_module_version",
            "torch_cuda_build",
            "distributions",
        },
        label,
    )
    for key in (
        "python_version",
        "python_implementation",
        "python_full_version",
        "torch_module_version",
        "torch_cuda_build",
    ):
        _require(
            isinstance(identity[key], str) and identity[key],
            f"{label}: missing {key}",
        )
    expected = {
        "torch",
        "triton",
        "tvm_ffi",
        "nvidia_cutlass_dsl",
        "nvidia_cutlass_dsl_libs_base",
        "nvidia_cutlass_dsl_libs_cu13",
    }
    _require(
        isinstance(identity["distributions"], dict) and set(identity["distributions"]) == expected,
        f"{label}: distribution set drift",
    )
    for logical_name, distribution in identity["distributions"].items():
        expected_keys = {"distribution", "version", "identity_source"}
        if logical_name == "tvm_ffi":
            expected_keys.add("installed_files")
        _keys(distribution, expected_keys, f"{label} {logical_name}")
        _require(
            (
                distribution["distribution"] is None
                or (isinstance(distribution["distribution"], str) and distribution["distribution"])
            )
            and isinstance(distribution["version"], str)
            and distribution["version"],
            f"{label} {logical_name}: missing identity",
        )
        _require(
            distribution["identity_source"]
            == ("distribution" if distribution["distribution"] is not None else "module"),
            f"{label} {logical_name}: identity source drift",
        )
    ffi_files = identity["distributions"]["tvm_ffi"]["installed_files"]
    _keys(
        ffi_files,
        {
            "schema",
            "aggregate_sha256",
            "entry_count",
            "total_file_bytes",
            "entries",
        },
        f"{label} tvm-ffi files",
    )
    _require(
        ffi_files["schema"] == "gdn-sm90a.installed-distribution-files.v1"
        and isinstance(ffi_files["entries"], list),
        f"{label}: tvm-ffi installed-files schema drift",
    )
    aggregate = hashlib.sha256()
    total_bytes = 0
    previous_path = None
    for index, entry in enumerate(ffi_files["entries"]):
        _keys(
            entry,
            {"path", "size_bytes", "sha256"},
            f"{label} tvm-ffi file {index}",
        )
        _require(
            isinstance(entry["path"], str)
            and entry["path"]
            and not Path(entry["path"]).is_absolute()
            and ".." not in Path(entry["path"]).parts
            and (previous_path is None or previous_path < entry["path"])
            and isinstance(entry["size_bytes"], int)
            and not isinstance(entry["size_bytes"], bool)
            and entry["size_bytes"] >= 0
            and _is_hex(entry["sha256"]),
            f"{label}: malformed tvm-ffi file entry",
        )
        previous_path = entry["path"]
        total_bytes += entry["size_bytes"]
        aggregate.update(_canonical_bytes(entry))
        aggregate.update(b"\n")
    _require(
        ffi_files["entry_count"] == len(ffi_files["entries"])
        and ffi_files["total_file_bytes"] == total_bytes
        and ffi_files["aggregate_sha256"] == aggregate.hexdigest(),
        f"{label}: tvm-ffi installed-files aggregate drift",
    )
    for logical_name in (
        "nvidia_cutlass_dsl",
        "nvidia_cutlass_dsl_libs_base",
        "nvidia_cutlass_dsl_libs_cu13",
    ):
        _require(
            identity["distributions"][logical_name]["version"] == "4.5.1",
            f"{label}: CuTeDSL distribution version drift",
        )


def _verify_source_lock(source: dict[str, Any], contract: dict[str, Any]) -> None:
    _keys(
        source,
        {
            "schema",
            "contract_sha256",
            "report_repository",
            "report_attestation_sha256",
            "report",
            "source_attestation_sha256",
            "locks",
            "observed",
        },
        "source lock",
    )
    _require(
        source["schema"] == "gdn-sm90a.public-source-lock.v1",
        "source-lock schema mismatch",
    )
    _require(
        source["contract_sha256"] == contract["contract_sha256"],
        "source-lock contract mismatch",
    )
    _require(
        source["report_repository"] == contract["report_repository"]
        and source["report_attestation_sha256"] == contract["report_attestation_sha256"]
        and source["report"] == contract["report_attestation"],
        "source-lock report harness mismatch",
    )
    _require(
        source["source_attestation_sha256"] == contract["source_attestation_sha256"],
        "source-lock attestation digest mismatch",
    )
    _require(source["locks"] == SOURCE_LOCKS, "source-lock identities drifted")
    _require(
        source["observed"] == _expected_observed_sources(),
        "source-lock observed identities drifted",
    )


def _verify_build_lock(build: dict[str, Any], contract: dict[str, Any]) -> None:
    _keys(
        build,
        {
            "schema",
            "contract_sha256",
            "tvm_build_attestation_sha256",
            "cutedsl_dependency_attestation_sha256",
            "runtime_identity_sha256",
            "libraries",
            "cutedsl_dependency",
            "runtime_identity",
            "contract_digest_matched",
            "before_after_matched",
        },
        "build lock",
    )
    _require(build["schema"] == "gdn-sm90a.public-build-lock.v1", "build-lock schema mismatch")
    _require(
        build["contract_sha256"] == contract["contract_sha256"],
        "build-lock contract mismatch",
    )
    _require(
        build["tvm_build_attestation_sha256"] == contract["tvm_build_attestation_sha256"],
        "build-lock digest mismatch",
    )
    _require(
        build["cutedsl_dependency_attestation_sha256"]
        == contract["cutedsl_dependency_attestation_sha256"]
        and build["runtime_identity_sha256"] == contract["runtime_identity_sha256"],
        "build-lock dependency/runtime digest mismatch",
    )
    _require(
        build["cutedsl_dependency"] == contract["cutedsl_dependency_attestation"],
        "build-lock dependency identity mismatch",
    )
    _verify_dependency_attestation(
        build["cutedsl_dependency"],
        build["cutedsl_dependency_attestation_sha256"],
        "build-lock dependency",
    )
    _require(
        build["runtime_identity"] == contract["runtime_identity"],
        "build-lock runtime identity mismatch",
    )
    _verify_runtime_identity(build["runtime_identity"], "build-lock runtime identity")
    _require(build["contract_digest_matched"] is True, "build digest was not matched")
    _require(build["before_after_matched"] is True, "build before/after did not match")
    _require(
        isinstance(build["libraries"], dict)
        and set(build["libraries"]) == {"compiler", "runtime", "ffi"},
        "build library set mismatch",
    )
    basenames = {
        "compiler": {"libtvm.so", "libtvm_compiler.so"},
        "runtime": {"libtvm_runtime.so"},
        "ffi": {"libtvm_ffi.so"},
    }
    for name, library in build["libraries"].items():
        _keys(library, {"basename", "size_bytes", "sha256"}, f"build library {name}")
        _require(library["basename"] in basenames[name], f"build library {name}: basename drift")
        _require(
            isinstance(library["size_bytes"], int)
            and not isinstance(library["size_bytes"], bool)
            and library["size_bytes"] > 0,
            f"build library {name}: invalid size",
        )
        _require(_is_hex(library["sha256"]), f"build library {name}: invalid digest")


def _verify_environment(environment: dict[str, Any]) -> str:
    _keys(
        environment,
        {
            "schema",
            "accelerator",
            "compute_capability",
            "target_arch",
            "memory_total_mib",
            "driver_version",
            "cuda_compiler_release",
            "torch_version",
            "torch_cuda_version",
            "logical_cuda_device_count",
            "physical_device_binding_verified",
            "receipt_identity_consistent",
        },
        "environment",
    )
    _require(
        environment["schema"] == "gdn-sm90a.public-environment.v1",
        "environment schema mismatch",
    )
    _require(
        isinstance(environment["accelerator"], str)
        and environment["accelerator"].startswith("NVIDIA H20"),
        "environment accelerator mismatch",
    )
    _require(environment["compute_capability"] == "9.0", "environment compute capability drift")
    _require(environment["target_arch"] == "sm_90a", "environment target drift")
    _require(
        isinstance(environment["memory_total_mib"], int) and environment["memory_total_mib"] > 0,
        "environment memory size is invalid",
    )
    for key in (
        "driver_version",
        "cuda_compiler_release",
        "torch_version",
        "torch_cuda_version",
    ):
        _require(
            isinstance(environment[key], str) and bool(environment[key]),
            f"environment {key} is missing",
        )
    _require(environment["logical_cuda_device_count"] == 1, "logical GPU count drift")
    _require(
        environment["physical_device_binding_verified"] is True
        and environment["receipt_identity_consistent"] is True,
        "environment binding verdict did not pass",
    )
    return _sha256_bytes(_canonical_bytes(environment))


def _verify_oracles(oracle: dict[str, Any], contract: dict[str, Any]) -> dict[str, str]:
    _keys(
        oracle,
        {
            "schema",
            "original_schema",
            "status",
            "contract_sha256",
            "report_attestation_sha256",
            "source_attestation_sha256",
            "tvm_build_attestation_sha256",
            "cutedsl_dependency_attestation_sha256",
            "runtime_identity_sha256",
            "entrypoint",
            "backend_identity",
            "row_count",
            "rows",
            "physical_device_binding_verified",
            "process_isolation_verified",
        },
        "oracle manifest",
    )
    _require(
        oracle["schema"] == "gdn-sm90a.public-oracle-manifest.v1",
        "oracle manifest schema mismatch",
    )
    _require(
        oracle["original_schema"] == "gdn-sm90a.cutedsl-oracle-manifest.v1",
        "oracle source schema mismatch",
    )
    _require(oracle["status"] == "PASS", "oracle manifest did not pass")
    for key in (
        "contract_sha256",
        "report_attestation_sha256",
        "source_attestation_sha256",
        "tvm_build_attestation_sha256",
        "cutedsl_dependency_attestation_sha256",
        "runtime_identity_sha256",
    ):
        _require(oracle[key] == contract[key], f"oracle manifest {key} mismatch")
    comparator = EXPECTED_IMPLEMENTATIONS["cutedsl"]
    _require(oracle["entrypoint"] == comparator["entrypoint"], "oracle entrypoint drift")
    _require(oracle["backend_identity"] == comparator["backend_identity"], "oracle backend drift")
    _require(
        oracle["physical_device_binding_verified"] is True
        and oracle["process_isolation_verified"] is True,
        "oracle isolation verdict did not pass",
    )
    _require(oracle["row_count"] == len(ROWS), "oracle row count mismatch")
    _require(isinstance(oracle["rows"], list), "oracle rows are missing")
    expected_ids = [row["row_id"] for row in ROWS]
    observed_ids = [row.get("row_id") for row in oracle["rows"] if isinstance(row, dict)]
    _require(observed_ids == expected_ids, "oracle row order or coverage drifted")
    result = {}
    for expected, row in zip(ROWS, oracle["rows"]):
        _keys(row, {"row_id", "oracle_sha256", "state_present"}, f"oracle {expected['row_id']}")
        _require(_is_hex(row["oracle_sha256"]), f"oracle {expected['row_id']}: invalid hash")
        _require(
            row["state_present"] is bool(expected["stateful"]),
            f"oracle {expected['row_id']}: state presence mismatch",
        )
        result[expected["row_id"]] = row["oracle_sha256"]
    return result


def _verify_metric(
    metric: Any, *, stateful: bool, max_abs: float, max_rms: float, label: str
) -> None:
    _keys(metric, {"present_match", "max_abs", "relative_rms", "allclose"}, label)
    _require(metric["present_match"] is True and metric["allclose"] is True, f"{label}: mismatch")
    if not stateful:
        _require(
            metric["max_abs"] is None and metric["relative_rms"] is None,
            f"{label}: absent tensor has numeric errors",
        )
    else:
        _require(
            _is_number(metric["max_abs"])
            and 0 <= metric["max_abs"] <= max_abs
            and _is_number(metric["relative_rms"])
            and 0 <= metric["relative_rms"] <= max_rms,
            f"{label}: error exceeds policy",
        )


def _verify_gpu_state(state: Any, environment: dict[str, Any], label: str) -> None:
    _keys(
        state,
        {
            "accelerator",
            "util_pct",
            "memory_used_mib",
            "pstate",
            "sm_clock_mhz",
            "memory_clock_mhz",
            "temperature_c",
            "power_draw_w",
        },
        label,
    )
    _require(state["accelerator"] == environment["accelerator"], f"{label}: GPU model drift")
    _require(
        _is_number(state["util_pct"]) and state["util_pct"] >= 0,
        f"{label}: invalid utilization",
    )
    _require(
        _is_number(state["memory_used_mib"]) and state["memory_used_mib"] >= 0,
        f"{label}: invalid memory use",
    )
    _require(isinstance(state["pstate"], str) and state["pstate"], f"{label}: invalid pstate")
    for key in ("sm_clock_mhz", "memory_clock_mhz", "temperature_c", "power_draw_w"):
        _require(
            state[key] is None or _is_number(state[key]),
            f"{label}: invalid {key}",
        )


def _verify_cutedsl_runtime(value: Any, contract: dict[str, Any], label: str) -> None:
    _keys(
        value,
        {
            "dependency_attestation_sha256",
            "cutlass_module_relative_path",
            "cutlass_cute_module_relative_path",
            "distribution_versions",
            "module_binding_verified",
        },
        label,
    )
    _require(
        value["dependency_attestation_sha256"] == contract["cutedsl_dependency_attestation_sha256"],
        f"{label}: dependency digest drift",
    )
    _require(
        isinstance(value["cutlass_module_relative_path"], str)
        and value["cutlass_module_relative_path"].endswith("cutlass/__init__.py")
        and isinstance(value["cutlass_cute_module_relative_path"], str)
        and value["cutlass_cute_module_relative_path"].endswith("cutlass/cute/__init__.py"),
        f"{label}: module path drift",
    )
    _require(
        value["distribution_versions"]
        == {
            "nvidia-cutlass-dsl": "4.5.1",
            "nvidia-cutlass-dsl-libs-base": "4.5.1",
            "nvidia-cutlass-dsl-libs-cu13": "4.5.1",
        }
        and value["module_binding_verified"] is True,
        f"{label}: runtime distribution binding drift",
    )


def _verify_runner_attestation(
    value: Any,
    implementation: str,
    contract: dict[str, Any],
    build: dict[str, Any],
    label: str,
) -> None:
    _require(isinstance(value, dict), f"{label}: expected an object")
    common = {"kind", "entrypoint", "fallback", "cutedsl_runtime"}
    spec = EXPECTED_IMPLEMENTATIONS[implementation]
    if implementation == "tirx":
        _keys(
            value,
            common
            | {
                "module_relative_path",
                "tvm_module_relative_path",
                "tvm_build_attestation_sha256",
                "loaded_core_libraries",
                "tvm_ffi",
            },
            label,
        )
        _require(
            value["kind"] == "tirx"
            and value["entrypoint"] == spec["entrypoint"]
            and value["fallback"] is False
            and value["module_relative_path"] == SOURCE_LOCKS["tirx"]["required_path"]
            and value["tvm_module_relative_path"] == SOURCE_LOCKS["tvm"]["required_path"]
            and value["tvm_build_attestation_sha256"] == contract["tvm_build_attestation_sha256"],
            f"{label}: TIRx module/build binding drift",
        )
        _keys(
            value["loaded_core_libraries"],
            {"tvm_runtime", "tvm_compiler"},
            f"{label}: loaded libraries",
        )
        expected_libraries = {
            "tvm_runtime": build["libraries"]["runtime"],
            "tvm_compiler": build["libraries"]["compiler"],
        }
        _require(
            value["loaded_core_libraries"] == expected_libraries,
            f"{label}: loaded library identity drift",
        )
        _keys(
            value["tvm_ffi"],
            {
                "distribution",
                "version",
                "installed_files_aggregate_sha256",
                "noneditable_installation_verified",
            },
            f"{label}: tvm-ffi",
        )
        ffi = contract["runtime_identity"]["distributions"]["tvm_ffi"]
        _require(
            value["tvm_ffi"]
            == {
                "distribution": ffi["distribution"],
                "version": ffi["version"],
                "installed_files_aggregate_sha256": ffi["installed_files"]["aggregate_sha256"],
                "noneditable_installation_verified": True,
            },
            f"{label}: tvm-ffi identity drift",
        )
    elif implementation == "cutedsl":
        _keys(
            value,
            common
            | {
                "module_relative_path",
                "cutedsl_version",
                "forbidden_gdn2_imported",
            },
            label,
        )
        _require(
            value["kind"] == "cutedsl"
            and value["entrypoint"] == spec["entrypoint"]
            and value["fallback"] is False
            and value["module_relative_path"] == SOURCE_LOCKS["cutedsl"]["required_path"]
            and value["cutedsl_version"] == "4.5.1"
            and value["forbidden_gdn2_imported"] is False,
            f"{label}: CuTe GDN runner binding drift",
        )
    else:
        _keys(
            value,
            common
            | {
                "public_module_relative_path",
                "chunk_module_relative_path",
                "backend_dispatch_disabled",
            },
            label,
        )
        _require(
            value["kind"] == "fla"
            and value["entrypoint"] == spec["entrypoint"]
            and value["fallback"] is False
            and value["public_module_relative_path"] == "fla/ops/gated_delta_rule/__init__.py"
            and value["chunk_module_relative_path"] == SOURCE_LOCKS["fla"]["required_path"]
            and value["backend_dispatch_disabled"] is True,
            f"{label}: FLA runner binding drift",
        )
    _verify_cutedsl_runtime(value["cutedsl_runtime"], contract, f"{label}: CuTeDSL runtime")


def _verify_receipts(
    *,
    raw: bytes,
    contract: dict[str, Any],
    environment: dict[str, Any],
    environment_sha256: str,
    oracle_hashes: dict[str, str],
    build: dict[str, Any],
) -> list[dict[str, Any]]:
    lines = raw.splitlines()
    _require(bool(lines), "timing receipt stream is empty")
    receipts = []
    for line_number, line in enumerate(lines, start=1):
        try:
            receipt = json.loads(line)
        except json.JSONDecodeError as error:
            raise VerificationError(f"receipt line {line_number}: invalid JSON: {error}") from error
        _require(isinstance(receipt, dict), f"receipt line {line_number}: expected object")
        _require(
            line == _canonical_bytes(receipt),
            f"receipt line {line_number}: encoding is not canonical",
        )
        receipts.append(receipt)

    top_keys = {
        "schema",
        "source_receipt_sha256",
        "status",
        "correctness_passed",
        "correctness_policy",
        "run_id",
        "contract_sha256",
        "report_attestation_sha256",
        "source_attestation_sha256",
        "tvm_build_attestation_sha256",
        "cutedsl_dependency_attestation_sha256",
        "runtime_identity_sha256",
        "environment_sha256",
        "row_id",
        "implementation",
        "process_index",
        "execution_order",
        "execution_sequence",
        "launch_token",
        "source_identity",
        "backend_identity",
        "public_entrypoint",
        "fallback",
        "runner_attestation",
        "wrapped_callable_chain",
        "compile_first_call_ms",
        "warmup_iters",
        "timed_iters",
        "timer",
        "raw_per_iter_ms",
        "summary",
        "gpu",
        "process_isolation",
        "output_metrics",
        "state_metrics",
        "output_hash",
        "state_hash",
        "oracle_sha256",
        "input_seed",
        "row",
        "software",
    }
    row_map = {row["row_id"]: row for row in ROWS}
    identities = set()
    tokens = set()
    source_hashes = set()
    for receipt in receipts:
        prefix = (
            f"{receipt.get('row_id')}/{receipt.get('implementation')}/p"
            f"{receipt.get('process_index')}"
        )
        _keys(receipt, top_keys, prefix)
        _require(
            receipt["schema"] == "gdn-sm90a.public-evidence-timing-receipt.v1",
            f"{prefix}: schema mismatch",
        )
        _require(receipt["status"] == "timing_ok", f"{prefix}: timing status failed")
        _require(receipt["correctness_passed"] is True, f"{prefix}: correctness failed")
        _require(receipt["correctness_policy"] == CORRECTNESS, f"{prefix}: policy drift")
        for key in (
            "run_id",
            "contract_sha256",
            "report_attestation_sha256",
            "source_attestation_sha256",
            "tvm_build_attestation_sha256",
            "cutedsl_dependency_attestation_sha256",
            "runtime_identity_sha256",
        ):
            _require(receipt[key] == contract[key], f"{prefix}: {key} mismatch")
        _require(
            receipt["environment_sha256"] == environment_sha256,
            f"{prefix}: environment digest mismatch",
        )
        _require(_is_hex(receipt["source_receipt_sha256"]), f"{prefix}: source digest invalid")
        _require(
            receipt["source_receipt_sha256"] not in source_hashes,
            f"{prefix}: duplicate source digest",
        )
        source_hashes.add(receipt["source_receipt_sha256"])
        row_id = receipt["row_id"]
        implementation = receipt["implementation"]
        process_index = receipt["process_index"]
        _require(row_id in row_map, f"{prefix}: unknown row")
        _require(implementation in IMPLEMENTATIONS, f"{prefix}: unknown implementation")
        _require(
            isinstance(process_index, int)
            and not isinstance(process_index, bool)
            and 0 <= process_index < TIMING["escalated_processes"],
            f"{prefix}: invalid process index",
        )
        _require(
            process_index < TIMING["base_processes"] or row_id == PACKED_ROW,
            f"{prefix}: unexpected escalation process",
        )
        identity = (row_id, implementation, process_index)
        _require(identity not in identities, f"{prefix}: duplicate identity")
        identities.add(identity)
        order = _rotation(process_index)
        _require(receipt["execution_sequence"] == list(order), f"{prefix}: order drift")
        _require(
            receipt["execution_order"] == order.index(implementation),
            f"{prefix}: position drift",
        )
        expected_token = _launch_token(
            contract["contract_sha256"], row_id, implementation, process_index
        )
        _require(receipt["launch_token"] == expected_token, f"{prefix}: launch token drift")
        _require(expected_token not in tokens, f"{prefix}: duplicate launch token")
        tokens.add(expected_token)
        spec = EXPECTED_IMPLEMENTATIONS[implementation]
        _require(receipt["source_identity"] == spec["source_identity"], f"{prefix}: source drift")
        _require(
            receipt["backend_identity"] == spec["backend_identity"],
            f"{prefix}: backend drift",
        )
        _require(receipt["public_entrypoint"] == spec["entrypoint"], f"{prefix}: entrypoint drift")
        _require(receipt["fallback"] is False, f"{prefix}: fallback is not false")
        _verify_runner_attestation(
            receipt["runner_attestation"],
            implementation,
            contract,
            build,
            f"{prefix}: runner attestation",
        )
        wrapped = receipt["wrapped_callable_chain"]
        _require(
            isinstance(wrapped, list)
            and wrapped
            and all(
                isinstance(item, str) and item and "/" not in item and "\\" not in item
                for item in wrapped
            ),
            f"{prefix}: callable chain is invalid",
        )
        _require(
            _is_number(receipt["compile_first_call_ms"], positive=True),
            f"{prefix}: invalid compile latency",
        )
        _require(receipt["warmup_iters"] == TIMING["warmup_iters"], f"{prefix}: warmup drift")
        _require(receipt["timed_iters"] == TIMING["timed_iters"], f"{prefix}: sample-count drift")
        _require(receipt["timer"] == "cuda_event", f"{prefix}: timer drift")
        samples = receipt["raw_per_iter_ms"]
        _require(
            isinstance(samples, list)
            and len(samples) == TIMING["timed_iters"]
            and all(_is_number(sample, positive=True) for sample in samples),
            f"{prefix}: invalid raw samples",
        )
        _keys(
            receipt["summary"],
            {"average_ms", "median_ms", "min_ms", "max_ms"},
            f"{prefix}: summary",
        )
        expected_summary = {
            "average_ms": statistics.fmean(samples),
            "median_ms": statistics.median(samples),
            "min_ms": min(samples),
            "max_ms": max(samples),
        }
        _require(
            _deep_same(receipt["summary"], expected_summary),
            f"{prefix}: summary does not match samples",
        )
        row = row_map[row_id]
        _require(receipt["row"] == row, f"{prefix}: embedded row drift")
        _require(receipt["input_seed"] == row["seed"], f"{prefix}: seed drift")
        _require(receipt["oracle_sha256"] == oracle_hashes[row_id], f"{prefix}: oracle drift")
        _require(_is_hex(receipt["output_hash"]), f"{prefix}: output hash invalid")
        _require(
            _is_hex(receipt["state_hash"]) if row["stateful"] else receipt["state_hash"] is None,
            f"{prefix}: state hash invalid",
        )
        _verify_metric(
            receipt["output_metrics"],
            stateful=True,
            max_abs=CORRECTNESS["output_max_abs"],
            max_rms=CORRECTNESS["output_relative_rms"],
            label=f"{prefix}: output metrics",
        )
        _verify_metric(
            receipt["state_metrics"],
            stateful=row["stateful"],
            max_abs=CORRECTNESS["state_max_abs"],
            max_rms=CORRECTNESS["state_relative_rms"],
            label=f"{prefix}: state metrics",
        )
        gpu = receipt["gpu"]
        _keys(
            gpu,
            {
                "accelerator",
                "compute_capability",
                "target_arch",
                "logical_device_count",
                "binding_verified",
                "before",
                "after",
                "quiet_gate",
            },
            f"{prefix}: GPU",
        )
        _require(gpu["accelerator"] == environment["accelerator"], f"{prefix}: GPU model drift")
        _require(gpu["compute_capability"] == "9.0", f"{prefix}: compute capability drift")
        _require(gpu["target_arch"] == "sm_90a", f"{prefix}: target drift")
        _require(gpu["logical_device_count"] == 1, f"{prefix}: device-count drift")
        _require(gpu["binding_verified"] is True, f"{prefix}: binding not verified")
        _verify_gpu_state(gpu["before"], environment, f"{prefix}: GPU before")
        _verify_gpu_state(gpu["after"], environment, f"{prefix}: GPU after")
        _require(
            gpu["before"]["util_pct"] <= TIMING["max_gpu_util_pct"],
            f"{prefix}: GPU was not quiet",
        )
        quiet = gpu["quiet_gate"]
        _keys(
            quiet,
            {
                "elapsed_s",
                "polls",
                "max_observed_util_pct",
                "timeout_s",
                "poll_interval_s",
            },
            f"{prefix}: quiet gate",
        )
        _require(
            _is_number(quiet["elapsed_s"])
            and isinstance(quiet["polls"], int)
            and not isinstance(quiet["polls"], bool)
            and quiet["polls"] >= 1
            and _is_number(quiet["max_observed_util_pct"])
            and quiet["timeout_s"] == TIMING["quiet_timeout_s"]
            and quiet["poll_interval_s"] == TIMING["quiet_poll_interval_s"],
            f"{prefix}: quiet gate drift",
        )
        _require(
            receipt["process_isolation"]
            == {
                "fresh_process_verified": True,
                "no_preexisting_compute_processes": True,
                "no_foreign_compute_processes": True,
            },
            f"{prefix}: process isolation verdict failed",
        )
        software = receipt["software"]
        _keys(
            software,
            {"python_version", "torch_version", "torch_cuda_version", "package_versions"},
            f"{prefix}: software",
        )
        _require(
            isinstance(software["python_version"], str) and software["python_version"],
            f"{prefix}: Python version missing",
        )
        _require(
            software["torch_version"] == environment["torch_version"],
            f"{prefix}: torch drift",
        )
        _require(
            software["torch_cuda_version"] == environment["torch_cuda_version"],
            f"{prefix}: CUDA runtime drift",
        )
        _require(
            software["torch_cuda_version"] == contract["runtime_identity"]["torch_cuda_build"],
            f"{prefix}: runtime identity CUDA build drift",
        )
        _require(
            isinstance(software["package_versions"], dict)
            and set(software["package_versions"]) == PACKAGE_KEYS
            and all(
                version is None or (isinstance(version, str) and version)
                for version in software["package_versions"].values()
            ),
            f"{prefix}: package-version set drift",
        )
        runtime_distributions = contract["runtime_identity"]["distributions"]
        _require(
            software["torch_version"] == contract["runtime_identity"]["torch_module_version"]
            and software["package_versions"]["triton"] == runtime_distributions["triton"]["version"]
            and software["package_versions"]["nvidia-cutlass-dsl"]
            == runtime_distributions["nvidia_cutlass_dsl"]["version"]
            and software["package_versions"]["nvidia-cutlass-dsl-libs-base"]
            == runtime_distributions["nvidia_cutlass_dsl_libs_base"]["version"]
            and software["package_versions"]["nvidia-cutlass-dsl-libs-cu13"]
            == runtime_distributions["nvidia_cutlass_dsl_libs_cu13"]["version"],
            f"{prefix}: package versions differ from runtime identity",
        )
        ffi = runtime_distributions["tvm_ffi"]
        ffi_name = ffi["distribution"].lower().replace("_", "-")
        ffi_key = "apache-tvm-ffi" if ffi_name == "apache-tvm-ffi" else "tvm-ffi"
        _require(
            software["package_versions"][ffi_key] == ffi["version"],
            f"{prefix}: tvm-ffi version differs from runtime identity",
        )

    row_order = {row["row_id"]: index for index, row in enumerate(ROWS)}
    impl_order = {implementation: index for index, implementation in enumerate(IMPLEMENTATIONS)}
    expected_order = sorted(
        receipts,
        key=lambda receipt: (
            row_order[receipt["row_id"]],
            receipt["process_index"],
            impl_order[receipt["implementation"]],
        ),
    )
    _require(receipts == expected_order, "receipt stream order is not deterministic")
    return receipts


def _verify_launches(
    *,
    raw: bytes,
    contract: dict[str, Any],
    receipts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    lines = raw.splitlines()
    _require(bool(lines), "public launch ledger is empty")
    top_keys = {
        "schema",
        "source_launch_sha256",
        "source_receipt_sha256",
        "run_id",
        "contract_sha256",
        "row_id",
        "implementation",
        "process_index",
        "execution_order",
        "execution_sequence",
        "launch_token",
        "parent_child_binding_verified",
        "fresh_process_launch_verified",
    }
    receipt_map = {
        (receipt["row_id"], receipt["implementation"], receipt["process_index"]): receipt
        for receipt in receipts
    }
    launches = []
    identities = set()
    tokens = set()
    source_hashes = set()
    for line_number, line in enumerate(lines, start=1):
        try:
            launch = json.loads(line)
        except json.JSONDecodeError as error:
            raise VerificationError(f"launch line {line_number}: invalid JSON: {error}") from error
        _require(isinstance(launch, dict), f"launch line {line_number}: expected object")
        _require(
            line == _canonical_bytes(launch),
            f"launch line {line_number}: encoding is not canonical",
        )
        _keys(launch, top_keys, f"launch line {line_number}")
        row_id = launch["row_id"]
        implementation = launch["implementation"]
        process_index = launch["process_index"]
        identity = (row_id, implementation, process_index)
        prefix = f"{row_id}/{implementation}/p{process_index}"
        _require(
            launch["schema"] == "gdn-sm90a.public-evidence-launch.v1",
            f"{prefix}: launch schema drift",
        )
        _require(identity in receipt_map, f"{prefix}: launch has no timing receipt")
        _require(identity not in identities, f"{prefix}: duplicate launch identity")
        identities.add(identity)
        _require(
            isinstance(process_index, int)
            and not isinstance(process_index, bool)
            and 0 <= process_index < TIMING["escalated_processes"],
            f"{prefix}: invalid launch process index",
        )
        order = _rotation(process_index)
        _require(
            launch["execution_sequence"] == list(order)
            and launch["execution_order"] == order.index(implementation),
            f"{prefix}: launch rotation drift",
        )
        token = _launch_token(contract["contract_sha256"], row_id, implementation, process_index)
        _require(launch["launch_token"] == token, f"{prefix}: launch token drift")
        _require(token not in tokens, f"{prefix}: duplicate launch token")
        tokens.add(token)
        _require(
            _is_hex(launch["source_launch_sha256"])
            and launch["source_launch_sha256"] not in source_hashes,
            f"{prefix}: source launch digest invalid or duplicate",
        )
        source_hashes.add(launch["source_launch_sha256"])
        receipt = receipt_map[identity]
        _require(
            launch["source_receipt_sha256"] == receipt["source_receipt_sha256"],
            f"{prefix}: launch/receipt source binding drift",
        )
        _require(
            launch["run_id"] == contract["run_id"]
            and launch["contract_sha256"] == contract["contract_sha256"],
            f"{prefix}: launch contract binding drift",
        )
        _require(
            launch["parent_child_binding_verified"] is True
            and launch["fresh_process_launch_verified"] is True,
            f"{prefix}: launch process binding verdict failed",
        )
        launches.append(launch)
    _require(set(receipt_map) == identities, "public launch coverage differs from timing receipts")
    row_order = {row["row_id"]: index for index, row in enumerate(ROWS)}
    impl_order = {implementation: index for index, implementation in enumerate(IMPLEMENTATIONS)}
    expected_order = sorted(
        launches,
        key=lambda launch: (
            row_order[launch["row_id"]],
            launch["process_index"],
            impl_order[launch["implementation"]],
        ),
    )
    _require(launches == expected_order, "public launch ledger order is not deterministic")
    return launches


def _geomean(values: Iterable[float]) -> float:
    values = list(values)
    _require(bool(values) and all(value > 0 for value in values), "geomean input invalid")
    return math.exp(statistics.fmean(math.log(value) for value in values))


def _escalates(ratio: float) -> bool:
    distance = abs(ratio - 1.0)
    limit = TIMING["noise_band_pct"] / 100.0
    return distance <= limit or math.isclose(distance, limit, rel_tol=0.0, abs_tol=1e-12)


def _recompute(receipts: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for receipt in receipts:
        grouped[(receipt["row_id"], receipt["implementation"])].append(receipt)
    packed_medians = {}
    for implementation in IMPLEMENTATIONS:
        items = [
            receipt
            for receipt in grouped[(PACKED_ROW, implementation)]
            if receipt["process_index"] < TIMING["base_processes"]
        ]
        _require(
            len(items) == TIMING["base_processes"],
            "packed-n10 base process set is incomplete",
        )
        packed_medians[implementation] = statistics.median(
            receipt["summary"]["average_ms"] for receipt in items
        )
    packed_ratios = {
        "tirx_over_cutedsl": packed_medians["tirx"] / packed_medians["cutedsl"],
        "tirx_over_fla": packed_medians["tirx"] / packed_medians["fla"],
    }
    escalated = any(_escalates(packed_ratios[name]) for name in TIMING["escalation_ratios"])
    expected_counts = {
        row["row_id"]: (
            TIMING["escalated_processes"]
            if escalated and row["row_id"] == PACKED_ROW
            else TIMING["base_processes"]
        )
        for row in ROWS
    }
    expected_total = sum(expected_counts.values()) * len(IMPLEMENTATIONS)
    _require(len(receipts) == expected_total, "receipt count does not match escalation policy")
    rows = {}
    primary_cutedsl = []
    primary_fla = []
    for row in ROWS:
        row_id = row["row_id"]
        medians = {}
        averages = {}
        observed = {}
        oracle_hashes = set()
        for implementation in IMPLEMENTATIONS:
            items = sorted(
                grouped[(row_id, implementation)], key=lambda item: item["process_index"]
            )
            expected_count = expected_counts[row_id]
            _require(
                [item["process_index"] for item in items] == list(range(expected_count)),
                f"{row_id}/{implementation}: process coverage mismatch",
            )
            observed[implementation] = len(items)
            averages[implementation] = [item["summary"]["average_ms"] for item in items]
            medians[implementation] = statistics.median(averages[implementation])
            oracle_hashes.update(item["oracle_sha256"] for item in items)
        _require(len(oracle_hashes) == 1, f"{row_id}: oracle hashes differ")
        cutedsl = medians["tirx"] / medians["cutedsl"]
        fla = medians["tirx"] / medians["fla"]
        rows[row_id] = {
            "primary": row["primary"],
            "critical": row["critical"],
            "expected_processes": expected_counts[row_id],
            "observed_processes": observed,
            "process_averages_ms": averages,
            "median_ms": medians,
            "oracle_sha256": next(iter(oracle_hashes)),
            "tirx_over_cutedsl": cutedsl,
            "tirx_over_fla": fla,
        }
        if row["primary"]:
            primary_cutedsl.append(cutedsl)
            primary_fla.append(fla)
    computed = {
        "packed_n10_base_ratios": packed_ratios,
        "packed_n10_escalated": escalated,
        "packed_n10_final_processes": expected_counts[PACKED_ROW],
        "rows": rows,
        "primary_geomean": {
            "tirx_over_cutedsl": _geomean(primary_cutedsl),
            "tirx_over_fla": _geomean(primary_fla),
        },
        "receipt_count": len(receipts),
    }
    return computed, expected_counts


def _verify_summary(
    summary: dict[str, Any], contract: dict[str, Any], computed: dict[str, Any]
) -> None:
    _keys(
        summary,
        {
            "schema",
            "status",
            "run_id",
            "contract_sha256",
            "report_attestation_sha256",
            "source_attestation_sha256",
            "tvm_build_attestation_sha256",
            "cutedsl_dependency_attestation_sha256",
            "runtime_identity_sha256",
            "base_processes",
            "packed_n10_base_ratios",
            "noise_band_pct",
            "packed_n10_escalated",
            "packed_n10_final_processes",
            "receipt_count",
            "fresh_process_launch_count",
            "physical_device_binding_verified",
            "process_isolation_verified",
        },
        "run summary",
    )
    _require(summary["schema"] == "gdn-sm90a.public-run-summary.v1", "run-summary schema mismatch")
    _require(summary["status"] == "PASS", "run summary did not pass")
    for key in (
        "run_id",
        "contract_sha256",
        "report_attestation_sha256",
        "source_attestation_sha256",
        "tvm_build_attestation_sha256",
        "cutedsl_dependency_attestation_sha256",
        "runtime_identity_sha256",
    ):
        _require(summary[key] == contract[key], f"run summary {key} mismatch")
    _require(summary["base_processes"] == TIMING["base_processes"], "base process count drift")
    _require(summary["noise_band_pct"] == TIMING["noise_band_pct"], "noise band drift")
    _require(
        _deep_same(summary["packed_n10_base_ratios"], computed["packed_n10_base_ratios"]),
        "packed-n10 base ratios drift",
    )
    _require(
        summary["packed_n10_escalated"] is computed["packed_n10_escalated"],
        "packed-n10 escalation decision drift",
    )
    _require(
        summary["packed_n10_final_processes"] == computed["packed_n10_final_processes"],
        "packed-n10 final process count drift",
    )
    _require(
        summary["receipt_count"] == computed["receipt_count"]
        and summary["fresh_process_launch_count"] == computed["receipt_count"],
        "run-summary process counts drift",
    )
    _require(
        summary["physical_device_binding_verified"] is True
        and summary["process_isolation_verified"] is True,
        "run-summary isolation verdict failed",
    )


def _verify_performance(
    performance: dict[str, Any], contract: dict[str, Any], computed: dict[str, Any]
) -> None:
    _keys(
        performance,
        {
            "schema",
            "run_id",
            "contract_sha256",
            "report_attestation_sha256",
            "source_attestation_sha256",
            "tvm_build_attestation_sha256",
            "cutedsl_dependency_attestation_sha256",
            "runtime_identity_sha256",
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
            "process_isolation_verified",
            "status",
            "decision_status",
        },
        "performance",
    )
    _require(
        performance["schema"] == "gdn-sm90a.public-performance.v1",
        "performance schema mismatch",
    )
    for key in (
        "run_id",
        "contract_sha256",
        "report_attestation_sha256",
        "source_attestation_sha256",
        "tvm_build_attestation_sha256",
        "cutedsl_dependency_attestation_sha256",
        "runtime_identity_sha256",
    ):
        _require(performance[key] == contract[key], f"performance {key} mismatch")
    _require(
        performance["ratio_direction"] == "TIRx latency / comparator latency; lower is faster",
        "ratio direction drift",
    )
    _require(
        performance["timer"] == "CUDA events around the public call"
        and performance["statistic"] == "median of per-process averages",
        "performance statistic drift",
    )
    _require(
        performance["warmup_iters"] == TIMING["warmup_iters"]
        and performance["timed_iters"] == TIMING["timed_iters"]
        and performance["base_processes"] == TIMING["base_processes"]
        and performance["packed_n10_escalated_processes"] == TIMING["escalated_processes"]
        and performance["packed_n10_noise_band_pct"] == TIMING["noise_band_pct"],
        "performance protocol drift",
    )
    _require(
        _deep_same(performance["packed_n10_base_ratios"], computed["packed_n10_base_ratios"]),
        "performance base ratios drift",
    )
    _require(
        performance["packed_n10_escalation_required"] is computed["packed_n10_escalated"],
        "performance escalation decision drift",
    )
    _require(_deep_same(performance["rows"], computed["rows"]), "performance rows drift")
    _require(
        _deep_same(performance["primary_geomean"], computed["primary_geomean"]),
        "performance geomean drift",
    )
    _require(
        performance["receipt_count"] == computed["receipt_count"]
        and performance["fresh_process_launch_count"] == computed["receipt_count"],
        "performance process counts drift",
    )
    _require(
        performance["process_isolation_verified"] is True
        and performance["status"] == "PASS"
        and performance["decision_status"] == "CHARACTERIZATION",
        "performance verdict drift",
    )


def _verify_publication(
    publication: dict[str, Any],
    contract: dict[str, Any],
    environment_sha256: str,
    receipts: list[dict[str, Any]],
    launches: list[dict[str, Any]],
) -> None:
    _keys(
        publication,
        {
            "schema",
            "status",
            "evidence_kind",
            "decision_status",
            "upstream_merge_claim",
            "run_id",
            "claim_scope",
            "contract_sha256",
            "report_attestation_sha256",
            "source_attestation_sha256",
            "tvm_build_attestation_sha256",
            "cutedsl_dependency_attestation_sha256",
            "runtime_identity_sha256",
            "environment_sha256",
            "receipt_count",
            "fresh_process_launch_count",
            "physical_device_binding_verified",
            "process_isolation_verified",
            "input_artifact_sha256",
        },
        "publication",
    )
    _require(
        publication["schema"] == "gdn-sm90a.public-fresh-evidence.v1",
        "publication schema mismatch",
    )
    _require(publication["status"] == "PASS", "publication status is not PASS")
    _require(
        publication["evidence_kind"] == "fresh-public-tag-h20-rerun",
        "publication evidence kind drift",
    )
    _require(
        publication["decision_status"] == "CHARACTERIZATION"
        and publication["upstream_merge_claim"] is False,
        "publication claim boundary drift",
    )
    for key in (
        "run_id",
        "claim_scope",
        "contract_sha256",
        "report_attestation_sha256",
        "source_attestation_sha256",
        "tvm_build_attestation_sha256",
        "cutedsl_dependency_attestation_sha256",
        "runtime_identity_sha256",
    ):
        _require(publication[key] == contract[key], f"publication {key} mismatch")
    _require(
        publication["environment_sha256"] == environment_sha256,
        "publication environment digest mismatch",
    )
    _require(
        publication["receipt_count"] == len(receipts)
        and publication["fresh_process_launch_count"] == len(launches)
        and len(launches) == len(receipts),
        "publication process counts mismatch",
    )
    _require(
        publication["physical_device_binding_verified"] is True
        and publication["process_isolation_verified"] is True,
        "publication isolation verdict failed",
    )
    input_hashes = publication["input_artifact_sha256"]
    _keys(
        input_hashes,
        {
            "benchmark_contract",
            "environment_check",
            "oracle_manifest",
            "run_summary",
            "benchmark_report",
            "launch_ledger",
            "launch_ledger_set",
            "timing_receipt_set",
        },
        "publication input digests",
    )
    _require(
        all(_is_hex(value) for value in input_hashes.values()),
        "publication input digest is invalid",
    )
    _require(
        input_hashes["benchmark_contract"] == contract["contract_sha256"],
        "private contract input digest mismatch",
    )
    expected_set = _sha256_bytes(
        _canonical_bytes(sorted(receipt["source_receipt_sha256"] for receipt in receipts))
    )
    _require(
        input_hashes["timing_receipt_set"] == expected_set,
        "private receipt-set digest mismatch",
    )
    expected_launch_set = _sha256_bytes(
        _canonical_bytes(sorted(launch["source_launch_sha256"] for launch in launches))
    )
    _require(
        input_hashes["launch_ledger_set"] == expected_launch_set,
        "private launch-ledger set digest mismatch",
    )


def verify_bundle(bundle: Path) -> dict[str, Any]:
    """Verify the sealed bundle without importing or trusting the producer."""

    bundle = bundle.expanduser().resolve()
    _require(bundle.is_dir(), "bundle directory is missing")
    _verify_manifest(bundle)
    publication = _read_json(bundle / "publication.json")
    contract = _read_json(bundle / "contract.json")
    source = _read_json(bundle / "source-lock.json")
    build = _read_json(bundle / "build-lock.json")
    environment = _read_json(bundle / "environment.json")
    oracle = _read_json(bundle / "oracle-manifest.json")
    summary = _read_json(bundle / "run-summary.json")
    performance = _read_json(bundle / "performance.json")
    launches_raw = (bundle / "launches.jsonl").read_bytes()
    receipts_raw = (bundle / "timing-receipts.jsonl").read_bytes()

    for name, value in (
        ("publication", publication),
        ("contract", contract),
        ("source lock", source),
        ("build lock", build),
        ("environment", environment),
        ("oracle manifest", oracle),
        ("run summary", summary),
        ("performance", performance),
    ):
        _scan_disclosure(value, name)
    for line_number, line in enumerate(launches_raw.splitlines(), start=1):
        try:
            launch_value = json.loads(line)
        except json.JSONDecodeError as error:
            raise VerificationError(f"launch line {line_number}: invalid JSON: {error}") from error
        _scan_disclosure(launch_value, f"launch line {line_number}")
    for line_number, line in enumerate(receipts_raw.splitlines(), start=1):
        try:
            receipt_value = json.loads(line)
        except json.JSONDecodeError as error:
            raise VerificationError(f"receipt line {line_number}: invalid JSON: {error}") from error
        _scan_disclosure(receipt_value, f"receipt line {line_number}")

    _verify_contract(contract)
    _verify_source_lock(source, contract)
    _verify_build_lock(build, contract)
    environment_sha256 = _verify_environment(environment)
    _require(
        environment["torch_version"] == contract["runtime_identity"]["torch_module_version"]
        and environment["torch_cuda_version"] == contract["runtime_identity"]["torch_cuda_build"],
        "environment differs from the contract runtime identity",
    )
    oracle_hashes = _verify_oracles(oracle, contract)
    receipts = _verify_receipts(
        raw=receipts_raw,
        contract=contract,
        environment=environment,
        environment_sha256=environment_sha256,
        oracle_hashes=oracle_hashes,
        build=build,
    )
    launches = _verify_launches(raw=launches_raw, contract=contract, receipts=receipts)
    computed, _ = _recompute(receipts)
    _verify_summary(summary, contract, computed)
    _verify_performance(performance, contract, computed)
    _verify_publication(publication, contract, environment_sha256, receipts, launches)
    return {
        "status": "PASS",
        "run_id": contract["run_id"],
        "receipt_count": len(receipts),
        "tirx_over_cutedsl": computed["primary_geomean"]["tirx_over_cutedsl"],
        "tirx_over_fla": computed["primary_geomean"]["tirx_over_fla"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Independently verify a sealed public fresh H20 evidence bundle."
    )
    parser.add_argument("--bundle", type=Path, required=True)
    args = parser.parse_args()
    result = verify_bundle(args.bundle)
    print(
        "GDN_FRESH_PUBLIC_EVIDENCE_VERIFY_PASS "
        f"run_id={result['run_id']} receipts={result['receipt_count']} "
        f"tirx/cutedsl={result['tirx_over_cutedsl']:.6f} "
        f"tirx/fla={result['tirx_over_fla']:.6f}"
    )


if __name__ == "__main__":
    main()
