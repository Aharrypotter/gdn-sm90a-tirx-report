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
"""Materialize and validate the frozen public-tag GDN benchmark contract."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from reproduce.seal_source_tree import build_manifest

SCHEMA = "gdn-sm90a.public-fresh-benchmark-contract.v1"
IMPLEMENTATION_ORDER = ("tirx", "cutedsl", "fla")
PACKED_N10_ROW_ID = "packed-n10-t4096-h8-mha-state"
REPORT_REPOSITORY = "https://github.com/Aharrypotter/gdn-sm90a-tirx-report"
CUTEDSL_DEPENDENCY_LABEL = "nvidia-cutlass-dsl-4.5.1"
CUTEDSL_DEPENDENCY_EXPECTED_AGGREGATE_SHA256 = (
    "41bc70784cde0774308db6883d52e61cdeefe90bedd95631f9da64cee32c5506"
)
CUTEDSL_DISTRIBUTIONS = (
    "nvidia-cutlass-dsl",
    "nvidia-cutlass-dsl-libs-base",
    "nvidia-cutlass-dsl-libs-cu13",
)
TVM_BUILD_LIBRARY_CANDIDATES = {
    # The pinned TVM revision names this target libtvm_compiler.so.  Accept the
    # legacy libtvm.so spelling as well so the attestation remains explicit
    # across the two supported out-of-tree layouts.
    "compiler": ("libtvm.so", "libtvm_compiler.so"),
    "runtime": ("libtvm_runtime.so",),
    "ffi": ("libtvm_ffi.so",),
}

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

IMPLEMENTATIONS: dict[str, dict[str, Any]] = {
    "tirx": {
        "entrypoint": "tirx_kernels.attention.gdn_sm90.chunk_gated_delta_rule",
        "backend_identity": "tirx.gdn.sm90a.wgmma.product-dispatch.packed.v3",
        "source_identity": (
            "tvm-git:acb1312de80b39340e09b0aaad818ff029e745d6;"
            "tirx-git:12ce3721f7c62c5fbd911103ae373de689e58385;"
            "tirx-runtime:90c9c62c84ecc452dd86602f0ea49a625845045c"
        ),
        "require_fallback_false": True,
        "require_unique_cache": True,
    },
    "cutedsl": {
        "entrypoint": "cula.gdn.prefill.chunk_gated_delta_rule",
        "backend_identity": "sm90_cutedsl_gdn",
        "source_identity": (
            "git:88737e9d906cf313995a092624656a89d74dd65e;tag:gdn-sm90a-comparator-r1;dsl:4.5.1"
        ),
        "required_dsl_version": "4.5.1",
        "require_fallback_false": True,
        "require_unique_cache": True,
    },
    "fla": {
        "entrypoint": "fla.ops.gated_delta_rule.chunk.chunk_gated_delta_rule",
        "backend_identity": "fla.gdn.chunk.triton.d1ce07369d58",
        "source_identity": (
            "git:d1ce07369d581813553f30a750af3b6b5f9af6a9;op:fla.ops.gated_delta_rule.chunk"
        ),
        "backend_dispatch_policy": "FLA_DISABLE_BACKEND_DISPATCH=1",
        "require_fallback_false": True,
        "require_unique_cache": True,
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

_EXPECTED_BACKENDS = {
    implementation: spec["backend_identity"] for implementation, spec in IMPLEMENTATIONS.items()
}

FROZEN_ROWS: tuple[dict[str, Any], ...] = (
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
        "expected_backends": _EXPECTED_BACKENDS,
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
        "expected_backends": _EXPECTED_BACKENDS,
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
        "expected_backends": _EXPECTED_BACKENDS,
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
        "expected_backends": _EXPECTED_BACKENDS,
    },
    {
        "row_id": PACKED_N10_ROW_ID,
        "sequence_lengths": [410, 410, 410, 410, 410, 410, 409, 409, 409, 409],
        "q_heads": 8,
        "v_heads": 8,
        "scale": 0.73,
        "seed": 240805,
        "stateful": True,
        "primary": True,
        "critical": False,
        "expected_backends": _EXPECTED_BACKENDS,
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
        "expected_backends": _EXPECTED_BACKENDS,
    },
)

_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class ContractError(ValueError):
    """Raised when a benchmark contract or source checkout violates its lock."""


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize a value deterministically for an identity digest."""

    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _paths_overlap(left: Path, right: Path) -> bool:
    left = left.resolve()
    right = right.resolve()
    return left == right or left.is_relative_to(right) or right.is_relative_to(left)


def cutedsl_dependency_python_paths(root: Path) -> tuple[Path, Path]:
    """Return the two explicit paths needed for metadata and ``cutlass`` imports."""

    root = root.expanduser().resolve()
    package_root = (root / "nvidia_cutlass_dsl" / "python_packages").resolve()
    if not root.is_dir():
        raise ContractError(f"CuTeDSL dependency root is not a directory: {root}")
    if not package_root.is_dir():
        raise ContractError(f"CuTeDSL Python package root is missing: {package_root}")
    for relative in (
        "cutlass/__init__.py",
        "cutlass/cute/__init__.py",
    ):
        path = package_root / relative
        if not path.is_file():
            raise ContractError(f"CuTeDSL dependency is missing {path}")
    return root, package_root


def activate_cutedsl_dependency_root(root: Path) -> tuple[Path, Path]:
    """Prepend the attested dependency paths without processing arbitrary ``.pth`` files."""

    paths = cutedsl_dependency_python_paths(root)
    rendered = [str(path) for path in paths]
    sys.path[:] = rendered + [item for item in sys.path if item not in rendered]
    return paths


def verify_cutedsl_dependency_root(
    root: Path,
    *,
    expected: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build and verify the path-private CuTeDSL 4.5.1 tree attestation."""

    root, _ = cutedsl_dependency_python_paths(root)
    observed = build_manifest(root, CUTEDSL_DEPENDENCY_LABEL)
    if observed.get("aggregate_sha256") != CUTEDSL_DEPENDENCY_EXPECTED_AGGREGATE_SHA256:
        raise ContractError(
            "CuTeDSL dependency aggregate drift: expected "
            f"{CUTEDSL_DEPENDENCY_EXPECTED_AGGREGATE_SHA256}, "
            f"observed {observed.get('aggregate_sha256')}"
        )
    if expected is not None and observed != expected:
        raise ContractError("CuTeDSL dependency tree attestation drifted")
    return observed


def verify_contract_cutedsl_dependency_root(contract: dict[str, Any]) -> dict[str, Any]:
    """Rebuild the exact CuTeDSL dependency manifest bound by the contract."""

    observed = verify_cutedsl_dependency_root(
        Path(contract["cutedsl_dependency_root"]),
        expected=contract["cutedsl_dependency_attestation"],
    )
    digest = sha256_bytes(canonical_json_bytes(observed))
    if digest != contract["cutedsl_dependency_attestation_sha256"]:
        raise ContractError("CuTeDSL dependency attestation digest mismatch")
    return observed


def _distribution_installed_files(
    distribution: importlib.metadata.Distribution,
) -> dict[str, Any]:
    """Hash installed distribution files without exposing absolute filesystem paths."""

    files = distribution.files
    if files is None:
        raise ContractError(
            f"{distribution.metadata.get('Name', '<unknown>')} has no installed-files metadata"
        )
    entries = []
    for relative in sorted(files, key=lambda item: item.as_posix()):
        relative_path = Path(relative.as_posix())
        if relative_path.suffix == ".pyc" or "__pycache__" in relative_path.parts:
            continue
        installed = Path(distribution.locate_file(relative)).resolve()
        if not installed.is_file():
            raise ContractError(f"installed distribution file is missing: {relative.as_posix()}")
        stat = installed.stat()
        entries.append(
            {
                "path": relative.as_posix(),
                "size_bytes": stat.st_size,
                "sha256": _sha256_file(installed),
            }
        )
    aggregate = hashlib.sha256()
    for entry in entries:
        aggregate.update(canonical_json_bytes(entry))
        aggregate.update(b"\n")
    return {
        "schema": "gdn-sm90a.installed-distribution-files.v1",
        "aggregate_sha256": aggregate.hexdigest(),
        "entry_count": len(entries),
        "total_file_bytes": sum(entry["size_bytes"] for entry in entries),
        "entries": entries,
    }


def _distribution_identity(
    candidates: tuple[str, ...],
    *,
    fingerprint_files: bool = False,
) -> dict[str, Any]:
    found: dict[str, importlib.metadata.Distribution] = {}
    for candidate in candidates:
        try:
            distribution = importlib.metadata.distribution(candidate)
        except importlib.metadata.PackageNotFoundError:
            continue
        canonical_name = str(distribution.metadata.get("Name") or candidate)
        found[canonical_name] = distribution
    if len(found) != 1:
        raise ContractError(
            f"expected exactly one installed distribution from {candidates!r}, "
            f"observed {sorted(found)}"
        )
    canonical_name, distribution = next(iter(found.items()))
    identity = {
        "distribution": canonical_name,
        "version": distribution.version,
        "metadata_root": str(Path(distribution.locate_file("")).resolve()),
    }
    if fingerprint_files:
        identity["installed_files"] = _distribution_installed_files(distribution)
    return identity


def _distribution_or_module_identity(
    candidates: tuple[str, ...], module_name: str
) -> dict[str, str | None]:
    """Use distribution metadata when present, otherwise bind a versioned module file."""

    try:
        distribution = _distribution_identity(candidates)
    except ContractError:
        module = importlib.import_module(module_name)
        version = getattr(module, "__version__", None)
        module_file = getattr(module, "__file__", None)
        if not isinstance(version, str) or not version or not module_file:
            raise ContractError(
                f"{module_name} has neither distribution metadata nor versioned module identity"
            ) from None
        return {
            "distribution": None,
            "version": version,
            "metadata_root": None,
            "module_file": str(Path(module_file).resolve()),
        }
    return {**distribution, "module_file": None}


def capture_runtime_identity(
    cutedsl_dependency_root: Path,
    *,
    require_cuda_uninitialized: bool = True,
) -> dict[str, Any]:
    """Freeze the interpreter and package versions before any CUDA initialization."""

    dependency_root, _ = activate_cutedsl_dependency_root(cutedsl_dependency_root)
    torch = importlib.import_module("torch")
    if require_cuda_uninitialized and torch.cuda.is_initialized():
        raise ContractError("runtime identity must be captured before CUDA initialization")
    distributions = {
        "torch": _distribution_identity(("torch",)),
        "triton": _distribution_or_module_identity(("triton",), "triton"),
        "tvm_ffi": _distribution_identity(
            ("apache-tvm-ffi", "tvm-ffi"),
            fingerprint_files=True,
        ),
        "nvidia_cutlass_dsl": _distribution_identity(("nvidia-cutlass-dsl",)),
        "nvidia_cutlass_dsl_libs_base": _distribution_identity(("nvidia-cutlass-dsl-libs-base",)),
        "nvidia_cutlass_dsl_libs_cu13": _distribution_identity(("nvidia-cutlass-dsl-libs-cu13",)),
    }
    for logical_name in (
        "nvidia_cutlass_dsl",
        "nvidia_cutlass_dsl_libs_base",
        "nvidia_cutlass_dsl_libs_cu13",
    ):
        distribution = distributions[logical_name]
        if distribution["version"] != "4.5.1":
            raise ContractError(
                f"{distribution['distribution']} version drift: "
                f"expected 4.5.1, observed {distribution['version']}"
            )
        metadata_root = Path(distribution["metadata_root"])
        if not (metadata_root == dependency_root or metadata_root.is_relative_to(dependency_root)):
            raise ContractError(
                f"{distribution['distribution']} metadata escaped the dependency root: "
                f"{metadata_root}"
            )
    cuda_build = getattr(getattr(torch, "version", None), "cuda", None)
    if not isinstance(cuda_build, str) or not cuda_build:
        raise ContractError("the selected torch distribution has no CUDA build string")
    module_version = getattr(torch, "__version__", None)
    if not isinstance(module_version, str) or not module_version:
        raise ContractError("the selected torch module has no version string")
    return {
        "sys_executable": os.path.abspath(sys.executable),
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_full_version": sys.version,
        "torch_module_version": module_version,
        "torch_cuda_build": cuda_build,
        "distributions": distributions,
    }


def verify_contract_runtime_identity(
    contract: dict[str, Any],
    *,
    require_cuda_uninitialized: bool = True,
) -> dict[str, Any]:
    """Re-capture the runtime identity and require exact contract equality."""

    observed = capture_runtime_identity(
        Path(contract["cutedsl_dependency_root"]),
        require_cuda_uninitialized=require_cuda_uninitialized,
    )
    if observed != contract["runtime_identity"]:
        raise ContractError("Python runtime or package identity drifted")
    digest = sha256_bytes(canonical_json_bytes(observed))
    if digest != contract["runtime_identity_sha256"]:
        raise ContractError("runtime identity digest mismatch")
    return observed


def verify_tvm_build_root(
    root: Path,
    *,
    expected: dict[str, Any] | None = None,
    verify_sha256: bool = True,
) -> dict[str, Any]:
    """Verify the explicit out-of-tree TVM build and its three shared libraries."""

    root = root.expanduser().resolve()
    lib_dir = (root / "lib").resolve()
    if not root.is_dir():
        raise ContractError(f"TVM build root is not a directory: {root}")
    if not lib_dir.is_dir():
        raise ContractError(f"TVM build lib directory is missing: {lib_dir}")
    expected_libraries = expected.get("libraries", {}) if expected is not None else {}
    libraries = {}
    for logical_name, candidates in TVM_BUILD_LIBRARY_CANDIDATES.items():
        expected_library = expected_libraries.get(logical_name)
        if expected_library is not None:
            candidates = (expected_library.get("basename"),)
        matches = [
            lib_dir / candidate
            for candidate in candidates
            if isinstance(candidate, str) and (lib_dir / candidate).is_file()
        ]
        if len(matches) != 1:
            raise ContractError(
                f"TVM build must contain exactly one {logical_name} library from "
                f"{candidates!r} in {lib_dir}; observed {matches}"
            )
        requested_path = matches[0]
        resolved_path = requested_path.resolve()
        if not resolved_path.is_relative_to(lib_dir):
            raise ContractError(
                f"TVM {logical_name} library resolves outside build/lib: {resolved_path}"
            )
        stat = resolved_path.stat()
        sha256 = (
            _sha256_file(resolved_path)
            if verify_sha256 or expected_library is None
            else expected_library["sha256"]
        )
        libraries[logical_name] = {
            "basename": requested_path.name,
            "path": str(requested_path),
            "resolved_path": str(resolved_path),
            "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "sha256": sha256,
        }
    observed = {
        "root": str(root),
        "lib_dir": str(lib_dir),
        "libraries": libraries,
    }
    if expected is not None and observed != expected:
        raise ContractError("out-of-tree TVM build attestation drifted")
    return observed


def verify_contract_runtime_root(
    contract: dict[str, Any], *, verify_sha256: bool = True
) -> dict[str, Any]:
    """Re-verify the exact out-of-tree TVM libraries bound by the contract."""

    observed = verify_tvm_build_root(
        Path(contract["tvm_build_root"]),
        expected=contract["tvm_build_attestation"],
        verify_sha256=verify_sha256,
    )
    digest = sha256_bytes(canonical_json_bytes(observed))
    if digest != contract["tvm_build_attestation_sha256"]:
        raise ContractError("TVM build attestation digest mismatch")
    return observed


def _git_text(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ("git", "-C", str(root), *arguments),
        check=False,
        text=True,
        capture_output=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise ContractError(f"git {' '.join(arguments)} failed for {root}: {detail}")
    return completed.stdout.strip()


def verify_report_checkout(root: Path) -> dict[str, Any]:
    """Verify the exact clean public checkout that supplies this benchmark harness."""

    root = root.expanduser().resolve()
    if not root.is_dir():
        raise ContractError(f"report root is not a directory: {root}")
    top_level = Path(_git_text(root, "rev-parse", "--show-toplevel")).resolve()
    if top_level != root:
        raise ContractError(
            f"report root must be the checkout top level: expected {top_level}, got {root}"
        )
    head = _git_text(root, "rev-parse", "HEAD^{commit}")
    tree = _git_text(root, "rev-parse", "HEAD^{tree}")
    for label, value in (("HEAD", head), ("tree", tree)):
        if len(value) != 40 or any(character not in "0123456789abcdef" for character in value):
            raise ContractError(f"report {label} is not a full Git object ID: {value!r}")
    status = _git_text(root, "status", "--porcelain", "--untracked-files=all")
    if status:
        raise ContractError(f"report checkout is not clean:\n{status}")
    origin = _git_text(root, "config", "--get", "remote.origin.url")
    normalized_origin = origin.removesuffix(".git")
    if normalized_origin != REPORT_REPOSITORY:
        raise ContractError(f"report origin drift: expected {REPORT_REPOSITORY}, observed {origin}")
    return {
        "root": str(root),
        "repository": REPORT_REPOSITORY,
        "head": head,
        "tree": tree,
        "clean_checkout": True,
    }


def verify_contract_report_root(
    contract: dict[str, Any],
    *,
    executing_file: Path | None = None,
) -> dict[str, Any]:
    """Re-verify the public harness checkout and optionally the executing module path."""

    report_root = Path(contract["report_root"]).resolve()
    if executing_file is not None:
        executing_file = executing_file.resolve()
        if not executing_file.is_relative_to(report_root):
            raise ContractError(
                f"benchmark module executed outside the bound report root: {executing_file}"
            )
        expected_package = report_root / "reproduce" / "benchmark"
        if not executing_file.is_relative_to(expected_package):
            raise ContractError(
                f"benchmark module executed outside reproduce/benchmark: {executing_file}"
            )
    observed = verify_report_checkout(report_root)
    if observed != contract["report_attestation"]:
        raise ContractError("report harness attestation drifted")
    digest = sha256_bytes(canonical_json_bytes(observed))
    if digest != contract["report_attestation_sha256"]:
        raise ContractError("report harness attestation digest mismatch")
    return observed


def verify_git_checkout(root: Path, source_name: str) -> dict[str, Any]:
    """Verify a checkout against the immutable public source lock."""

    if source_name not in SOURCE_LOCKS:
        raise ContractError(f"unknown source lock {source_name!r}")
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise ContractError(f"{source_name} root is not a directory: {root}")
    lock = SOURCE_LOCKS[source_name]
    top_level = Path(_git_text(root, "rev-parse", "--show-toplevel")).resolve()
    if top_level != root:
        raise ContractError(
            f"{source_name} root must be the checkout top level: expected {top_level}, got {root}"
        )
    head = _git_text(root, "rev-parse", "HEAD^{commit}")
    if head != lock["commit"]:
        raise ContractError(f"{source_name} HEAD drift: expected {lock['commit']}, observed {head}")
    tree = _git_text(root, "rev-parse", "HEAD^{tree}")
    if tree != lock["tree"]:
        raise ContractError(f"{source_name} tree drift: expected {lock['tree']}, observed {tree}")
    checkout_status = _git_text(root, "status", "--porcelain", "--untracked-files=all")
    if checkout_status:
        raise ContractError(f"{source_name} checkout is not clean:\n{checkout_status}")
    required_path = root / lock["required_path"]
    if not required_path.is_file():
        raise ContractError(f"{source_name} is missing locked path {lock['required_path']}")

    tag = lock.get("tag")
    tag_object = None
    peeled_commit = None
    if tag is not None:
        tag_object = _git_text(root, "rev-parse", f"refs/tags/{tag}^{{tag}}")
        if tag_object != lock["tag_object"]:
            raise ContractError(
                f"{source_name} tag-object drift: expected {lock['tag_object']}, "
                f"observed {tag_object}"
            )
        if _git_text(root, "cat-file", "-t", tag_object) != "tag":
            raise ContractError(f"{source_name} tag {tag!r} is not annotated")
        peeled_commit = _git_text(root, "rev-parse", f"refs/tags/{tag}^{{commit}}")
        if peeled_commit != lock["commit"]:
            raise ContractError(
                f"{source_name} tag peel drift: expected {lock['commit']}, observed {peeled_commit}"
            )

    runtime_commit = lock.get("runtime_commit")
    if runtime_commit is not None:
        ancestor = subprocess.run(
            ("git", "-C", str(root), "merge-base", "--is-ancestor", runtime_commit, head),
            check=False,
        )
        if ancestor.returncode != 0:
            raise ContractError(
                f"{source_name} runtime commit {runtime_commit} is not an ancestor of {head}"
            )

    return {
        "source": source_name,
        "root": str(root),
        "head": head,
        "tree": tree,
        "tracked_clean": True,
        "clean_checkout": True,
        "tag": tag,
        "tag_object": tag_object,
        "peeled_commit": peeled_commit,
        "runtime_commit": runtime_commit,
        "required_path": lock["required_path"],
    }


def verify_contract_source_roots(contract: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Re-verify every source root instead of trusting materialization-time state."""

    roots = contract["source_roots"]
    observed = {
        source_name: verify_git_checkout(Path(roots[source_name]), source_name)
        for source_name in SOURCE_LOCKS
    }
    if observed != contract["source_attestations"]:
        raise ContractError("source attestation drifted after the contract was materialized")
    digest = sha256_bytes(canonical_json_bytes(observed))
    if digest != contract["source_attestation_sha256"]:
        raise ContractError("source attestation digest mismatch")
    return observed


def frozen_rows() -> list[dict[str, Any]]:
    """Return a defensive copy of the six frozen benchmark rows."""

    return copy.deepcopy(list(FROZEN_ROWS))


def _timing_contract(physical_gpu_index: int, gpu_uuid: str) -> dict[str, Any]:
    return {
        "physical_gpu_index": physical_gpu_index,
        "expected_gpu_uuid": gpu_uuid,
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
        "escalation_row_id": PACKED_N10_ROW_ID,
        "escalation_ratios": ["tirx_over_cutedsl", "tirx_over_fla"],
        "timer": "cuda_event",
        "statistic": "median_of_process_averages",
        "post_util_policy": "record_only",
        "require_rotating_three_way_order": True,
        "require_unique_cache": True,
    }


def build_contract(
    *,
    run_id: str,
    source_roots: dict[str, Path],
    report_root: Path,
    tvm_build_root: Path,
    cutedsl_dependency_root: Path,
    oracle_root: Path,
    physical_gpu_index: int,
    gpu_uuid: str,
) -> dict[str, Any]:
    """Build a contract after verifying each caller-supplied checkout."""

    resolved_roots = {
        source_name: str(source_roots[source_name].expanduser().resolve())
        for source_name in SOURCE_LOCKS
    }
    attestations = {
        source_name: verify_git_checkout(Path(resolved_roots[source_name]), source_name)
        for source_name in SOURCE_LOCKS
    }
    resolved_report_root = report_root.expanduser().resolve()
    report_attestation = verify_report_checkout(resolved_report_root)
    resolved_build_root = tvm_build_root.expanduser().resolve()
    build_attestation = verify_tvm_build_root(resolved_build_root)
    resolved_cutedsl_dependency_root = cutedsl_dependency_root.expanduser().resolve()
    cutedsl_dependency_attestation = verify_cutedsl_dependency_root(
        resolved_cutedsl_dependency_root
    )
    runtime_identity = capture_runtime_identity(resolved_cutedsl_dependency_root)
    contract = {
        "schema": SCHEMA,
        "run_id": run_id,
        "claim_scope": "fresh public-tag H20 six-row characterization",
        "report_root": str(resolved_report_root),
        "report_repository": REPORT_REPOSITORY,
        "report_attestation": report_attestation,
        "report_attestation_sha256": sha256_bytes(canonical_json_bytes(report_attestation)),
        "source_roots": resolved_roots,
        "source_locks": copy.deepcopy(SOURCE_LOCKS),
        "source_attestations": attestations,
        "source_attestation_sha256": sha256_bytes(canonical_json_bytes(attestations)),
        "tvm_build_root": str(resolved_build_root),
        "tvm_build_attestation": build_attestation,
        "tvm_build_attestation_sha256": sha256_bytes(canonical_json_bytes(build_attestation)),
        "cutedsl_dependency_root": str(resolved_cutedsl_dependency_root),
        "cutedsl_dependency_attestation": cutedsl_dependency_attestation,
        "cutedsl_dependency_attestation_sha256": sha256_bytes(
            canonical_json_bytes(cutedsl_dependency_attestation)
        ),
        "runtime_identity": runtime_identity,
        "runtime_identity_sha256": sha256_bytes(canonical_json_bytes(runtime_identity)),
        "oracle_root": str(oracle_root.expanduser().resolve()),
        "implementations": copy.deepcopy(IMPLEMENTATIONS),
        "correctness": copy.deepcopy(CORRECTNESS),
        "timing": _timing_contract(physical_gpu_index, gpu_uuid),
        "rows": frozen_rows(),
        "ratio": {"numerator": "tirx", "denominator": "cutedsl"},
        "secondary_ratios": [{"name": "tirx_over_fla", "numerator": "tirx", "denominator": "fla"}],
    }
    validate_contract(contract)
    return contract


def validate_contract(contract: dict[str, Any]) -> None:
    """Reject any change to the frozen identities, rows, or timing protocol."""

    if contract.get("schema") != SCHEMA:
        raise ContractError(f"expected schema {SCHEMA!r}")
    run_id = contract.get("run_id")
    if not isinstance(run_id, str) or not _RUN_ID.fullmatch(run_id):
        raise ContractError("run_id must contain only letters, numbers, dot, underscore, or dash")
    if contract.get("source_locks") != SOURCE_LOCKS:
        raise ContractError("source locks differ from the public release identities")
    if contract.get("implementations") != IMPLEMENTATIONS:
        raise ContractError("implementation identities or entrypoints drifted")
    comparator_entrypoint = contract["implementations"]["cutedsl"]["entrypoint"]
    if comparator_entrypoint != "cula.gdn.prefill.chunk_gated_delta_rule":
        raise ContractError("CuTe comparator must use cula.gdn.prefill, never cula.gdn2")
    if "gdn2" in comparator_entrypoint:
        raise ContractError("CuTe comparator must not use the GDN2 namespace")
    if contract.get("correctness") != CORRECTNESS:
        raise ContractError("correctness policy drifted")
    if contract.get("rows") != frozen_rows():
        raise ContractError("benchmark rows differ from the frozen six-row matrix")
    if contract.get("ratio") != {"numerator": "tirx", "denominator": "cutedsl"}:
        raise ContractError("primary ratio must remain TIRx latency over CuTe latency")
    if contract.get("secondary_ratios") != [
        {"name": "tirx_over_fla", "numerator": "tirx", "denominator": "fla"}
    ]:
        raise ContractError("secondary FLA ratio drifted")

    if contract.get("report_repository") != REPORT_REPOSITORY:
        raise ContractError("report repository differs from the public harness repository")
    report_root = contract.get("report_root")
    if not isinstance(report_root, str) or not Path(report_root).is_absolute():
        raise ContractError("report_root must be an absolute caller-supplied path")
    resolved_report_root = Path(report_root).resolve()
    report_attestation = contract.get("report_attestation")
    if not isinstance(report_attestation, dict):
        raise ContractError("report harness attestation is missing")
    if (
        report_attestation.get("root") != report_root
        or report_attestation.get("repository") != REPORT_REPOSITORY
        or report_attestation.get("clean_checkout") is not True
    ):
        raise ContractError("report harness root, repository, or clean state drifted")
    for key in ("head", "tree"):
        value = report_attestation.get(key)
        if (
            not isinstance(value, str)
            or len(value) != 40
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ContractError(f"report harness {key} is not a full Git object ID")
    expected_report_digest = sha256_bytes(canonical_json_bytes(report_attestation))
    if contract.get("report_attestation_sha256") != expected_report_digest:
        raise ContractError("report harness attestation digest mismatch")

    roots = contract.get("source_roots")
    if not isinstance(roots, dict) or set(roots) != set(SOURCE_LOCKS):
        raise ContractError("source_roots must contain tvm, tirx, cutedsl, and fla")
    if any(not Path(root).is_absolute() for root in roots.values()):
        raise ContractError("all source roots must be absolute")
    for source_name, root in roots.items():
        if _paths_overlap(Path(root), resolved_report_root):
            raise ContractError(f"{source_name} source root must not overlap the report checkout")
    tvm_build_root = contract.get("tvm_build_root")
    if not isinstance(tvm_build_root, str) or not Path(tvm_build_root).is_absolute():
        raise ContractError("tvm_build_root must be an absolute caller-supplied path")
    resolved_build_root = Path(tvm_build_root).resolve()
    if any(
        resolved_build_root == Path(root).resolve()
        or resolved_build_root.is_relative_to(Path(root).resolve())
        for root in roots.values()
    ):
        raise ContractError("tvm_build_root must be outside all four source checkouts")
    if _paths_overlap(resolved_build_root, resolved_report_root):
        raise ContractError("tvm_build_root must not overlap the report checkout")
    build_attestation = contract.get("tvm_build_attestation")
    if not isinstance(build_attestation, dict):
        raise ContractError("TVM build attestation is missing")
    if build_attestation.get("root") != tvm_build_root:
        raise ContractError("TVM build attestation root differs from tvm_build_root")
    expected_lib_dir = str((resolved_build_root / "lib").resolve())
    if build_attestation.get("lib_dir") != expected_lib_dir:
        raise ContractError("TVM build attestation must bind the absolute build/lib directory")
    libraries = build_attestation.get("libraries")
    if not isinstance(libraries, dict) or set(libraries) != set(TVM_BUILD_LIBRARY_CANDIDATES):
        raise ContractError("TVM build attestation must contain compiler, runtime, and ffi")
    for logical_name, library in libraries.items():
        if not isinstance(library, dict):
            raise ContractError(f"TVM {logical_name} library attestation is malformed")
        if library.get("basename") not in TVM_BUILD_LIBRARY_CANDIDATES[logical_name]:
            raise ContractError(f"TVM {logical_name} library basename is unsupported")
        for key in ("path", "resolved_path"):
            path = library.get(key)
            if (
                not isinstance(path, str)
                or not Path(path).is_absolute()
                or not Path(path).resolve().is_relative_to(Path(expected_lib_dir))
            ):
                raise ContractError(f"TVM {logical_name} {key} must be inside the bound build/lib")
        if (
            not isinstance(library.get("size_bytes"), int)
            or library["size_bytes"] <= 0
            or not isinstance(library.get("mtime_ns"), int)
            or library["mtime_ns"] <= 0
        ):
            raise ContractError(f"TVM {logical_name} stat attestation is invalid")
        sha256 = library.get("sha256")
        if (
            not isinstance(sha256, str)
            or len(sha256) != 64
            or any(character not in "0123456789abcdef" for character in sha256)
        ):
            raise ContractError(f"TVM {logical_name} SHA-256 attestation is invalid")
    expected_build_digest = sha256_bytes(canonical_json_bytes(build_attestation))
    if contract.get("tvm_build_attestation_sha256") != expected_build_digest:
        raise ContractError("TVM build attestation digest mismatch")

    cutedsl_dependency_root = contract.get("cutedsl_dependency_root")
    if (
        not isinstance(cutedsl_dependency_root, str)
        or not Path(cutedsl_dependency_root).is_absolute()
    ):
        raise ContractError("cutedsl_dependency_root must be an absolute caller-supplied path")
    resolved_dependency_root = Path(cutedsl_dependency_root).resolve()
    protected_roots = [
        *(Path(root).resolve() for root in roots.values()),
        resolved_report_root,
        resolved_build_root,
    ]
    if any(_paths_overlap(resolved_dependency_root, root) for root in protected_roots):
        raise ContractError(
            "cutedsl_dependency_root must not overlap source, report, or TVM build roots"
        )
    dependency_attestation = contract.get("cutedsl_dependency_attestation")
    if not isinstance(dependency_attestation, dict):
        raise ContractError("CuTeDSL dependency attestation is missing")
    if dependency_attestation.get("root_label") != CUTEDSL_DEPENDENCY_LABEL:
        raise ContractError("CuTeDSL dependency root label drifted")
    if (
        dependency_attestation.get("aggregate_sha256")
        != CUTEDSL_DEPENDENCY_EXPECTED_AGGREGATE_SHA256
    ):
        raise ContractError("CuTeDSL dependency aggregate differs from the preserved tree")
    if not isinstance(dependency_attestation.get("entries"), list) or dependency_attestation.get(
        "entry_count"
    ) != len(dependency_attestation["entries"]):
        raise ContractError("CuTeDSL dependency entries are malformed")
    expected_dependency_digest = sha256_bytes(canonical_json_bytes(dependency_attestation))
    if contract.get("cutedsl_dependency_attestation_sha256") != expected_dependency_digest:
        raise ContractError("CuTeDSL dependency attestation digest mismatch")

    runtime_identity = contract.get("runtime_identity")
    if not isinstance(runtime_identity, dict):
        raise ContractError("runtime identity is missing")
    executable = runtime_identity.get("sys_executable")
    if not isinstance(executable, str) or not Path(executable).is_absolute():
        raise ContractError("runtime sys.executable must be absolute")
    for key in (
        "python_version",
        "python_implementation",
        "python_full_version",
        "torch_module_version",
        "torch_cuda_build",
    ):
        if not isinstance(runtime_identity.get(key), str) or not runtime_identity[key]:
            raise ContractError(f"runtime field {key!r} is missing")
    distributions = runtime_identity.get("distributions")
    expected_distributions = {
        "torch",
        "triton",
        "tvm_ffi",
        "nvidia_cutlass_dsl",
        "nvidia_cutlass_dsl_libs_base",
        "nvidia_cutlass_dsl_libs_cu13",
    }
    if not isinstance(distributions, dict) or set(distributions) != expected_distributions:
        raise ContractError("runtime distribution identity set is incomplete")
    for logical_name, distribution in distributions.items():
        if not isinstance(distribution, dict):
            raise ContractError(f"runtime distribution {logical_name} is malformed")
        if not isinstance(distribution.get("version"), str) or not distribution["version"]:
            raise ContractError(f"runtime distribution {logical_name} lacks a version")
        distribution_name = distribution.get("distribution")
        metadata_root = distribution.get("metadata_root")
        if logical_name == "triton" and distribution_name is None:
            module_file = distribution.get("module_file")
            if (
                metadata_root is not None
                or not isinstance(module_file, str)
                or not Path(module_file).is_absolute()
            ):
                raise ContractError(
                    "runtime triton fallback must bind an absolute versioned module file"
                )
        elif (
            not isinstance(distribution_name, str)
            or not distribution_name
            or not isinstance(metadata_root, str)
            or not Path(metadata_root).is_absolute()
        ):
            raise ContractError(
                f"runtime distribution {logical_name} lacks name or absolute metadata_root"
            )
    for logical_name in (
        "nvidia_cutlass_dsl",
        "nvidia_cutlass_dsl_libs_base",
        "nvidia_cutlass_dsl_libs_cu13",
    ):
        distribution = distributions[logical_name]
        if distribution["version"] != "4.5.1":
            raise ContractError(f"runtime distribution {logical_name} must be version 4.5.1")
        metadata_root = Path(distribution["metadata_root"]).resolve()
        if not (
            metadata_root == resolved_dependency_root
            or metadata_root.is_relative_to(resolved_dependency_root)
        ):
            raise ContractError(
                f"runtime distribution {logical_name} escaped cutedsl_dependency_root"
            )
    ffi_distribution = distributions["tvm_ffi"]["distribution"].lower().replace("_", "-")
    if ffi_distribution not in {"apache-tvm-ffi", "tvm-ffi"}:
        raise ContractError("runtime tvm-ffi distribution name is unsupported")
    ffi_files = distributions["tvm_ffi"].get("installed_files")
    if (
        not isinstance(ffi_files, dict)
        or ffi_files.get("schema") != "gdn-sm90a.installed-distribution-files.v1"
        or not isinstance(ffi_files.get("entries"), list)
        or ffi_files.get("entry_count") != len(ffi_files["entries"])
        or not _is_sha256(ffi_files.get("aggregate_sha256"))
    ):
        raise ContractError("runtime tvm-ffi installed-files fingerprint is malformed")
    total_file_bytes = 0
    aggregate = hashlib.sha256()
    for entry in ffi_files["entries"]:
        if (
            not isinstance(entry, dict)
            or not isinstance(entry.get("path"), str)
            or not entry["path"]
            or Path(entry["path"]).is_absolute()
            or not isinstance(entry.get("size_bytes"), int)
            or entry["size_bytes"] < 0
            or not _is_sha256(entry.get("sha256"))
        ):
            raise ContractError("runtime tvm-ffi installed-file entry is malformed")
        total_file_bytes += entry["size_bytes"]
        aggregate.update(canonical_json_bytes(entry))
        aggregate.update(b"\n")
    if (
        ffi_files.get("total_file_bytes") != total_file_bytes
        or ffi_files["aggregate_sha256"] != aggregate.hexdigest()
    ):
        raise ContractError("runtime tvm-ffi installed-files aggregate mismatch")
    expected_runtime_digest = sha256_bytes(canonical_json_bytes(runtime_identity))
    if contract.get("runtime_identity_sha256") != expected_runtime_digest:
        raise ContractError("runtime identity digest mismatch")

    oracle_root = contract.get("oracle_root")
    if not isinstance(oracle_root, str) or not Path(oracle_root).is_absolute():
        raise ContractError("oracle_root must be an absolute caller-supplied path")

    timing = contract.get("timing")
    if not isinstance(timing, dict):
        raise ContractError("timing contract is missing")
    fixed_timing = {
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
        "escalation_row_id": PACKED_N10_ROW_ID,
        "escalation_ratios": ["tirx_over_cutedsl", "tirx_over_fla"],
        "timer": "cuda_event",
        "statistic": "median_of_process_averages",
        "post_util_policy": "record_only",
        "require_rotating_three_way_order": True,
        "require_unique_cache": True,
    }
    for key, expected in fixed_timing.items():
        if timing.get(key) != expected:
            raise ContractError(f"timing field {key!r} must remain {expected!r}")
    physical_index = timing.get("physical_gpu_index")
    if not isinstance(physical_index, int) or physical_index < 0:
        raise ContractError("physical_gpu_index must be a non-negative integer")
    gpu_uuid = timing.get("expected_gpu_uuid")
    if not isinstance(gpu_uuid, str) or not gpu_uuid.startswith("GPU-"):
        raise ContractError("expected_gpu_uuid must be an NVIDIA GPU UUID")

    attestations = contract.get("source_attestations")
    if not isinstance(attestations, dict) or set(attestations) != set(SOURCE_LOCKS):
        raise ContractError("source attestations are missing")
    for source_name, attestation in attestations.items():
        lock = SOURCE_LOCKS[source_name]
        if attestation.get("root") != roots[source_name]:
            raise ContractError(f"{source_name} attested root differs from source_roots")
        if attestation.get("head") != lock["commit"] or attestation.get("tree") != lock["tree"]:
            raise ContractError(f"{source_name} attestation differs from its commit/tree lock")
        if attestation.get("tracked_clean") is not True:
            raise ContractError(f"{source_name} attestation is not tracked-clean")
        if attestation.get("clean_checkout") is not True:
            raise ContractError(f"{source_name} attestation is not fully clean")
        if attestation.get("tag") != lock.get("tag"):
            raise ContractError(f"{source_name} tag attestation drifted")
        if attestation.get("tag_object") != lock.get("tag_object"):
            raise ContractError(f"{source_name} tag object attestation drifted")
    expected_digest = sha256_bytes(canonical_json_bytes(attestations))
    if contract.get("source_attestation_sha256") != expected_digest:
        raise ContractError("source attestation digest mismatch")


def load_contract(path: Path) -> tuple[dict[str, Any], str]:
    """Load, validate, and hash an exact contract file."""

    raw = path.read_bytes()
    try:
        contract = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ContractError(f"invalid JSON contract {path}: {error}") from error
    validate_contract(contract)
    return contract, sha256_bytes(raw)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Materialize the frozen six-row fresh public-tag benchmark contract."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--tvm-root", type=Path, required=True)
    parser.add_argument("--tirx-root", type=Path, required=True)
    parser.add_argument("--cutedsl-root", type=Path, required=True)
    parser.add_argument("--fla-root", type=Path, required=True)
    parser.add_argument("--report-root", type=Path, required=True)
    parser.add_argument("--tvm-build-root", type=Path, required=True)
    parser.add_argument("--cutedsl-dependency-root", type=Path, required=True)
    parser.add_argument("--oracle-root", type=Path, required=True)
    parser.add_argument("--physical-gpu-index", type=int, required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite contract: {args.output}")
    executing_report_root = Path(__file__).resolve().parents[2]
    if executing_report_root != args.report_root.expanduser().resolve():
        raise ContractError(
            "the executing contract module is outside --report-root: "
            f"{executing_report_root} != {args.report_root.expanduser().resolve()}"
        )
    contract = build_contract(
        run_id=args.run_id,
        source_roots={
            "tvm": args.tvm_root,
            "tirx": args.tirx_root,
            "cutedsl": args.cutedsl_root,
            "fla": args.fla_root,
        },
        report_root=args.report_root,
        tvm_build_root=args.tvm_build_root,
        cutedsl_dependency_root=args.cutedsl_dependency_root,
        oracle_root=args.oracle_root,
        physical_gpu_index=args.physical_gpu_index,
        gpu_uuid=args.gpu_uuid,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(contract, indent=2, sort_keys=True) + "\n"
    args.output.write_text(raw)
    print(f"GDN_CONTRACT_OK sha256={sha256_bytes(raw.encode())} path={args.output}")


if __name__ == "__main__":
    main()
