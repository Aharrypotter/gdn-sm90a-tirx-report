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
"""CPU-only unit tests; no CUDA runtime, source checkout, or network is required."""

from __future__ import annotations

import importlib.metadata
import math
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from reproduce.benchmark.contract import (
    CUTEDSL_DEPENDENCY_EXPECTED_AGGREGATE_SHA256,
    CUTEDSL_DEPENDENCY_LABEL,
    IMPLEMENTATIONS,
    PACKED_N10_ROW_ID,
    REPORT_REPOSITORY,
    SOURCE_LOCKS,
    ContractError,
    build_contract,
    canonical_json_bytes,
    frozen_rows,
    sha256_bytes,
    validate_contract,
    verify_contract_cutedsl_dependency_root,
    verify_contract_report_root,
    verify_contract_runtime_identity,
    verify_tvm_build_root,
)
from reproduce.benchmark.report import geometric_mean, median_process_average
from reproduce.benchmark.run import compose_pythonpath, escalation_required
from reproduce.benchmark.worker import (
    _versions,
    expected_launch_token,
    parse_gpu_state_row,
    path_is_within_any_root,
    rotation_for_process,
)


class FrozenContractTest(unittest.TestCase):
    def test_six_rows_and_sequence_totals_are_frozen(self) -> None:
        rows = frozen_rows()
        self.assertEqual(len(rows), 6)
        self.assertEqual(rows[4]["row_id"], PACKED_N10_ROW_ID)
        self.assertEqual(sum(rows[4]["sequence_lengths"]), 4096)
        self.assertEqual(sum(rows[5]["sequence_lengths"]), 8192)
        self.assertEqual([row["seed"] for row in rows], list(range(240801, 240807)))

    def test_frozen_rows_returns_a_defensive_copy(self) -> None:
        first = frozen_rows()
        first[0]["sequence_lengths"][0] = 1
        self.assertEqual(frozen_rows()[0]["sequence_lengths"], [512])

    def test_corrected_cutedsl_identity_is_explicit(self) -> None:
        self.assertEqual(
            IMPLEMENTATIONS["cutedsl"]["entrypoint"],
            "cula.gdn.prefill.chunk_gated_delta_rule",
        )
        self.assertNotIn("gdn2", IMPLEMENTATIONS["cutedsl"]["entrypoint"])
        self.assertEqual(
            SOURCE_LOCKS["cutedsl"]["commit"],
            "88737e9d906cf313995a092624656a89d74dd65e",
        )
        self.assertEqual(SOURCE_LOCKS["cutedsl"]["tag"], "gdn-sm90a-comparator-r1")
        self.assertEqual(
            IMPLEMENTATIONS["fla"]["backend_dispatch_policy"],
            "FLA_DISABLE_BACKEND_DISPATCH=1",
        )


class SchedulingHelpersTest(unittest.TestCase):
    def test_three_way_rotation_repeats(self) -> None:
        self.assertEqual(rotation_for_process(0), ("tirx", "cutedsl", "fla"))
        self.assertEqual(rotation_for_process(1), ("cutedsl", "fla", "tirx"))
        self.assertEqual(rotation_for_process(2), ("fla", "tirx", "cutedsl"))
        self.assertEqual(rotation_for_process(3), rotation_for_process(0))
        with self.assertRaises(ValueError):
            rotation_for_process(-1)

    def test_noise_band_is_inclusive(self) -> None:
        self.assertTrue(escalation_required(1.0))
        self.assertTrue(escalation_required(0.98))
        self.assertTrue(escalation_required(1.02))
        self.assertFalse(escalation_required(0.9799))
        self.assertFalse(escalation_required(1.0201))
        with self.assertRaises(ValueError):
            escalation_required(float("nan"))

    def test_launch_identity_changes_with_each_dimension(self) -> None:
        base = expected_launch_token("a" * 64, "row", "tirx", 0)
        self.assertEqual(len(base), 64)
        self.assertEqual(base, expected_launch_token("a" * 64, "row", "tirx", 0))
        self.assertNotEqual(base, expected_launch_token("a" * 64, "row", "tirx", 1))
        self.assertNotEqual(base, expected_launch_token("a" * 64, "row", "cutedsl", 0))


class AuditHelpersTest(unittest.TestCase):
    def test_gpu_state_parser_handles_optional_telemetry(self) -> None:
        state = parse_gpu_state_row(
            ["0", "GPU-test", "NVIDIA H20", "0", "12", "P0", "1980", "2619", "32", "N/A"]
        )
        self.assertEqual(state["physical_index"], 0)
        self.assertEqual(state["uuid"], "GPU-test")
        self.assertIsNone(state["power_draw_w"])
        self.assertEqual(state["sm_clock_mhz"], 1980.0)

    def test_process_statistic_and_geomean(self) -> None:
        receipts = [
            {"summary": {"average_ms": 3.0}},
            {"summary": {"average_ms": 1.0}},
            {"summary": {"average_ms": 2.0}},
        ]
        self.assertEqual(median_process_average(receipts), 2.0)
        self.assertTrue(math.isclose(geometric_mean([1.0, 4.0]), 2.0))
        with self.assertRaises(ValueError):
            geometric_mean([0.0])

    def test_triton_version_falls_back_to_exact_module_version(self) -> None:
        def distribution_version(name: str) -> str:
            if name == "triton":
                raise importlib.metadata.PackageNotFoundError(name)
            return f"{name}-distribution-version"

        with (
            mock.patch(
                "reproduce.benchmark.worker.importlib.metadata.version",
                side_effect=distribution_version,
            ),
            mock.patch(
                "reproduce.benchmark.worker.importlib.import_module",
                return_value=mock.Mock(__version__="3.6.0"),
            ),
        ):
            self.assertEqual(_versions()["triton"], "3.6.0")

        with (
            mock.patch(
                "reproduce.benchmark.worker.importlib.metadata.version",
                side_effect=distribution_version,
            ),
            mock.patch(
                "reproduce.benchmark.worker.importlib.import_module",
                return_value=mock.Mock(spec=[]),
            ),
            self.assertRaisesRegex(RuntimeError, "neither distribution metadata"),
        ):
            _versions()


class RuntimePathTest(unittest.TestCase):
    def _fake_build(self, root: Path) -> None:
        lib = root / "lib"
        lib.mkdir(parents=True)
        for name in ("libtvm.so", "libtvm_runtime.so", "libtvm_ffi.so"):
            (lib / name).write_bytes(f"test-{name}".encode())

    def _fake_dependency_attestation(self) -> dict:
        return {
            "schema": "gdn-sm90a.transferred-source-tree.v1",
            "root_label": CUTEDSL_DEPENDENCY_LABEL,
            "aggregate_sha256": CUTEDSL_DEPENDENCY_EXPECTED_AGGREGATE_SHA256,
            "entry_count": 0,
            "file_count": 0,
            "symlink_count": 0,
            "total_file_bytes": 0,
            "entries": [],
        }

    def _fake_runtime_identity(self, temporary: Path, dependency_root: Path) -> dict:
        ffi_entry = {
            "path": "tvm_ffi/__init__.py",
            "size_bytes": 1,
            "sha256": "0" * 64,
        }
        ffi_aggregate = sha256_bytes(canonical_json_bytes(ffi_entry) + b"\n")

        def dependency_distribution(name: str) -> dict[str, str]:
            return {
                "distribution": name,
                "version": "4.5.1",
                "metadata_root": str(dependency_root.resolve()),
            }

        return {
            "sys_executable": str((temporary / "env" / "bin" / "python").resolve()),
            "python_version": "3.12.3",
            "python_implementation": "CPython",
            "python_full_version": "3.12.3 (test)",
            "torch_module_version": "2.11.0a0+eb65b36914.nv26.02",
            "torch_cuda_build": "13.1",
            "distributions": {
                "torch": {
                    "distribution": "torch",
                    "version": "2.11.0a0+eb65b36914.nv26.2",
                    "metadata_root": str((temporary / "system-site").resolve()),
                },
                "triton": {
                    "distribution": None,
                    "version": "3.6.0",
                    "metadata_root": None,
                    "module_file": str((temporary / "system-site" / "triton.py").resolve()),
                },
                "tvm_ffi": {
                    "distribution": "apache-tvm-ffi",
                    "version": "0.1.13",
                    "metadata_root": str((temporary / "env-site").resolve()),
                    "installed_files": {
                        "schema": "gdn-sm90a.installed-distribution-files.v1",
                        "aggregate_sha256": ffi_aggregate,
                        "entry_count": 1,
                        "total_file_bytes": 1,
                        "entries": [ffi_entry],
                    },
                },
                "nvidia_cutlass_dsl": dependency_distribution("nvidia-cutlass-dsl"),
                "nvidia_cutlass_dsl_libs_base": dependency_distribution(
                    "nvidia-cutlass-dsl-libs-base"
                ),
                "nvidia_cutlass_dsl_libs_cu13": dependency_distribution(
                    "nvidia-cutlass-dsl-libs-cu13"
                ),
            },
        }

    def test_out_of_tree_build_attestation_and_contract_field(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            build_root = temporary / "build"
            self._fake_build(build_root)
            attestation = verify_tvm_build_root(build_root)
            self.assertEqual(attestation["lib_dir"], str((build_root / "lib").resolve()))
            self.assertEqual(set(attestation["libraries"]), {"compiler", "runtime", "ffi"})

            source_roots = {name: temporary / "sources" / name for name in SOURCE_LOCKS}
            report_root = temporary / "report"
            dependency_root = temporary / "cutedsl-dependency"
            dependency_attestation = self._fake_dependency_attestation()
            runtime_identity = self._fake_runtime_identity(temporary, dependency_root)

            def fake_git_attestation(root: Path, source_name: str) -> dict:
                lock = SOURCE_LOCKS[source_name]
                return {
                    "source": source_name,
                    "root": str(root.resolve()),
                    "head": lock["commit"],
                    "tree": lock["tree"],
                    "tracked_clean": True,
                    "clean_checkout": True,
                    "tag": lock.get("tag"),
                    "tag_object": lock.get("tag_object"),
                    "peeled_commit": lock["commit"] if lock.get("tag") else None,
                    "runtime_commit": lock.get("runtime_commit"),
                    "required_path": lock["required_path"],
                }

            report_attestation = {
                "root": str(report_root.resolve()),
                "repository": REPORT_REPOSITORY,
                "head": "a" * 40,
                "tree": "b" * 40,
                "clean_checkout": True,
            }
            with (
                mock.patch(
                    "reproduce.benchmark.contract.verify_git_checkout",
                    side_effect=fake_git_attestation,
                ),
                mock.patch(
                    "reproduce.benchmark.contract.verify_report_checkout",
                    return_value=report_attestation,
                ),
                mock.patch(
                    "reproduce.benchmark.contract.verify_cutedsl_dependency_root",
                    return_value=dependency_attestation,
                ),
                mock.patch(
                    "reproduce.benchmark.contract.capture_runtime_identity",
                    return_value=runtime_identity,
                ),
            ):
                contract = build_contract(
                    run_id="cpu-contract-test",
                    source_roots=source_roots,
                    report_root=report_root,
                    tvm_build_root=build_root,
                    cutedsl_dependency_root=dependency_root,
                    oracle_root=temporary / "oracles",
                    physical_gpu_index=0,
                    gpu_uuid="GPU-00000000-0000-0000-0000-000000000000",
                )
            validate_contract(contract)
            self.assertEqual(contract["tvm_build_root"], str(build_root.resolve()))
            self.assertEqual(contract["tvm_build_attestation"], attestation)
            self.assertEqual(contract["report_attestation"], report_attestation)
            self.assertEqual(
                contract["cutedsl_dependency_attestation"],
                dependency_attestation,
            )
            self.assertEqual(contract["runtime_identity"], runtime_identity)

            (build_root / "lib" / "libtvm.so").write_bytes(b"drifted")
            with self.assertRaises(ContractError):
                verify_tvm_build_root(build_root, expected=attestation)

    def test_pythonpath_preserves_external_site_without_source_tvm_ffi(self) -> None:
        contract = {
            "report_root": "/locked/report",
            "cutedsl_dependency_root": "/deps/cutedsl-4.5.1",
            "source_roots": {
                "tvm": "/locked/tvm",
                "tirx": "/locked/tirx",
                "cutedsl": "/locked/cula",
                "fla": "/locked/fla",
            },
        }
        external = os.pathsep.join(("/venv/tvm-ffi-site", "/venv/cutedsl-4.5.1"))
        paths = compose_pythonpath(contract, external).split(os.pathsep)
        self.assertIn("/venv/tvm-ffi-site", paths)
        self.assertIn("/venv/cutedsl-4.5.1", paths)
        self.assertEqual(paths[0], "/locked/report")
        self.assertIn("/deps/cutedsl-4.5.1/nvidia_cutlass_dsl/python_packages", paths)
        self.assertNotIn("/locked/tvm/.local/python", paths)
        self.assertNotIn("/locked/tvm/3rdparty/tvm-ffi/python", paths)

    def test_noneditable_dependency_must_be_outside_all_sources(self) -> None:
        roots = (Path("/sources/tvm"), Path("/sources/tirx"))
        self.assertTrue(path_is_within_any_root(Path("/sources/tvm/python/tvm"), roots))
        self.assertFalse(path_is_within_any_root(Path("/venv/site/tvm_ffi"), roots))

    def test_report_dependency_and_runtime_drift_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            dependency_root = temporary / "dependency"
            report_attestation = {
                "root": str((temporary / "report").resolve()),
                "repository": REPORT_REPOSITORY,
                "head": "a" * 40,
                "tree": "b" * 40,
                "clean_checkout": True,
            }
            dependency_attestation = self._fake_dependency_attestation()
            runtime_identity = self._fake_runtime_identity(temporary, dependency_root)
            contract = {
                "report_root": report_attestation["root"],
                "report_attestation": report_attestation,
                "report_attestation_sha256": sha256_bytes(canonical_json_bytes(report_attestation)),
                "cutedsl_dependency_root": str(dependency_root.resolve()),
                "cutedsl_dependency_attestation": dependency_attestation,
                "cutedsl_dependency_attestation_sha256": sha256_bytes(
                    canonical_json_bytes(dependency_attestation)
                ),
                "runtime_identity": runtime_identity,
                "runtime_identity_sha256": sha256_bytes(canonical_json_bytes(runtime_identity)),
            }

            with mock.patch(
                "reproduce.benchmark.contract.verify_report_checkout",
                return_value={**report_attestation, "tree": "c" * 40},
            ):
                with self.assertRaises(ContractError):
                    verify_contract_report_root(contract)
            with mock.patch(
                "reproduce.benchmark.contract.verify_cutedsl_dependency_root",
                return_value={**dependency_attestation, "entry_count": 1},
            ):
                with self.assertRaises(ContractError):
                    verify_contract_cutedsl_dependency_root(contract)
            with mock.patch(
                "reproduce.benchmark.contract.capture_runtime_identity",
                return_value={
                    **runtime_identity,
                    "torch_cuda_build": "12.9",
                },
            ):
                with self.assertRaises(ContractError):
                    verify_contract_runtime_identity(contract)


if __name__ == "__main__":
    unittest.main()
