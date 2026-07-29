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

"""Fail-closed NVIDIA H20 / compute-capability-9.0 environment gate."""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any


def command(*args: str) -> str:
    return subprocess.run(
        args,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--physical-gpu-index", required=True, type=int)
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional new JSON output path. Existing files are never overwritten.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    errors: list[str] = []
    facts: dict[str, Any] = {
        "schema": "gdn-sm90a.h20-environment-check.v1",
        "accelerator_required": "NVIDIA H20",
        "compute_capability_required": "9.0",
        "target_arch": "sm_90a",
        "private_device_identifiers_emitted": False,
    }
    if args.physical_gpu_index < 0:
        errors.append("physical GPU index must be nonnegative")

    physical_uuid = None
    physical_name = None
    try:
        rows_text = command(
            "nvidia-smi",
            "--query-gpu=index,name,uuid,compute_cap,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        )
        rows = list(csv.reader(io.StringIO(rows_text), skipinitialspace=True))
        matches = [row for row in rows if len(row) == 6 and int(row[0]) == args.physical_gpu_index]
        if len(matches) != 1:
            errors.append("requested physical GPU index did not resolve exactly once")
        else:
            _, physical_name, physical_uuid, compute_cap, driver, memory_mib = matches[0]
            physical_name = physical_name.strip()
            compute_cap = compute_cap.strip()
            if not physical_name.startswith("NVIDIA H20"):
                errors.append(f"physical GPU is {physical_name!r}, not NVIDIA H20")
            if compute_cap != "9.0":
                errors.append(f"physical GPU compute capability is {compute_cap!r}, not 9.0")
            facts.update(
                accelerator=physical_name,
                compute_capability=compute_cap,
                driver_version=driver.strip(),
                memory_total_mib=int(memory_mib.strip()),
            )
    except (FileNotFoundError, ValueError, subprocess.CalledProcessError) as err:
        errors.append(f"physical NVIDIA query failed: {err}")

    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not visible:
        errors.append("CUDA_VISIBLE_DEVICES must explicitly select exactly one physical H20")
    else:
        tokens = [token.strip() for token in visible.split(",") if token.strip()]
        if len(tokens) != 1:
            errors.append("CUDA_VISIBLE_DEVICES must contain exactly one token")
        elif physical_uuid is not None:
            token = tokens[0]
            if token == str(args.physical_gpu_index):
                facts["binding_mode"] = "physical_index"
            elif token == physical_uuid:
                facts["binding_mode"] = "full_gpu_uuid"
            else:
                errors.append(
                    "CUDA_VISIBLE_DEVICES does not match the requested physical index or full UUID"
                )

    nvcc = shutil.which("nvcc")
    if nvcc is None:
        errors.append("nvcc is not available on PATH")
    else:
        try:
            nvcc_output = command(nvcc, "--version")
            match = re.search(r"release ([0-9]+\.[0-9]+)", nvcc_output)
            if match is None:
                errors.append("could not parse CUDA compiler version")
            else:
                facts["cuda_compiler_release"] = match.group(1)
        except subprocess.CalledProcessError as err:
            errors.append(f"nvcc query failed: {err}")

    try:
        import torch

        if not torch.cuda.is_available():
            errors.append("PyTorch reports CUDA unavailable")
        else:
            count = torch.cuda.device_count()
            facts["logical_cuda_device_count"] = count
            if count != 1:
                errors.append(f"PyTorch exposes {count} CUDA devices, expected exactly one")
            if count >= 1:
                torch_name = torch.cuda.get_device_name(0)
                capability = torch.cuda.get_device_capability(0)
                facts.update(
                    torch_version=torch.__version__,
                    torch_cuda_version=torch.version.cuda,
                    torch_logical_device_name=torch_name,
                    torch_logical_compute_capability=f"{capability[0]}.{capability[1]}",
                )
                if not torch_name.startswith("NVIDIA H20"):
                    errors.append(f"PyTorch logical device is {torch_name!r}, not NVIDIA H20")
                if capability != (9, 0):
                    errors.append(
                        f"PyTorch logical compute capability is {capability}, expected (9, 0)"
                    )
                if physical_name is not None and torch_name != physical_name:
                    errors.append("physical and PyTorch H20 names do not match")
    except (ImportError, RuntimeError) as err:
        errors.append(f"PyTorch CUDA identity check failed: {err}")

    result = {
        **facts,
        "status": "PASS" if not errors else "FAIL",
        "physical_device_binding_verified": not errors,
        "error_count": len(errors),
        "errors": errors,
    }
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        output = args.output.expanduser().resolve()
        if output.exists():
            raise SystemExit(f"refusing to overwrite existing output: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload)
    print(payload, end="")
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
