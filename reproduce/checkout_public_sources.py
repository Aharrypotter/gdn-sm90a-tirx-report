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

"""Create isolated checkouts of the exact public source locks.

Every destination is caller supplied and must not exist.  The program performs
fetch-only network operations and never removes, resets, pushes, or tags a
repository.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Source:
    key: str
    repository: str
    commit: str
    tag: str | None = None
    fetch_depth: int = 1


SOURCES = (
    Source(
        key="tvm",
        repository="https://github.com/Aharrypotter/tvm.git",
        tag="gdn-sm90a-compiler-r0",
        commit="acb1312de80b39340e09b0aaad818ff029e745d6",
    ),
    Source(
        key="tirx",
        repository="https://github.com/Aharrypotter/tirx-kernels.git",
        tag="gdn-sm90a-kernel-r0",
        commit="12ce3721f7c62c5fbd911103ae373de689e58385",
        fetch_depth=3,
    ),
    Source(
        key="cutedsl",
        repository="https://github.com/Aharrypotter/cuLA.git",
        tag="gdn-sm90a-comparator-r1",
        commit="88737e9d906cf313995a092624656a89d74dd65e",
    ),
    Source(
        key="fla",
        repository="https://github.com/fla-org/flash-linear-attention.git",
        commit="d1ce07369d581813553f30a750af3b6b5f9af6a9",
    ),
)


def run(*command: str, cwd: Path | None = None) -> str:
    result = subprocess.run(
        command,
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def checkout(source: Source, destination: Path) -> None:
    run("git", "init", "--quiet", str(destination))
    run("git", "remote", "add", "origin", source.repository, cwd=destination)
    if source.tag is not None:
        refspec = f"refs/tags/{source.tag}:refs/tags/{source.tag}"
        depth = f"--depth={source.fetch_depth}"
        run("git", "fetch", "--quiet", depth, "origin", refspec, cwd=destination)
        run("git", "checkout", "--quiet", "--detach", source.tag, cwd=destination)
    else:
        run("git", "fetch", "--quiet", "--depth=1", "origin", source.commit, cwd=destination)
        run("git", "checkout", "--quiet", "--detach", "FETCH_HEAD", cwd=destination)
    actual = run("git", "rev-parse", "HEAD", cwd=destination)
    if actual != source.commit:
        raise RuntimeError(f"{source.key}: fetched {actual}, expected {source.commit}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tvm-dir", required=True, type=Path)
    parser.add_argument("--tirx-dir", required=True, type=Path)
    parser.add_argument("--cutedsl-dir", required=True, type=Path)
    parser.add_argument("--fla-dir", required=True, type=Path)
    parser.add_argument(
        "--initialize-tvm-ffi",
        action="store_true",
        help="Initialize only TVM's pinned 3rdparty/tvm-ffi submodule.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    destinations = {
        "tvm": args.tvm_dir.expanduser().resolve(),
        "tirx": args.tirx_dir.expanduser().resolve(),
        "cutedsl": args.cutedsl_dir.expanduser().resolve(),
        "fla": args.fla_dir.expanduser().resolve(),
    }
    if len(set(destinations.values())) != len(destinations):
        raise SystemExit("all four destination directories must be different")
    for key, destination in destinations.items():
        if destination.exists():
            raise SystemExit(f"{key}: destination already exists: {destination}")
        if destination == Path(destination.anchor):
            raise SystemExit(f"{key}: refusing filesystem-root destination")

    completed: list[str] = []
    try:
        for source in SOURCES:
            destination = destinations[source.key]
            destination.parent.mkdir(parents=True, exist_ok=True)
            checkout(source, destination)
            completed.append(source.key)
        if args.initialize_tvm_ffi:
            run(
                "git",
                "submodule",
                "update",
                "--init",
                "--recursive",
                "3rdparty/tvm-ffi",
                cwd=destinations["tvm"],
            )
    except Exception as err:
        detail = {
            "status": "FAIL",
            "completed": completed,
            "error": str(err),
            "cleanup_performed": False,
        }
        print(json.dumps(detail, indent=2, sort_keys=True))
        raise SystemExit(1) from err

    print(
        json.dumps(
            {
                "status": "PASS",
                "sources": {source.key: source.commit for source in SOURCES},
                "tvm_ffi_initialized": args.initialize_tvm_ffi,
                "remote_writes": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
