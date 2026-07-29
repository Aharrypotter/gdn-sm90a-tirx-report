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

"""Create a deterministic manifest for blobs in an exact committed Git delta."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


def git(repo: Path, *args: str, input_bytes: bytes | None = None) -> bytes:
    return subprocess.run(
        ("git", "-C", str(repo), *args),
        input=input_bytes,
        check=True,
        capture_output=True,
    ).stdout


def resolve_commit(repo: Path, revision: str) -> str:
    return git(repo, "rev-parse", "--verify", f"{revision}^{{commit}}").decode().strip()


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def make_manifest(repo: Path, base_revision: str, head_revision: str) -> dict[str, Any]:
    base = resolve_commit(repo, base_revision)
    head = resolve_commit(repo, head_revision)
    ancestry = subprocess.run(
        ("git", "-C", str(repo), "merge-base", "--is-ancestor", base, head),
        capture_output=True,
    )
    if ancestry.returncode != 0:
        raise ValueError("base revision is not an ancestor of head revision")

    names = git(
        repo,
        "diff",
        "--name-only",
        "--diff-filter=ACMR",
        "-z",
        f"{base}..{head}",
    )
    paths = [item.decode() for item in names.split(b"\0") if item]
    paths.sort(key=str.encode)
    if len(paths) != len(set(paths)):
        raise ValueError("Git delta produced duplicate paths")

    files = []
    aggregate = hashlib.sha256()
    for path in paths:
        if Path(path).is_absolute() or ".." in Path(path).parts:
            raise ValueError(f"unsafe Git path: {path}")
        payload = git(repo, "show", f"{head}:{path}")
        digest = sha256(payload)
        files.append({"path": path, "sha256": digest, "size_bytes": len(payload)})
        aggregate.update(f"{digest}  {path}\n".encode())

    return {
        "schema": "gdn-sm90a.git-delta-seal.v1",
        "base_commit": base,
        "head_commit": head,
        "head_tree": git(repo, "rev-parse", f"{head}^{{tree}}").decode().strip(),
        "diff_filter": "ACMR",
        "file_count": len(files),
        "aggregate": {
            "algorithm": "sha256",
            "entry": "<sha256(file bytes)>  <repo-relative path>\\n",
            "ordering": "repo-relative path bytewise ascending",
            "sha256": aggregate.hexdigest(),
        },
        "files": files,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--base", required=True)
    parser.add_argument("--head", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--expected-aggregate",
        help="Fail unless the generated aggregate equals this SHA-256.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo = args.repo.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not repo.is_dir():
        raise SystemExit("repository path is not a directory")
    if output.exists():
        raise SystemExit(f"refusing to overwrite existing output: {output}")
    try:
        manifest = make_manifest(repo, args.base, args.head)
    except (ValueError, subprocess.CalledProcessError) as err:
        raise SystemExit(f"cannot seal Git delta: {err}") from err
    aggregate = manifest["aggregate"]["sha256"]
    if args.expected_aggregate is not None and aggregate != args.expected_aggregate:
        raise SystemExit(
            f"aggregate mismatch: generated {aggregate}, expected {args.expected_aggregate}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "status": "PASS",
                "file_count": manifest["file_count"],
                "aggregate_sha256": aggregate,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
