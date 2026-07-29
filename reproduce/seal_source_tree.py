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

"""Create a deterministic, path-private manifest for an rsynced source tree."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

EXCLUDED_NAMES = {".git", ".pytest_cache", ".ruff_cache", "__pycache__"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()


def collect_entries(root: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for directory, directory_names, file_names in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        traversable_directories = []
        for name in sorted(directory_names):
            if name in EXCLUDED_NAMES:
                continue
            path = directory_path / name
            if path.is_symlink():
                entries.append(
                    {
                        "path": path.relative_to(root).as_posix(),
                        "type": "symlink",
                        "target": os.readlink(path),
                    }
                )
            else:
                traversable_directories.append(name)
        directory_names[:] = traversable_directories
        for name in sorted(file_names):
            if name in EXCLUDED_NAMES:
                continue
            path = directory_path / name
            relative = path.relative_to(root).as_posix()
            if path.is_symlink():
                entries.append(
                    {
                        "path": relative,
                        "type": "symlink",
                        "target": os.readlink(path),
                    }
                )
                continue
            if not path.is_file():
                raise RuntimeError(f"unsupported source-tree entry: {relative}")
            stat = path.stat()
            entries.append(
                {
                    "executable": bool(stat.st_mode & 0o111),
                    "path": relative,
                    "sha256": sha256_file(path),
                    "size_bytes": stat.st_size,
                    "type": "file",
                }
            )
    return sorted(entries, key=lambda entry: entry["path"])


def build_manifest(root: Path, label: str) -> dict[str, Any]:
    entries = collect_entries(root)
    aggregate = hashlib.sha256()
    for entry in entries:
        aggregate.update(canonical_bytes(entry))
        aggregate.update(b"\n")
    regular_files = [entry for entry in entries if entry["type"] == "file"]
    symlinks = [entry for entry in entries if entry["type"] == "symlink"]
    return {
        "schema": "gdn-sm90a.transferred-source-tree.v1",
        "root_label": label,
        "aggregate_sha256": aggregate.hexdigest(),
        "entry_count": len(entries),
        "file_count": len(regular_files),
        "symlink_count": len(symlinks),
        "total_file_bytes": sum(entry["size_bytes"] for entry in regular_files),
        "entries": entries,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Compare the computed manifest with an existing output file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"source root is not a directory: {root}")
    if output == Path(output.anchor):
        raise SystemExit("refusing filesystem-root output")

    manifest = build_manifest(root, args.label)
    rendered = json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if args.check:
        if not output.is_file():
            raise SystemExit(f"manifest does not exist: {output}")
        if output.read_text() != rendered:
            raise SystemExit(f"source tree differs from manifest: {args.label}")
        print(
            json.dumps(
                {
                    "status": "PASS",
                    "root_label": args.label,
                    "aggregate_sha256": manifest["aggregate_sha256"],
                    "entry_count": manifest["entry_count"],
                },
                sort_keys=True,
            )
        )
        return

    if output.exists():
        raise SystemExit(f"refusing to overwrite existing output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered)
    print(
        json.dumps(
            {
                "status": "PASS",
                "root_label": args.label,
                "aggregate_sha256": manifest["aggregate_sha256"],
                "entry_count": manifest["entry_count"],
                "output": str(output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
