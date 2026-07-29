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

"""Fail when a repository-local Markdown link resolves outside or nowhere."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from urllib.parse import unquote

MARKDOWN_LINK = re.compile(
    r"!?\[[^\]]*\]\((?P<target><[^>\n]+>|[^)\s\n]+)(?:\s+[\"'][^\"']*[\"'])?\)"
)
SKIP_DIRECTORIES = {".git", ".pytest_cache", ".ruff_cache", "__pycache__", "build", "scratch"}


def markdown_files(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*.md")
        if not any(part in SKIP_DIRECTORIES for part in path.relative_to(root).parts)
    )


def check_links(root: Path) -> dict[str, object]:
    errors = []
    checked_links = 0
    files = markdown_files(root)
    for path in files:
        relative_source = path.relative_to(root).as_posix()
        for match in MARKDOWN_LINK.finditer(path.read_text()):
            target = match.group("target").strip("<>")
            if target.startswith(("#", "/", "mailto:", "data:", "http://", "https://")):
                continue
            checked_links += 1
            path_part = unquote(target.split("#", 1)[0].split("?", 1)[0])
            resolved = (path.parent / path_part).resolve()
            if not resolved.is_relative_to(root):
                errors.append(f"{relative_source}: link escapes repository: {target}")
            elif not resolved.exists():
                errors.append(f"{relative_source}: missing link target: {target}")
    return {
        "schema": "gdn-sm90a.markdown-link-check.v1",
        "status": "PASS" if not errors else "FAIL",
        "file_count": len(files),
        "checked_local_link_count": checked_links,
        "error_count": len(errors),
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Report repository root.",
    )
    args = parser.parse_args()
    result = check_links(args.root.resolve())
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
