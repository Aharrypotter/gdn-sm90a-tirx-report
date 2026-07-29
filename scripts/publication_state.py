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

"""Reject obsolete pre-rerun guidance from current publication payloads."""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import PurePosixPath

TEXT_SUFFIXES = {".cff", ".json", ".md", ".svg", ".txt"}
STALE_FRESH_STATE_PATTERNS = (
    (
        "fresh_pending_or_future",
        re.compile(
            r"\bfresh(?: public-tag)?(?: 66-receipt)? (?:rerun|run)\b"
            r"[^\n]{0,100}\b(?:pending|not complete|still required|"
            r"not yet represented|future gate)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "fresh_rerun_pending_marker",
        re.compile(r"\bfresh-rerun-pending\b", re.IGNORECASE),
    ),
    (
        "required_future_rerun",
        re.compile(r"\brequired future rerun\b", re.IGNORECASE),
    ),
    (
        "uncompleted_public_tag_rerun",
        re.compile(r"\buncompleted public-tag rerun\b", re.IGNORECASE),
    ),
    (
        "fresh_pending_zh",
        re.compile(
            r"(?:fresh|public[- ]tags?).{0,40}(?:仍待|尚未完成|未完成|未来门)",
            re.IGNORECASE,
        ),
    ),
)


def scan_stale_fresh_state(payloads: Mapping[str, bytes | str]) -> list[dict[str, object]]:
    """Return stable, line-addressed findings for publication-facing text."""

    findings = []
    for path in sorted(payloads):
        if PurePosixPath(path).suffix.lower() not in TEXT_SUFFIXES:
            continue
        value = payloads[path]
        try:
            text = value.decode("utf-8") if isinstance(value, bytes) else value
        except UnicodeDecodeError:
            findings.append(
                {
                    "path": path,
                    "pattern_id": "non_utf8_publication_text",
                    "line": None,
                }
            )
            continue
        for pattern_id, pattern in STALE_FRESH_STATE_PATTERNS:
            for match in pattern.finditer(text):
                findings.append(
                    {
                        "path": path,
                        "pattern_id": pattern_id,
                        "line": text.count("\n", 0, match.start()) + 1,
                    }
                )
    return findings
