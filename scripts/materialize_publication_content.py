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

"""Materialize publication templates from the canonical public claim registry."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

import validate_claims

SOURCE_FILES = (
    "master-en.md",
    "master-zh.md",
    "wechat.md",
    "x-thread-en.md",
    "x-thread-zh.md",
    "zhihu.md",
)
EXPECTED_LANGUAGE = {
    "master-en.md": "en",
    "master-zh.md": "zh",
    "wechat.md": "zh",
    "x-thread-en.md": "en",
    "x-thread-zh.md": "zh",
    "zhihu.md": "zh",
}
CLAIM_TOKEN = re.compile(r"\{\{claim:(C\d{2}):(en|zh)\}\}")
ANY_CLAIM_TOKEN = re.compile(r"\{\{claim:[^{}\n]+\}\}")
TEMPLATE_ONLY_BLOCK = re.compile(r"<!--\s*TEMPLATE_ONLY\b.*?-->\s*", re.DOTALL)
MARKDOWN_LINK = re.compile(r"(?P<prefix>!?\[[^\]]*\]\()(?P<target>[^)\s]+)(?P<suffix>\))")


class MaterializationError(ValueError):
    """A publication template cannot be materialized safely."""


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def ensure_inside(path: Path, root: Path, context: str) -> Path:
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise MaterializationError(f"{context} escapes repository root: {path}")
    return resolved


def rebase_relative_links(text: str, source: Path, output: Path, root: Path) -> str:
    """Keep repository-relative Markdown links valid after moving into dist."""

    def replace(match: re.Match[str]) -> str:
        target = match.group("target")
        if (
            "://" in target
            or target.startswith(("#", "/", "mailto:", "data:"))
            or target.startswith("{{")
        ):
            return match.group(0)

        path_part, fragment = (target.split("#", 1) + [""])[:2]
        path_part, query = (path_part.split("?", 1) + [""])[:2]
        source_target = ensure_inside(source.parent / path_part, root, f"{source}:{target}")
        if not source_target.exists():
            raise MaterializationError(f"{source}: relative link target does not exist: {target}")
        rebased = Path(os.path.relpath(source_target, output.parent.resolve())).as_posix()
        if query:
            rebased = f"{rebased}?{query}"
        if fragment:
            rebased = f"{rebased}#{fragment}"
        return f"{match.group('prefix')}{rebased}{match.group('suffix')}"

    return MARKDOWN_LINK.sub(replace, text)


def materialize_one(
    *,
    source: Path,
    output: Path,
    root: Path,
    claims_by_id: dict[str, dict[str, Any]],
    rendered_claims: dict[str, dict[str, str]],
) -> bytes:
    text = source.read_text()
    template_decimal_literals = validate_claims.PERFORMANCE_LITERAL.findall(text)
    if template_decimal_literals:
        raise MaterializationError(
            f"{source}: decimal literals are forbidden in publication templates: "
            f"{template_decimal_literals}"
        )

    text, marker_count = TEMPLATE_ONLY_BLOCK.subn("", text)
    if marker_count != 1:
        raise MaterializationError(
            f"{source}: expected exactly one TEMPLATE_ONLY block, found {marker_count}"
        )

    expected_language = EXPECTED_LANGUAGE[source.name]
    inserted_claims: list[str] = []

    def replace_claim(match: re.Match[str]) -> str:
        claim_id, language = match.groups()
        claim = claims_by_id.get(claim_id)
        if claim is None:
            raise MaterializationError(f"{source}: unknown claim token {claim_id}")
        if claim_id == "C12":
            raise MaterializationError(
                f"{source}: C12 cannot be materialized before its gate passes"
            )
        if claim.get("enabled") is not True:
            reason = claim.get("disabled_reason", "claim is not enabled")
            raise MaterializationError(f"{source}: disabled claim token {claim_id}: {reason}")
        if language != expected_language:
            raise MaterializationError(
                f"{source}: token {claim_id}:{language} does not match {expected_language}"
            )
        try:
            rendered = rendered_claims[claim_id][language]
        except KeyError as error:
            raise MaterializationError(
                f"{source}: canonical statement is unavailable for {claim_id}:{language}"
            ) from error
        inserted_claims.append(rendered)
        return rendered

    text = CLAIM_TOKEN.sub(replace_claim, text)
    residual = ANY_CLAIM_TOKEN.findall(text)
    if residual:
        raise MaterializationError(f"{source}: residual or malformed claim tokens: {residual}")
    if "TEMPLATE_ONLY" in text or "Publication source template" in text or "发布源模板" in text:
        raise MaterializationError(f"{source}: template-only publication instructions remain")

    text = rebase_relative_links(text, source, output, root)
    text = text.rstrip() + "\n"

    allowed_decimals = collections.Counter(
        literal
        for rendered in inserted_claims
        for literal in validate_claims.PERFORMANCE_LITERAL.findall(rendered)
    )
    output_decimals = collections.Counter(validate_claims.PERFORMANCE_LITERAL.findall(text))
    if output_decimals != allowed_decimals:
        raise MaterializationError(
            f"{source}: materialized decimal facts drifted from canonical claims: "
            f"expected={dict(allowed_decimals)}, actual={dict(output_decimals)}"
        )
    return text.encode()


def expected_outputs(root: Path) -> dict[Path, bytes]:
    source_root = ensure_inside(root / "content", root, "content root")
    output_root = ensure_inside(root / "dist/content", root, "output root")
    registry = validate_claims.load_json(root / "contracts/claim-registry.json")
    validation = validate_claims.validate(root, [source_root])
    if validation["status"] != "PASS":
        raise MaterializationError(
            "canonical claim validation failed: " + "; ".join(validation["errors"])
        )

    claims_by_id = {
        claim["id"]: claim
        for claim in registry.get("claims", [])
        if isinstance(claim, dict) and isinstance(claim.get("id"), str)
    }
    expected: dict[Path, bytes] = {}
    for name in SOURCE_FILES:
        source = source_root / name
        if not source.is_file():
            raise MaterializationError(f"missing publication template: {source}")
        output = output_root / name
        expected[output] = materialize_one(
            source=source,
            output=output,
            root=root,
            claims_by_id=claims_by_id,
            rendered_claims=validation["rendered_claims"],
        )
    return expected


def actual_output_files(output_root: Path) -> set[Path]:
    if not output_root.exists():
        return set()
    return {path for path in output_root.rglob("*") if path.is_file()}


def apply_or_check(root: Path, *, check: bool) -> dict[str, Any]:
    expected = expected_outputs(root)
    output_root = root / "dist/content"
    expected_paths = set(expected)
    actual_paths = actual_output_files(output_root)
    extra = sorted(path.relative_to(root).as_posix() for path in actual_paths - expected_paths)
    if extra:
        raise MaterializationError(f"unexpected generated publication files: {extra}")

    mismatches = []
    if check:
        for path, payload in expected.items():
            if not path.is_file():
                mismatches.append(f"{path.relative_to(root)}: missing")
            elif path.read_bytes() != payload:
                mismatches.append(f"{path.relative_to(root)}: content mismatch")
        if mismatches:
            raise MaterializationError("; ".join(mismatches))
    else:
        output_root.mkdir(parents=True, exist_ok=True)
        for path, payload in expected.items():
            if not path.is_file() or path.read_bytes() != payload:
                path.write_bytes(payload)

    return {
        "schema": "gdn-sm90a.publication-content-materialization.v1",
        "status": "PASS",
        "mode": "check" if check else "write",
        "file_count": len(expected),
        "files": [
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": sha256_bytes(payload),
                "size_bytes": len(payload),
            }
            for path, payload in sorted(expected.items())
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Report repository root.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify dist/content is byte-identical to deterministic materialization.",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    try:
        result = apply_or_check(root, check=args.check)
    except (MaterializationError, OSError, ValueError) as error:
        result = {
            "schema": "gdn-sm90a.publication-content-materialization.v1",
            "status": "FAIL",
            "mode": "check" if args.check else "write",
            "error": str(error),
        }
        print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
        raise SystemExit(1) from error
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
