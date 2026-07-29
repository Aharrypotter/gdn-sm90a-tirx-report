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

"""Validate and materialize the public claim registry from canonical evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import string
from pathlib import Path
from typing import Any

EXPECTED_CLAIM_IDS = tuple(f"C{index:02d}" for index in range(1, 15))
CANONICAL_PERFORMANCE_PATH = (
    "evidence/historical/gdn-sm90a-h20-20260728-v1/results/performance.json"
)
PUBLIC_MANIFEST_PATH = "evidence/historical/gdn-sm90a-h20-20260728-v1/manifest.json"
PUBLIC_MANIFEST_DOC_PATH = "docs/evidence-provenance.md"
EXPECTED_CUTEDSL_TAG = "gdn-sm90a-comparator-r1"
EXPECTED_CUTEDSL_COMMIT = "88737e9d906cf313995a092624656a89d74dd65e"
EXCLUDED_CUTEDSL_TAG = "gdn2-sm90a-comparator-r0"
CONTENT_SUFFIXES = {".md", ".txt"}
PERFORMANCE_LITERAL = re.compile(r"(?<![A-Za-z0-9_-])\d+\.\d+(?![A-Za-z0-9_-])")


class ContractError(ValueError):
    """A machine-readable public claim contract is invalid."""


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ContractError(f"{path}: expected a JSON object")
    return value


def resolve_pointer(document: Any, pointer: str) -> Any:
    if pointer == "":
        return document
    if not pointer.startswith("/"):
        raise ContractError(f"invalid JSON pointer: {pointer!r}")
    value = document
    for raw_token in pointer[1:].split("/"):
        token = raw_token.replace("~1", "/").replace("~0", "~")
        if isinstance(value, dict):
            if token not in value:
                raise ContractError(f"{pointer}: missing object key {token!r}")
            value = value[token]
        elif isinstance(value, list):
            try:
                value = value[int(token)]
            except (IndexError, ValueError) as error:
                raise ContractError(f"{pointer}: invalid list index {token!r}") from error
        else:
            raise ContractError(f"{pointer}: cannot descend through {type(value).__name__}")
    return value


def compare(left: Any, operator: str, right: Any) -> bool:
    operators = {
        "eq": lambda: left == right,
        "lt": lambda: left < right,
        "le": lambda: left <= right,
        "gt": lambda: left > right,
        "ge": lambda: left >= right,
    }
    if operator not in operators:
        raise ContractError(f"unsupported comparison operator: {operator!r}")
    return operators[operator]()


def evaluate_binding(binding: dict[str, Any], sources: dict[str, Any]) -> Any:
    source_name = binding.get("source")
    if source_name not in sources:
        raise ContractError(f"binding names unknown source {source_name!r}")
    value = resolve_pointer(sources[source_name], binding.get("pointer", ""))
    operation = binding.get("operation")
    if operation == "value":
        return value
    if operation == "map_length":
        if not isinstance(value, dict):
            raise ContractError("map_length requires an object")
        return len(value)
    if operation == "count_where":
        if not isinstance(value, dict):
            raise ContractError("count_where requires an object")
        field = binding.get("field")
        operator = binding.get("operator")
        operand = binding.get("operand")
        count = 0
        for row_id, row in value.items():
            if not isinstance(row, dict) or field not in row:
                raise ContractError(f"count_where row {row_id!r} lacks field {field!r}")
            if compare(row[field], operator, operand):
                count += 1
        return count
    if operation == "percent_above_one":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ContractError("percent_above_one requires a number")
        return (value - 1.0) * 100.0
    raise ContractError(f"unsupported binding operation: {operation!r}")


def format_value(value: Any, format_name: str) -> str:
    if format_name == "exact":
        if isinstance(value, str):
            return value
        return json.dumps(value, ensure_ascii=False, allow_nan=False)
    if format_name == "integer":
        if isinstance(value, bool) or not isinstance(value, int):
            raise ContractError(f"integer format requires int, found {value!r}")
        return str(value)
    if format_name == "fixed_2":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ContractError(f"fixed_2 format requires number, found {value!r}")
        if not math.isfinite(value):
            raise ContractError("fixed_2 cannot render a non-finite number")
        return f"{value:.2f}"
    if format_name == "fixed_4":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ContractError(f"fixed_4 format requires number, found {value!r}")
        if not math.isfinite(value):
            raise ContractError("fixed_4 cannot render a non-finite number")
        return f"{value:.4f}"
    raise ContractError(f"unsupported rendering format: {format_name!r}")


def template_fields(template: str) -> set[str]:
    fields = set()
    for _, field_name, format_spec, conversion in string.Formatter().parse(template):
        if field_name is not None:
            if format_spec or conversion:
                raise ContractError("formatting belongs in bindings, not statement templates")
            fields.add(field_name)
    return fields


def materialize_claim(
    claim: dict[str, Any],
    sources: dict[str, Any],
) -> tuple[dict[str, str], dict[str, Any]]:
    values = {}
    rendered_values = {}
    bindings = claim.get("bindings", {})
    if not isinstance(bindings, dict):
        raise ContractError(f"{claim.get('id')}: bindings must be an object")
    for name, binding in bindings.items():
        if not isinstance(binding, dict):
            raise ContractError(f"{claim.get('id')}/{name}: binding must be an object")
        values[name] = evaluate_binding(binding, sources)
        rendered_values[name] = format_value(values[name], binding.get("format"))

    statements = claim.get("statement")
    if not isinstance(statements, dict) or set(statements) != {"en", "zh"}:
        raise ContractError(f"{claim.get('id')}: statement must contain exactly en and zh")
    for language, template in statements.items():
        if not isinstance(template, str):
            raise ContractError(f"{claim.get('id')}/{language}: statement must be text")
        fields = template_fields(template)
        if fields != set(bindings):
            raise ContractError(
                f"{claim.get('id')}/{language}: template fields {sorted(fields)} "
                f"do not match bindings {sorted(bindings)}"
            )
    rendered = {
        language: template.format_map(rendered_values) for language, template in statements.items()
    }
    return rendered, values


def validate_registry_structure(
    registry: dict[str, Any],
    caveats: dict[str, Any],
) -> list[str]:
    errors = []
    if registry.get("schema") != "gdn-sm90a.public-claim-registry.v1":
        errors.append("unexpected claim-registry schema")
    if caveats.get("schema") != "gdn-sm90a.public-claim-caveats.v1":
        errors.append("unexpected caveats schema")
    performance_source = registry.get("canonical_sources", {}).get("performance", {})
    if performance_source.get("path") != CANONICAL_PERFORMANCE_PATH:
        errors.append("claim registry does not name the canonical performance source")
    if caveats.get("canonical_performance_source") != CANONICAL_PERFORMANCE_PATH:
        errors.append("caveats do not name the canonical performance source")

    claims = registry.get("claims")
    if not isinstance(claims, list):
        return [*errors, "claims must be a list"]
    claim_ids = [claim.get("id") for claim in claims if isinstance(claim, dict)]
    if tuple(claim_ids) != EXPECTED_CLAIM_IDS:
        errors.append(f"claim IDs must be exactly {list(EXPECTED_CLAIM_IDS)}")
    if len(claim_ids) != len(set(claim_ids)):
        errors.append("claim IDs are not unique")

    mandatory = caveats.get("mandatory_caveats")
    if not isinstance(mandatory, list):
        return [*errors, "mandatory_caveats must be a list"]
    caveat_ids = [item.get("id") for item in mandatory if isinstance(item, dict)]
    if len(caveat_ids) != len(set(caveat_ids)):
        errors.append("caveat IDs are not unique")
    known_caveats = set(caveat_ids)
    for claim in claims:
        if not isinstance(claim, dict):
            errors.append("every claim must be an object")
            continue
        for caveat_id in claim.get("required_caveats", []):
            if caveat_id not in known_caveats:
                errors.append(f"{claim.get('id')}: unknown caveat {caveat_id!r}")
        statements = claim.get("statement", {})
        if claim.get("category", "").startswith("performance"):
            for language, template in statements.items():
                if PERFORMANCE_LITERAL.search(template):
                    errors.append(
                        f"{claim.get('id')}/{language}: copied decimal performance literal"
                    )
        for name, binding in claim.get("bindings", {}).items():
            if binding.get("fact_class") == "performance" and binding.get("source") != (
                "performance"
            ):
                errors.append(
                    f"{claim.get('id')}/{name}: performance fact does not use performance.json"
                )

    by_id = {claim.get("id"): claim for claim in claims if isinstance(claim, dict)}
    if by_id.get("C12", {}).get("enabled") is not False:
        errors.append("C12 must remain disabled until a fresh public-tag bundle exists")
    if by_id.get("C12", {}).get("enablement_gate", {}).get("must_not_use") != (
        "evidence/historical/gdn-sm90a-h20-20260728-v1"
    ):
        errors.append("C12 does not explicitly exclude the historical bundle")
    return errors


def validate_forbidden_phrases(caveats: dict[str, Any]) -> list[str]:
    errors = []
    entries = caveats.get("forbidden_phrases")
    if not isinstance(entries, list) or not entries:
        return ["forbidden_phrases must be a non-empty list"]
    ids = []
    patterns = []
    for entry in entries:
        if not isinstance(entry, dict):
            errors.append("every forbidden phrase must be an object")
            continue
        entry_id = entry.get("id")
        match = entry.get("match")
        pattern = entry.get("pattern")
        ids.append(entry_id)
        patterns.append((match, pattern))
        if match not in {"literal_casefold", "regex_casefold"}:
            errors.append(f"{entry_id}: unsupported forbidden-phrase matcher")
        if not isinstance(pattern, str) or not pattern:
            errors.append(f"{entry_id}: forbidden phrase has no pattern")
        elif match == "regex_casefold":
            try:
                re.compile(pattern, re.IGNORECASE)
            except re.error as error:
                errors.append(f"{entry_id}: invalid regular expression: {error}")
    if len(ids) != len(set(ids)):
        errors.append("forbidden phrase IDs are not unique")
    if len(patterns) != len(set(patterns)):
        errors.append("forbidden phrase patterns are not unique")
    return errors


def validate_link_map(link_map: dict[str, Any], source_lock: dict[str, Any]) -> list[str]:
    errors = []
    if link_map.get("schema") != "gdn-sm90a.public-link-map.v1":
        errors.append("unexpected link-map schema")
    comparator = link_map.get("links", {}).get("cutedsl_comparator", {})
    if comparator.get("tag") != EXPECTED_CUTEDSL_TAG:
        errors.append("CuTeDSL comparator does not use corrected r1 tag")
    if comparator.get("commit") != EXPECTED_CUTEDSL_COMMIT:
        errors.append("CuTeDSL comparator does not use corrected 88737e9 commit")
    if EXPECTED_CUTEDSL_TAG not in comparator.get("url", ""):
        errors.append("CuTeDSL comparator URL is not pinned to the corrected r1 tag")
    excluded = link_map.get("excluded_links", {}).get(EXCLUDED_CUTEDSL_TAG, {})
    if excluded.get("tag") != EXCLUDED_CUTEDSL_TAG:
        errors.append("gdn2 r0 tag is not explicitly excluded")
    if excluded.get("status") != "NOT_USED_BY_HISTORICAL_GDN_RECEIPTS":
        errors.append("gdn2 r0 exclusion status is missing")

    locked_comparator = source_lock.get("public_sources", {}).get("cutedsl_comparator", {})
    locked_release = locked_comparator.get("public_release", {})
    locked_exact = locked_comparator.get("exact_runtime_source", {})
    if locked_release.get("tag") != comparator.get("tag"):
        errors.append("link map comparator tag differs from public source lock")
    if locked_release.get("commit") != comparator.get("commit"):
        errors.append("link map comparator commit differs from public source lock")
    if locked_exact.get("entrypoint") != comparator.get("entrypoint"):
        errors.append("link map comparator entrypoint differs from public source lock")
    locked_excluded = source_lock.get("superseded_artifacts", {}).get(EXCLUDED_CUTEDSL_TAG, {})
    if locked_excluded.get("status") != excluded.get("status"):
        errors.append("link map gdn2 exclusion differs from public source lock")
    return errors


def validate_performance_semantics(performance: dict[str, Any]) -> list[str]:
    errors = []
    rows = performance.get("rows")
    if not isinstance(rows, dict) or not rows:
        return ["performance rows must be a non-empty object"]
    cutedsl_ratios = {
        row_id: row.get("tirx_over_cutedsl")
        for row_id, row in rows.items()
        if isinstance(row, dict)
    }
    fla_ratios = {
        row_id: row.get("tirx_over_fla") for row_id, row in rows.items() if isinstance(row, dict)
    }
    if set(cutedsl_ratios) != set(rows) or set(fla_ratios) != set(rows):
        errors.append("every performance row must contain both comparator ratios")
        return errors
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
        for value in [*cutedsl_ratios.values(), *fla_ratios.values()]
    ):
        errors.append("performance ratios must be positive finite numbers")
        return errors

    non_wins = [row_id for row_id, value in cutedsl_ratios.items() if value >= 1.0]
    if non_wins != ["packed-n10-t4096-h8-mha-state"]:
        errors.append("packed-n10 must be the only non-winning CuTeDSL row")
    if any(value >= 1.0 for value in fla_ratios.values()):
        errors.append("every frozen FLA ratio must remain below one")

    packed = performance.get("packed_n10_interpretation", {})
    packed_ratio = cutedsl_ratios.get("packed-n10-t4096-h8-mha-state")
    if packed.get("tirx_over_cutedsl") != packed_ratio:
        errors.append("packed-n10 interpretation ratio differs from the row summary")
    noise_band_pct = packed.get("noise_band_pct")
    if (
        isinstance(noise_band_pct, bool)
        or not isinstance(noise_band_pct, (int, float))
        or packed_ratio is None
        or packed_ratio > 1.0 + noise_band_pct / 100.0
    ):
        errors.append("packed-n10 is not inside the declared noise band")

    expected_receipts = 0
    for row_id, row in rows.items():
        process_averages = row.get("process_averages_ms", {})
        if set(process_averages) != {"tirx", "cutedsl", "fla"}:
            errors.append(f"{row_id}: incomplete process-average implementations")
            continue
        expected_receipts += sum(len(values) for values in process_averages.values())
    if performance.get("receipt_count") != expected_receipts:
        errors.append("receipt_count does not match process-average membership")
    if performance.get("unique_cache_count") != expected_receipts:
        errors.append("unique_cache_count does not match process-average membership")
    return errors


def find_content_files(root: Path, requested: list[Path]) -> list[Path]:
    if requested:
        candidates = []
        for requested_path in requested:
            path = requested_path if requested_path.is_absolute() else root / requested_path
            if path.is_dir():
                candidates.extend(
                    child
                    for child in path.rglob("*")
                    if child.is_file() and child.suffix.lower() in CONTENT_SUFFIXES
                )
            elif path.is_file():
                candidates.append(path)
            else:
                raise ContractError(f"content path does not exist: {path}")
        return sorted(set(candidates))

    candidates = []
    readme = root / "README.md"
    if readme.is_file():
        candidates.append(readme)
    for directory_name in ("content", "reports", "docs", "releases"):
        directory = root / directory_name
        if directory.is_dir():
            candidates.extend(
                child
                for child in directory.rglob("*")
                if child.is_file() and child.suffix.lower() in CONTENT_SUFFIXES
            )
    return sorted(set(candidates))


def scan_content(
    root: Path,
    paths: list[Path],
    caveats: dict[str, Any],
) -> list[str]:
    errors = []
    for path in paths:
        text = path.read_text()
        relative = path.resolve().relative_to(root.resolve()).as_posix()
        for entry in caveats["forbidden_phrases"]:
            pattern = entry["pattern"]
            if entry["match"] == "literal_casefold":
                matched = pattern.casefold() in text.casefold()
            else:
                matched = re.search(pattern, text, re.IGNORECASE) is not None
            if matched:
                errors.append(f"{relative}: contains forbidden phrase {entry['id']}")
    return errors


def validate_public_manifest_reference(root: Path) -> list[str]:
    manifest_path = root / PUBLIC_MANIFEST_PATH
    documentation_path = root / PUBLIC_MANIFEST_DOC_PATH
    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    expected_line = f"{digest}  manifest.json"
    if expected_line not in documentation_path.read_text():
        return [
            f"{PUBLIC_MANIFEST_DOC_PATH}: public manifest reference does not match "
            f"{PUBLIC_MANIFEST_PATH}"
        ]
    return []


def validate(root: Path, requested_content: list[Path]) -> dict[str, Any]:
    errors = []
    registry = load_json(root / "contracts/claim-registry.json")
    caveats = load_json(root / "contracts/caveats.json")
    link_map = load_json(root / "contracts/link-map.json")
    source_lock = load_json(
        root / "evidence/historical/gdn-sm90a-h20-20260728-v1/metadata/source-lock.json"
    )

    errors.extend(validate_registry_structure(registry, caveats))
    errors.extend(validate_forbidden_phrases(caveats))
    errors.extend(validate_link_map(link_map, source_lock))
    errors.extend(validate_public_manifest_reference(root))

    sources = {}
    for source_name, source in registry.get("canonical_sources", {}).items():
        source_path = (root / source["path"]).resolve()
        if not source_path.is_relative_to(root.resolve()):
            errors.append(f"canonical source escapes repository: {source['path']}")
            continue
        sources[source_name] = load_json(source_path)
    performance = sources.get("performance")
    if performance is not None:
        errors.extend(validate_performance_semantics(performance))

    rendered_claims = {}
    raw_values = {}
    for claim in registry.get("claims", []):
        claim_id = claim.get("id")
        if claim.get("enabled") is not True:
            continue
        try:
            rendered, values = materialize_claim(claim, sources)
        except (ContractError, KeyError, TypeError, ValueError) as error:
            errors.append(f"{claim_id}: {error}")
            continue
        rendered_claims[claim_id] = rendered
        raw_values[claim_id] = values

    try:
        content_files = find_content_files(root, requested_content)
        errors.extend(scan_content(root, content_files, caveats))
    except (ContractError, ValueError) as error:
        errors.append(str(error))
        content_files = []

    return {
        "schema": "gdn-sm90a.public-claim-validation.v1",
        "status": "PASS" if not errors else "FAIL",
        "error_count": len(errors),
        "errors": errors,
        "canonical_performance_source": CANONICAL_PERFORMANCE_PATH,
        "enabled_claim_count": len(rendered_claims),
        "disabled_claims": [
            claim.get("id")
            for claim in registry.get("claims", [])
            if claim.get("enabled") is not True
        ],
        "content_files_checked": [
            path.resolve().relative_to(root.resolve()).as_posix() for path in content_files
        ],
        "rendered_claims": rendered_claims,
        "resolved_values": raw_values,
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
        "--check-content",
        action="append",
        default=[],
        type=Path,
        help="Additional content file or directory to scan; may be repeated.",
    )
    parser.add_argument(
        "--show-rendered",
        action="store_true",
        help="Include source-derived rendered claims in stdout.",
    )
    args = parser.parse_args()

    result = validate(args.root.resolve(), args.check_content)
    if not args.show_rendered:
        result.pop("rendered_claims")
        result.pop("resolved_values")
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
