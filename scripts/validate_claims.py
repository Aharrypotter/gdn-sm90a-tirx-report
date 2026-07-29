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
import importlib.util
import json
import math
import re
import statistics
import string
from pathlib import Path
from typing import Any

EXPECTED_CLAIM_IDS = tuple(f"C{index:02d}" for index in range(1, 15))
CANONICAL_PERFORMANCE_PATH = (
    "evidence/historical/gdn-sm90a-h20-20260728-v1/results/performance.json"
)
FRESH_BUNDLE_PATH = "evidence/fresh/gdn-sm90a-public-tags-h20-20260729-v1"
FRESH_PERFORMANCE_PATH = f"{FRESH_BUNDLE_PATH}/performance.json"
FRESH_PUBLICATION_PATH = f"{FRESH_BUNDLE_PATH}/publication.json"
FRESH_ENVIRONMENT_PATH = f"{FRESH_BUNDLE_PATH}/environment.json"
FRESH_REPORT_PATH = "reports/fresh-public-tag-performance.md"
PUBLIC_MANIFEST_PATH = "evidence/historical/gdn-sm90a-h20-20260728-v1/manifest.json"
PUBLIC_MANIFEST_DOC_PATH = "docs/evidence-provenance.md"
EXPECTED_CUTEDSL_TAG = "gdn-sm90a-comparator-r1"
EXPECTED_CUTEDSL_COMMIT = "88737e9d906cf313995a092624656a89d74dd65e"
EXCLUDED_CUTEDSL_TAG = "gdn2-sm90a-comparator-r0"
FRESH_CLAIM_SCOPE = "fresh public-tag H20 six-row characterization"
FRESH_EVIDENCE_KIND = "fresh-public-tag-h20-rerun"
REPORT_TAG = "gdn-sm90a-r1"
SUPERSEDED_REPORT_TAG = "gdn-sm90a-r0"
FRESH_ROW_ORDER = (
    "single-t512-h8-mha-zero",
    "single-t1024-h8-mha-state",
    "single-t1024-h8-hv16-gva-state",
    "single-t4096-h16-mha-zero",
    "packed-n10-t4096-h8-mha-state",
    "packed-n20-t8192-h8-hv16-gva-state",
)
FRESH_TEMPLATE_LANGUAGE = {
    "master-en.md": "en",
    "master-zh.md": "zh",
    "wechat.md": "zh",
    "x-thread-en.md": "en",
    "x-thread-zh.md": "zh",
    "zhihu.md": "zh",
}
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
    if format_name == "fixed_6":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ContractError(f"fixed_6 format requires number, found {value!r}")
        if not math.isfinite(value):
            raise ContractError("fixed_6 cannot render a non-finite number")
        return f"{value:.6f}"
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
    fresh_sources = {
        "fresh_performance": FRESH_PERFORMANCE_PATH,
        "fresh_publication": FRESH_PUBLICATION_PATH,
        "fresh_environment": FRESH_ENVIRONMENT_PATH,
    }
    for source_name, expected_path in fresh_sources.items():
        source = registry.get("canonical_sources", {}).get(source_name, {})
        if source.get("path") != expected_path:
            errors.append(
                f"claim registry does not name canonical {source_name} source {expected_path}"
            )
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
            if binding.get("fact_class") != "performance":
                continue
            expected_source = "fresh_performance" if claim.get("id") == "C12" else "performance"
            if binding.get("source") != expected_source:
                errors.append(
                    f"{claim.get('id')}/{name}: performance fact does not use "
                    f"{expected_source}.json"
                )

    by_id = {claim.get("id"): claim for claim in claims if isinstance(claim, dict)}
    c12 = by_id.get("C12", {})
    if c12.get("enabled") is not True:
        errors.append("C12 must be enabled for the sealed fresh public-tag bundle")
    expected_gate = {
        "required_bundle_path": FRESH_BUNDLE_PATH,
        "required_bundle_class": FRESH_EVIDENCE_KIND,
        "required_claim_scope": FRESH_CLAIM_SCOPE,
        "required_status": "PASS",
        "required_decision_status": "CHARACTERIZATION",
        "required_upstream_merge_claim": False,
        "receipt_count_policy": "derive_from_bundle",
        "must_not_use": "evidence/historical/gdn-sm90a-h20-20260728-v1",
    }
    if c12.get("enablement_gate") != expected_gate:
        errors.append("C12 enablement gate does not exactly bind the fresh evidence boundary")
    if set(c12.get("required_caveats", [])) != {"K01", "K02", "K07", "K09"}:
        errors.append("C12 does not carry the required fresh performance and fork caveats")
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


def validate_link_map(
    root: Path,
    link_map: dict[str, Any],
    source_lock: dict[str, Any],
    fresh_source_lock: dict[str, Any],
) -> list[str]:
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

    fresh_link = link_map.get("links", {}).get("fresh_evidence", {})
    expected_fresh_url = (
        "https://github.com/Aharrypotter/gdn-sm90a-tirx-report/tree/"
        f"{REPORT_TAG}/"
        f"{FRESH_BUNDLE_PATH}"
    )
    if fresh_link.get("kind") != "repository_path" or fresh_link.get("url") != expected_fresh_url:
        errors.append("link map does not name the canonical fresh evidence root")
    if fresh_link.get("evidence_kind") != FRESH_EVIDENCE_KIND:
        errors.append("link map fresh evidence kind drifted")
    if fresh_link.get("decision_status") != "CHARACTERIZATION":
        errors.append("link map fresh decision status drifted")
    manifest_digest = hashlib.sha256(
        (root / FRESH_BUNDLE_PATH / "manifest.json").read_bytes()
    ).hexdigest()
    if fresh_link.get("manifest_sha256") != manifest_digest:
        errors.append("link map fresh manifest digest drifted")

    report_link = link_map.get("links", {}).get("fresh_performance_report", {})
    expected_report_url = (
        "https://github.com/Aharrypotter/gdn-sm90a-tirx-report/blob/"
        f"{REPORT_TAG}/"
        f"{FRESH_REPORT_PATH}"
    )
    if report_link.get("kind") != "repository_path" or report_link.get("url") != (
        expected_report_url
    ):
        errors.append("link map does not name the canonical fresh performance report")

    report_tag = link_map.get("links", {}).get("report_tag", {})
    if report_tag != {
        "kind": "immutable_tag",
        "tag": REPORT_TAG,
        "url": (f"https://github.com/Aharrypotter/gdn-sm90a-tirx-report/tree/{REPORT_TAG}"),
    }:
        errors.append("link map does not bind the final report tag")
    report_release = link_map.get("links", {}).get("report_release", {})
    if report_release != {
        "kind": "release",
        "tag": REPORT_TAG,
        "url": (f"https://github.com/Aharrypotter/gdn-sm90a-tirx-report/releases/tag/{REPORT_TAG}"),
    }:
        errors.append("link map does not bind the final report release")

    historical_link = link_map.get("links", {}).get("historical_evidence", {})
    expected_historical_url = (
        "https://github.com/Aharrypotter/gdn-sm90a-tirx-report/tree/"
        f"{REPORT_TAG}/evidence/historical/gdn-sm90a-h20-20260728-v1"
    )
    if historical_link.get("url") != expected_historical_url:
        errors.append("link map historical evidence URL is not pinned to the report tag")

    superseded_report = link_map.get("excluded_links", {}).get(SUPERSEDED_REPORT_TAG, {})
    if superseded_report != {
        "tag": SUPERSEDED_REPORT_TAG,
        "tag_object": "81c0ec29ebbceed192d871f8d91794d7170bba18",
        "peeled_commit": "e1fd180b12b65552183a63ea2a0b62f21c3b8634",
        "tree": "48f8e4688e5cae8c0aedef6171dc38675b9c3c84",
        "status": "SUPERSEDED_BEFORE_RELEASE",
        "reason": (
            "The tag-event checkout dereferenced the annotated tag object, so the "
            "required tag-object-only asset CI correctly failed. The tag remains "
            "immutable and has no GitHub release; r1 restores the exact tag object "
            "before building."
        ),
    }:
        errors.append("superseded report r0 tag identity or reason drifted")

    fresh_comparator = fresh_source_lock.get("locks", {}).get("cutedsl", {})
    if fresh_comparator.get("tag") != comparator.get("tag"):
        errors.append("link map comparator tag differs from fresh source lock")
    if fresh_comparator.get("commit") != comparator.get("commit"):
        errors.append("link map comparator commit differs from fresh source lock")
    if fresh_comparator.get("required_path") != "cula/gdn/prefill.py":
        errors.append("fresh source lock does not name the GDN comparator entrypoint path")
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


def verify_fresh_bundle(root: Path) -> tuple[dict[str, Any] | None, list[str]]:
    """Run the independent fresh-bundle verifier used by the public workflow."""

    verifier_path = root / "reproduce/fresh_evidence/verify.py"
    try:
        spec = importlib.util.spec_from_file_location(
            "gdn_sm90a_fresh_evidence_verify", verifier_path
        )
        if spec is None or spec.loader is None:
            raise ContractError("cannot load fresh evidence verifier")
        verifier = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(verifier)
        result = verifier.verify_bundle(root / FRESH_BUNDLE_PATH)
    except Exception as error:  # The verifier owns its detailed error taxonomy.
        return None, [f"{FRESH_BUNDLE_PATH}: independent verification failed: {error}"]
    if not isinstance(result, dict) or result.get("status") != "PASS":
        return None, [f"{FRESH_BUNDLE_PATH}: independent verifier did not return PASS"]
    return result, []


def validate_fresh_performance_semantics(
    root: Path,
    performance: dict[str, Any],
    publication: dict[str, Any],
    environment: dict[str, Any],
    verifier_result: dict[str, Any] | None,
) -> list[str]:
    """Validate fresh facts without assuming whether escalation produced 54 or 66 receipts."""

    errors = []
    if performance.get("schema") != "gdn-sm90a.public-performance.v1":
        errors.append("fresh performance schema drifted")
    if performance.get("status") != "PASS":
        errors.append("fresh performance status is not PASS")
    if performance.get("decision_status") != "CHARACTERIZATION":
        errors.append("fresh performance decision is not CHARACTERIZATION")

    rows = performance.get("rows")
    if not isinstance(rows, dict) or set(rows) != set(FRESH_ROW_ORDER):
        return [*errors, "fresh performance rows are not the exact six-row matrix"]

    expected_receipts = 0
    implementations = {"tirx", "cutedsl", "fla"}
    for row_id in FRESH_ROW_ORDER:
        row = rows[row_id]
        if not isinstance(row, dict):
            errors.append(f"fresh/{row_id}: row must be an object")
            continue
        observed = row.get("observed_processes")
        averages = row.get("process_averages_ms")
        if not isinstance(observed, dict) or set(observed) != implementations:
            errors.append(f"fresh/{row_id}: observed-process membership drifted")
            continue
        if not isinstance(averages, dict) or set(averages) != implementations:
            errors.append(f"fresh/{row_id}: process-average membership drifted")
            continue
        expected_processes = row.get("expected_processes")
        for implementation in sorted(implementations):
            count = observed[implementation]
            values = averages[implementation]
            if (
                isinstance(count, bool)
                or not isinstance(count, int)
                or count <= 0
                or count != expected_processes
            ):
                errors.append(f"fresh/{row_id}/{implementation}: process count drifted")
                continue
            if not isinstance(values, list) or len(values) != count:
                errors.append(f"fresh/{row_id}/{implementation}: receipt-derived averages drifted")
                continue
            if any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
                for value in values
            ):
                errors.append(f"fresh/{row_id}/{implementation}: process averages are invalid")
            expected_receipts += count

    receipt_lines = (root / FRESH_BUNDLE_PATH / "timing-receipts.jsonl").read_bytes().splitlines()
    launch_lines = (root / FRESH_BUNDLE_PATH / "launches.jsonl").read_bytes().splitlines()
    count_fields = {
        "performance.receipt_count": performance.get("receipt_count"),
        "performance.fresh_process_launch_count": performance.get("fresh_process_launch_count"),
        "publication.receipt_count": publication.get("receipt_count"),
        "publication.fresh_process_launch_count": publication.get("fresh_process_launch_count"),
        "timing receipt ledger": len(receipt_lines),
        "launch ledger": len(launch_lines),
    }
    if verifier_result is not None:
        count_fields["independent verifier"] = verifier_result.get("receipt_count")
    for name, count in count_fields.items():
        if count != expected_receipts:
            errors.append(
                f"fresh {name}={count!r} does not match bundle-derived "
                f"receipt count {expected_receipts}"
            )

    packed = rows["packed-n10-t4096-h8-mha-state"]
    base_processes = performance.get("base_processes")
    if isinstance(base_processes, bool) or not isinstance(base_processes, int):
        errors.append("fresh base_processes is not an integer")
    else:
        tirx_values = packed.get("process_averages_ms", {}).get("tirx", [])
        cutedsl_values = packed.get("process_averages_ms", {}).get("cutedsl", [])
        if len(tirx_values) >= base_processes and len(cutedsl_values) >= base_processes:
            derived_base_ratio = statistics.median(
                tirx_values[:base_processes]
            ) / statistics.median(cutedsl_values[:base_processes])
            recorded_base_ratio = performance.get("packed_n10_base_ratios", {}).get(
                "tirx_over_cutedsl"
            )
            if not math.isclose(
                derived_base_ratio,
                recorded_base_ratio,
                rel_tol=0.0,
                abs_tol=1e-15,
            ):
                errors.append("fresh packed-n10 base-three-process ratio is not receipt-derived")
        else:
            errors.append("fresh packed-n10 base-three-process set is incomplete")

    primary_rows = [rows[row_id] for row_id in FRESH_ROW_ORDER if rows[row_id].get("primary")]
    for ratio_name in ("tirx_over_cutedsl", "tirx_over_fla"):
        ratios = [row.get(ratio_name) for row in primary_rows]
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
            for value in ratios
        ):
            errors.append(f"fresh {ratio_name} contains invalid row ratios")
            continue
        derived_geomean = math.exp(statistics.fmean(math.log(value) for value in ratios))
        recorded_geomean = performance.get("primary_geomean", {}).get(ratio_name)
        if not math.isclose(
            derived_geomean,
            recorded_geomean,
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            errors.append(f"fresh {ratio_name} geometric mean is not row-derived")

    if publication.get("schema") != "gdn-sm90a.public-fresh-evidence.v1":
        errors.append("fresh publication schema drifted")
    if publication.get("status") != "PASS":
        errors.append("fresh publication status is not PASS")
    if publication.get("evidence_kind") != FRESH_EVIDENCE_KIND:
        errors.append("fresh publication evidence kind drifted")
    if publication.get("claim_scope") != FRESH_CLAIM_SCOPE:
        errors.append("fresh publication claim scope drifted")
    if publication.get("decision_status") != "CHARACTERIZATION":
        errors.append("fresh publication decision is not CHARACTERIZATION")
    if publication.get("upstream_merge_claim") is not False:
        errors.append("fresh publication must not claim an upstream merge")
    if publication.get("process_isolation_verified") is not True:
        errors.append("fresh publication does not verify process isolation")
    if publication.get("physical_device_binding_verified") is not True:
        errors.append("fresh publication does not verify physical-device binding")
    if environment.get("target_arch") != "sm_90a":
        errors.append("fresh environment does not target sm_90a")
    if not str(environment.get("accelerator", "")).startswith("NVIDIA H20"):
        errors.append("fresh environment is not the frozen NVIDIA H20 target")
    return errors


def render_fresh_performance_report(
    performance: dict[str, Any],
    publication: dict[str, Any],
    environment: dict[str, Any],
    manifest_sha256: str,
) -> str:
    """Render the additive fresh report directly from the sealed bundle."""

    def fixed_6(value: Any) -> str:
        return format_value(value, "fixed_6")

    packed_escalated = performance["packed_n10_escalation_required"]
    packed_processes = performance["rows"]["packed-n10-t4096-h8-mha-state"]["expected_processes"]
    if packed_escalated:
        packed_interpretation = (
            "The packed-n10 trigger ratio is intentionally the preregistered "
            "base-three-process value. The table reports the final row ratio after "
            f"the triggered {packed_processes}-process measurement."
        )
    else:
        packed_interpretation = (
            "The packed-n10 base-three-process ratio did not trigger escalation. "
            "The table therefore reports the same three-process measurement set."
        )

    lines = [
        "# Fresh public-tag H20 performance characterization",
        "",
        "This additive report is rendered from the separately sealed fresh evidence "
        "bundle. It does not replace or mutate the historical performance report.",
        "",
        "## Evidence identity",
        "",
        f"- Evidence root: [`{FRESH_BUNDLE_PATH}`](../{FRESH_BUNDLE_PATH}/)",
        f"- Evidence kind: `{publication['evidence_kind']}`",
        f"- Claim scope: `{publication['claim_scope']}`",
        f"- Decision status: `{publication['decision_status']}`",
        f"- Environment: `{environment['accelerator']}`, target `{environment['target_arch']}`",
        f"- Bundle-derived timing receipts: {performance['receipt_count']}",
        f"- Manifest SHA-256: `{manifest_sha256}`",
        "",
        "## Six-row timing table",
        "",
        f"Ratio direction: {performance['ratio_direction']}. Latencies are the "
        f"{performance['statistic']} in milliseconds.",
        "",
        "| Row | TIRx ms | CuTeDSL ms | FLA ms | TIRx/CuTeDSL | TIRx/FLA | "
        "Processes per implementation |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row_id in FRESH_ROW_ORDER:
        row = performance["rows"][row_id]
        lines.append(
            f"| `{row_id}` | {fixed_6(row['median_ms']['tirx'])} | "
            f"{fixed_6(row['median_ms']['cutedsl'])} | "
            f"{fixed_6(row['median_ms']['fla'])} | "
            f"{fixed_6(row['tirx_over_cutedsl'])} | "
            f"{fixed_6(row['tirx_over_fla'])} | {row['expected_processes']} |"
        )
    lines.extend(
        [
            "",
            "## Aggregate and packed-n10 trigger",
            "",
            "- Six-row geometric-mean TIRx/CuTeDSL ratio: "
            f"`{fixed_6(performance['primary_geomean']['tirx_over_cutedsl'])}`.",
            "- Six-row geometric-mean TIRx/FLA ratio: "
            f"`{fixed_6(performance['primary_geomean']['tirx_over_fla'])}`.",
            "- packed-n10 base-three-process TIRx/CuTeDSL trigger ratio: "
            f"`{fixed_6(performance['packed_n10_base_ratios']['tirx_over_cutedsl'])}`.",
            "- packed-n10 escalation required: "
            f"`{str(packed_escalated).lower()}`; "
            f"final processes per implementation: "
            f"`{packed_processes}`.",
            "",
            packed_interpretation,
            "",
            "## Evidence boundary",
            "",
            "This bundle verifies exact public source tags/commits, fresh-process "
            "launch identity, physical H20 binding, receipt-level correctness, and "
            "the six-row public-call timing characterization. Its decision is "
            "`CHARACTERIZATION`, not a universal performance or upstream-release claim.",
            "",
            "This fresh run does **not** reproduce the historical host-sync audit, "
            "Compute Sanitizer gates, or full codegen/resource reseals. Those remain "
            "historical-only evidence and are not promoted into this fresh bundle.",
            "",
            "The TVM, tirx-kernels, and cuLA sources are unofficial fork artifacts. "
            "No upstream merge, endorsement, or official release is claimed.",
            "",
        ]
    )
    return "\n".join(lines)


def validate_fresh_publication_content(
    root: Path,
    rendered_c12: dict[str, str],
) -> list[str]:
    errors = []
    for language, statement in rendered_c12.items():
        weight = sum(1 if ord(character) < 0x1100 else 2 for character in statement)
        if weight > 280:
            errors.append(f"C12/{language}: conservative X weight {weight} exceeds 280")
    for name, language in FRESH_TEMPLATE_LANGUAGE.items():
        path = root / "content" / name
        token = f"{{{{claim:C12:{language}}}}}"
        if token not in path.read_text():
            errors.append(f"content/{name}: missing source-derived C12 token")

    readme = (root / "README.md").read_text()
    if rendered_c12["en"] not in readme:
        errors.append("README.md: missing the exact source-derived English C12 statement")

    required_fresh_links = {
        "README.md": FRESH_BUNDLE_PATH,
        "evidence/README.md": "fresh/gdn-sm90a-public-tags-h20-20260729-v1/",
        "docs/limitations.md": FRESH_BUNDLE_PATH,
    }
    for relative, required_text in required_fresh_links.items():
        text = (root / relative).read_text()
        if required_text not in text:
            errors.append(f"{relative}: missing the canonical fresh evidence link")
        if "CHARACTERIZATION" not in text:
            errors.append(f"{relative}: missing the fresh CHARACTERIZATION boundary")

    stale_markers = (
        "A fresh public-tag rerun is still pending",
        "Fresh public-tag rerun 仍待执行",
        "A fresh public-tag 66-receipt rerun is still required",
        "does not yet contain a fresh public-tag execution bundle",
    )
    for relative in [
        "README.md",
        "evidence/README.md",
        "docs/limitations.md",
        *(f"content/{name}" for name in FRESH_TEMPLATE_LANGUAGE),
    ]:
        text = (root / relative).read_text()
        for marker in stale_markers:
            if marker.casefold() in text.casefold():
                errors.append(f"{relative}: contains stale fresh-evidence marker {marker!r}")
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
    fresh_source_lock = load_json(root / FRESH_BUNDLE_PATH / "source-lock.json")
    verifier_result, verifier_errors = verify_fresh_bundle(root)

    errors.extend(validate_registry_structure(registry, caveats))
    errors.extend(validate_forbidden_phrases(caveats))
    errors.extend(validate_link_map(root, link_map, source_lock, fresh_source_lock))
    errors.extend(validate_public_manifest_reference(root))
    errors.extend(verifier_errors)

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
    fresh_performance = sources.get("fresh_performance")
    fresh_publication = sources.get("fresh_publication")
    fresh_environment = sources.get("fresh_environment")
    if (
        fresh_performance is not None
        and fresh_publication is not None
        and fresh_environment is not None
    ):
        errors.extend(
            validate_fresh_performance_semantics(
                root,
                fresh_performance,
                fresh_publication,
                fresh_environment,
                verifier_result,
            )
        )

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

    if "C12" in rendered_claims:
        errors.extend(validate_fresh_publication_content(root, rendered_claims["C12"]))
    else:
        errors.append("C12 did not materialize from the fresh bundle")

    if (
        fresh_performance is not None
        and fresh_publication is not None
        and fresh_environment is not None
    ):
        manifest_sha256 = hashlib.sha256(
            (root / FRESH_BUNDLE_PATH / "manifest.json").read_bytes()
        ).hexdigest()
        expected_report = render_fresh_performance_report(
            fresh_performance,
            fresh_publication,
            fresh_environment,
            manifest_sha256,
        )
        report_path = root / FRESH_REPORT_PATH
        if not report_path.is_file():
            errors.append(f"{FRESH_REPORT_PATH}: missing")
        elif report_path.read_text() != expected_report:
            errors.append(f"{FRESH_REPORT_PATH}: not source-derived from the fresh bundle")

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
        "canonical_fresh_performance_source": FRESH_PERFORMANCE_PATH,
        "fresh_bundle_receipt_count": (
            verifier_result.get("receipt_count") if verifier_result is not None else None
        ),
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
