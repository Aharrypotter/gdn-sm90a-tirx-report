#!/usr/bin/env python3
# Copyright 2026 Hongyi Wu
# Licensed under the Apache License, Version 2.0.
"""Render the historical performance table from the canonical JSON result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROW_ORDER = (
    ("single-t512-h8-mha-zero", "single-512 MHA zero"),
    ("single-t1024-h8-mha-state", "single-1024 MHA state"),
    ("single-t1024-h8-hv16-gva-state", "single-1024 GVA state"),
    ("single-t4096-h16-mha-zero", "single-4096 MHA zero"),
    ("packed-n10-t4096-h8-mha-state", "packed-10 MHA state"),
    ("packed-n20-t8192-h8-hv16-gva-state", "packed-20 GVA state"),
)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"{path}: expected a JSON object")
    return value


def render(source: Path) -> str:
    performance = load_json(source)
    lines = [
        "# Historical six-row performance",
        "",
        "> Evidence class: `HISTORICAL_EVIDENCE_BOUND`. This is not yet an",
        "> independent rerun from the public Git tags.",
        "",
        "Timer: CUDA events around the final public callable. Each displayed",
        "latency is the median of independent process-average latencies. Ratio",
        "direction is TIRx latency divided by comparator latency; lower is faster.",
        "",
        "| Row | TIRx (ms) | CuTeDSL (ms) | FLA (ms) | TIRx/CuTeDSL | TIRx/FLA | Processes |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row_id, label in ROW_ORDER:
        row = performance["rows"][row_id]
        lines.append(
            "| "
            + " | ".join(
                (
                    label,
                    f"{row['median_ms']['tirx']:.6f}",
                    f"{row['median_ms']['cutedsl']:.6f}",
                    f"{row['median_ms']['fla']:.6f}",
                    f"{row['tirx_over_cutedsl']:.4f}",
                    f"{row['tirx_over_fla']:.4f}",
                    str(row["n_processes"]),
                )
            )
            + " |"
        )
    geomean = performance["primary_geomean"]
    packed_ratio = performance["rows"]["packed-n10-t4096-h8-mha-state"]["tirx_over_cutedsl"]
    packed_delta_pct = (packed_ratio - 1) * 100
    lines.extend(
        (
            "| **six-row geometric mean** | — | — | — | "
            f"**{geomean['tirx_over_cutedsl']:.4f}** | "
            f"**{geomean['tirx_over_fla']:.4f}** | — |",
            "",
            f"Five of {len(ROW_ORDER)} TIRx/CuTeDSL row ratios are below one. "
            "Packed-10 is "
            f"{packed_delta_pct:.2f}% "
            "above CuTeDSL and remains inside the preregistered "
            f"±{performance['packed_n10_interpretation']['noise_band_pct']:.0f}% "
            "noise band; it is not a speed win.",
            "",
            "Scope: one NVIDIA H20 environment, the frozen six-row GDN prefill",
            "matrix, and public-call latency only. This is not end-to-end model",
            "throughput and does not generalize to every Hopper GPU or shape.",
            "",
            f"Canonical machine-readable source: [`{source.as_posix()}`](../{source.as_posix()}).",
            "",
        )
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("evidence/historical/gdn-sm90a-h20-20260728-v1/results/performance.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/historical-performance.md"),
    )
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    payload = render(args.source)
    if args.check:
        if not args.output.is_file() or args.output.read_text() != payload:
            raise SystemExit(f"generated report is stale: {args.output}")
        print(f"PERFORMANCE_MARKDOWN_CURRENT output={args.output}")
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload)
    print(f"PERFORMANCE_MARKDOWN_RENDERED output={args.output}")


if __name__ == "__main__":
    main()
