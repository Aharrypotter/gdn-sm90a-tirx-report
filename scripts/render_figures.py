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

"""Render deterministic, evidence-bound figures for the public GDN report."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import struct
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch  # noqa: E402
from matplotlib.ticker import MultipleLocator  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    ROOT / "evidence" / "historical" / "gdn-sm90a-h20-20260728-v1" / "results" / "performance.json"
)
DEFAULT_OUTPUT = ROOT / "assets" / "figures"

ROW_IDS = (
    "single-t512-h8-mha-zero",
    "single-t1024-h8-mha-state",
    "single-t1024-h8-hv16-gva-state",
    "single-t4096-h16-mha-zero",
    "packed-n10-t4096-h8-mha-state",
    "packed-n20-t8192-h8-hv16-gva-state",
)
ROW_LABELS = {
    "single-t512-h8-mha-zero": "Single · T512 · H8 MHA · zero",
    "single-t1024-h8-mha-state": "Single · T1024 · H8 MHA · state",
    "single-t1024-h8-hv16-gva-state": "Single · T1024 · H8/Hv16 GVA · state",
    "single-t4096-h16-mha-zero": "Single · T4096 · H16 MHA · zero",
    "packed-n10-t4096-h8-mha-state": "Packed-10 · T4096 · H8 MHA · state",
    "packed-n20-t8192-h8-hv16-gva-state": "Packed-20 · T8192 · H8/Hv16 GVA · state",
}

PAPER = "#FBFCFD"
WHITE = "#FFFFFF"
INK = "#17212B"
MUTED = "#5B6773"
GRID = "#D7DEE5"
NEUTRAL = "#E8ECEF"
NEUTRAL_DARK = "#36424D"
BLUE = "#246B9E"
BLUE_DARK = "#174A70"
BLUE_LIGHT = "#DCEAF4"
GOLD = "#C58B22"
GOLD_DARK = "#79550E"
GOLD_LIGHT = "#F4E8C6"

FIGURE_SIZE = (14.0, 8.5)
PNG_DPI = 160
EXPECTED_PNG_SIZE = (2240, 1360)
FIGURE_NAMES = (
    "latency_by_row",
    "ratios_by_row",
    "architecture_evidence_chain",
)
PRIVATE_PATTERNS = (
    b"/Users/",
    b"/home/",
    b"/workspace/",
    b"file://",
    b"BEGIN PRIVATE KEY",
    b"BEGIN OPENSSH PRIVATE KEY",
)

matplotlib.rcParams.update(
    {
        "figure.facecolor": PAPER,
        "axes.facecolor": PAPER,
        "savefig.facecolor": PAPER,
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.edgecolor": NEUTRAL_DARK,
        "xtick.color": MUTED,
        "ytick.color": INK,
        "axes.titlecolor": INK,
        "axes.linewidth": 0.8,
        "grid.color": GRID,
        "grid.linewidth": 0.7,
        "hatch.linewidth": 1.0,
        "svg.fonttype": "path",
        "svg.hashsalt": "gdn-sm90a-tirx-report-figures-v1",
    }
)


@dataclass(frozen=True)
class Row:
    row_id: str
    label: str
    median_ms: dict[str, float]
    ratios: dict[str, float]
    n_processes: int


@dataclass(frozen=True)
class Evidence:
    rows: tuple[Row, ...]
    evidence_class: str
    receipt_count: int
    noise_band_pct: float
    packed_status: str
    gate_decision: str
    decision_status: str


def load_evidence(path: Path) -> Evidence:
    value = json.loads(path.read_text())
    if value.get("schema") != "gdn-sm90a.public-performance-summary.v1":
        raise ValueError("unexpected performance-summary schema")
    if value.get("evidence_class") != "HISTORICAL_EVIDENCE_BOUND":
        raise ValueError("figures may only render the historical-bound public evidence")
    if value.get("receipt_count") != 66:
        raise ValueError("expected exactly 66 timing receipts")
    if set(value.get("rows", {})) != set(ROW_IDS):
        raise ValueError("performance summary does not contain the exact six frozen rows")

    rows = []
    for row_id in ROW_IDS:
        source = value["rows"][row_id]
        medians = {key: float(source["median_ms"][key]) for key in ("tirx", "cutedsl", "fla")}
        ratios = {
            "cutedsl": float(source["tirx_over_cutedsl"]),
            "fla": float(source["tirx_over_fla"]),
        }
        numeric_values = (*medians.values(), *ratios.values())
        if not all(math.isfinite(item) and item > 0 for item in numeric_values):
            raise ValueError(f"{row_id}: non-positive or non-finite performance value")
        for comparator in ("cutedsl", "fla"):
            calculated = medians["tirx"] / medians[comparator]
            if not math.isclose(calculated, ratios[comparator], rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError(f"{row_id}: published {comparator} ratio does not match medians")
        rows.append(
            Row(
                row_id=row_id,
                label=ROW_LABELS[row_id],
                median_ms=medians,
                ratios=ratios,
                n_processes=int(source["n_processes"]),
            )
        )

    packed = value["packed_n10_interpretation"]
    noise_band_pct = float(packed["noise_band_pct"])
    packed_ratio = rows[4].ratios["cutedsl"]
    lower = 1.0 - noise_band_pct / 100.0
    upper = 1.0 + noise_band_pct / 100.0
    if not lower <= packed_ratio <= upper:
        raise ValueError("packed-10 ratio is not inside the declared noise band")
    if packed["status"] != "WITHIN_PREREGISTERED_NOISE_BAND_NOT_A_SPEED_WIN":
        raise ValueError("packed-10 status does not preserve the no-speed-win boundary")
    if value.get("ratio_direction") != "tirx_latency / comparator_latency; lower is faster":
        raise ValueError("unexpected ratio direction")

    return Evidence(
        rows=tuple(rows),
        evidence_class=value["evidence_class"],
        receipt_count=int(value["receipt_count"]),
        noise_band_pct=noise_band_pct,
        packed_status=packed["status"],
        gate_decision=value["gate_evaluation"]["decision"],
        decision_status=value["decision_status"],
    )


def title_block(fig: plt.Figure, title: str, subtitle: str) -> None:
    fig.text(0.035, 0.965, title, ha="left", va="top", fontsize=20, fontweight="bold")
    fig.text(0.035, 0.925, subtitle, ha="left", va="top", fontsize=10.5, color=MUTED)


def source_note(fig: plt.Figure, evidence: Evidence) -> None:
    fig.text(
        0.035,
        0.028,
        (
            "Source: public performance.json · "
            f"{evidence.evidence_class} · NVIDIA H20 · BF16 / D=128 · "
            f"{evidence.receipt_count} timing receipts · "
            "historical sealed evidence, not a fresh public-tag rerun"
        ),
        ha="left",
        va="bottom",
        fontsize=8.5,
        color=MUTED,
    )


def render_latency(evidence: Evidence) -> plt.Figure:
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    title_block(
        fig,
        "Median operator latency across six frozen GDN rows",
        (
            "NVIDIA H20 · BF16 · D=128 · median of independent-process means · "
            "100 timed iterations/process · absolute milliseconds from zero · lower is better"
        ),
    )
    fig.subplots_adjust(left=0.34, right=0.965, top=0.81, bottom=0.16)

    y = list(range(len(evidence.rows)))
    height = 0.18
    series = (
        ("tirx", "TIRx", -0.22, BLUE, BLUE_DARK, ""),
        ("cutedsl", "CuTeDSL", 0.0, WHITE, BLUE_DARK, "///"),
        ("fla", "FLA", 0.22, GOLD_LIGHT, GOLD_DARK, "xx"),
    )
    maximum = max(row.median_ms[key] for row in evidence.rows for key, *_ in series)
    ax.set_xlim(0.0, maximum * 1.19)
    for key, _label, offset, face, edge, hatch in series:
        positions = [item + offset for item in y]
        values = [row.median_ms[key] for row in evidence.rows]
        ax.barh(
            positions,
            values,
            height=height,
            color=face,
            edgecolor=edge,
            linewidth=1.2,
            hatch=hatch,
            zorder=3,
        )
        for position, value in zip(positions, values, strict=True):
            ax.text(
                value + maximum * 0.012,
                position,
                f"{value:.3f} ms",
                ha="left",
                va="center",
                fontsize=8.5,
                color=INK,
            )

    ax.set_yticks(y, [row.label for row in evidence.rows])
    ax.invert_yaxis()
    ax.set_xlabel("Median operator latency (ms) — lower is better", labelpad=10)
    ax.xaxis.set_major_locator(MultipleLocator(0.1))
    ax.grid(axis="x", zorder=0)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0, pad=10, labelsize=9.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(NEUTRAL_DARK)
    ax.spines["bottom"].set_color(NEUTRAL_DARK)

    legend = (
        Patch(facecolor=BLUE, edgecolor=BLUE_DARK, label="TIRx · solid"),
        Patch(facecolor=WHITE, edgecolor=BLUE_DARK, hatch="///", label="CuTeDSL · diagonal"),
        Patch(facecolor=GOLD_LIGHT, edgecolor=GOLD_DARK, hatch="xx", label="FLA · cross-hatch"),
    )
    ax.legend(
        handles=legend,
        loc="lower right",
        frameon=False,
        ncol=3,
        bbox_to_anchor=(1.0, 1.01),
        borderaxespad=0,
        fontsize=9,
    )
    ax.text(
        1.0,
        -0.115,
        "Independent processes: 3 per row, except packed-10: 7.",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8.5,
        color=MUTED,
    )
    source_note(fig, evidence)
    return fig


def _ratio_label(ax: plt.Axes, value: float, y: float) -> None:
    if value >= 0.94:
        x = value - 0.014
        align = "right"
    else:
        x = value + 0.014
        align = "left"
    ax.text(x, y, f"{value:.3f}", ha=align, va="center", fontsize=8.5, color=INK)


def render_ratios(evidence: Evidence) -> plt.Figure:
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    title_block(
        fig,
        "TIRx-to-comparator latency ratios across six frozen GDN rows",
        (
            "Ratio = TIRx latency / comparator latency · 1.0 is parity · "
            f"neutral band is ±{evidence.noise_band_pct:.0f}% · lower is better"
        ),
    )
    fig.subplots_adjust(left=0.34, right=0.965, top=0.81, bottom=0.16)

    lower = 1.0 - evidence.noise_band_pct / 100.0
    upper = 1.0 + evidence.noise_band_pct / 100.0
    ax.axvspan(
        lower,
        upper,
        facecolor=NEUTRAL,
        edgecolor=NEUTRAL_DARK,
        hatch="////",
        linewidth=0.6,
        alpha=0.8,
        zorder=0,
    )
    ax.axvline(1.0, color=NEUTRAL_DARK, linewidth=1.4, linestyle=(0, (5, 4)), zorder=1)
    packed_index = ROW_IDS.index("packed-n10-t4096-h8-mha-state")
    ax.axhspan(packed_index - 0.45, packed_index + 0.45, color="#F2F4F6", zorder=-1)

    for index, row in enumerate(evidence.rows):
        cute_y = index - 0.12
        fla_y = index + 0.12
        cute_ratio = row.ratios["cutedsl"]
        fla_ratio = row.ratios["fla"]
        ax.plot(
            [cute_ratio, 1.0],
            [cute_y, cute_y],
            color=BLUE_DARK,
            linewidth=1.4,
            linestyle="-",
            zorder=2,
        )
        ax.plot(
            [fla_ratio, 1.0],
            [fla_y, fla_y],
            color=GOLD_DARK,
            linewidth=1.4,
            linestyle=(0, (4, 3)),
            zorder=2,
        )
        ax.scatter(
            [cute_ratio],
            [cute_y],
            s=58,
            marker="o",
            facecolor=BLUE,
            edgecolor=BLUE_DARK,
            linewidth=1.0,
            zorder=4,
        )
        ax.scatter(
            [fla_ratio],
            [fla_y],
            s=60,
            marker="D",
            facecolor=GOLD_LIGHT,
            edgecolor=GOLD_DARK,
            linewidth=1.2,
            zorder=4,
        )
        _ratio_label(ax, cute_ratio, cute_y)
        _ratio_label(ax, fla_ratio, fla_y)

    packed_ratio = evidence.rows[packed_index].ratios["cutedsl"]
    ax.annotate(
        (
            f"Packed-10 TIRx/CuTeDSL = {packed_ratio:.3f}\n"
            f"inside ±{evidence.noise_band_pct:.0f}% band; not a speed win"
        ),
        xy=(packed_ratio, packed_index - 0.12),
        xytext=(0.69, packed_index - 0.95),
        arrowprops={
            "arrowstyle": "-|>",
            "color": NEUTRAL_DARK,
            "linewidth": 1.0,
            "connectionstyle": "arc3,rad=-0.12",
        },
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": WHITE,
            "edgecolor": NEUTRAL_DARK,
            "linewidth": 0.9,
        },
        ha="left",
        va="center",
        fontsize=9,
        color=INK,
        zorder=6,
    )

    ax.set_xlim(0.0, 1.08)
    ax.set_ylim(-0.62, len(evidence.rows) - 0.38)
    ax.set_yticks(list(range(len(evidence.rows))), [row.label for row in evidence.rows])
    ax.invert_yaxis()
    ax.set_xlabel("TIRx latency / comparator latency — lower is better", labelpad=10)
    ax.xaxis.set_major_locator(MultipleLocator(0.1))
    ax.grid(axis="x", zorder=0)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0, pad=10, labelsize=9.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(NEUTRAL_DARK)
    ax.spines["bottom"].set_color(NEUTRAL_DARK)

    legend = (
        Line2D(
            [0],
            [0],
            color=BLUE_DARK,
            marker="o",
            markerfacecolor=BLUE,
            linewidth=1.4,
            label="TIRx / CuTeDSL · solid circle",
        ),
        Line2D(
            [0],
            [0],
            color=GOLD_DARK,
            marker="D",
            markerfacecolor=GOLD_LIGHT,
            linestyle=(0, (4, 3)),
            linewidth=1.4,
            label="TIRx / FLA · dashed diamond",
        ),
        Patch(
            facecolor=NEUTRAL,
            edgecolor=NEUTRAL_DARK,
            hatch="////",
            label=f"±{evidence.noise_band_pct:.0f}% around parity",
        ),
    )
    ax.legend(
        handles=legend,
        loc="lower right",
        frameon=False,
        ncol=3,
        bbox_to_anchor=(1.0, 1.01),
        borderaxespad=0,
        fontsize=9,
    )
    source_note(fig, evidence)
    return fig


def add_box(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    body: str,
    *,
    facecolor: str,
    edgecolor: str,
    hatch: str = "",
    linestyle: str = "-",
) -> None:
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.008,rounding_size=0.012",
        transform=ax.transAxes,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=1.25,
        hatch=hatch,
        linestyle=linestyle,
    )
    ax.add_patch(patch)
    ax.text(
        x + 0.018,
        y + height - 0.026,
        title,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        fontweight="bold",
        color=INK,
    )
    ax.text(
        x + 0.018,
        y + height - 0.068,
        body,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.8,
        color=MUTED,
        linespacing=1.35,
    )


def add_arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    linestyle: str = "-",
    color: str = NEUTRAL_DARK,
    mutation_scale: float = 14,
) -> None:
    arrow = FancyArrowPatch(
        start,
        end,
        transform=ax.transAxes,
        arrowstyle="-|>",
        mutation_scale=mutation_scale,
        linewidth=1.4,
        linestyle=linestyle,
        color=color,
        shrinkA=2,
        shrinkB=2,
    )
    ax.add_patch(arrow)


def render_architecture(evidence: Evidence) -> plt.Figure:
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    title_block(
        fig,
        "Compiler-to-evidence architecture for the SM90a GDN release",
        (
            "Layered source and evidence flow · solid arrows are current historical-bound assets · "
            "dashed arrow is a required future rerun, not current evidence"
        ),
    )
    fig.subplots_adjust(left=0.03, right=0.97, top=0.83, bottom=0.12)
    ax.set_axis_off()

    columns = (
        (0.02, 0.275, "1 · TVM compiler capability", BLUE, BLUE_DARK, WHITE, ""),
        (0.365, 0.275, "2 · TIRx GDN product", BLUE_LIGHT, BLUE_DARK, WHITE, ""),
        (0.71, 0.27, "3 · Validation / release evidence", GOLD_LIGHT, GOLD_DARK, WHITE, "//"),
    )
    for x, width, title, header_face, edge, body_face, hatch in columns:
        header = FancyBboxPatch(
            (x, 0.775),
            width,
            0.085,
            boxstyle="round,pad=0.008,rounding_size=0.012",
            transform=ax.transAxes,
            facecolor=header_face,
            edgecolor=edge,
            linewidth=1.3,
            hatch=hatch,
        )
        ax.add_patch(header)
        ax.text(
            x + 0.018,
            0.817,
            title,
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=11.5,
            fontweight="bold",
            color=INK if header_face != BLUE else WHITE,
        )
        outline = FancyBboxPatch(
            (x, 0.285),
            width,
            0.465,
            boxstyle="round,pad=0.01,rounding_size=0.015",
            transform=ax.transAxes,
            facecolor=body_face,
            edgecolor=edge,
            linewidth=1.1,
        )
        ax.add_patch(outline)

    add_box(
        ax,
        0.037,
        0.602,
        0.241,
        0.12,
        "Target / codegen",
        "sm_90a target\nWGMMA N16 / N32 / N64 / N128",
        facecolor=BLUE_LIGHT,
        edgecolor=BLUE_DARK,
    )
    add_box(
        ax,
        0.037,
        0.452,
        0.241,
        0.12,
        "Async movement",
        "TMA + mbarrier\nbounded, swizzle-aware copies",
        facecolor=WHITE,
        edgecolor=BLUE_DARK,
    )
    add_box(
        ax,
        0.037,
        0.302,
        0.241,
        0.12,
        "Compiler source lock",
        "gdn-sm90a-compiler-r0\nacb1312de80b…",
        facecolor=WHITE,
        edgecolor=BLUE_DARK,
        linestyle="--",
    )

    add_box(
        ax,
        0.382,
        0.602,
        0.241,
        0.12,
        "Contract + API",
        "BF16 Q / K / V · D=128\nFP32 gates and recurrent state",
        facecolor=BLUE_LIGHT,
        edgecolor=BLUE_DARK,
    )
    add_box(
        ax,
        0.382,
        0.452,
        0.241,
        0.12,
        "Algebra + references",
        "64-token chunk semantics\nMHA / GQA / GVA ownership",
        facecolor=WHITE,
        edgecolor=BLUE_DARK,
    )
    add_box(
        ax,
        0.382,
        0.302,
        0.241,
        0.12,
        "Schedules + dispatch",
        "general · register replay\nbounded predecessor replay",
        facecolor=WHITE,
        edgecolor=BLUE_DARK,
        linestyle="--",
    )

    add_box(
        ax,
        0.727,
        0.602,
        0.236,
        0.12,
        "Validation layers",
        "semantics · correctness · safety\ncodegen · resources · timing",
        facecolor=GOLD_LIGHT,
        edgecolor=GOLD_DARK,
        hatch="//",
    )
    add_box(
        ax,
        0.727,
        0.452,
        0.236,
        0.12,
        "Public evidence state",
        f"{evidence.evidence_class}\n{evidence.receipt_count} timing receipts",
        facecolor=WHITE,
        edgecolor=GOLD_DARK,
        hatch="//",
    )
    add_box(
        ax,
        0.727,
        0.302,
        0.236,
        0.12,
        "Decision boundary",
        f"{evidence.gate_decision}\nreport status: {evidence.decision_status}",
        facecolor=WHITE,
        edgecolor=GOLD_DARK,
        linestyle="--",
    )

    add_arrow(ax, (0.302, 0.535), (0.355, 0.535), color=BLUE_DARK)
    add_arrow(ax, (0.647, 0.535), (0.7, 0.535), color=GOLD_DARK)

    ax.text(
        0.02,
        0.235,
        "Evidence-state progression",
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=10,
        fontweight="bold",
        color=INK,
    )
    timeline = (
        (0.02, 0.075, 0.18, "Sealed A46-S3", "immutable historical source"),
        (0.245, 0.075, 0.22, "Public derivation", "allowlisted + privacy-safe"),
        (0.51, 0.075, 0.19, "Current report", "historical-bound claims"),
    )
    for x, y, width, title, body in timeline:
        add_box(
            ax,
            x,
            y,
            width,
            0.11,
            title,
            body,
            facecolor=WHITE,
            edgecolor=NEUTRAL_DARK,
        )
    add_arrow(ax, (0.205, 0.13), (0.238, 0.13), color=NEUTRAL_DARK)
    add_arrow(ax, (0.47, 0.13), (0.503, 0.13), color=NEUTRAL_DARK)
    add_box(
        ax,
        0.755,
        0.075,
        0.225,
        0.11,
        "Fresh public-tag rerun",
        "REQUIRED · not yet represented",
        facecolor=PAPER,
        edgecolor=NEUTRAL_DARK,
        linestyle="--",
    )
    add_arrow(
        ax,
        (0.705, 0.13),
        (0.748, 0.13),
        color=NEUTRAL_DARK,
        linestyle="--",
    )

    fig.text(
        0.035,
        0.025,
        (
            "Unofficial personal-fork research release · no upstream merge or endorsement · "
            "performance state read from public performance.json"
        ),
        ha="left",
        va="bottom",
        fontsize=8.5,
        color=MUTED,
    )
    return fig


def strip_dynamic_svg_metadata(path: Path) -> None:
    text = path.read_text()
    text = re.sub(r"\s*<metadata>.*?</metadata>", "", text, count=1, flags=re.DOTALL)
    text = "\n".join(line.rstrip() for line in text.splitlines()) + "\n"
    path.write_text(text)


def save_figure(fig: plt.Figure, output_dir: Path, name: str) -> None:
    svg_path = output_dir / f"{name}.svg"
    png_path = output_dir / f"{name}.png"
    fig.savefig(
        svg_path,
        format="svg",
        dpi=PNG_DPI,
        metadata={"Date": None, "Creator": "gdn-sm90a-tirx-report render_figures.py"},
    )
    strip_dynamic_svg_metadata(svg_path)
    fig.savefig(
        png_path,
        format="png",
        dpi=PNG_DPI,
        metadata={"Software": "gdn-sm90a-tirx-report render_figures.py"},
    )
    plt.close(fig)


def png_dimensions(path: Path) -> tuple[int, int]:
    payload = path.read_bytes()
    if len(payload) < 24 or payload[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"{path.name}: invalid PNG signature")
    if payload[12:16] != b"IHDR":
        raise ValueError(f"{path.name}: PNG does not start with IHDR")
    return struct.unpack(">II", payload[16:24])


def validate_svg(path: Path) -> list[str]:
    errors = []
    payload = path.read_bytes()
    for pattern in PRIVATE_PATTERNS:
        if pattern in payload:
            errors.append(f"{path.name}: contains forbidden payload {pattern!r}")
    try:
        root = ET.fromstring(payload)
    except ET.ParseError as err:
        return [f"{path.name}: malformed SVG: {err}"]
    view_box = root.attrib.get("viewBox", "").split()
    if len(view_box) != 4:
        errors.append(f"{path.name}: missing four-value viewBox")
    else:
        try:
            width = float(view_box[2])
            height = float(view_box[3])
            if width < 1000 or height < 600:
                errors.append(f"{path.name}: SVG viewBox is too small: {width}x{height}")
        except ValueError:
            errors.append(f"{path.name}: non-numeric SVG viewBox")
    for element in root.iter():
        for key, value in element.attrib.items():
            if key.endswith("href") and not value.startswith("#"):
                errors.append(f"{path.name}: external href is forbidden: {value}")
            for target in re.findall(r"url\(([^)]+)\)", value):
                if not target.startswith("#"):
                    errors.append(f"{path.name}: external URL is forbidden: {target}")
    return errors


def validate_outputs(output_dir: Path) -> dict[str, Any]:
    errors = []
    hashes = {}
    expected_files = {f"{name}.{suffix}" for name in FIGURE_NAMES for suffix in ("svg", "png")}
    actual_files = {
        path.name
        for path in output_dir.iterdir()
        if path.is_file() and path.suffix in {".svg", ".png"}
    }
    if actual_files != expected_files:
        missing = sorted(expected_files - actual_files)
        extra = sorted(actual_files - expected_files)
        errors.append(f"figure membership mismatch: missing={missing}, extra={extra}")
    for name in FIGURE_NAMES:
        svg_path = output_dir / f"{name}.svg"
        png_path = output_dir / f"{name}.png"
        if svg_path.is_file():
            errors.extend(validate_svg(svg_path))
            hashes[svg_path.name] = hashlib.sha256(svg_path.read_bytes()).hexdigest()
        if png_path.is_file():
            payload = png_path.read_bytes()
            for pattern in PRIVATE_PATTERNS:
                if pattern in payload:
                    errors.append(f"{png_path.name}: contains forbidden payload {pattern!r}")
            try:
                dimensions = png_dimensions(png_path)
                if dimensions != EXPECTED_PNG_SIZE:
                    message = (
                        f"{png_path.name}: dimensions are {dimensions}, "
                        f"expected {EXPECTED_PNG_SIZE}"
                    )
                    errors.append(message)
            except ValueError as err:
                errors.append(str(err))
            hashes[png_path.name] = hashlib.sha256(payload).hexdigest()
    return {
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "hashes": dict(sorted(hashes.items())),
        "png_dimensions": list(EXPECTED_PNG_SIZE),
    }


def render_all(evidence: Evidence, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    figures = {
        "latency_by_row": render_latency(evidence),
        "ratios_by_row": render_ratios(evidence),
        "architecture_evidence_chain": render_architecture(evidence),
    }
    for name in FIGURE_NAMES:
        save_figure(figures[name], output_dir, name)


def check_determinism(evidence: Evidence, output_dir: Path) -> dict[str, Any]:
    existing = validate_outputs(output_dir)
    errors = list(existing["errors"])
    with tempfile.TemporaryDirectory(prefix="gdn-sm90a-figures-check-") as temporary:
        regenerated_dir = Path(temporary)
        render_all(evidence, regenerated_dir)
        regenerated = validate_outputs(regenerated_dir)
        errors.extend(regenerated["errors"])
        for name in FIGURE_NAMES:
            for suffix in ("svg", "png"):
                filename = f"{name}.{suffix}"
                current = output_dir / filename
                candidate = regenerated_dir / filename
                both_exist = current.is_file() and candidate.is_file()
                if both_exist and current.read_bytes() != candidate.read_bytes():
                    errors.append(f"{filename}: generated bytes differ from the checked-in asset")
    return {
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "hashes": existing["hashes"],
        "png_dimensions": existing["png_dimensions"],
        "deterministic_byte_comparison": not errors,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    evidence = load_evidence(input_path)
    if args.check:
        if not output_dir.is_dir():
            raise SystemExit("figure output directory is missing")
        result = check_determinism(evidence, output_dir)
    else:
        render_all(evidence, output_dir)
        result = validate_outputs(output_dir)
    result.update(
        {
            "schema": "gdn-sm90a.figure-render-verification.v1",
            "source_schema": "gdn-sm90a.public-performance-summary.v1",
            "evidence_class": evidence.evidence_class,
            "receipt_count": evidence.receipt_count,
        }
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
