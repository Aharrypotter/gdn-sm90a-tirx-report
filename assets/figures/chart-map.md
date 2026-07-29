# Figure chart map

All quantitative values are rendered from
`evidence/historical/gdn-sm90a-h20-20260728-v1/results/performance.json`.
The rendering surface is deterministic static SVG plus sealed PNG.  The public
evidence class is `HISTORICAL_EVIDENCE_BOUND`; none of these figures represents
a fresh rerun from the public tags.

## `latency_by_row`

- **Question:** What are the absolute public-operator latencies for TIRx,
  CuTeDSL, and FLA on each of the six frozen H20 rows?
- **Supported takeaway:** Absolute latency varies materially by row.  TIRx is
  lower on five CuTeDSL comparisons; packed-10 must be read separately and is
  not claimed as a speed win.
- **Family / variant:** Comparison & Ranking / grouped horizontal bar.
- **Fields:** `rows.<row_id>.median_ms.{tirx,cutedsl,fla}`,
  `rows.<row_id>.n_processes`, `evidence_class`, `receipt_count`.
- **Scale:** Milliseconds, linear, forced to a zero baseline; lower is better.
- **Palette / non-color:** Blue solid TIRx; white/blue-outline diagonal-hatch
  CuTeDSL; gold cross-hatch FLA.  Direct values and the legend preserve
  distinction without color.
- **Paths:** `assets/figures/latency_by_row.svg` and
  `assets/figures/latency_by_row.png`.
- **QA:** Six rows and three implementations present; zero baseline visible;
  exact direct values fit; long labels fit; source, H20/BF16/D128 scope,
  process count, receipt count, and historical-bound caveat visible.

## `ratios_by_row`

- **Question:** How does TIRx latency compare with each comparator on every
  frozen row, and where is the preregistered parity/noise region?
- **Supported takeaway:** Five TIRx/CuTeDSL ratios are below parity.  Packed-10
  is above parity but inside the preregistered ±2% band and is explicitly
  labelled **not a speed win**.  All TIRx/FLA ratios are below parity.
- **Family / variant:** Uncertainty & Benchmark / paired dot-and-reference
  chart.
- **Fields:** `rows.<row_id>.tirx_over_cutedsl`,
  `rows.<row_id>.tirx_over_fla`,
  `packed_n10_interpretation.noise_band_pct`,
  `packed_n10_interpretation.status`.
- **Scale:** Ratio from zero to 1.08; 1.0 parity line; shaded ±2% band; lower is
  better.
- **Palette / non-color:** Blue circle/solid connector for CuTeDSL; gold
  diamond/dashed connector for FLA; neutral hatched parity band and dark
  reference line.
- **Paths:** `assets/figures/ratios_by_row.svg` and
  `assets/figures/ratios_by_row.png`.
- **QA:** Ratios are recomputed from medians before rendering; 1.0 line and
  ±2% band visible; packed-10 annotation does not imply a win; exact ratio
  labels fit; source and historical-bound caveat visible.

## `architecture_evidence_chain`

- **Question:** How do the compiler primitives, TIRx GDN product modules, and
  validation/release evidence connect without promoting historical evidence
  to a fresh rerun?
- **Supported takeaway:** The deliverable is a layered chain from SM90a
  compiler capability through a frozen GDN product contract and schedules to
  a public, historical-bound evidence package.  A fresh public-tag rerun is a
  separate future gate.
- **Family / variant:** Decomposition & Progression / layered architecture and
  evidence-state flow.
- **Fields:** `evidence_class`, `receipt_count`,
  `gate_evaluation.decision`, `decision_status`; module/capability labels are
  release-source structure, not performance values.
- **Palette / non-color:** Blue compiler layer, open blue product layer, gold
  hatched evidence layer, and neutral dashed future state.  Headers, borders,
  hatches, arrow styles, and explicit state labels provide non-color
  distinction.
- **Paths:** `assets/figures/architecture_evidence_chain.svg` and
  `assets/figures/architecture_evidence_chain.png`.
- **QA:** Current historical path uses solid arrows; future rerun uses a
  dashed arrow and says “required, not yet represented”; unofficial-fork and
  no-upstream-merge caveats visible.

## Shared export QA

- Renderer fixes figure size, DPI, fonts, palette, SVG hash salt, and metadata.
- `scripts/render_figures.py --check` regenerates in a temporary directory,
  byte-compares the three SVGs, verifies frozen SHA256 seals for all six
  checked-in assets, and structurally validates the regenerated PNGs.
- PNG dimensions are 2240×1360.  SVG view boxes are at least 1000×600 points.
- SVGs contain no external image/font/script references; internal fragment
  references are allowed.
- SVG/PNG payloads are scanned for `/Users/`, `/home/`, `/workspace/`,
  `file://`, and private-key markers.
- Final PNGs must be inspected in a local image viewer for clipping,
  collisions, grayscale/non-color distinction, and caveat legibility.

## Final QA status

- **Automated SVG byte-for-byte regeneration:** PASS.
- **Checked-in SVG/PNG SHA256 seals:** PASS.
- **Cross-platform PNG signature, dimensions, privacy, and structural
  regeneration:** PASS. PNG raster bytes are not compared across operating
  systems because the native font/raster stack is platform-dependent.
- **Local visual inspection (2026-07-29):** PASS.  Titles, subtitles, legends,
  long row labels, direct values, and footnotes are not clipped.  The latency
  chart preserves a zero baseline; hatch patterns distinguish all three
  implementations.  The ratio chart visibly anchors parity at 1.0, shows the
  ±2% band, and attaches “not a speed win” directly to packed-10.  The
  architecture diagram keeps current historical-bound assets on solid arrows
  and the uncompleted public-tag rerun on a dashed arrow and dashed container.
