# Figure chart map

The latency and ratio figures render quantitative values only from
`evidence/historical/gdn-sm90a-h20-20260728-v1/results/performance.json`.
The architecture figure additionally reads the separately sealed fresh
`performance.json` and `publication.json` to show its completed public-tag H20
`CHARACTERIZATION`. The rendering surface is deterministic static SVG plus
sealed PNG. Historical and fresh evidence classes remain additive and their
aggregates are never merged.

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
  two separately sealed validation/release evidence classes connect without
  promoting or merging either class?
- **Supported takeaway:** The deliverable is a layered chain from SM90a
  compiler capability through a frozen GDN product contract and schedules to
  an immutable historical-bound evidence package. A second, additive path
  records the completed exact-public-tag H20 `CHARACTERIZATION`.
- **Family / variant:** Decomposition & Progression / layered architecture and
  evidence-state flow.
- **Fields:** `evidence_class`, `receipt_count`,
  `gate_evaluation.decision`, `decision_status`; fresh `status`,
  `decision_status`, `evidence_kind`, and `receipt_count`; module/capability
  labels are release-source structure, not performance values.
- **Palette / non-color:** Blue compiler layer, open blue product layer, gold
  hatched historical evidence layer, and a neutral dashed additive fresh
  state. Headers, borders, hatches, arrow styles, and explicit state labels
  provide non-color
  distinction.
- **Paths:** `assets/figures/architecture_evidence_chain.svg` and
  `assets/figures/architecture_evidence_chain.png`.
- **QA:** The historical path and completed fresh `CHARACTERIZATION` are both
  visible; the dashed container means “separate additive class,” not
  promotion. The 66-receipt counts, unofficial-fork boundary, and
  no-upstream-merge caveat are visible.

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
  architecture diagram keeps the immutable historical package on the solid
  path and the completed public-tag H20 `CHARACTERIZATION` in a separate
  dashed container; the two aggregates are not connected or merged.
