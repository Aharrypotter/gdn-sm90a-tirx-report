# Benchmark methodology

This document describes the frozen historical benchmark contract.  The
current result is `HISTORICAL_EVIDENCE_BOUND`; it is not a fresh run from the
public tags.

## Implementations

| Name | Exact historical identity |
|---|---|
| TIRx | backend `tirx.gdn.sm90a.wgmma.product-dispatch.packed.v3`; sealed compiler+kernel source bundle byte-mapped to the public TVM and tirx-kernels runtime deltas |
| CuTeDSL | commit [`88737e9…`](https://github.com/Aharrypotter/cuLA/commit/88737e9d906cf313995a092624656a89d74dd65e), entrypoint `cula.gdn.prefill.chunk_gated_delta_rule`, backend `sm90_cutedsl_gdn`, CuTe DSL 4.5.1 |
| FLA | upstream commit [`d1ce073…`](https://github.com/fla-org/flash-linear-attention/commit/d1ce07369d581813553f30a750af3b6b5f9af6a9), operator `fla.ops.gated_delta_rule.chunk` |

The corrected CuTeDSL public tag is
[`gdn-sm90a-comparator-r1`](https://github.com/Aharrypotter/cuLA/tree/gdn-sm90a-comparator-r1).
The earlier GDN2 comparator tag is not evidence for these receipts.

## Hardware and software scope

- accelerator: one NVIDIA H20 execution environment;
- target: CUDA `sm_90a`;
- physical-device binding: verified, with private identifiers omitted;
- PyTorch: `2.11.0a0+eb65b36914.nv26.02`;
- PyTorch-reported CUDA version: 13.1;
- CuTe DSL packages: 4.5.1.

BF16 Q/K/V and head dimension 128 are frozen by the public tagged source
contract.  They were not explicit fields in the original historical benchmark
JSON, so the public evidence records that provenance rather than inventing
fields.

## Workload matrix

| Row | Sequence layout | Heads | State | Route |
|---|---|---|---|---|
| `single-t512-h8-mha-zero` | `[512]` | MHA 8/8/8 | zero, no final state | pipeline |
| `single-t1024-h8-mha-state` | `[1024]` | MHA 8/8/8 | initial + final | pipeline |
| `single-t1024-h8-hv16-gva-state` | `[1024]` | GVA 8/8/16 | initial + final | tail-predecessor |
| `single-t4096-h16-mha-zero` | `[4096]` | MHA 16/16/16 | zero, no final state | tail-predecessor |
| `packed-n10-t4096-h8-mha-state` | `6×410 + 4×409` | MHA 8/8/8 | initial + final | short register replay |
| `packed-n20-t8192-h8-hv16-gva-state` | `5000 + 18×170 + 132` | GVA 8/8/16 | initial + final | tail-predecessor |

Every row uses scale 0.73 and a frozen per-row seed.  Exact sequence lengths,
seeds, expected backends, and critical-row flags are in
[`contracts/benchmark.json`](../evidence/historical/gdn-sm90a-h20-20260728-v1/contracts/benchmark.json).

## Timing protocol

- wait for at most 120 seconds for GPU utilization at or below 5%;
- poll quiet state once per second;
- 20 warmup iterations;
- 100 timed iterations per receipt;
- one fresh cache per receipt;
- 3 independent processes per implementation and row by default;
- 7 processes for packed-10 after its preregistered escalation;
- a preregistered ±2% noise band.

This process matrix produces the receipt set sealed by the canonical result;
its source-derived receipt and cache counts are rendered in
[`reports/historical-performance.md`](../reports/historical-performance.md).
Each receipt retains its per-iteration timing samples and correctness result in
[`receipts/timing.jsonl`](../evidence/historical/gdn-sm90a-h20-20260728-v1/receipts/timing.jsonl).

The per-process average is computed from its timed samples.  The row latency is
the median of the independent process averages.

## Ratio and aggregate

For comparator `C`:

```text
row ratio = median TIRx latency / median C latency
```

Lower is faster.  The primary aggregate is the geometric mean of the frozen
row ratios.  A latency ratio must not be rewritten as a reciprocal “times
faster” claim without an explicit derivation and scope.

## Historical results

The human-readable table is generated, never transcribed:
[`reports/historical-performance.md`](../reports/historical-performance.md).
Its sole numerical source is
[`results/performance.json`](../evidence/historical/gdn-sm90a-h20-20260728-v1/results/performance.json);
`scripts/render_performance_markdown.py --check` rejects a stale table.

## Claim boundary

These are public-call kernel/operator latencies for the frozen BF16, D=128 GDN
prefill rows on one H20 environment.  They are not end-to-end model
throughput, not arbitrary-shape coverage, not every Hopper GPU, and not an
upstream comparison endorsed by TVM, tirx-kernels, cuLA, or FLA.
