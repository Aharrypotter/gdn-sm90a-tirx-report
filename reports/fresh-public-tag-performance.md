# Fresh public-tag H20 performance characterization

This additive report is rendered from the separately sealed fresh evidence bundle. It does not replace or mutate the historical performance report.

## Evidence identity

- Evidence root: [`evidence/fresh/gdn-sm90a-public-tags-h20-20260729-v1`](../evidence/fresh/gdn-sm90a-public-tags-h20-20260729-v1/)
- Evidence kind: `fresh-public-tag-h20-rerun`
- Claim scope: `fresh public-tag H20 six-row characterization`
- Decision status: `CHARACTERIZATION`
- Environment: `NVIDIA H20-3e`, target `sm_90a`
- Bundle-derived timing receipts: 66
- Manifest SHA-256: `d20b748ac54a3e14786d957353e3f85bee2566ceb913adc8679b8e660d2f282d`

## Six-row timing table

Ratio direction: TIRx latency / comparator latency; lower is faster. Latencies are the median of per-process averages in milliseconds.

| Row | TIRx ms | CuTeDSL ms | FLA ms | TIRx/CuTeDSL | TIRx/FLA | Processes per implementation |
|---|---:|---:|---:|---:|---:|---:|
| `single-t512-h8-mha-zero` | 0.102550 | 0.121454 | 0.373282 | 0.844350 | 0.274724 | 3 |
| `single-t1024-h8-mha-state` | 0.157430 | 0.174915 | 0.389517 | 0.900037 | 0.404167 | 3 |
| `single-t1024-h8-hv16-gva-state` | 0.143251 | 0.177971 | 0.394927 | 0.804915 | 0.362729 | 3 |
| `single-t4096-h16-mha-zero` | 0.263054 | 0.451661 | 0.442887 | 0.582414 | 0.593953 | 3 |
| `packed-n10-t4096-h8-mha-state` | 0.173384 | 0.170712 | 0.392756 | 1.015654 | 0.441455 | 7 |
| `packed-n20-t8192-h8-hv16-gva-state` | 0.520107 | 0.555996 | 0.629735 | 0.935451 | 0.825913 | 3 |

## Aggregate and packed-n10 trigger

- Six-row geometric-mean TIRx/CuTeDSL ratio: `0.834812`.
- Six-row geometric-mean TIRx/FLA ratio: `0.453700`.
- packed-n10 base-three-process TIRx/CuTeDSL trigger ratio: `1.017445`.
- packed-n10 escalation required: `true`; final processes per implementation: `7`.

The packed-n10 trigger ratio is intentionally the preregistered base-three-process value. The table reports the final row ratio after the triggered 7-process measurement.

## Evidence boundary

This bundle verifies exact public source tags/commits, fresh-process launch identity, physical H20 binding, receipt-level correctness, and the six-row public-call timing characterization. Its decision is `CHARACTERIZATION`, not a universal performance or upstream-release claim.

This fresh run does **not** reproduce the historical host-sync audit, Compute Sanitizer gates, or full codegen/resource reseals. Those remain historical-only evidence and are not promoted into this fresh bundle.

The TVM, tirx-kernels, and cuLA sources are unofficial fork artifacts. No upstream merge, endorsement, or official release is claimed.
