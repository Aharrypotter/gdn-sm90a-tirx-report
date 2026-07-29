# Historical six-row performance

> Evidence class: `HISTORICAL_EVIDENCE_BOUND`. This is not yet an
> independent rerun from the public Git tags.

Timer: CUDA events around the final public callable. Each displayed
latency is the median of independent process-average latencies. Ratio
direction is TIRx latency divided by comparator latency; lower is faster.

| Row | TIRx (ms) | CuTeDSL (ms) | FLA (ms) | TIRx/CuTeDSL | TIRx/FLA | Processes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| single-512 MHA zero | 0.102857 | 0.122897 | 0.418859 | 0.8369 | 0.2456 | 3 |
| single-1024 MHA state | 0.157265 | 0.177869 | 0.440610 | 0.8842 | 0.3569 | 3 |
| single-1024 GVA state | 0.143440 | 0.176857 | 0.429780 | 0.8111 | 0.3338 | 3 |
| single-4096 MHA zero | 0.260955 | 0.453872 | 0.471597 | 0.5750 | 0.5533 | 3 |
| packed-10 MHA state | 0.174118 | 0.171608 | 0.438900 | 1.0146 | 0.3967 | 7 |
| packed-20 GVA state | 0.520388 | 0.556828 | 0.634935 | 0.9346 | 0.8196 | 3 |
| **six-row geometric mean** | — | — | — | **0.8301** | **0.4171** | — |

Five of 6 TIRx/CuTeDSL row ratios are below one. Packed-10 is 1.46% above CuTeDSL and remains inside the preregistered ±2% noise band; it is not a speed win.

Scope: one NVIDIA H20 environment, the frozen six-row GDN prefill
matrix, and public-call latency only. This is not end-to-end model
throughput and does not generalize to every Hopper GPU or shape.

Canonical machine-readable source: [`evidence/historical/gdn-sm90a-h20-20260728-v1/results/performance.json`](../evidence/historical/gdn-sm90a-h20-20260728-v1/results/performance.json).
