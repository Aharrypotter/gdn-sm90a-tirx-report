# X thread — English

## 1

I published a TIRx-on-Hopper study: a bounded SM90a compiler slice, a
productized GDN prefill operator, and evidence that keeps source, semantics,
codegen, safety, and timing separate.

This is an unofficial personal-fork experiment for TIRx on Hopper SM90a.

## 2

The compiler work was more than selecting `sm_90a`: WGMMA SS/RS lowering,
warpgroup layouts, swizzled and K-major descriptors, explicit-stride TMA
TensorMaps, bounded ragged-tail copies, and the host TensorMap ABI all had to
agree.

## 3

The compiler fork supplies the SM90a WGMMA/TMA-oriented capabilities required by this operator; it is not a complete-SM90 support claim.

Compiler scope:
https://github.com/Aharrypotter/tvm/tree/gdn-sm90a-compiler-r0

## 4

For a recurrent operator, schedule-first tuning is risky. I froze the token
recurrence, FP32 V-first state, MHA/GQA/GVA mapping, precision-visible inverse
ladder, and packed-boundary behavior before treating performance as a product
question.

## 5

The TIRx fork publishes a productized GDN prefill operator with exact optimized routes and a documented pipeline route.

## 6

The wrapper dispatches from host-visible metadata only. Exact allowlisted keys
get a specialized route; every valid near miss uses the general TIRx pipeline.
No route calls an external GDN fallback.

## 7

Three product schedules:

- general prepare + recurrent scan/output pipeline;
- one exact fused short register-replay path;
- exact tail-predecessor paths with bounded replay and co-resident value
  warpgroups.

## 8

Validation was layered: CPU semantics, public GPU behavior, allowlist and
near-miss dispatch, stream liveness, packed redzones, host-sync audit, Compute
Sanitizer, source-bound WGMMA/TMA codegen/resources, then isolated public-call
timing.

## 9

The historical benchmark covers 6 exact BF16/D128 rows on one NVIDIA H20 environment.

Ratio direction: tirx_latency / comparator_latency; lower is faster.

## 10

This is public-call operator latency, not end-to-end model throughput. The
scope must not be generalized beyond the frozen matrix.

## 11

TIRx has lower latency than CuTeDSL on 5 of 6 frozen rows.

## 12

packed-n10 has an exact TIRx/CuTeDSL ratio of 1.0146, or 1.46% higher latency, inside the preregistered 2.0% noise band; it is not a speed win.

The packed exception belongs in the headline, not a footnote.

## 13

TIRx has lower latency than FLA on 6 of 6 frozen rows.

Generated table:
https://github.com/Aharrypotter/gdn-sm90a-tirx-report

## 14

Comparator provenance required a correction:

The CuTeDSL comparator is gdn-sm90a-comparator-r1 at 88737e9d906cf313995a092624656a89d74dd65e, using cula.gdn.prefill.chunk_gated_delta_rule; gdn2-sm90a-comparator-r0 is excluded from this report.

## 15

Corrected CuTeDSL source:
https://github.com/Aharrypotter/cuLA/tree/gdn-sm90a-comparator-r1

The public package status is HISTORICAL_EVIDENCE_BOUND; unofficial-personal-fork is true and upstream-merge is false.

## 16

Exact public tags, H20 6 rows: 66 bundle-derived receipts; CHARACTERIZATION; TIRx/CuTeDSL 0.834812, TIRx/FLA 0.453700, packed-n10 base-three-process 1.017445. Does not reproduce historical host-sync/sanitizer/full codegen/resource reseals. Unofficial forks; no upstream merge.

## 17

Fresh source-derived report:
https://github.com/Aharrypotter/gdn-sm90a-tirx-report/blob/gdn-sm90a-r0/reports/fresh-public-tag-performance.md
