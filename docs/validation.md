# Validation

Validation is split into semantic, public-path, safety, codegen/resource, and
performance gates.  Passing one layer is not evidence for another.

## Status vocabulary

Two uses of “fresh” must not be conflated:

- The historical A46-S3 release ran a fresh full canonical timing matrix for
  its sealed final source.  It did not inherit its full performance conclusion
  from an older codegen baseline.
- The public tags were created after that seal.  The current public bundle is
  therefore `HISTORICAL_EVIDENCE_BOUND` and still requires a fresh 66-receipt
  rerun from the public tags.

The machine decision is
[`results/release-decision.json`](../evidence/historical/gdn-sm90a-h20-20260728-v1/results/release-decision.json).

## Historical gate summary

| Gate | Historical result | Evidence |
|---|---|---|
| Compiler tests | 92 passed, 88 skipped | [correctness summary](../evidence/historical/gdn-sm90a-h20-20260728-v1/results/correctness.json) |
| Public GPU semantics | 10 passed | [correctness summary](../evidence/historical/gdn-sm90a-h20-20260728-v1/results/correctness.json) |
| Public route tests | 15 passed | [correctness summary](../evidence/historical/gdn-sm90a-h20-20260728-v1/results/correctness.json) |
| Per-receipt correctness | all 66 passed | [correctness summary](../evidence/historical/gdn-sm90a-h20-20260728-v1/results/correctness.json) |
| Host-visible synchronization audit | PASS; 9 profiled public calls, zero violations | [safety summary](../evidence/historical/gdn-sm90a-h20-20260728-v1/results/safety.json) |
| Packed redzones | PASS; four guards, 19 adjacent boundaries, inputs immutable | [safety summary](../evidence/historical/gdn-sm90a-h20-20260728-v1/results/safety.json) |
| Compute Sanitizer | PASS; four memcheck cases, zero race hazards, zero sync errors | [safety summary](../evidence/historical/gdn-sm90a-h20-20260728-v1/results/safety.json) |
| Codegen/resources | PASS; six cases, 11 stages, zero spill and stack bytes | [codegen inventory](../evidence/historical/gdn-sm90a-h20-20260728-v1/results/codegen-resources.json) |
| Canonical timing | PASS; 66 isolated receipts | [performance result](../evidence/historical/gdn-sm90a-h20-20260728-v1/results/performance.json) |
| Artifact seal | PASS locally and remotely; 380 files | [controls](../evidence/historical/gdn-sm90a-h20-20260728-v1/results/controls.json) |

## Correctness policy

All three implementations used tensors derived from the same seeded inputs.
CuTeDSL outputs and states were the frozen per-receipt oracle.  The policy
applied allclose, maximum-absolute-error, and relative-RMS bounds to outputs
and, where requested, final states.

The six-row result contains 22 receipts per implementation:

- 3 processes for each implementation on five rows;
- 7 processes for each implementation on packed-10 after its preregistered
  escalation;
- 66 total receipts.

Every receipt passed correctness.  Exact per-row maxima are retained in
[`results/correctness.json`](../evidence/historical/gdn-sm90a-h20-20260728-v1/results/correctness.json);
the tolerance contract is
[`contracts/benchmark.json`](../evidence/historical/gdn-sm90a-h20-20260728-v1/contracts/benchmark.json).

## Safety boundary

The host-sync audit covers dispatcher-visible operations for the traced public
workload.  It found device-to-device copies but no dispatcher-visible host
synchronization violations.  It does not inspect arbitrary custom-extension
internals; that would require Nsight Systems or CUPTI.

The authoritative sanitizer attempt is `sanitizer-r1`.  An earlier control
attempt was superseded because recursive child-process tracking remained
attached after a shell child exited; no kernel error was observed, and product
source did not change between attempts.  The supersession is preserved rather
than erased in
[`results/controls.json`](../evidence/historical/gdn-sm90a-h20-20260728-v1/results/controls.json).

## Codegen and resources

The formal inventory covers all six benchmark cases and 11 generated stages.
Across that bounded set:

- maximum registers per thread: 238;
- maximum dynamic/shared-memory inventory value: 182,288 bytes;
- maximum spill-load bytes: 0;
- maximum spill-store bytes: 0;
- maximum stack-frame bytes: 0;
- maximum SASS local-load/store instruction count: 0.

Two accepted optimized routes had byte-identical generated CUDA comparisons,
but the full six-case historical cubin/SASS baseline was incomplete.
Consequently timing inheritance was ineligible and the historical campaign ran
the full canonical matrix.  This distinction is recorded in
[`results/codegen-resources.json`](../evidence/historical/gdn-sm90a-h20-20260728-v1/results/codegen-resources.json).

## What is not yet validated

The current repository does not yet contain a fresh public-tag execution
bundle.  Until that rerun is published, historical results must not be
described as independently reproduced from:

- TVM `gdn-sm90a-compiler-r0`;
- tirx-kernels `gdn-sm90a-kernel-r0`;
- corrected CuTeDSL comparator `gdn-sm90a-comparator-r1`;
- FLA commit `d1ce07369d581813553f30a750af3b6b5f9af6a9`.

The historical bundle remains valid evidence for the sealed run, and its
source mapping is explicit, but its provenance class must not be promoted.
