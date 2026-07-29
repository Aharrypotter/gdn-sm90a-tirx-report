# Limitations and non-claims

This document is part of the result.  The release should not be cited without
these boundaries.

## Evidence age

The published bundle is `HISTORICAL_EVIDENCE_BOUND`.  It is a verified
allowlisted derivation from an immutable 380-file release seal, but it is not
an independent rerun from the public Git tags.  A fresh public-tag 66-receipt
rerun is still required.

See
[`PUBLICATION.json`](../evidence/historical/gdn-sm90a-h20-20260728-v1/PUBLICATION.json)
and
[`results/release-decision.json`](../evidence/historical/gdn-sm90a-h20-20260728-v1/results/release-decision.json).

## Hardware and compiler scope

- Measurements cover one NVIDIA H20 execution environment and target
  `sm_90a`.
- No H100, H200, other Hopper, Blackwell, or multi-GPU result is claimed.
- The TVM fork implements the bounded SM90a mechanisms required by this
  operator.  It is not full TVM support for SM90, SM90a, or all Hopper
  instructions and layouts.
- The runtime requires compute capability 9.0, but measured compatibility and
  performance are only established for the frozen H20 environment.

## Operator scope

- Q, K, and V are BF16.
- Key and value head dimensions are fixed at 128.
- The public state is FP32 and V-first `[V, K]`.
- Only the documented MHA, GQA, and GVA head relationships are supported.
- The default path assumes valid, non-empty, strictly increasing packed
  boundaries; it deliberately avoids synchronously copying `cu_seqlens` to
  the host.
- Only GDN prefill is covered.  Decode, backward, training, quantized modes,
  and end-to-end model integration are outside this release.

## Schedule scope

Only four exact metadata keys select specialized routes.  All valid near
misses use the general TIRx pipeline.  Performance measured on an allowlisted
row must not be generalized to arbitrary lengths, sequence-count
distributions, optional-gate combinations, or state modes.

## Performance scope

The performance claim is limited to the frozen public-call rows.  One
packed-10 comparison remains inside the preregistered noise band and is not a
speed win.  The complete generated table is
[`reports/historical-performance.md`](../reports/historical-performance.md).

This does not establish a universal TIRx or CuTeDSL ordering.  Any reciprocal
speedup statement would need to name the exact aggregate, ratio direction,
derivation, and workload.  This is not end-to-end model latency or throughput.

The canonical numerical evidence is in
[`results/performance.json`](../evidence/historical/gdn-sm90a-h20-20260728-v1/results/performance.json)
and
[`receipts/timing.jsonl`](../evidence/historical/gdn-sm90a-h20-20260728-v1/receipts/timing.jsonl).

## Comparator scope

The CuTeDSL comparator is specifically:

- commit `88737e9d906cf313995a092624656a89d74dd65e`;
- `cula.gdn.prefill.chunk_gated_delta_rule`;
- backend `sm90_cutedsl_gdn`;
- CuTe DSL 4.5.1;
- public reference tag
  [`gdn-sm90a-comparator-r1`](https://github.com/Aharrypotter/cuLA/tree/gdn-sm90a-comparator-r1).

The earlier `gdn2-sm90a-comparator-r0` tag is not the historical GDN
comparator and is not evidence for these numbers.

FLA is bound to exact upstream commit
[`d1ce07369d581813553f30a750af3b6b5f9af6a9`](https://github.com/fla-org/flash-linear-attention/commit/d1ce07369d581813553f30a750af3b6b5f9af6a9)
and `fla.ops.gated_delta_rule.chunk`.

## Validation scope

- CPU references prove semantics, not GPU execution.
- GPU correctness is bounded by the frozen matrix and tolerance policy.
- The host-sync audit covers dispatcher-visible operations in the traced
  workload; it is not a complete custom-extension trace.
- Codegen/resource results cover six cases and 11 stages, not every valid
  specialization.
- Safety and performance evidence belong to exact source, environment,
  workload, and measurement identities.

## Publication status

The TVM, tirx-kernels, and cuLA artifacts are unofficial personal-fork
snapshots.  Release commits and annotated tags are unsigned.  No upstream pull
request was created, and no upstream project has merged or endorsed this
work.
