# GDN SM90a / TIRx public evidence r2

This r2 release is a publication-guidance correction. It does not change
runtime source, either evidence bundle, any measured latency or ratio, or any
source coordinate.

The release remains an unofficial personal-fork research artifact. It is not
an Apache TVM, mlc-ai, inclusionAI, FLA, or NVIDIA release or endorsement, and
no upstream pull request was created as part of this publication.

## Why r2

The immutable r1 release passed tag CI and public asset audit, and its evidence
and performance facts remain valid. Its publication-content ZIP nevertheless
retained obsolete pre-rerun guidance in:

- `releases/platform-checklist.md`;
- `releases/publishing-order.md`;
- `releases/rollback-and-corrections.md`;
- `assets/figures/architecture_evidence_chain.{png,svg}`;
- `assets/figures/chart-map.md`; and
- the scope wording of claim C14.

That guidance conflicted with the completed, separately sealed 66-receipt H20
`CHARACTERIZATION` in the same package. The exact r1 release and five asset
hashes are preserved in the
[machine supersession record](https://github.com/Aharrypotter/gdn-sm90a-tirx-report/blob/gdn-sm90a-r2/releases/gdn-sm90a-r1-supersession.json)
and its
[human-readable summary](https://github.com/Aharrypotter/gdn-sm90a-tirx-report/blob/gdn-sm90a-r2/releases/gdn-sm90a-r1-supersession.md).
The r1 publication-content ZIP has SHA-256
`0316cfb489c78520bd84815668f4e715d281d8d6ce5bb670d597d680609249f2`.
It remains public and unmodified for audit, but should not be used for new
platform posts.

r2 updates the guidance and architecture diagram, narrows C14 to the
historical package, and adds a package-wide machine gate that rejects obsolete
pre-rerun state wording before asset construction and again during independent
asset verification. The historical latency and ratio figure bytes remain
unchanged.

## Immutable source and evidence boundary

This release binds one annotated report tag to:

- the [public TVM SM90a compiler snapshot](https://github.com/Aharrypotter/tvm/releases/tag/gdn-sm90a-compiler-r0);
- the [public TIRx GDN prefill kernel snapshot](https://github.com/Aharrypotter/tirx-kernels/releases/tag/gdn-sm90a-kernel-r0);
- the [corrected CuTe DSL GDN comparator snapshot](https://github.com/Aharrypotter/cuLA/releases/tag/gdn-sm90a-comparator-r1);
- the [exact FLA comparator commit](https://github.com/fla-org/flash-linear-attention/commit/d1ce07369d581813553f30a750af3b6b5f9af6a9);
- the preserved historical `HISTORICAL_EVIDENCE_BOUND` bundle; and
- the separately sealed fresh public-tag H20 `CHARACTERIZATION` bundle.

The measured boundary remains operator latency for the frozen six-row,
BF16/head-dimension-128 GDN prefill matrix on one NVIDIA H20 environment. It
does not establish performance for every Hopper GPU, shape, dtype, model, or
end-to-end workload. Historical and fresh aggregates remain distinct.
Historical host-sync, sanitizer, and full codegen/resource gates are not
promoted into the fresh characterization.

Read the
[fresh public-tag performance characterization](https://github.com/Aharrypotter/gdn-sm90a-tirx-report/blob/gdn-sm90a-r2/reports/fresh-public-tag-performance.md)
for the six-row table, aggregate ratios, packed-n10 trigger, and precise
evidence boundary.

## Assets

- `gdn-sm90a-r2-source.tar` — normalized archive of every tracked file in the
  report tag.
- `gdn-sm90a-r2-public-evidence.tar` — historical and fresh public evidence
  plus offline verification code.
- `gdn-sm90a-r2-publication-content.zip` — corrected, materialized bilingual
  drafts, figures, reports, claims, limitations, and publication guidance.
- `gdn-sm90a-r2-release-manifest.json` — report tag, source, evidence, claim,
  release-note, and asset identities.
- `SHA256SUMS` — detached SHA-256 checksums for the three packages and release
  manifest.

Verify all downloaded assets before use:

```bash
shasum -a 256 -c SHA256SUMS
```

Published tags and evidence roots are immutable. Any later correction that
changes source, evidence, or a technical claim will use another release tag
and retain both r1 and r2 for audit.
