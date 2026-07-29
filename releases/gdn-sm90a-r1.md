# GDN SM90a / TIRx public evidence r1

This is an unofficial personal-fork research release. It is not an Apache TVM,
mlc-ai, inclusionAI, FLA, or NVIDIA release or endorsement, and no upstream
pull request was created as part of this publication.

The release binds one immutable report tag to:

- the [public TVM SM90a compiler snapshot](https://github.com/Aharrypotter/tvm/releases/tag/gdn-sm90a-compiler-r0);
- the [public TIRx GDN prefill kernel snapshot](https://github.com/Aharrypotter/tirx-kernels/releases/tag/gdn-sm90a-kernel-r0);
- the [corrected CuTe DSL GDN comparator snapshot](https://github.com/Aharrypotter/cuLA/releases/tag/gdn-sm90a-comparator-r1);
- the [exact FLA comparator commit](https://github.com/fla-org/flash-linear-attention/commit/d1ce07369d581813553f30a750af3b6b5f9af6a9);
- the preserved historical evidence bundle; and
- a separately sealed fresh public-tag H20 characterization bundle.

The measured boundary remains operator latency for the frozen six-row,
BF16/head-dimension-128 GDN prefill matrix on one NVIDIA H20 environment. It
does not establish performance for every Hopper GPU, shape, dtype, model, or
end-to-end workload. Exact measured values, process counts, packed-n10
escalation status, source identities, and limitations are machine-derived from
the attached evidence and release manifest.

Read the
[fresh public-tag performance characterization](https://github.com/Aharrypotter/gdn-sm90a-tirx-report/blob/gdn-sm90a-r1/reports/fresh-public-tag-performance.md)
for the six-row table, aggregate ratios, packed-n10 trigger, and the precise
boundary between fresh and historical evidence.

## Why r1

The immutable `gdn-sm90a-r0` report tag passed ordinary verification but was
not released: its tag-event checkout dereferenced the annotated tag object, so
the required tag-object-only asset gate correctly failed. The
[supersession record](https://github.com/Aharrypotter/gdn-sm90a-tirx-report/blob/gdn-sm90a-r1/releases/gdn-sm90a-r0-supersession.md)
binds the exact r0 tag object, commit, tree, and failed-before-release status.

This r1 correction restores and verifies the remote annotated tag object before
building. It does not change runtime source, evidence, or performance facts.

## Assets

- `gdn-sm90a-r1-source.tar` — normalized archive of every tracked file in the
  report tag.
- `gdn-sm90a-r1-public-evidence.tar` — historical and fresh public evidence
  plus offline verification code.
- `gdn-sm90a-r1-publication-content.zip` — materialized bilingual drafts,
  figures, reports, claims, and limitations.
- `gdn-sm90a-r1-release-manifest.json` — report tag, source, evidence, claim,
  release-note, and asset identities.
- `SHA256SUMS` — detached SHA-256 checksums for the three packages and release
  manifest.

Verify all downloaded assets before use:

```bash
shasum -a 256 -c SHA256SUMS
```

Published tags and evidence roots are immutable. Any correction that changes
source, evidence, or a technical claim will use a new release tag and retain
this release for audit.
