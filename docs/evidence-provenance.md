# Evidence provenance

The historical public evidence is a deterministic, privacy-safe projection of
an immutable release seal. It is not a copied-and-regex-scrubbed log
directory. A separately sealed fresh evidence root records the exact
public-tag H20 timing characterization without changing the historical bundle.

## Provenance chain

```text
immutable private release seal (380 files)
    ├── exact source bundle
    ├── release decision
    └── artifact manifest
            ↓ explicit field allowlist
public historical bundle
    ├── contracts and source locks
    ├── 66 numerical timing receipts
    ├── correctness/safety/codegen summaries
    ├── field-level provenance map
    └── manifest + manifest seal
```

The original seal identities are:

| Object | SHA-256 |
|---|---|
| source bundle | `dd161b457810ad6fea91cc305c9f27b40f627ad4acf3cb426f5a798d5326531c` |
| release decision | `39670f9e73282f5c336c70ee70b6d79791cc5b5529fd8c529defa7339722bcad` |
| artifact manifest | `f090d0ce736b8e162307b9e2a8838cb4e1559dfce1c17e5b8102e38b23dc7c9a` |
| artifact membership | 380 files |

They are recorded in
[`PUBLICATION.json`](../evidence/historical/gdn-sm90a-h20-20260728-v1/PUBLICATION.json).

## Public bundle seal

The public manifest declares 18 payload files, excluding `manifest.json` and
its detached text seal.  The manifest itself is sealed as:

```text
54e733e9edc053afea9a41f720ed6688ff5106b0c346af2f002f37045f3b1b50  manifest.json
```

See
[`manifest.json`](../evidence/historical/gdn-sm90a-h20-20260728-v1/manifest.json)
and
[`MANIFEST.sha256`](../evidence/historical/gdn-sm90a-h20-20260728-v1/MANIFEST.sha256).

Every output field is mapped to its allowlisted historical source in
[`provenance/field-map.json`](../evidence/historical/gdn-sm90a-h20-20260728-v1/provenance/field-map.json).
Raw logs, job scripts, filesystem roots, cache paths, executable paths, host
and container identifiers, SSH aliases, GPU UUIDs, process identifiers,
profiler traces, generated CUDA/cubin/SASS, and the private reviewer report
are excluded.  The policy is documented in
[`REDACTION.md`](../evidence/historical/gdn-sm90a-h20-20260728-v1/REDACTION.md).

## Source qualification

The public source lock makes different strengths of mapping explicit:

- TIRx compiler source is byte-mapped to the exact runtime delta at
  [`acb1312…`](https://github.com/Aharrypotter/tvm/commit/acb1312de80b39340e09b0aaad818ff029e745d6).
- TIRx kernel source is byte-mapped to the exact runtime delta at
  [`90c9c62…`](https://github.com/Aharrypotter/tirx-kernels/commit/90c9c62c84ecc452dd86602f0ea49a625845045c).
- FLA is bound to declared upstream commit
  [`d1ce073…`](https://github.com/fla-org/flash-linear-attention/commit/d1ce07369d581813553f30a750af3b6b5f9af6a9)
  and operator `fla.ops.gated_delta_rule.chunk`.
- CuTeDSL is bound by exact commit, entrypoint, backend identity, and DSL
  version.

The full machine-readable mapping and per-repository tree/tag objects are in
[`metadata/source-lock.json`](../evidence/historical/gdn-sm90a-h20-20260728-v1/metadata/source-lock.json).

## Corrected CuTeDSL comparator

All 22 historical CuTeDSL receipts name:

- commit `88737e9d906cf313995a092624656a89d74dd65e`;
- entrypoint `cula.gdn.prefill.chunk_gated_delta_rule`;
- backend `sm90_cutedsl_gdn`;
- CuTe DSL 4.5.1.

The corrected public tag
[`gdn-sm90a-comparator-r1`](https://github.com/Aharrypotter/cuLA/tree/gdn-sm90a-comparator-r1)
points directly to that exact commit.  The callable is visible at the
immutable
[`cula/gdn/prefill.py`](https://github.com/Aharrypotter/cuLA/blob/88737e9d906cf313995a092624656a89d74dd65e/cula/gdn/prefill.py).

The earlier `gdn2-sm90a-comparator-r0` tag points to a different GDN2 source
snapshot.  It is retained immutably but is explicitly
`NOT_USED_BY_HISTORICAL_GDN_RECEIPTS`; it must not be cited as comparator
source for this report.

## Historical versus fresh evidence

The historical sealed run reached `RELEASE_READY` and selected
`USE_FRESH_CANONICAL_TIMING` within that campaign.  Publication of new source
tags does not retroactively turn it into a public-tag rerun.

Accordingly:

- historical package status: `HISTORICAL_EVIDENCE_BOUND`;
- fresh public-tag bundle status: `PASS`, decision `CHARACTERIZATION`;
- fresh evidence:
  [`gdn-sm90a-public-tags-h20-20260729-v1`](../evidence/fresh/gdn-sm90a-public-tags-h20-20260729-v1/);
- fresh receipt count: derived from the sealed bundle and its escalation
  policy, never copied from the historical result;
- fresh scope: exact public source/build/runtime identity, physical H20
  binding, fresh-process launches, receipt correctness, and six-row timing;
- excluded from the fresh scope: the historical host-sync audit, Compute
  Sanitizer gates, and full codegen/resource reseals;
- immutable rule: never edit or relabel the historical bundle to represent
  fresh evidence.

## Local verification

From the repository root:

```bash
python3 scripts/verify_public_evidence.py \
  --bundle evidence/historical/gdn-sm90a-h20-20260728-v1
```

The verifier checks manifest membership and hashes, receipt uniqueness and
sample counts, summary and ratio re-derivation, forbidden private fields and
patterns, comparator qualification, and the evidence-status boundary.

Verify the independently sealed fresh bundle with:

```bash
python3 -m reproduce.fresh_evidence.verify \
  --bundle evidence/fresh/gdn-sm90a-public-tags-h20-20260729-v1
```
