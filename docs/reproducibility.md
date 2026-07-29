# Reproducibility

Reproducibility has three distinct levels:

1. **Source retrieval**: obtain the exact public commits and tags.
2. **Historical re-derivation**: verify the sanitized bundle and recompute its
   timing summaries from raw samples.
3. **Fresh execution**: build and run the exact public sources on H20 under the
   frozen contract.

Levels 1 and 2 are available now.  Level 3 is required before the evidence
status can be promoted beyond `HISTORICAL_EVIDENCE_BOUND`; no fresh public-tag
execution bundle is currently claimed.

## 1. Retrieve exact sources

```bash
git clone https://github.com/Aharrypotter/tvm.git tvm-sm90a
git -C tvm-sm90a checkout --detach acb1312de80b39340e09b0aaad818ff029e745d6
git -C tvm-sm90a submodule update --init --recursive

git clone https://github.com/Aharrypotter/tirx-kernels.git tirx-kernels-sm90a
git -C tirx-kernels-sm90a checkout --detach 12ce3721f7c62c5fbd911103ae373de689e58385

git clone https://github.com/Aharrypotter/cuLA.git cula-gdn-comparator
git -C cula-gdn-comparator checkout --detach 88737e9d906cf313995a092624656a89d74dd65e

git clone https://github.com/fla-org/flash-linear-attention.git fla
git -C fla checkout --detach d1ce07369d581813553f30a750af3b6b5f9af6a9
```

The expected public tag objects and peeled commits are:

| Repository tag | Tag object | Peeled commit |
|---|---|---|
| `gdn-sm90a-compiler-r0` | `18e2172e54aefcba7e11f3e62fa8bfa137b480d4` | `acb1312de80b39340e09b0aaad818ff029e745d6` |
| `gdn-sm90a-kernel-r0` | `f233dcbfc314415b9af496e3fd855554d81d662c` | `12ce3721f7c62c5fbd911103ae373de689e58385` |
| `gdn-sm90a-comparator-r1` | `0e2c50a4f39b58811e234466682a62f8926998c4` | `88737e9d906cf313995a092624656a89d74dd65e` |

Inspect them without trusting a moving branch:

```bash
git -C tvm-sm90a rev-parse gdn-sm90a-compiler-r0^{tag}
git -C tvm-sm90a rev-parse gdn-sm90a-compiler-r0^{}
git -C tirx-kernels-sm90a rev-parse gdn-sm90a-kernel-r0^{tag}
git -C tirx-kernels-sm90a rev-parse gdn-sm90a-kernel-r0^{}
git -C cula-gdn-comparator rev-parse gdn-sm90a-comparator-r1^{tag}
git -C cula-gdn-comparator rev-parse gdn-sm90a-comparator-r1^{}
```

The kernel runtime is commit
[`90c9c62…`](https://github.com/Aharrypotter/tirx-kernels/commit/90c9c62c84ecc452dd86602f0ea49a625845045c);
the release tag peels to the subsequent documentation commit
[`12ce372…`](https://github.com/Aharrypotter/tirx-kernels/commit/12ce3721f7c62c5fbd911103ae373de689e58385).
The runtime tree is unchanged by that documentation commit.

## 2. Verify historical evidence

From this repository root:

```bash
python3 scripts/verify_public_evidence.py \
  --bundle evidence/historical/gdn-sm90a-h20-20260728-v1
```

The verifier independently checks:

- the public manifest seal, membership, file sizes, and SHA-256 values;
- exactly 66 unique `(row, implementation, process)` identities;
- exactly 100 raw samples per receipt;
- per-process averages, row medians, ratios, and six-row geometric means;
- correctness status for every receipt;
- forbidden private fields, paths, identifiers, and raw-artifact suffixes;
- the corrected CuTeDSL comparator mapping;
- preservation of `HISTORICAL_EVIDENCE_BOUND`.

Equivalent repository command:

```bash
make verify-historical-evidence
```

## 3. Fresh execution contract

A valid fresh public-tag rerun must satisfy all of the following before it can
be published:

- build the exact TVM compiler, tirx-kernels release, corrected CuTeDSL GDN
  comparator, and FLA commit listed above;
- target `sm_90a` on a physically verified NVIDIA H20;
- use CuTe DSL 4.5.1 and record the complete software environment;
- execute the public TIRx dispatch with backend
  `tirx.gdn.sm90a.wgmma.product-dispatch.packed.v3` and no external fallback;
- rerun the same six rows, seeds, scale, state modes, and sequence lengths from
  [`contracts/benchmark.json`](../evidence/historical/gdn-sm90a-h20-20260728-v1/contracts/benchmark.json);
- use 20 warmups, 100 timed iterations, independent processes, unique empty
  caches, the 5% quiet threshold, and the frozen 3/7 process counts;
- capture all 66 raw timing receipts and per-receipt correctness;
- rerun public GPU semantics, exact route/near-miss, safety, and
  source-bound codegen/resource gates;
- seal source, receipts, summaries, and artifacts in a new evidence root.

The fresh package must be additive, for example under `evidence/fresh/`.  It
must never overwrite, mutate, or relabel
[`evidence/historical/gdn-sm90a-h20-20260728-v1`](../evidence/historical/gdn-sm90a-h20-20260728-v1/).

## Environment boundary

The historical environment reports PyTorch
`2.11.0a0+eb65b36914.nv26.02`, PyTorch CUDA 13.1, and CuTe DSL 4.5.1.  Exact
environment metadata is in
[`metadata/environment.json`](../evidence/historical/gdn-sm90a-h20-20260728-v1/metadata/environment.json).

Source checkout alone does not reproduce performance.  Results also belong to
the exact build, target, device, workload, timing, cache, correctness, and
contention contracts.
