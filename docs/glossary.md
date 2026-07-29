# Glossary

**A46-S3**

The internal historical release checkpoint from which the sanitized public
evidence was derived.  It is an evidence identity, not a public source tag.

**BF16**

Bfloat16.  The public Q/K/V and output dtype in this release.

**CTA**

Cooperative thread array, commonly called a CUDA thread block.

**Chunk**

A 64-token semantic and scheduling unit in the released GDN implementation.

**Codegen/resource gate**

A source-bound inspection of generated CUDA/device artifacts and launch
resources.  It is separate from source review and measured performance.

**Comparator**

The exact implementation against which TIRx latency is divided.  Here the
comparators are corrected cuLA GDN commit `88737e9…` and FLA commit
`d1ce073…`.

**CuTeDSL**

NVIDIA CUTLASS DSL, used by the cuLA GDN comparator.  Historical receipts bind
version 4.5.1, backend `sm90_cutedsl_gdn`, and entrypoint
`cula.gdn.prefill.chunk_gated_delta_rule`.

**Fallback**

Execution delegated to another implementation.  The TIRx product contract has
no external GDN fallback; non-allowlisted valid shapes use the general TIRx
pipeline.

**Fresh canonical timing**

A complete timing matrix executed for the exact final source under the frozen
benchmark contract.  The historical campaign used fresh canonical timing
within its seal. The separately sealed public-tag characterization is a
different evidence identity.

**Fresh public-tag rerun**

A new execution built from the published TVM, tirx-kernels, and corrected
cuLA tags plus the exact FLA commit. The current instance is stored under
[`evidence/fresh/gdn-sm90a-public-tags-h20-20260729-v1`](../evidence/fresh/gdn-sm90a-public-tags-h20-20260729-v1/)
with decision `CHARACTERIZATION`, separately from the historical bundle.

**GDN**

Gated Delta Net, the recurrent operator implemented here.

**Geometric mean**

The multiplicative aggregate of the frozen per-row latency ratios.  It gives
equal multiplicative weight to each row.

**GQA**

Grouped-query head mapping: `Hq` is a multiple of `Hk = Hv`.

**GVA**

Grouped-value head mapping: `Hv` is a multiple of `Hq = Hk`.

**H20**

The NVIDIA Hopper accelerator used by the frozen measurements.  Results are
not generalized to every Hopper product.

**Historical evidence bound**

The publication class of the original bundle. It means that bundle is a
verified derivation from the immutable historical seal and is not represented
as a rerun from the new public tags.

**Ho**

The output-head count, `max(Hq, Hv)`.

**mbarrier**

Hopper memory barrier primitives used to coordinate asynchronous producers and
consumers.

**MHA**

Multi-head mapping with `Hq = Hk = Hv`.

**Noise band**

The preregistered interpretation interval used by this benchmark.  Being
inside it is not a speed win.  Packed-10 is classified this way; the exact
source-derived value is in the generated performance report.

**Packed variable length**

Multiple independent sequences stored in one token-major tensor and delimited
by CUDA-resident `cu_seqlens`.

**Pipeline route**

The general two-stage TIRx schedule: 256-thread chunk preparation followed by
a 128-thread recurrent scan/output kernel.

**Public path**

Invocation through the documented wrapper, including validation and product
dispatch.  Directly calling an internal PrimFunc does not prove public-path
behavior.

**Receipt**

One implementation/row/process result containing 100 raw timing samples,
summary statistics, source/backend identity, and correctness status.

**Register replay**

The single-kernel 512-thread specialized schedule for the exact packed-10
allowlist key.

**Release reseal**

The process of binding final source to correctness, safety, codegen,
performance, and immutable artifact evidence after productization.

**Ratio**

`TIRx latency / comparator latency`.  Lower is faster; a ratio below one favors
TIRx.

**RS / SS**

WGMMA register/shared and shared/shared operand forms.

**SM90 / SM90a**

SM90 denotes Hopper compute capability 9.0.  `sm_90a` is the
architecture-specific target required by the released WGMMA path.  Generic
SM90 and SM90a are not interchangeable in compiler dispatch.

**Source lock**

Machine-readable mapping from a result to exact repositories, bases, commits,
trees, tags, backend identities, and source-manifest digests.

**Tail-predecessor route**

The specialized two-stage schedule that replays at most one predecessor
chunk, with two co-resident value warpgroups and consumer-relative Q-barrier
phases.

**TIRx**

The tile-oriented TIR dialect and compiler path used to express and lower the
GDN kernels.

**TMA**

Tensor Memory Accelerator, used for asynchronous global/shared tensor
transfers and TensorMap descriptors on Hopper.

**Upstream endorsement**

Maintainer acceptance or merge by an upstream project.  None is claimed:
these are unofficial personal-fork artifacts and no upstream PR was created.

**Warpgroup**

Four warps, or 128 CUDA threads, cooperating on one WGMMA operation.

**WGMMA**

Warpgroup matrix multiply-accumulate instructions used by the bounded SM90a
compiler and kernel paths.
