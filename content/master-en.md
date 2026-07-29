# TIRx on Hopper: From a bounded SM90a compiler slice to productized GDN prefill

<!-- TEMPLATE_ONLY
Publication source template.
Resolve every {{claim:Cxx:en}} token from contracts/claim-registry.json with
scripts/validate_claims.py --show-rendered. Never hand-edit a resolved number.
-->

{{claim:C01:en}}

The point of this project is not that one kernel happened to benchmark well.
The more interesting result is the full path from compiler capability to a
public operator: add the missing Hopper mechanisms at the compiler layer,
freeze the recurrent semantics independently of any schedule, build several
qualified TIRx schedules, route public calls without hidden fallback, and bind
the final source to correctness, safety, code generation, and reproducible
performance evidence.

This is also a deliberately bounded result. {{claim:C02:en}} The measured
operator contract is BF16 with head dimension 128, and the current evidence is
limited to a frozen matrix on one NVIDIA H20 environment. It is not a claim
about every Hopper GPU, every GDN shape, or end-to-end model throughput.

## Why this required compiler work

Writing a Hopper kernel in a DSL is not just a matter of selecting
`sm_90a`. The compiler must carry a collection of contracts all the way to
generated CUDA:

- fail-closed target qualification for the architecture-specific path;
- native WGMMA lowering for the exact shared/shared and register/shared
  fragments used by the operator;
- explicit warpgroup accumulator and register layouts;
- swizzle-aware shared-memory descriptors, including K-major forms;
- TMA TensorMap construction for explicit-stride global views;
- bounded global/shared copies for ragged tails;
- correctly aligned TensorMap storage in generated host code.

The public compiler tag implements the slice needed by this operator. It does
not attempt to cover arbitrary WGMMA shapes, every TMA mode, Blackwell
TCGEN05, or every possible SM90 workload. The exact boundary is documented in
[the compiler capability note](../docs/compiler-capability.md).

## Freeze semantics before tuning schedules

GDN is recurrent. A schedule can produce plausible tensors while still
changing the state orientation, placing a gate on the wrong side of a matrix,
rounding at a different stage, or letting a packed tail cross a sequence
boundary.

The public semantic contract therefore exists independently of the GPU
implementation. The state is FP32 and V-first. Q, K, and V are BF16. MHA,
GQA, and GVA head mappings are explicit. Each sequence is divided into
64-token chunks, and the precision-visible chunk algebra fixes:

1. the alpha prefix;
2. QK and KK transfer scaling;
3. row-side beta before the lower-triangular inverse;
4. the FP16-visible inverse ladder;
5. column-side beta when the inverse becomes a BF16 operand;
6. the order of prior-state output, state projection, corrected value,
   within-chunk output, and terminal-state update.

Ragged tail values are identities or zeros as appropriate, and no replay may
cross `cu_seqlens`. The details and literal token recurrence are in
[the GDN semantic contract](../docs/gdn-semantics.md).

## Three product schedules, one public API

{{claim:C03:en}}

The general route is a two-stage pipeline: chunk-parallel preparation followed
by the genuinely recurrent scan/output stage. It remains the safe product path
for every valid non-allowlisted specialization.

Two additional schedules are selected only by exact metadata:

- a fused short register-replay route for one qualified packed shape;
- a tail-predecessor route that bounds replay to one prior chunk and uses
  co-resident value warpgroups with consumer-relative barrier phases.

The optimized dispatch is an exact allowlist. A near miss, a different state
mode, or missing explicit gates returns to the documented pipeline route. The
wrapper chooses using host-visible tensor metadata only; it does not read
CUDA-resident sequence boundaries to select a schedule and it never calls
FLA, Triton, CuTeDSL, or a C++ GDN fallback.

See [schedules and product dispatch](../docs/schedules-and-dispatch.md) for the
full route map.

## Validation is a ladder, not one green check

The release evidence separates:

- CPU semantic and auxiliary-reference agreement;
- public GPU semantics and exact route/near-miss behavior;
- repeat and non-default-stream liveness;
- packed redzones and input immutability;
- dispatcher-visible host synchronization;
- Compute Sanitizer memcheck, racecheck, and synccheck;
- source-bound WGMMA/TMA codegen and resource inventory;
- isolated timing receipts through the final public callable.

This separation matters. Source inspection is not GPU correctness. Correct
output is not memory safety. A clean sanitizer run is not proof of native
WGMMA codegen. A fast internal PrimFunc is not evidence for public dispatch.

The historical release had to use a fresh full canonical timing matrix because
the complete device-artifact baseline was not eligible for timing
inheritance. The resulting validation summary is in
[validation](../docs/validation.md).

## What the historical performance evidence says

{{claim:C04:en}}

{{claim:C05:en}}

{{claim:C08:en}}

{{claim:C09:en}}

{{claim:C10:en}}

The wording above is generated from the canonical performance JSON. The
complete source-derived table, including row latencies and geometric means,
is [the generated historical performance report](../reports/historical-performance.md).
The timing region is the final public callable under the frozen benchmark
contract, not an end-to-end model benchmark.

## Comparator identity matters

{{claim:C13:en}}

That correction is important. The historical receipts identify the GDN
callable, backend, commit, and CuTe DSL version. An earlier GDN2 publication
tag points at different source and cannot be used to explain these GDN
measurements. The FLA comparator is likewise pinned to one upstream commit and
one callable.

The complete mapping is in [the evidence provenance note](../docs/evidence-provenance.md)
and the machine-readable [link map](../contracts/link-map.json).

## Evidence status and what comes next

{{claim:C14:en}}

The sanitized bundle is deterministic, privacy-safe, and bound to an immutable
historical release seal. It contains numerical receipts and compact
correctness, safety, codegen, and release summaries, while excluding raw logs,
private host/device identifiers, cache paths, profiler artifacts, and process
metadata.

It is still historical evidence. The newly published public tags have not yet
been independently rerun as one separately sealed release. A fresh run must
build the public compiler, kernel, corrected CuTeDSL comparator, and exact FLA
commit; rerun the same matrix and validation gates; then publish a new additive
evidence root. The historical bundle must never be relabeled or overwritten.

## Public artifacts

- [Report and evidence repository](https://github.com/Aharrypotter/gdn-sm90a-tirx-report)
- [TVM compiler tag](https://github.com/Aharrypotter/tvm/tree/gdn-sm90a-compiler-r0)
- [TIRx GDN kernel tag](https://github.com/Aharrypotter/tirx-kernels/tree/gdn-sm90a-kernel-r0)
- [Corrected CuTeDSL GDN comparator tag](https://github.com/Aharrypotter/cuLA/tree/gdn-sm90a-comparator-r1)
- [Exact FLA comparator commit](https://github.com/fla-org/flash-linear-attention/commit/d1ce07369d581813553f30a750af3b6b5f9af6a9)

These are unofficial personal-fork artifacts. No upstream project has merged,
endorsed, or released this work.
