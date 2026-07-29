# X thread — English

<!-- TEMPLATE_ONLY
Publication source template. Resolve {{claim:Cxx:en}} from the claim registry.
Check every rendered post in the platform composer before publishing.
-->

## 1

I published a TIRx-on-Hopper study: a bounded SM90a compiler slice, a
productized GDN prefill operator, and evidence that keeps source, semantics,
codegen, safety, and timing separate.

{{claim:C01:en}}

## 2

The compiler work was more than selecting `sm_90a`: WGMMA SS/RS lowering,
warpgroup layouts, swizzled and K-major descriptors, explicit-stride TMA
TensorMaps, bounded ragged-tail copies, and the host TensorMap ABI all had to
agree.

## 3

{{claim:C02:en}}

Compiler scope:
https://github.com/Aharrypotter/tvm/tree/gdn-sm90a-compiler-r0

## 4

For a recurrent operator, schedule-first tuning is risky. I froze the token
recurrence, FP32 V-first state, MHA/GQA/GVA mapping, precision-visible inverse
ladder, and packed-boundary behavior before treating performance as a product
question.

## 5

{{claim:C03:en}}

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

{{claim:C04:en}}

{{claim:C05:en}}

## 10

This is public-call operator latency, not end-to-end model throughput. The
scope must not be generalized beyond the frozen matrix.

## 11

{{claim:C08:en}}

## 12

{{claim:C09:en}}

The packed exception belongs in the headline, not a footnote.

## 13

{{claim:C10:en}}

Generated table:
https://github.com/Aharrypotter/gdn-sm90a-tirx-report

## 14

Comparator provenance required a correction:

{{claim:C13:en}}

## 15

Corrected CuTeDSL source:
https://github.com/Aharrypotter/cuLA/tree/gdn-sm90a-comparator-r1

{{claim:C14:en}}

## 16

{{claim:C12:en}}

## 17

Fresh source-derived report:
https://github.com/Aharrypotter/gdn-sm90a-tirx-report/blob/gdn-sm90a-r1/reports/fresh-public-tag-performance.md
