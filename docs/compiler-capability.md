# Bounded SM90a compiler capability

The TVM fork adds the compiler mechanisms required by this GDN operator.  It
does **not** claim complete SM90, SM90a, Hopper, or CUDA feature coverage.

The exact public compiler delta is commit
[`acb1312de80b39340e09b0aaad818ff029e745d6`](https://github.com/Aharrypotter/tvm/commit/acb1312de80b39340e09b0aaad818ff029e745d6),
published as
[`gdn-sm90a-compiler-r0`](https://github.com/Aharrypotter/tvm/tree/gdn-sm90a-compiler-r0).
Its frozen upstream base is
[`5b2693d96e06a3b635c5fdeb6e044d2fa13a0349`](https://github.com/apache/tvm/commit/5b2693d96e06a3b635c5fdeb6e044d2fa13a0349).

## Implemented slice

| Capability | Accepted contract |
|---|---|
| Architecture qualification | An explicit `sm_90a`-qualified dispatch; malformed, generic `sm_90`, or out-of-range targets fail the WGMMA predicate |
| WGMMA scope | One complete 128-thread warpgroup |
| WGMMA shapes | `M=64`; logical/native `N ∈ {16, 32, 64, 128}`; `K` is at least 16 and divisible by 16 |
| WGMMA types | FP16 or BF16 inputs with FP32 register accumulators |
| Operand forms | Shared/shared (SS) and register/shared (RS); RS supports the bounded FP32-to-BF16 left-operand cast used by GDN |
| Fragment layouts | Explicit A-register and FP32 accumulator layouts, including aligned output-column slices from a larger resident accumulator |
| Shared descriptors | Swizzle-aware WGMMA descriptors for MN-major and K-major shared-memory layouts |
| TMA | Explicit-stride global views, host-side TensorMap ABI storage/alignment, and descriptor dtype normalization |
| Bounded copies | Logical valid-prefix masking and opt-in rematerialization of swizzled shared-memory offsets |
| Host codegen | TensorMap-pointer bindings and portable 16-bit storage representation needed by the generated host wrapper |

The WGMMA implementation and its fail-closed checks are visible in
[`gemm_async/wgmma.py`](https://github.com/Aharrypotter/tvm/blob/acb1312de80b39340e09b0aaad818ff029e745d6/python/tvm/backend/cuda/operator/tile_primitive/gemm_async/wgmma.py).
The architecture predicate is in
[`common.py`](https://github.com/Aharrypotter/tvm/blob/acb1312de80b39340e09b0aaad818ff029e745d6/python/tvm/backend/cuda/operator/tile_primitive/common.py).

## Why these pieces are coupled

Native WGMMA emission alone is insufficient for this operator:

- Q, K, and V tiles arrive through TMA and must agree with explicit global
  strides and swizzled shared-memory layouts.
- WGMMA accumulators are distributed across a warpgroup.  A consumer that
  writes or reuses only an aligned N slice needs the correct logical
  register offset, not merely a shape-compatible buffer.
- Ragged sequence tails require bounded global/shared copies.  The logical
  prefix must be proven to match physical traversal, and the swizzle offset
  must be rematerialized for the masked path.
- TensorMap descriptors cross the generated host/device boundary and require
  the correct size and alignment in host C codegen.

Relevant immutable source:

- [WGMMA accumulator and A-register layouts](https://github.com/Aharrypotter/tvm/blob/acb1312de80b39340e09b0aaad818ff029e745d6/python/tvm/tirx/layout.py)
- [TMA shared-layout and descriptor utilities](https://github.com/Aharrypotter/tvm/blob/acb1312de80b39340e09b0aaad818ff029e745d6/python/tvm/backend/cuda/operator/tile_primitive/tma_utils.py)
- [TMA explicit-stride handling](https://github.com/Aharrypotter/tvm/blob/acb1312de80b39340e09b0aaad818ff029e745d6/python/tvm/backend/cuda/operator/tile_primitive/copy_async/tma.py)
- [bounded global/shared copy lowering](https://github.com/Aharrypotter/tvm/blob/acb1312de80b39340e09b0aaad818ff029e745d6/python/tvm/backend/cuda/operator/tile_primitive/copy/gmem_smem.py)
- [host TensorMap ABI codegen](https://github.com/Aharrypotter/tvm/blob/acb1312de80b39340e09b0aaad818ff029e745d6/src/target/source/codegen_c_host.cc)

## Dispatch and instruction boundary

The WGMMA tile dispatch has priority only when the target is exactly in its
accepted SM90a range and the execution scope is a full warpgroup.  It lowers
to native WGMMA SS or RS instructions, with the group commit/wait protocol
left visible to the caller so multi-operation pipelines can group work
deliberately.

For the released GDN specializations, generated code is required to contain
WGMMA, TMA, and mbarrier mechanisms and to exclude TCGEN05, TMEM, and external
fallback calls.  The historical codegen inventory covers six cases and 11
stages; see
[`results/codegen-resources.json`](../evidence/historical/gdn-sm90a-h20-20260728-v1/results/codegen-resources.json).

## Explicit non-goals

This compiler tag does not establish:

- full Apache TVM support for Hopper;
- arbitrary WGMMA shapes, dtypes, scopes, sparsity, or block scaling;
- every TMA rank, layout, multicast, reduction, or cluster feature;
- SM90 support without the `a` feature suffix;
- Blackwell/TCGEN05 support;
- upstream compatibility or maintainer acceptance.

The fork is an unofficial, source-bound research release.  No upstream TVM
pull request was created.
