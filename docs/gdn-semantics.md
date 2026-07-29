# GDN semantic contract

The operator implements packed-variable-length Gated Delta Net prefill with a
resident FP32 recurrent state.  The semantic contract is frozen independently
of any particular GPU schedule.

Immutable sources:

- [public contract](https://github.com/Aharrypotter/tirx-kernels/blob/90c9c62c84ecc452dd86602f0ea49a625845045c/tirx_kernels/attention/_gdn_sm90/contract.py)
- [tokenwise and chunkwise references](https://github.com/Aharrypotter/tirx-kernels/blob/90c9c62c84ecc452dd86602f0ea49a625845045c/tirx_kernels/attention/_gdn_sm90/reference.py)
- [precision-visible collective inverse reference](https://github.com/Aharrypotter/tirx-kernels/blob/90c9c62c84ecc452dd86602f0ea49a625845045c/tirx_kernels/attention/_gdn_sm90/inverse_reference.py)

## Public tensors

| Input or output | Shape | Dtype |
|---|---|---|
| `q` | `[total_tokens, Hq, 128]` | BF16 |
| `k` | `[total_tokens, Hk, 128]` | BF16 |
| `v` | `[total_tokens, Hv, 128]` | BF16 |
| optional `alpha` | `[total_tokens, Ho]` | FP32 |
| optional `beta` | `[total_tokens, Ho]` | FP32 |
| `cu_seqlens` | `[num_sequences + 1]` | CUDA INT32 or INT64 |
| optional initial/final state | `[num_sequences, Ho, 128, 128]`, public `[V, K]` order | FP32 |
| output | `[total_tokens, Ho, 128]` | BF16 |

`Ho = max(Hq, Hv)`.  All tensors must be contiguous, share one CUDA device,
and satisfy the public alignment checks.

Supported head relationships are:

- MHA: `Hq = Hk = Hv`;
- GQA: `Hq` is an integer multiple of `Hk = Hv`;
- GVA: `Hv` is an integer multiple of `Hq = Hk`.

Each output head maps to exactly one Q, K, and V owner.  Unsupported head
relationships fail validation rather than falling back.

## Literal token recurrence

For one mapped output head, let the public state be
`H_t ∈ R^(V×K)`.  With scalar gates `alpha_t` and `beta_t`:

```text
H_decay = alpha_t * H_(t-1)
prediction = H_decay @ k_t
update = beta_t * (v_t - prediction)
H_t = H_decay + update outer k_t
o_t = scale * (H_t @ q_t)
```

An absent `alpha` is semantically one; an absent `beta` is semantically one.
An absent initial state is zero.  `output_final_state=False` suppresses the
returned state but does not change the output recurrence.

## Frozen 64-token chunk algebra

The GPU implementation is required to preserve this precision-visible order
for each 64-token chunk:

1. `cp = exp(cumsum(log(alpha)))`.
2. WGMMA forms QK and KK, then applies the transfer
   `cp[row] / cp[col]`.
3. `beta[row]` is applied to physical KK before inversion.
4. The inclusive unit-lower KK operator is inverted with FP16-visible stages
   `8 → 16 → 32 → 64`.
5. `beta[col]` is applied when the inverse is published as a BF16 operand.
6. Prior-state output O1, state/key projection SK, corrected value NewV,
   within-chunk output O2, and the terminal-state update occur in that order.

Changing the side on which `beta` is applied, the state orientation, the
rounding points, or the inverse ladder is a semantic change even when a
high-level recurrence looks algebraically equivalent.

## Ragged and packed isolation

Every sequence is evaluated independently.  For a partial final chunk:

- invalid Q/K/V elements are zero;
- invalid `alpha` is one;
- invalid `beta` is zero;
- the inverse has an identity diagonal in the tail;
- no value, state, or predecessor replay may cross a `cu_seqlens` boundary.

The default wrapper validates only host-visible metadata and does not copy
`cu_seqlens` values to the host.  Callers must provide boundaries that begin
at zero, end at `total_tokens`, are strictly increasing, and represent
non-empty sequences.

## Numerical acceptance

The frozen historical correctness policy is:

| Quantity | `atol` | `rtol` | maximum absolute error | relative RMS limit |
|---|---:|---:|---:|---:|
| output | 0.01 | 0.01 | 0.075 | 0.15 |
| final state | 0.005 | 0.001 | 0.075 | 0.15 |

The policy and all 66 per-process results are machine-readable in
[`contracts/benchmark.json`](../evidence/historical/gdn-sm90a-h20-20260728-v1/contracts/benchmark.json)
and
[`results/correctness.json`](../evidence/historical/gdn-sm90a-h20-20260728-v1/results/correctness.json).

CPU-reference agreement establishes the recurrence and auxiliary orientation;
it does not establish GPU correctness, route selection, memory safety,
codegen, or performance.  Those are separate validation gates.
