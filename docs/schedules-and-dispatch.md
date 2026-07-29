# Schedules and product dispatch

The public wrapper uses exact, metadata-only dispatch.  Schedule selection is
part of the product contract: it is not an autotuner result chosen at runtime
and it never delegates to another GDN implementation.

Immutable dispatch source:
[`api.py`](https://github.com/Aharrypotter/tirx-kernels/blob/90c9c62c84ecc452dd86602f0ea49a625845045c/tirx_kernels/attention/_gdn_sm90/api.py).

## Dispatch key

The route key is:

```text
(total_tokens, num_sequences, Hq, Hk, Hv,
 has_initial_state, output_final_state)
```

The specialized routes additionally require explicit `alpha` and `beta`.
Only host-visible tensor metadata is used.  Device values in `cu_seqlens`
are not read by the host to choose a route.

## The three schedules

### General pipeline

The general route factors the 64-token chunk algebra into:

1. a 256-thread chunk-parallel prepare kernel for KKT/inverse and WY data;
2. a 128-thread recurrent scan/output kernel.

Chunk slots use device-computable sparse numbering bounded by
`ceildiv(total_tokens, 64) + num_sequences`, avoiding a host read of packed
boundaries.  This route handles every valid non-allowlisted shape and every
near miss.

Source:
[`pipeline.py`](https://github.com/Aharrypotter/tirx-kernels/blob/90c9c62c84ecc452dd86602f0ea49a625845045c/tirx_kernels/attention/_gdn_sm90/pipeline.py).

### Short register replay

This is a single 512-thread fused kernel for exactly:

```text
(4096 tokens, 10 sequences, Hq=Hk=Hv=8,
 initial state present, final state requested)
```

It keeps the qualified short packed recurrence in a four-role,
register-replay schedule.  It is not a generic packed fast path.

Source:
[`short_four_role.py`](https://github.com/Aharrypotter/tirx-kernels/blob/90c9c62c84ecc452dd86602f0ea49a625845045c/tirx_kernels/attention/_gdn_sm90/short_four_role.py).

### Tail-predecessor replay

This route uses a 256-thread prepare kernel and a 256-thread replay/output
kernel with two co-resident value warpgroups.  It bounds replay to one
predecessor chunk and uses a consumer-relative Q-barrier phase.  The public
specialization enables the associated bounded replay, recomputation,
double-buffering, dead-ladder pruning, and register-carried corrected-U
contracts as one qualified schedule.

It is enabled for exactly three keys:

- `(1024, 1, Hq=8, Hk=8, Hv=16, initial state, final state)`;
- `(4096, 1, Hq=Hk=Hv=16, zero initial state, no final state)`;
- `(8192, 20, Hq=8, Hk=8, Hv=16, initial state, final state)`.

## Frozen six-row route map

| Benchmark row | Head mode | State mode | Product route |
|---|---|---|---|
| `single-t512-h8-mha-zero` | MHA | zero / no final state | general pipeline |
| `single-t1024-h8-mha-state` | MHA | initial + final state | general pipeline |
| `single-t1024-h8-hv16-gva-state` | GVA | initial + final state | tail-predecessor |
| `single-t4096-h16-mha-zero` | MHA | zero / no final state | tail-predecessor |
| `packed-n10-t4096-h8-mha-state` | MHA | initial + final state | short register replay |
| `packed-n20-t8192-h8-hv16-gva-state` | GVA | initial + final state | tail-predecessor |

The machine-readable mapping is in
[`contracts/product.json`](../evidence/historical/gdn-sm90a-h20-20260728-v1/contracts/product.json).

## Fail-closed behavior

- A missing specialized key uses the general TIRx route.
- A matching shape without both explicit gates uses the general route.
- A different optional-state combination uses the general route.
- An unsupported head relationship, dtype, rank, device, alignment, or state
  shape raises an input error.
- A non-compute-capability-9.0 device raises an architecture error.
- No valid route falls back to FLA, Triton, CuTeDSL, or a C++ extension.

Route tests must therefore cover both allowlisted keys and near misses.  A
benchmark that directly calls an internal PrimFunc is not evidence for public
dispatch.
