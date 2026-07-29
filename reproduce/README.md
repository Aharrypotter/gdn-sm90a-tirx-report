# Reproduction scaffold

This directory checks out and verifies the exact public source snapshots used
by the report.  It also provides a fail-closed NVIDIA H20 environment check,
an out-of-tree TVM build, and the CPU-only GDN semantic/mechanism test gate.

The benchmark harness below can execute the fresh GPU correctness and timing
portion of the campaign.  It does **not** replace the separate safety,
code-generation, resource, or final sealing gates.  The historical evidence
remains `HISTORICAL_EVIDENCE_BOUND` until a complete fresh public-tag run and
its remaining gates are sealed.

## Exact source locks

| Component | Public source | Exact revision |
|---|---|---|
| TVM compiler | `Aharrypotter/tvm` tag `gdn-sm90a-compiler-r0` | `acb1312de80b39340e09b0aaad818ff029e745d6` |
| TIRx kernels | `Aharrypotter/tirx-kernels` tag `gdn-sm90a-kernel-r0` | release `12ce3721f7c62c5fbd911103ae373de689e58385`; runtime `90c9c62c84ecc452dd86602f0ea49a625845045c` |
| CuTeDSL comparator | `Aharrypotter/cuLA` tag `gdn-sm90a-comparator-r1` | `88737e9d906cf313995a092624656a89d74dd65e` |
| FLA comparator | `fla-org/flash-linear-attention` | `d1ce07369d581813553f30a750af3b6b5f9af6a9` |

The corrected CuTeDSL tag is intentionally `r1`.  The earlier GDN2 tag is not
the implementation named by the historical GDN receipts.

## 1. Check out isolated sources

Choose four new, non-existent destination directories.  The script refuses to
reuse or delete a destination and performs no remote write:

```bash
python reproduce/checkout_public_sources.py \
  --tvm-dir /absolute/path/sources/tvm \
  --tirx-dir /absolute/path/sources/tirx-kernels \
  --cutedsl-dir /absolute/path/sources/cuLA \
  --fla-dir /absolute/path/sources/flash-linear-attention \
  --initialize-tvm-ffi
```

Only the pinned `3rdparty/tvm-ffi` submodule is initialized.  No package is
installed by the checkout step.

## 2. Verify the source identities

All paths are explicit.  Verification fails on a wrong commit, tree, annotated
tag object, remote repository, runtime ancestry, dirty index, or untracked
file:

```bash
python reproduce/verify_source_locks.py \
  --source-lock /absolute/path/report/config/public-source-lock.json \
  --tvm-dir /absolute/path/sources/tvm \
  --tirx-dir /absolute/path/sources/tirx-kernels \
  --cutedsl-dir /absolute/path/sources/cuLA \
  --fla-dir /absolute/path/sources/flash-linear-attention
```

Run this gate before building and again before sealing results.  Keep build,
cache, and Python package directories outside all four source trees so the
source gate stays clean.

## 3. Bind the H20 environment

Select exactly one physical GPU.  The checker requires:

- a visible NVIDIA H20;
- compute capability exactly 9.0;
- one-token `CUDA_VISIBLE_DEVICES` binding to the requested physical index or
  its full GPU UUID;
- one logical PyTorch CUDA device with the same H20/9.0 identity;
- a visible CUDA compiler.

It compares the UUID internally but does not emit it:

```bash
CUDA_VISIBLE_DEVICES=3 \
python reproduce/check_environment.py --physical-gpu-index 3
```

The script exits nonzero on any ambiguity.  Do not kill or alter unrelated GPU
processes to make this check pass.

## 4. Build TVM without editable installs

The TVM and `tvm-ffi` Python modules must never be installed with
`pip install -e`.  Editable installs can silently import another worktree.
This build uses an explicit, out-of-tree CMake directory and installs the
pinned `tvm-ffi` submodule non-editably into an explicit target directory:

```bash
/absolute/path/venv/bin/python -m pip install \
  -r reproduce/requirements-build.txt

bash reproduce/build_tvm.sh \
  --tvm-dir /absolute/path/sources/tvm \
  --build-dir /absolute/path/build/tvm-sm90a \
  --python-site-dir /absolute/path/python-site/tvm-sm90a \
  --python /absolute/path/venv/bin/python \
  --cuda-root /absolute/path/cuda
```

The build enables the CUDA runtime.  The GDN compiler target is the exact TIR
target `sm_90a`; the final import check constructs that target explicitly.  To
match the frozen H20 build, `USE_LLVM=OFF`, `USE_OPENMP=OFF`, and `USE_RPC=ON`
are passed explicitly.  No extra CMake CUDA-architecture or Python-module
override is added to the frozen configuration. The pinned build-front-end
versions match the fresh public-tag build receipt; they are not runtime
performance claims.

## 5. Run the CPU-only static gate

This gate disables CUDA visibility and runs the four source/reference suites
that previously produced 27 passing tests.  It is a semantic and mechanism
gate, not GPU correctness evidence:

```bash
bash reproduce/run_static_tests.sh \
  --tvm-dir /absolute/path/sources/tvm \
  --tvm-build-dir /absolute/path/build/tvm-sm90a \
  --python-site-dir /absolute/path/python-site/tvm-sm90a \
  --tirx-dir /absolute/path/sources/tirx-kernels \
  --python /absolute/path/venv/bin/python
```

The frozen workload and timing contracts are:

- [`configs/six-rows.json`](configs/six-rows.json)
- [`configs/timing-protocol.json`](configs/timing-protocol.json)

These files define the future fresh-run inputs and aggregation rules.  They do
not themselves constitute a benchmark runner or a performance result.

## 6. Run the fresh three-way benchmark on H20

Run from a committed, clean checkout of
`https://github.com/Aharrypotter/gdn-sm90a-tirx-report`.  The materializer
requires `--report-root` to equal that checkout's top level and freezes its
origin, exact HEAD, exact tree, and clean status.  Keep the contract, oracle,
cache, timing, and report outputs outside the checkout so later revalidation
does not dirty it.

Use the exact interpreter that owns the noneditable `tvm-ffi` installation.
The materializer freezes `sys.executable`, Python, Torch plus its CUDA build,
Triton, `tvm-ffi`, and all three CuTe DSL distribution identities.  It also
hashes every installed `tvm-ffi` file except `pyc`/`__pycache__`; absolute
installed paths are not included in that file manifest.

`--cutedsl-dependency-root` must name the preserved, path-private CuTe DSL
4.5.1 tree.  Its expected aggregate is
`41bc70784cde0774308db6883d52e61cdeefe90bedd95631f9da64cee32c5506`.
The root must not overlap any source checkout, the report checkout, or the TVM
build.  The harness explicitly prepends both the metadata root and
`nvidia_cutlass_dsl/python_packages`, then proves that `cutlass` and
`cutlass.cute` load from that tree.

Keep any additional noneditable dependency paths, including the selected
`tvm-ffi` site, visible to the same interpreter:

```bash
export PYTHONPATH="/absolute/path/python-site/tvm-sm90a"
```

Materialize the immutable contract.  `--tvm-build-root` is the out-of-tree
directory passed to `build_tvm.sh --build-dir`; the materializer hashes and
binds its absolute `lib/` directory, compiler library (`libtvm_compiler.so` or
the legacy `libtvm.so` spelling), `libtvm_runtime.so`, and `libtvm_ffi.so`.
The CuTe checkout must be detached at `gdn-sm90a-comparator-r1`, never the
superseded GDN2 tag:

```bash
python -m reproduce.benchmark.contract \
  --run-id gdn-sm90a-public-tags-h20-r0 \
  --report-root /absolute/path/gdn-sm90a-tirx-report \
  --tvm-root /absolute/path/sources/tvm \
  --tvm-build-root /absolute/path/build/tvm-sm90a \
  --tirx-root /absolute/path/sources/tirx-kernels \
  --cutedsl-root /absolute/path/sources/cuLA \
  --cutedsl-dependency-root /absolute/path/deps/nvidia-cutlass-dsl-4.5.1 \
  --fla-root /absolute/path/sources/flash-linear-attention \
  --oracle-root /absolute/path/fresh/oracles \
  --physical-gpu-index 3 \
  --gpu-uuid GPU-00000000-0000-0000-0000-000000000000 \
  --output /absolute/path/fresh/benchmark-contract.json
```

Replace the example UUID with the full UUID mapped to the selected physical
index.  Generate the CuTe correctness oracles, run the fresh-process benchmark,
then audit it:

```bash
python -m reproduce.benchmark.oracle \
  --contract /absolute/path/fresh/benchmark-contract.json \
  --output /absolute/path/fresh/oracles \
  --cache-root /absolute/path/fresh/oracle-cache

python -m reproduce.benchmark.run \
  --contract /absolute/path/fresh/benchmark-contract.json \
  --output /absolute/path/fresh/run \
  --cache-root /absolute/path/fresh/run-cache

python -m reproduce.benchmark.report \
  --contract /absolute/path/fresh/benchmark-contract.json \
  --receipts /absolute/path/fresh/run/timing \
  --output /absolute/path/fresh/run/benchmark-report.json
```

Oracle, run, cache, and report outputs are fail-closed and must not already
exist.  Each timing child binds logical CUDA device 0 with the full GPU UUID,
then independently verifies the physical-index/UUID mapping.  A busy GPU,
foreign compute process, report/source/build/dependency/runtime drift, wrong
backend, fallback, reused cache, or correctness failure invalidates the
attempt.  Oracle generation, every worker, the run orchestrator, and the
receipt reporter revalidate the report checkout, CuTe dependency tree, and
runtime identity before and after their work.  Receipts, run summary, and final
report carry those attestations.  The runtime identity locks the PyTorch module
version and the PyTorch distribution-metadata version as distinct exact fields;
this preserves vendor version strings such as `nv26.02` even when package
metadata normalizes the corresponding segment to `nv26.2`.  The harness never
kills a conflicting process.

## Optional: seal a committed source delta

`seal_git_delta.py` reads blobs from Git objects rather than the working tree:

```bash
python scripts/seal_git_delta.py \
  --repo /absolute/path/sources/tirx-kernels \
  --base 5f714f508a6bbfffeff449288c001377ab616f44 \
  --head 90c9c62c84ecc452dd86602f0ea49a625845045c \
  --output /absolute/path/output/tirx-runtime-delta.json
```

The output path must not already exist.  This helper performs no checkout,
reset, deletion, push, tag creation, or other remote mutation.
