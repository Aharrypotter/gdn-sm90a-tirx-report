# Fresh public evidence derivation

This directory turns a completed private H20 rerun into a deterministic,
publishable evidence bundle. It reconstructs every output document from an
explicit allowlist. It never copies the private input documents wholesale.

The public bundle retains:

- public source repositories, tags, commits, trees, and observed clean-source
  attestations;
- TVM shared-library basenames, sizes, and SHA-256 identities;
- H20 model, compute capability, `sm_90a`, CUDA compiler/runtime, NVIDIA
  driver, and dependency versions;
- the frozen rows, seeds, timing samples, summaries, correctness metrics,
  semantic hashes, process indices, process counts, and aggregate ratios.
- a path-free per-launch ledger bound to each private child PID/worker receipt,
  plus path-free runner attestations for TIRx, CuTeDSL, and FLA.

It intentionally omits absolute filesystem paths, machine or container
identity, GPU UUIDs and physical indices, process IDs, private workspace
identifiers, and per-process compiler-cache locations. The private harness
must have validated those facts before derivation. The public bundle records
only the resulting process-isolation and binding verdicts that can be
cross-checked without disclosing their private identifiers.

The environment input is the JSON emitted by
`python reproduce/check_environment.py --output ...`. It is mandatory because
the timing receipts carry the CUDA runtime version but do not carry the NVIDIA
driver or CUDA compiler version.

Derive a new bundle:

```bash
python -m reproduce.fresh_evidence.derive \
  --contract benchmark-contract.json \
  --environment environment-check.json \
  --oracle-manifest oracles/manifest.json \
  --receipts run/timing \
  --launches run/launches.jsonl \
  --run-summary run/run-summary.json \
  --report run/benchmark-report.json \
  --output public-evidence
```

The output directory must not already exist. Derivation requires `PASS`
oracle, run-summary, report, correctness, source/build, GPU-binding,
fresh-process, and isolation gates. A historical or incomplete run cannot be
promoted by this tool.

Verify a bundle using only its public contents:

```bash
python -m reproduce.fresh_evidence.verify --bundle public-evidence
```

The verifier checks the sealed deterministic manifest, strict document
shapes, disclosure policy, immutable public source locks, frozen six-row
contract, receipt identities, launch/receipt coverage, rotating execution
order, raw-sample statistics, correctness thresholds, oracle bindings,
escalation decision, per-row ratios, and primary geometric means. It does not
import the benchmark producer or trust its aggregate report.

Run the CPU-only tests:

```bash
PYTHONDONTWRITEBYTECODE=1 \
  python -m unittest discover reproduce/fresh_evidence/tests
```

The tests cover deterministic reconstruction, missing or tampered launch
ledgers, truncated runner attestations, source-before drift, ordinary manifest
tampering, tampering followed by a forged reseal, forbidden private-field
injection, and duplicate receipt identities.
