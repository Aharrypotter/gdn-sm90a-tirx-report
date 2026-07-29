# Redaction and derivation policy

This package was reconstructed field by field from an immutable historical
release seal.  It is not a regex-scrubbed copy of the private evidence tree.

Included data is limited to benchmark contracts, numerical timing samples,
correctness metrics, bounded safety summaries, codegen resource inventories,
release controls, and immutable source identities.

Excluded data includes raw logs and job scripts, local or remote filesystem
roots, caches and Python executable paths, host and container identifiers,
SSH aliases, GPU UUIDs, process identifiers, raw profiler traces, generated
CUDA/cubin/SASS artifacts, and the private reviewer report.

`provenance/field-map.json` records the source file and allowlisted JSON paths
for every public output.  `manifest.json` seals the resulting public payload.
