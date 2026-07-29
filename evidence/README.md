# Evidence classes

`historical/` contains privacy-safe derivations from an earlier immutable
release seal. A historical bundle can preserve a valid decision without
claiming that newly published Git tags were independently rerun.

`fresh/` contains additive results generated from exact public source
tags/commits. The current
[`gdn-sm90a-public-tags-h20-20260729-v1`](fresh/gdn-sm90a-public-tags-h20-20260729-v1/)
bundle is a six-row, bundle-derived 66-receipt H20
`CHARACTERIZATION`. It seals source/build/runtime identity, process launches,
receipt-level correctness, timing, and physical-device binding.

That fresh scope does not reproduce the historical host-sync audit, Compute
Sanitizer gates, or full codegen/resource reseals. Those gates remain
historical-only and must not be promoted into the fresh evidence class.

Never merge historical and fresh receipts into one aggregate. Every figure
and article claim must identify its evidence class.
