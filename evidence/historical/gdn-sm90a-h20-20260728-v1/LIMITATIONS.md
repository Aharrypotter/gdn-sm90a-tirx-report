# Limitations

1. This is historical evidence bound to the original immutable release seal.
   It is not an independent benchmark rerun from the newly published tags.
2. All 22 historical CuTeDSL receipts identify commit `88737e9`, CuTe DSL
   4.5.1, backend `sm90_cutedsl_gdn`, and callable
   `cula.gdn.prefill.chunk_gated_delta_rule`.  The corrected public comparator
   tag points directly to that commit.  The earlier GDN2 tag is not evidence
   for this GDN report.
3. BF16 and head dimension 128 are properties of the public tagged source
   contract, not explicit fields in the historical benchmark JSON.
4. The scope is six GDN prefill rows on one NVIDIA H20 environment.  It is not
   a claim about every Hopper GPU, arbitrary shapes, or end-to-end model speed.
5. Five rows have a lower TIRx/CuTeDSL latency ratio.  Packed-10 is 1.46% higher
   and is classified only as inside the preregistered ±2% noise band.
6. All branches and tags are unofficial personal-fork artifacts.  No upstream
   pull request, merge, endorsement, or full TVM SM90 support claim is implied.
