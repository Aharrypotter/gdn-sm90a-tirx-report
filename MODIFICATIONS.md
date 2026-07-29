# Modifications and source ownership

This is an unofficial personal research release. It does not claim upstream
merge or endorsement.

- `Aharrypotter/tvm:gdn-sm90a-compiler-r0` adds the SM90a compiler
  capabilities used by this experiment on top of frozen Apache TVM commit
  `5b2693d96e06a3b635c5fdeb6e044d2fa13a0349`.
- `Aharrypotter/tirx-kernels:gdn-sm90a-kernel-r0` adds the productized GDN
  prefill path on top of frozen `mlc-ai/tirx-kernels` commit
  `5f714f508a6bbfffeff449288c001377ab616f44`.
- `Aharrypotter/cuLA:gdn-sm90a-comparator-r1` is an annotated reference to
  exact historical comparator commit
  `88737e9d906cf313995a092624656a89d74dd65e`; no comparator source is copied
  into this report repository.
- FLA is referenced at exact upstream commit
  `d1ce07369d581813553f30a750af3b6b5f9af6a9`; no FLA source is copied here.

All copied or adapted source files retain their original license headers.
Machine-derived evidence is labelled by provenance class and sealed separately
from source code.
