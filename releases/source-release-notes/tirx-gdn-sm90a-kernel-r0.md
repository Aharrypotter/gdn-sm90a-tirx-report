# Unofficial TIRx GDN SM90a kernel snapshot

This is an unofficial personal-fork research snapshot, not an mlc-ai release
or endorsement.

- tag: `gdn-sm90a-kernel-r0`
- release commit: `12ce3721f7c62c5fbd911103ae373de689e58385`
- release tree: `cc04daa65ff52014348c8e078721e9afb017467a`
- accepted runtime commit: `90c9c62c84ecc452dd86602f0ea49a625845045c`
- required TVM tag: `Aharrypotter/tvm:gdn-sm90a-compiler-r0`

The snapshot exposes the product wrapper, exact optimized dispatch allowlist,
tests, and fallback policy for the bounded GDN prefill contract described in
the companion report. Optimization coverage is limited to the documented
dispatch rows, and the work was not merged upstream.

Technical report and evidence:
<https://github.com/Aharrypotter/gdn-sm90a-tirx-report>

No upstream pull request was created for this release.
