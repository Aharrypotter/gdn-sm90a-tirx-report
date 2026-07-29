# Publishing order

This sequence keeps every platform downstream of one validated source of
truth. Do not start a platform post while the GitHub assets or claim
materialization gate is incomplete.

## Gate 0 — Freeze the publication payload

1. Verify the three public tags and the exact FLA commit against
   [`contracts/link-map.json`](../contracts/link-map.json).
2. Verify the historical evidence bundle:

   ```bash
   make verify-historical-evidence
   ```

3. Independently verify the additive fresh public-tag bundle:

   ```bash
   make verify-fresh-evidence
   ```

4. Regenerate and check both human-readable performance reports:

   ```bash
   python3 scripts/render_performance_markdown.py --check
   python3 scripts/validate_claims.py
   ```

5. Validate claims and every publication template:

   ```bash
   python3 scripts/validate_claims.py \
     --check-content content \
     --check-content releases
   ```

6. Run repository static checks and inspect the complete staged scope when a
   commit is authorized.
7. Record the report commit, release tag, both evidence-manifest digests, and claim
   registry digest in the publication ledger.

Stop if any check fails. Do not repair a failed number by editing prose.
Repair the canonical evidence, renderer, or registry and regenerate.

## Gate 1 — Publish the GitHub source of truth

Publish or verify, in this order:

1. [TVM compiler tag](https://github.com/Aharrypotter/tvm/tree/gdn-sm90a-compiler-r0)
2. [TIRx GDN kernel tag](https://github.com/Aharrypotter/tirx-kernels/tree/gdn-sm90a-kernel-r0)
3. [Corrected CuTeDSL GDN comparator tag](https://github.com/Aharrypotter/cuLA/tree/gdn-sm90a-comparator-r1)
4. [Report and evidence repository](https://github.com/Aharrypotter/gdn-sm90a-tirx-report)
5. A report-repository release containing the immutable source coordinates,
   evidence class, generated performance report, and limitations

The report repository is the canonical destination for all social links.
Do not create an upstream pull request as part of publication.

## Gate 2 — Materialize the master drafts

The files in `content/` are source templates. Performance-dependent sentences
are represented by `{{claim:Cxx:language}}` tokens.

1. Run:

   ```bash
   python3 scripts/materialize_publication_content.py
   python3 scripts/materialize_publication_content.py --check
   python3 scripts/validate_claims.py --check-content dist/content
   ```

2. Use the generated files in `dist/content/` as the platform payloads. Do not
   hand-replace claim tokens.
3. Do not edit digits, signs, ratios, percentages, row counts, or receipt
   counts in a generated payload.
4. Compare the generated payload to both
   [`reports/historical-performance.md`](../reports/historical-performance.md)
   and
   [`reports/fresh-public-tag-performance.md`](../reports/fresh-public-tag-performance.md).
5. Confirm that no generated payload contains an unresolved token:

   ```bash
   if rg '\{\{claim:' dist/content; then exit 1; fi
   ```

The source templates remain tokenized in Git. The materialized payloads are
checked in as sealed release artifacts and must remain byte-identical
to the materializer output.

## Gate 3 — Publish discovery and long-form channels

Recommended order:

1. English X thread from
   [`dist/content/x-thread-en.md`](../dist/content/x-thread-en.md)
2. Chinese X thread from
   [`dist/content/x-thread-zh.md`](../dist/content/x-thread-zh.md)
3. Zhihu long-form article from
   [`dist/content/zhihu.md`](../dist/content/zhihu.md)
4. WeChat public-account article from
   [`dist/content/wechat.md`](../dist/content/wechat.md)

The English and Chinese master drafts remain the editorial reference:

- [`content/master-en.md`](../content/master-en.md)
- [`content/master-zh.md`](../content/master-zh.md)

If platform review or formatting delays one channel, continue only with
channels whose payloads independently pass the checklist. Do not weaken a
caveat to synchronize publication times.

## Gate 4 — Post-publication verification

For every platform:

1. Open the public post in a logged-out or private browser.
2. Verify title, line breaks, code blocks, images, and all links.
3. Confirm that packed-n10 is presented as a non-win inside the preregistered
   noise band.
4. Confirm that the historical bundle remains
   `HISTORICAL_EVIDENCE_BOUND`, the separate fresh public-tag bundle is a
   completed `CHARACTERIZATION`, and their aggregates are not merged.
5. Confirm the corrected CuTeDSL r1 tag and explicit GDN2 r0 exclusion.
6. Confirm unofficial-fork and no-upstream-merge/endorsement language.
7. Record the public URL, publication time, payload hash, and screenshot in
   the publication ledger.

## Gate 5 — Follow-up cadence

- First pass: verify all channels immediately after publication.
- Early follow-up: monitor broken links, misunderstood ratio direction, and
  comparator confusion.
- Later follow-up: consolidate questions into an FAQ or report-repository
  issue rather than changing source claims ad hoc.
- Subsequent rerun: publish any later execution as a new additive evidence
  release and clearly labelled follow-up. Never rewrite either existing
  evidence class as though the later run had always existed.
