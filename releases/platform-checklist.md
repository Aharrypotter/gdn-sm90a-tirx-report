# Platform publication checklist

Use this checklist for every materialized payload. A checked template is not
enough; preview the exact text that will be sent to the platform.

## Global claim checks

- [ ] The payload was materialized from a recorded report-repository commit.
- [ ] `scripts/validate_claims.py` passes for `content/` and `releases/`.
- [ ] No `{{claim:...}}` token remains in the platform payload.
- [ ] No performance digit was typed or edited manually after materialization.
- [ ] The payload distinguishes the immutable historical
      `HISTORICAL_EVIDENCE_BOUND` bundle from the separately sealed fresh
      public-tag H20 `CHARACTERIZATION`.
- [ ] The fresh bundle is described as a completed 66-receipt
      characterization, not as a full release reseal.
- [ ] Historical and fresh ratios, receipt sets, and decisions are never
      merged into one aggregate.
- [ ] The fresh bundle does not inherit the historical host-sync, sanitizer,
      or full codegen/resource gates.
- [ ] Scope is limited to H20, BF16/D128, and the frozen matrix.
- [ ] The timing region is the final public callable, not end-to-end model
      throughput.
- [ ] Ratio direction says TIRx latency divided by comparator latency; lower
      is faster.
- [ ] The packed-n10 comparison is explicitly a non-win inside the
      preregistered noise band.
- [ ] No universal TIRx/CuTeDSL or TVM/SM90 claim appears.
- [ ] Forks are described as unofficial.
- [ ] No upstream merge, release, or endorsement is implied.

## Provenance and link checks

- [ ] CuTeDSL uses `gdn-sm90a-comparator-r1`.
- [ ] CuTeDSL commit is `88737e9d906cf313995a092624656a89d74dd65e`.
- [ ] The callable is `cula.gdn.prefill.chunk_gated_delta_rule`.
- [ ] `gdn2-sm90a-comparator-r0` is explicitly excluded.
- [ ] FLA uses the exact commit and callable in
      [`contracts/link-map.json`](../contracts/link-map.json).
- [ ] Every external URL appears in the link map.
- [ ] Every relative link resolves in the report repository.
- [ ] The fresh evidence root and source-derived fresh performance report are
      linked explicitly.
- [ ] Social posts link to the report repository, not to a moving local branch
      or private evidence path.
- [ ] Links were opened from the final public post, not only from the editor.

## Technical-content checks

- [ ] Compiler capability is described as the slice required by this operator,
      not complete SM90 support.
- [ ] WGMMA, TMA, layouts, bounded copies, and host TensorMap support are
      attributed to the compiler layer.
- [ ] Recurrence, state orientation, gates, chunk precision, and packed
      isolation are attributed to the semantic layer.
- [ ] Pipeline, short register-replay, tail-predecessor, and exact allowlist
      are attributed to the schedule/dispatch layer.
- [ ] Correctness, safety, codegen, timing, and release evidence are not
      collapsed into one claim.
- [ ] No direct internal PrimFunc timing is presented as public-path evidence.

## X checklist

- [ ] Each post is understandable in sequence and does not rely on a later
      caveat to make an earlier claim true.
- [ ] The scope caveat appears before the first performance statement.
- [ ] Packed-n10 appears in the same post or immediately after the
      CuTeDSL-comparison statement.
- [ ] URLs and line breaks were previewed in the composer.
- [ ] Thread numbering and reply order are correct.
- [ ] The final post links to the report repository and repeats the unofficial
      fork/no-endorsement boundary.

## Zhihu checklist

- [ ] Heading hierarchy renders correctly.
- [ ] Code, tensor shapes, and inline identifiers remain legible.
- [ ] Relative repository links were converted or preserved as valid public
      links by the publishing workflow.
- [ ] The generated performance report is linked instead of copying its table.
- [ ] The comparator correction has its own visible section.
- [ ] The evidence-status section is not hidden behind a collapsed block.

## WeChat checklist

- [ ] The title remains factual and avoids a universal performance claim.
- [ ] The opening scope paragraph is visible before any “result” section.
- [ ] Markdown-only formatting was converted to the editor's supported style.
- [ ] Long code identifiers and links wrap on mobile preview.
- [ ] The generated report and source tags remain clickable.
- [ ] The final disclaimer is visible and not reduced to tiny footnote text.

## Accessibility and archival checks

- [ ] Every figure has descriptive alt text or an adjacent prose explanation.
- [ ] Color is not the only encoding for wins, non-wins, or evidence states.
- [ ] The materialized payload is archived with a SHA-256 digest and passes
      the publication-state scan.
- [ ] Publication time, public URL, platform, source commit, and operator are
      recorded.
- [ ] A screenshot or PDF capture exists for later correction audits.
