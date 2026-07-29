# Rollback and correction playbook

The goal of rollback is to stop propagation of an unsupported claim while
preserving an auditable record. Never move an immutable source tag or silently
rewrite an evidence bundle.

## Severity levels

### R0 — Presentation defect

Examples: broken line breaks, inaccessible image, typo that does not change a
technical claim, or a tracking URL that fails.

Action:

1. Repair the platform presentation if editing is supported.
2. Reopen every link.
3. Record the change in the publication ledger.
4. Do not issue a technical correction unless meaning changed.

### R1 — Ambiguous scope or caveat

Examples: H20/frozen-matrix scope is not visible, packed-n10 can be read as a
win, historical evidence can be mistaken for a fresh rerun, or unofficial
fork status is unclear.

Action:

1. Pause queued downstream posts.
2. Add a visible correction to the affected post or publish an immediate
   follow-up when editing is unavailable.
3. Link to [`docs/limitations.md`](../docs/limitations.md) and the generated
   [performance report](../reports/historical-performance.md).
4. Correct the source template through normal Git history; do not rewrite an
   already published release tag.
5. Rematerialize and revalidate every downstream payload.

### R2 — Incorrect performance or provenance fact

Examples: a copied number differs from canonical evidence, ratio direction is
reversed, the wrong comparator tag is cited, GDN2 r0 is used, or an upstream
merge/endorsement is claimed.

Action:

1. Stop all scheduled publication immediately.
2. Capture the affected public payload and URL before changing it.
3. Publish a visible correction with the exact canonical source link.
4. Remove or withdraw the incorrect post if a correction cannot prevent
   continued misquotation.
5. Run:

   ```bash
   python3 scripts/validate_claims.py \
     --check-content content \
     --check-content releases \
     --show-rendered
   ```

6. Determine whether the defect came from evidence, registry, renderer,
   materialization, or manual platform editing.
7. Fix the earliest incorrect layer, regenerate all downstream payloads, and
   publish a new report release or correction note.
8. Preserve the superseded artifact and its status. Never delete the audit
   trail or move an existing release tag.

### R3 — Evidence integrity failure

Examples: manifest mismatch, source identity cannot be reconstructed,
unexpected private data disclosure, or a supposedly fresh run is not bound to
the public tags.

Action:

1. Withdraw affected release assets and social claims.
2. Mark the report release as withdrawn or invalid without deleting the
   historical record.
3. Rotate or revoke any exposed credential through the relevant provider; do
   not place credential material in the correction.
4. Rebuild evidence from a clean source and re-run every required seal and
   disclosure check.
5. Resume publication only under a new version/tag after independent review.

## Correction template — English

```text
Correction — [UTC date and time]

Affected publication:
[public URL]

What was wrong:
[one factual sentence]

Correct statement:
[exact rendered claim from contracts/claim-registry.json]

Evidence:
[generated report or canonical repository URL from contracts/link-map.json]

Scope:
This remains HISTORICAL_EVIDENCE_BOUND evidence for the frozen H20 BF16/D128
matrix. A fresh public-tag rerun is pending. The artifacts are unofficial
personal forks with no upstream merge or endorsement.

Supersession:
[old payload/release identifier] is superseded by
[new payload/release identifier]. The old artifact is retained for audit.
```

## 勘误模板 — 中文

```text
勘误 — [UTC 日期与时间]

受影响内容：
[公开 URL]

原内容的问题：
[一个事实句]

正确表述：
[从 contracts/claim-registry.json 解析得到的完整 claim]

证据：
[contracts/link-map.json 中的 generated report 或 canonical repository URL]

边界：
该结果仍属于冻结 H20 BF16/D128 矩阵的 HISTORICAL_EVIDENCE_BOUND 证据；
fresh public-tag rerun 尚未完成。相关产物是非官方个人 fork，没有 upstream
merge 或 endorsement。

替代关系：
[旧 payload/release 标识] 已由 [新 payload/release 标识] 替代。
旧产物保留用于审计。
```

## Withdrawal template

```text
Publication withdrawn — [UTC date and time]

Reason:
[integrity or provenance defect]

Affected claims:
[claim IDs]

Canonical status:
[current status from PUBLICATION.json]

Next action:
[re-derive, rerun, or republish under a new immutable version]

No result from the withdrawn payload should be cited. The withdrawal record is
retained to preserve the audit trail.
```

## Fresh-rerun update policy

A future fresh public-tag result is a new evidence class and release. It must:

- use a new additive evidence root;
- name exact public source tags and environment;
- regenerate the performance report and claims;
- state whether the historical conclusion was confirmed, changed, or rejected;
- preserve the historical bundle and all prior corrections;
- trigger a new platform payload rather than silently editing old posts.
