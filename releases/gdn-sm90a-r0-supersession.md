# Report tag r0 supersession

The annotated report tag `gdn-sm90a-r0` is immutable at:

- tag object: `81c0ec29ebbceed192d871f8d91794d7170bba18`;
- peeled commit: `e1fd180b12b65552183a63ea2a0b62f21c3b8634`;
- tree: `48f8e4688e5cae8c0aedef6171dc38675b9c3c84`.

Its ordinary verification job passed, but its required release-asset job
correctly failed before publication because the tag-event checkout materialized
the local tag ref as the peeled commit rather than the annotated tag object.
The tag-object-only builder refused that ambiguous identity.

The failure is preserved in
[GitHub Actions run 30438780264](https://github.com/Aharrypotter/gdn-sm90a-tirx-report/actions/runs/30438780264);
the ordinary verify job passed and the release-asset job stopped at the tag
type check before constructing or publishing assets.

The r0 tag was never moved, deleted, or attached to a GitHub release. It is
superseded before release by `gdn-sm90a-r1`, whose workflow explicitly restores
and verifies the exact remote annotated tag object before asset construction.

Evidence, runtime source, performance facts, and historical artifacts did not
change. The correction is limited to release transport and immutable-tag
qualification.
