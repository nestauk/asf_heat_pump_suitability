---
title: Cross-version comparison script — cluster geometry checks
status: draft
github_issue: https://github.com/nestauk/asf_heat_pump_suitability/issues/472
pr:
asana: https://app.asana.com/1/5571817120120/project/1214222223606748/task/1216704619895657
created: 2026-08-13
---

## Problem

The comparison script's tabular checks (#447, PR #451) can't see
geometry-level drift — a clustering change could alter cluster count, total
area, or size distribution while every tabular check still passes. This is
by design in the base checks: `load_transform_df_stage_output` drops
geoarrow geometry columns because polars cannot read them, so nothing
downstream of the loader ever sees a polygon.

Third issue in the comparison-script stack (see
`docs/specs/447_addCompareVersionsScript.md` and the stack's drafts doc):
stacks on `447_addCompareVersionsScript`, and the shared distribution
helper introduced here is reused by the distribution-checks follow-on
(issue 5 in the stack).

## Proposal

Extend the report, for the clustering and contextual-features stages, with:

- cluster count delta (`n_unique(cluster_id)` per version)
- total area delta, in m² (EPSG:27700), with CRS and units stated
- cluster-area distribution comparison: Q1, Q3, min, max, mean per version
- UPRNs-per-cluster distribution comparison with the same statistics
  (the `n_UPRNs` count computed at the contextual-features stage)
- an old-vs-new overlaid plot per distribution, saved as a PNG next to the
  markdown report and embedded in it via an image link (matplotlib is
  already a main dependency)

Decisions settled during kickoff interview (2026-08-13):

- **Area is reported at both stages, with a caveat at the
  contextual-features stage.** The cluster stage's geoparquet carries exact
  EPSG:27700 geometry; the contextual-features geojson carries the same
  clusters reprojected to EPSG:4326 and simplified
  (`compute_contextual_features.py` simplifies before saving). Both stages
  get area checks — the contextual-features output is what the front-end
  tool consumes, so leaving it blind to area drift defeats the check — but
  its geometries are reprojected back to EPSG:27700 before measuring and
  the report states that its areas are computed on simplified geometry, so
  a reader doesn't chase simplification artefacts as drift.
- **`base.yaml` lists real columns only.** The per-stage target-column
  lists for the shared distribution helper (settled 2026-08-10 for the
  stack) name columns that exist in the stage output (e.g. `n_UPRNs` for
  the contextual-features stage), so config stays verifiable by test. The
  derived `area_m2` is computed from geometry in code for geometry-bearing
  stages and fed through the same shared helper — code owns derived
  metrics, config owns stored columns.
- **IQR is reported as the Q1 and Q3 quartiles**, old vs new, not as a
  single width. Both quartiles show whether a distribution shifted or
  widened; a lone width hides a shift where both quartiles move together.
- **Overlaid distribution plots land in this issue**, for its two
  distributions (cluster area, UPRNs per cluster). Building the plotting
  arm now, while there are only two distributions, means the
  distribution-checks follow-on inherits it rather than retrofitting it.

## Alternatives considered

- **Area checks at the cluster stage only** — rejected; avoids approximate
  areas but leaves the final, tool-facing stage blind to area drift.
- **Both stages with no simplified-geometry caveat** — rejected; a reader
  could chase reprojection/simplification artefacts as drift.
- **Listing `area_m2` as a pseudo-column in `base.yaml`** — rejected;
  config would name a column no output contains, and the follow-on issue's
  config inherits that ambiguity.
- **Single IQR width (Q3 − Q1)** — rejected; cannot distinguish a shifted
  distribution from a stable one.
- **Deferring plots to the distribution-checks follow-on, or recording
  them as an open question only** — rejected; the helper would be
  retrofitted just as distributions multiply, and the geometry
  distributions would get plots retroactively.

## Out of scope

- Categorical mix deltas, numeric drift (stacked follow-on issue 5)
- Turning deltas into an automatic expected/suspicious flag (stacked
  follow-on issue 6)
- PSI or other distribution-drift scalars

## Open questions

- Which of the follow-on issue's distributions (tenure, EPC, outdoor
  space…) are "important" enough to plot — this issue builds the plotting
  mechanism; the follow-on decides where else to apply it.

## Verification

- [ ] Cluster count delta reported for both stages
- [ ] Total area delta reported for both stages with CRS/units stated;
      simplified-geometry caveat in the contextual-features section
- [ ] Cluster-area distribution reported: Q1, Q3, min, max, mean per version
- [ ] UPRNs-per-cluster distribution reported with the same statistics
- [ ] One shared column-parameterized distribution function; target columns
      per stage in `base.yaml`
- [ ] Overlaid old-vs-new plots for cluster area and UPRNs per cluster,
      embedded in the report
- [ ] Unit tests cover at least one genuine geometry-drift case and one
      stable case
