---
title: Cross-version comparison script — cluster geometry checks
status: in-review
github_issue: https://github.com/nestauk/asf_heat_pump_suitability/issues/472
pr: https://github.com/nestauk/asf_heat_pump_suitability/pull/475
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

Implementation decisions within the spec's frame (2026-08-13):

- **Target columns live in `compare_versions.distribution_columns`** in
  `base.yaml`, keyed by stage (`cluster: []`,
  `compute_contextual_features: [n_UPRNs]`); a test pins the keys to known
  stages. The geometry-bearing stages themselves are a module constant
  (`GEOMETRY_STAGES`) — which outputs carry geometry is code behaviour,
  not a tunable.
- **The shared helper is `generate_dict_distribution_stats(df, column)`**
  (min/Q1/mean/Q3/max, linear-interpolated quartiles, nulls dropped);
  `get_dict_distribution_frames` pairs each distribution with the frames
  carrying it — derived `area_m2` from the geometry loader
  (`load_df_cluster_areas`, which reprojects non-EPSG:27700 outputs before
  measuring), configured columns from the tabular outputs.
- **Plots share the report's filename stem**
  (`{stage}_{la}_{old}_vs_{new}_{column}.png`, saved next to the report);
  both versions share histogram bins so the shapes are comparable. A
  distribution with a missing column or no values on one side skips its
  plot with a warning — its stats section already notes the gap.

Implementation decisions from the first acceptance run on real data
(2026-08-26):

- **The geometry checks filter to the clusters layer.** As of ~August 2026
  the contextual-features geojson is a multi-layer front-end file: a
  `layer` column tags each row (`clusters_with_contextual_features`,
  `ward_boundaries`, `anchor_loads`,
  `areas_of_district_heat_network_potential`). Aggregating all layers let
  six whole-county ward polygons swamp the cluster signal (+679.5M m²
  total-area delta on East Lothian 20260708 vs 20260806). Cluster count,
  total area, distribution stats and plots now cover the clusters layer
  only — one shared filter (`filter_df_clusters_layer`) applied to both
  the tabular and geometry-derived frames. A version without a `layer`
  column (pre-August outputs) is treated as all-clusters, so comparisons
  across the format change keep working.
- **The clusters layer name is config, not code:**
  `compare_versions.cluster_layer` in `base.yaml`, verified against the
  20260806 East Lothian output — a front-end rename is a config change.
- **A per-layer summary table keeps excluded layers visible.** The
  geometry section tabulates rows and total area (m²) per layer over the
  union of both versions' layers, so non-cluster layers are surfaced
  rather than silently dropped; a version without a `layer` column shows
  as a single `(pre-layers output)` row, and a side lacking a layer
  renders as a dash, not a misleading zero. When filtering was applied,
  the section states that the headline checks cover the clusters layer
  only.
- **Stats render consistently across dtypes:** an integral float renders
  like an int in `_format_stat` (1738.0 → "1,738"), so a version whose
  column round-trips to Float64 no longer renders "1,738.0" beside the
  other version's "1,165"; non-integral floats keep one decimal place.

## Verification

- [x] Cluster count delta reported for both stages
- [x] Total area delta reported for both stages with CRS/units stated;
      simplified-geometry caveat in the contextual-features section
- [x] Cluster-area distribution reported: Q1, Q3, min, max, mean per version
- [x] UPRNs-per-cluster distribution reported with the same statistics
- [x] One shared column-parameterized distribution function; target columns
      per stage in `base.yaml`
- [x] Overlaid old-vs-new plots for cluster area and UPRNs per cluster,
      embedded in the report
- [x] Unit tests cover at least one genuine geometry-drift case and one
      stable case (a cluster merge vs identical versions; acceptance runs
      against Plymouth 20260806 vs 20260812 exercised both stages on S3)
- [x] Multi-layer outputs: cluster count, total area, distribution stats
      and plots cover the config-named clusters layer only
      (`compare_versions.cluster_layer`), with the scope stated in the
      report
- [x] Per-layer summary table reports rows and total area (m²) per layer,
      old vs new, over the union of both versions' layers
- [x] A version without a `layer` column is treated as all-clusters
      (back-compat across the format change) and shown as a single
      pre-layers row in the per-layer table
- [x] Statistics render consistently across versions: integral floats
      render like ints, non-integral floats keep one decimal place
      (acceptance re-run against East Lothian 20260708 vs 20260806
      exercised the multi-layer path on S3)
