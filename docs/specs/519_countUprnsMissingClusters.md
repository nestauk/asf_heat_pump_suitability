---
title: "Cross-version comparison report: count UPRNs missing clusters"
status: in-review
github_issue: https://github.com/nestauk/asf_heat_pump_suitability/issues/519
pr: https://github.com/nestauk/asf_heat_pump_suitability/pull/520
asana: https://app.asana.com/1/5571817120120/project/1214222223606748/task/1218394231045431
created: 2026-09-28
---

## Problem

`compute_contextual_features.py` maps domestic UPRNs to clusters with
`cluster.map_df_uprns_to_clusters`. A UPRN whose building does not match a cluster gets a
null `cluster_id` (the `default=None` from PR #506). `extend_df_contextual_features` then
groups by `cluster_id` and left-joins the result onto the clusters. The null group has no
cluster to join to, so these UPRNs leave the output with no message.

The comparison report (#447, #472) does not show this loss. The suggestion came from the
#475 review round, through the Asana task linked above. The behaviour and the acceptance
criteria are in the GitHub issue. This spec records the design decisions.

## Proposal

1. **Tier 2 report only.** Add one section to the comparison report. Do not add a Tier 1
   pandera gate.
   _Rationale:_ the Asana task asks for the report. `pipeline_validation_checks.md` puts a
   bounded reconciliation in Tier 1, and that stays a possible later issue.
2. **Branch from #475's branch, not from #474.** The base is
   `472_addClusterGeometryChecksToCompareVersions` at `92640bc`, so this is PR 4 of the
   472 stack. The new code goes into the current single `compare_versions.py` module.
   _Rationale:_ the #474 decomposition is deferred until later. The code must reuse
   `filter_df_clusters_layer`, and that function is only on the 472 stack.
3. **The count is a reconciliation of two stages from one release.** For each version:

   - _UPRNs in:_ distinct `UPRN` in the `add_features` output (`domestic_uprns_with_features`)
     of the same release date and LA. This is the frame that `compute_contextual_features`
     reads before it maps UPRNs to clusters.
   - _UPRNs in clusters:_ sum of `n_UPRNs` over the clusters layer only
     (`filter_df_clusters_layer`).
   - _Missing:_ UPRNs in minus UPRNs in clusters. _Share:_ missing / UPRNs in.

   _Rationale:_ no saved output holds both `UPRN` and `cluster_id`, so a count is the most
   that we can get from saved data. The count is exact. `map_df_uprns_to_clusters` gives
   each UPRN at most one `cluster_id`, and `n_UPRNs` is `n_unique()` for each cluster. The
   old DESNZ-zone double-count concern is not in `cluster.py` on `dev` any more.

4. **Section shape: old, new and delta, with share.** One row for each measure above. The
   columns are old, new and change, the same layout as the existing count sections.
   _Rationale:_ it shows if the leak grew between versions, not only its size now.
5. **Runs only for `--stage compute_contextual_features`.** Load the add_features output
   for each version as a second stage, in the same way as `load_tuple_df_buildings` loads
   the buildings dataset for the decision-tree stage. The one-stage CLI does not change.
6. **Report-only, no threshold.** The section does not fail the run and does not add a
   config tolerance. The thresholding subtask is on hold.
7. **Missing input is not an error.** If a version has no add_features output for its
   release date, the section says that the count is unavailable for that version. The rest
   of the report still renders.

## Alternatives considered

- **Tier 1 gate only, or Tier 1 and Tier 2 together.** Rejected for now. Tier 1 bounds the
  loss in one run. It does not show the change between versions, and the Asana task asked
  for the report.
- **Stack on `474_decomposeCompareVersions`.** Rejected. Aidan deferred the decomposition.
  #474 must then move this section into `checks.py` / `report.py` when it is rebased.
- **New-version count only.** Rejected. It does not show if the loss grew.
- **Also count empty clusters** (clusters in `tech_clusters` that the `n_UPRNs` null filter
  drops). Rejected, UPRNs only. See Open questions.
- **Save the UPRN-to-cluster mapping so that the lost UPRNs can be named.** Not done. It
  changes a pipeline output and is out of scope.

## Out of scope

See the issue. In short: no Tier 1 gate, no threshold, no empty-cluster count, no fix to
the cause of unmatched UPRNs, no #474 decomposition.

## Open questions

- Should the empty-cluster drop at `compute_contextual_features.py` (the
  `# TODO identify source of empty clusters` filter) get its own count, in this section or
  in a Tier 1 gate?
- Is a UPRN with a null building ID counted as "missing"? The reconciliation counts it, but
  the cause is different from a building that is not in a cluster. Do we need to show the
  two causes separately?
- When #474 is picked up again, it must include this section in the split.

## Verification

The acceptance criteria are in
[#519](https://github.com/nestauk/asf_heat_pump_suitability/issues/519). In addition:

- [x] `uv run python -m pytest asf_heat_pump_suitability/pipeline/validate/tests/test_compare_versions.py`
      passes, with new tests for the count, the layer filter and a missing add_features output
- [x] East Lothian acceptance run:
      `uv run python -m asf_heat_pump_suitability.pipeline.validate.compare_versions --stage compute_contextual_features --local_authority east_lothian`
      gives a missing count of 0 or more, and the count agrees with a manual recount from the two outputs (2026-09-29, 20260806 vs 20260907, 291 to 292 missing, recount matched)
