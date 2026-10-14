---
title: Build a GB summary table from the comparison JSON files
status: draft
github_issue: https://github.com/nestauk/asf_heat_pump_suitability/issues/531
pr:
asana:
created: 2026-10-14
---

## Problem

With `--compare`, each pipeline run (#526, PR #534) writes one comparison
JSON for each local authority (LA) and stage to
`outputs/data/{la}/{release_date}/comparisons/`. A GB run gives about 1,750
files (about 350 LAs x 5 stages). Nothing reads them together, so there is no
quick answer to two questions after a GB run:

1. Did every LA and stage run?
2. Which LAs changed most since their last release?

The issue body has the behaviour and the acceptance criteria. This spec
records the design decisions and why we made them.

## Proposal

1. **A separate entrypoint, run by hand after the Orbit jobs finish.**
   New module `pipeline/validate/summarise_comparisons.py`, next to
   `compare_versions.py`. CLI: `--release_date` (required),
   `--local_authorities` (names, or `GB` for the full list, as in
   `orbit_pipeline.py`), `--report_dir` (local output) and `--save`.
   _Rationale:_ `orbit launch` submits the Batch job and returns at once
   (`arm_orbit/cli/launch.py` prints the job ID and exits). So
   `orbit_pipeline.py` cannot know when the comparisons are done, and the
   summary cannot be its last step. This was already an open question in
   the #526 spec. An optional `--wait` on `orbit_pipeline.py` that polls the
   jobs and then runs this summary is issue #536.

2. **The expected rows are the expected LAs x the five comparison stages.**
   The LAs come from `config["data"]["processed"]["valid_la_names"]`
   (`LAD23NM`, as `orbit_pipeline.py --local_authorities GB` reads them) or
   from `--local_authorities`. They are made into slugs with
   `make_str_slug`. The stages are the keys of
   `config["compare_versions"]["stage_output_datasets"]`.
   _Rationale:_ the table must show an LA that has no files at all. A list
   of the files found cannot show that.

3. **Each row gets one status: `compared`, `baseline`, `stage failed` or
   `comparison missing`.** `compared` and `baseline` come from the JSON.
   When there is no JSON, the reader checks if the stage output exists for
   the release (`get_str_stage_output_path` from `compare_versions`). No
   output gives `stage failed`. Output with no JSON gives
   `comparison missing` (the comparison failed, or `--compare` was off).
   _Rationale:_ the pipeline failure CSVs from #521 are never written now
   (see Open questions), and a check of the output shows what really
   happened for this release, also after an LA is run again by hand. The
   extra S3 lookup is needed only for the rows with no JSON.

4. **The JSON contract is the 14 keys of #534, with `schema_version` 1.**
   A JSON without `schema_version` is version 0 (the 11 keys before
   2026-10-13). The reader fills the keys that version 0 does not have with
   null. The three keys added in #534 (`schema_version`, `tech_counts`,
   `uprns_missing_clusters`) were agreed in this kickoff. They copy only
   numbers that the comparison already computes from data that it already
   loads.
   _Rationale:_ test files of version 0 are on S3 already (East Lothian,
   20261013). The version number lets the reader handle a later layout
   change without guessing from the keys.

5. **The table is long: one row for each LA and stage.** Fixed-shape keys
   are flattened into columns: `counts` (`rows_old`, `rows_new`,
   `rows_delta`, `uprns_*`), `uprn_churn` (`n_added`, `n_removed`,
   `removed_share`, `added_share`, ...), `cluster_counts` (`clusters_*`),
   `total_area` (`area_m2_*`) and `uprns_missing_clusters` (`uprns_in_old`,
   `uprns_missing_new`, `missing_share_new`, ...). `tech_counts` stays one
   list-of-struct column, null on the four stages that are not
   `decision_tree`. `df.explode("tech_counts").unnest("tech_counts")`
   gives one row for each tech.
   _Rationale:_ flatten when the shape is fixed, nest when the shape
   depends on the data. Tech labels are data, so columns for each tech
   would change the schema when a label changes. A new JSON key (for
   example from #535) adds columns, not a new layout.

6. **Rows are sorted by `max_rel_change`.** It is the largest absolute
   value of `|delta| / old` for rows, UPRNs, clusters and area, the churn
   `removed_share` and `added_share`, and `|n_delta| / n_old` for each
   tech. `max_rel_change_metric` names the metric, for example
   `tech:Networked heat pump`. A value with `old` = 0 is not used.
   `uprns_missing_clusters` is shown but not used in the sort. Baselines
   and rows with no JSON come after all compared rows.
   _Rationale:_ a relative change makes LAs of different sizes comparable.
   Churn shares find "same total, different UPRNs", which a net delta
   hides. Small missing-cluster counts give large, unstable percentages.

7. **No new thresholds.** The table shows the existing
   `tolerance_warnings` from the JSON and sorts. It does not flag rows.
   _Rationale:_ limits chosen before two GB runs would be guesses. The
   known result: the cluster-level stages still never get a flag (their
   JSON has no `uprn_churn`). A person reads the sorted list.

8. **When there is more than one JSON for an LA and stage, the newest by
   S3 write time is used.** The table has an `n_json_files` column, and the
   markdown lists the files that were not used.
   _Rationale:_ a second JSON appears when a comparison is run again with a
   different old date. The last run is normally what the person wants. A
   rule that uses the latest old date is wrong when an earlier date was
   pinned on purpose with `--old_release_date`.

9. **The output is a parquet table and a short markdown summary.** The
   markdown has the run-health counts by status and stage, the rows that
   are not `compared` or `baseline`, the top changes (with the tech line for
   rows where a tech moved most), the tolerance warnings, the extra JSON
   files, and the S3 path of each per-LA report it names. The number of top
   changes is a config value.
   _Rationale:_ the parquet is for analysis in polars. The markdown is the
   view for people, so they do not need to open the nested column.

10. **`--save` writes both files to
    `s3://asf-local-heat-planning-tool/outputs/comparisons/{release_date}/`.**
    The path is a new key in `base.yaml` under `compare_versions`. Without
    `--save`, the files go to `--report_dir` only.
    _Rationale:_ one place for each release, apart from the 350 LA folders.
    It has the same shape as `outputs/logs/pipeline_failure/{release_date}/`.

11. **The file names say which LAs the summary covers.** The files are
    `comparison_summary_{scope}.parquet` and `.md`. The first rule that
    matches gives `{scope}`:

    - the full GB list gives `gb`
    - 1 to 3 LAs give their slugs, sorted and joined with `__`, for example
      `east_lothian__glasgow_city`
    - more than 3 LAs give the count and the first and last slug, sorted,
      for example `12_las_barnet_to_westminster`

    The markdown lists every LA of the run at the top. A run with the same
    LAs replaces its own files. There is no `--label` flag.
    _Rationale:_ a release date is fixed for each run, so two subset runs, or
    a subset run and a GB run, on the same date would overwrite one summary.
    The name must be readable, so a hash of the LA list was rejected. Sorting
    makes the name the same for any order of the LAs.

### Branch and stacking

The branch `531_gbComparisonSummary` is made from
`526_compareVersionsInPipeline` (PR #534). The stack is #521 -> #534 ->
#531. The reader needs the `comparisons_dir` config key and the JSON keys
that #534 adds. The PR targets `526_compareVersionsInPipeline` and is
retargeted when #534 merges.

## Alternatives considered

- **Read the pipeline failure CSVs to label a missing JSON** (agreed in the
  first round, then rejected). `pipeline.py` never writes them (see Open
  questions). Fixing #521 is not in our scope, and the output check is
  correct also without the logs.
- **Run the summary at the end of `orbit_pipeline.py`.** Not possible:
  `orbit launch` does not wait for the job.
- **A wide table, one row for each LA.** Easier in a spreadsheet, but each
  JSON key adds five columns.
- **A column for each tech, or a second parquet for tech counts.** Columns
  for each tech change the schema with the data. A second file must be kept
  together with the first.
- **Relative-change thresholds in `base.yaml` for rows, UPRNs, clusters,
  area and tech share.** Rejected for now (decision 7). It can come back
  after two GB runs.
- **Use the JSON with the latest old date, or keep a row for each JSON.**
  See decision 8.
- **Other file-name rules** (decision 11). A short hash of the sorted
  slugs is unique but not readable. A time in the name never overwrites,
  but files pile up when a run is repeated. One summary for each release
  over every LA found on S3 cannot show an LA that wrote nothing. A
  `--label` flag and a `sample` name for the six sample LAs were not added
  for now.
- **Baseline numbers and the extra JSON keys in #534 itself.** Baseline
  numbers need a new loading path in `compare_versions.py`, which is not
  the integration job of #534. They moved to #535, together with the
  cross-LA peer checks. Only the three keys that copy existing numbers went
  into #534.
- **Put the JSON extension into #488 (checks as data).** It was considered
  as the home for the baseline numbers. #535 holds them now and refers to
  #488.

## Out of scope

- Baseline numbers and cross-LA peer checks (#535)
- New flag thresholds for cluster counts or area
- Running the comparisons (#526)
- Splitting or refactoring `compare_versions.py` (#474, #488)
- Fixing the failure-log code of #521
- Running the summary automatically after the Orbit jobs (#536)
- Alerts or notifications

## Open questions

- **The #521 failure CSV is never written.** In `pipeline.py`,
  `failed["local_authority"] = la` puts a value into an empty DataFrame,
  which gives 0 rows, so `len(failed) > 0` is never true. The S3 folder
  `outputs/logs/pipeline_failure/` is empty. Best guess: report it in the
  "Found while testing" section of PR #534 for the #521 author. Decision 3
  does not depend on it.
- **How many top changes does the markdown show?** Best guess: 20, as a
  config value. Change it after the first GB run.
- ~~**Is one S3 listing for each LA fast enough?**~~ Answered: yes. A GB
  summary (361 LAs) took 128 s on 2026-10-14.
- **Does the `(null)` tech count in the sort?** Best guess: yes. A large
  change in UPRNs with no tech is a signal.
- ~~**The newest file needs the S3 write time.**~~ Answered:
  `s3fs.S3FileSystem().ls(dir, detail=True)` returns `LastModified`, so no
  new helper in `s3_utils` is necessary.
- **Two sets of more than 3 LAs can get the same file name** when they
  have the same count and the same first and last slug (decision 11). The
  second summary then overwrites the first. Best guess: rare, and the
  markdown lists the LAs. If it happens, add a short hash or a `--label`.

## Verification

From the acceptance criteria of
https://github.com/nestauk/asf_heat_pump_suitability/issues/531:

- [ ] The table has one row for each expected LA and stage in the release
- [ ] A baseline shows as `baseline`, not as a change
- [ ] A missing JSON for a stage with no output shows as `stage failed`
- [ ] A missing JSON for a stage with output shows as `comparison missing`
- [ ] Rows are sorted by the largest relative change, and each row names
      that metric
- [ ] A second JSON for the same LA and stage is counted and listed, and
      the newest is used
- [ ] A version 0 JSON and a version 1 JSON are both read
- [ ] The markdown summary gives run-health counts and the top changes
- [ ] A subset run writes files named for its LAs and does not overwrite
      the GB summary of the same release
- [ ] With `--save` the parquet and markdown are in
      `outputs/comparisons/{release_date}/` (writes to S3, ask Aidan first)
- [ ] A run over the sample LAs gives the expected rows

Unit tests cover the status logic, the flattening (version 0 and 1), the
sort key and the duplicate rule, with JSON fixtures copied from the real
East Lothian files. The S3 listing and lookups are mocked.
