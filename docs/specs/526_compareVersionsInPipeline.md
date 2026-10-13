---
title: Run the cross-version comparison as part of a pipeline run
status: in-review
github_issue: https://github.com/nestauk/asf_heat_pump_suitability/issues/526
pr: https://github.com/nestauk/asf_heat_pump_suitability/pull/534
asana:
created: 2026-10-01
---

## Problem

The cross-version comparison (`pipeline/validate/compare_versions.py`, from
#447, #472 and #519) shows how one version of a stage output differs from
the version before it. Today it runs only by hand, for one stage and one local
authority (LA) at a time:

- Without dates, it compares the latest two release dates on S3. In a
  script, "latest two" can be wrong: a run of an older release date
  (`--release_date` set to the past) or another run that writes a later
  date folder means the latest two are not this run and the one before it.
- With fewer than two versions it raises `FileNotFoundError`. On the first
  GB run, about 350 LAs have no earlier version.
- It writes the report and plots to a local `report_dir`. #521 runs the
  pipeline in Orbit containers, and a container's local disk is lost when
  the container stops.
- It returns no machine-readable result, so nothing can collect the results
  across LAs.

So at GB scale nobody will see an unexpected change unless they run 5 stages
x 350 LAs by hand.

## Proposal

Each decision below was settled in the kickoff interview on 2026-10-01.

1. **The comparison runs inside `pipeline.py`, after an LA's five stages
   succeed.** It runs in the same Orbit container as the stages, so the
   comparisons run in parallel across chunks. `pipeline.py` calls
   `compare_versions` as a subprocess, the same way it calls each stage, so
   `compare_versions` needs no refactor to be called.
   _Rationale:_ a separate post-run script would run one after another from
   one machine, with slow geojson loads for each of 350 LAs.

2. **It is opt-in: `--compare` on `pipeline.py` and `orbit_pipeline.py`.**
   Without the flag a run behaves exactly as it does now. The Orbit launcher
   passes the flag to every chunk.
   _Rationale:_ test runs and the first GB baseline do not pay for extra S3
   loads, or risk new failures, unless someone asks.

3. **All five stages are compared:** `uprns`, `add_features`,
   `decision_tree`, `cluster`, `compute_contextual_features`.
   _Rationale:_ full coverage per LA. The cost is about 10 output loads per
   LA, plus the geometry reloads for the cluster stages, until #488 reads
   each object once.

4. **The old version is the latest earlier release date, chosen per stage.**
   `compare_versions` accepts `--new_release_date` on its own. For that
   stage it lists the release dates on S3 and takes the most recent one
   before the new date. Giving both dates still works as now. Giving only
   `--old_release_date` stays an error.
   _Rationale:_ a stage that failed in an earlier run still finds its own
   last good version. One LA's five reports can therefore compare against
   different old dates; each report and JSON states its old date.

5. **No earlier version means baseline, not error.** When a stage has no
   release date before the new one, `compare_versions` logs "baseline,
   nothing to compare", writes a JSON with `status: baseline`, and exits 0.
   _Rationale:_ the later GB table can then tell "no earlier version" apart
   from "comparison never ran or failed".

6. **The trigger is an optional pass-through.** `pipeline.py` and
   `orbit_pipeline.py` accept `--trigger` (`methodology_change` or
   `input_release`) and pass it to every comparison. Without it the reports
   show the numbers with no tolerance warnings, as now. `--trigger`
   without `--compare` is an argument error in both launchers (review item
   S4, 2026-10-08).
   _Rationale:_ the trigger is a fact only the person who starts the run
   knows, which is why the run manifest does not store it, and a test run
   with `--compare` should not need to invent one. Only the comparisons use
   the trigger, so `--trigger` alone would give a run with no tolerance
   checks and no message.

7. **Results go to S3 in the new release's folder:**
   `outputs/data/{la}/{release_date}/comparisons/`. For each stage this holds
   the markdown report, its plots, and a JSON of headline numbers. File
   names keep the current stem, `{stage}_{la}_{old}_vs_{new}`; a baseline
   JSON uses `{stage}_{la}_{new}_baseline`. The folder is a new
   `compare_versions` config entry in `base.yaml`, not a hard-coded path.
   The upload uses `save_utils.upload_file_to_s3`. Following the repo
   convention that scripts write to S3 only with `--save`, `compare_versions`
   gets a `--save` flag and `pipeline.py` passes it. Without `--save`, the
   report, plots and JSON go to the local `report_dir` only, as now.
   _Rationale:_ everything about one run lives in one folder, and is deleted
   with it. `--save` keeps the hand-run CLI working on a laptop.

8. **The JSON holds headline numbers only.** It has fourteen keys:
   `schema_version` (1), `stage`, `local_authority`, `release_date_old`
   (null for a baseline), `release_date_new`, `trigger`, `status`
   (`compared` or `baseline`), `counts`, `uprn_churn`, `tech_counts`,
   `cluster_counts`, `total_area`, `uprns_missing_clusters` and
   `tolerance_warnings`. The number keys hold what the report already
   builds: `generate_dict_count_delta`, `generate_dict_uprn_churn`,
   `generate_df_tech_counts` as a list of rows (UPRN level,
   `decision_tree` only), `generate_dict_cluster_count_delta` and
   `generate_dict_total_area_delta` (clusters layer only), and
   `generate_dict_uprns_missing_clusters` for each version under `old` and
   `new` (`compute_contextual_features` only, a version is null when its
   add*features output is missing). A key that does not apply (a baseline,
   or a key for another stage) is null, not missing, so every JSON has the
   same keys. `schema_version` is raised whenever keys change; a JSON
   without it counts as version 0.
   \_Rationale:* read by the GB table in #531, which sorts and flags LAs from
   these files without loading the outputs again. The keys copy only
   numbers the script already computes from data it already loads.
   Baseline numbers and cross-LA peer checks are #535, and the
   check-results-as-data refactor is #488. The same keys in every file let
   the table read them all the same way. Keys added 2026-10-13, agreed in
   the #531 kickoff, before any production run wrote a JSON.

9. **A failed comparison never fails the LA or the run.** `pipeline.py` logs
   a comparison's non-zero exit code with the stage and LA, adds it to a
   separate "comparisons failed" list in the end-of-run summary, and goes
   on. The LA still counts as succeeded. Comparison failures are not added
   to the stage-failure CSV that #521 writes to S3, so that file still means
   "a stage failed".
   _Rationale:_ the comparison is a report, not a gate. In the two-tier
   validation design, gates belong in the stages (Tier 1) and the
   comparison is Tier 2.

10. **`compare_versions` accepts an LA's official name as well as its slug.**
    It turns `--local_authority` into a slug with
    `local_authority.make_str_slug`, the helper the stages use, so "East
    Lothian" and `east_lothian` find the same outputs. Found during
    implementation: `pipeline.py` passes official names, and the old
    lowercasing gave "east lothian", which matches no S3 path. A value that
    is already a slug (only lowercase letters, digits, `_` and `-`) stays
    unchanged, because `make_str_slug` would turn the hyphen in a multi-LA
    slug such as `glasgow-midlothian` into an underscore (code review,
    2026-10-13).

11. **An optional `--old_release_date` pins every comparison to one
    release.** `pipeline.py` and `orbit_pipeline.py` accept it and pass it
    to every comparison. It needs `--compare` and must be earlier than the
    run's release date; both launchers stop with an argument error
    otherwise. A stage with no output at that date fails its comparison and
    goes into the "comparisons failed" list; there is no fallback to
    another date. Without the flag, decision 4 applies. Decided 2026-10-13.
    _Rationale:_ a test run saved after the last production run moves each
    stage's latest earlier date, so the next production run would compare
    against the test run. A fixed old date also gives release-to-release
    reports across all LAs. A silent fallback would label a comparison with
    a date the user did not ask for.

### Branch and stacking

The branch `526_compareVersionsInPipeline` is off `515_loop_pipeline` (#521),
which adds `pipeline.py` and `orbit_pipeline.py`. It has
`519_countUprnsMissingClusters` (#520) merged in, because this work edits
`compare_versions.py` too. The PR is stacked on #521; retarget it to `dev`
when #521 merges.

### Change outside the main scope

The branch also corrects the `make_str_slug` docstring in
`pipeline/transform/local_authority.py`. The docstring said the function
uses hyphens, but it uses underscores within a name and joins a list of
names with hyphens. The fix is here because decision 10 makes
`compare_versions` depend on this helper to find each LA's outputs, so a
wrong description of it now affects this feature.

## Alternatives considered

- **A separate post-run script over S3.** It keeps `pipeline.py` simple, but
  runs 350 LAs one after another from one machine, and someone must remember
  to run it after every Orbit chunk finishes. Rejected for (1).
- **On by default with `--no_compare`.** Nothing is forgotten, but every test
  run gets slower and pays for reports nobody reads. Rejected for (2).
- **Decision tree and final stage only, or the final stage only.** Cheaper
  per LA. Rejected for (3): the user chose full coverage.
- **One shared old date for all five stages.** The five reports always line
  up, but an LA with a partial earlier run gets no comparison at all.
  Rejected for (4).
- **Skip baselines silently in `pipeline.py`.** Less code, but a missing JSON
  then means either baseline or failure. Rejected for (5).
- **`--trigger` required with `--compare`.** Every automated report gets
  tolerance warnings, but test runs must invent a reason. Rejected for (6).
- **A separate tree, `outputs/comparisons/{release_date}/{la}/`.** One prefix
  lists every LA for a run, which suits the later GB table, but the reports
  sit away from the outputs they describe. Rejected for (7); the GB table can
  list `outputs/data/*/{release_date}/comparisons/` instead.
- **Rename `pipeline.py`'s `--release_date` to `--new_release_date`.** It
  would match `compare_versions`. Rejected for (11): the pipeline writes
  one release, every stage script takes `--release_date`, and #521 owns and
  documents that interface. `run_comparison` maps the one name to the other
  in one place.
- **Fall back to the latest earlier date when `--old_release_date` has no
  output for a stage.** Every stage would get a report, but some would
  compare against a date the user did not ask for. Rejected for (11).
- **Always save to S3, with no `--save` flag.** One less flag, and no
  report stays on a laptop by mistake. Rejected for (7) on 2026-10-07:
  `pipeline.py` always passes `--save`, so the flag costs a pipeline run
  nothing, and it stops exploratory hand runs from writing reports into a
  release folder next to the official ones.
- **The full GB summary table in this PR.** One PR delivers the whole loop,
  but it overlaps #488 and roughly doubles the PR. Deferred to #531.
- **Run comparisons in CI on pull requests.** A PR produces no outputs on
  S3, and running the pipeline in CI needs AWS access from GitHub Actions
  and minutes of runtime. Out of scope.

## Out of scope

- The GB-wide summary table built from the per-LA JSONs (#531, alongside
  #488).
- Running comparisons in CI or on pull requests.
- Tier 1 pandera gates for stages 2 to 5.
- Reading each S3 object once and separating checks from rendering (#488),
  and decomposing `compare_versions.py` (#474).
- The `{local_authority}` / `{local_authorities}` path placeholder mismatch
  (#514, #425); the comparison keeps using the existing path helpers.

## Open questions

- Does the Orbit image have a `git` binary? `.orbitignore` does not exclude
  `.git`, but without `git` the run manifest records `git_commit: unknown`
  and the report's commit-log section is reduced to a note. Best guess:
  check on the first real Orbit run with `--compare`; fixing the image is a
  separate issue. Still open: no Orbit run was made for this PR.
- How much does comparing five stages add to each LA's runtime on Orbit?
  Best guess: measure it on the first real Orbit run with `--compare`; if it
  is large, #488's single read is the fix, not fewer stages. Still open: no
  Orbit run was made for this PR.
- `orbit_pipeline.py` launches chunks and does not wait for them, so nothing
  knows when all comparisons for a run are done. Best guess: the follow-up
  GB table is run by hand once the Orbit jobs finish.

## Verification

From the issue's acceptance criteria:

- [ ] Without `--compare`, a pipeline run behaves exactly as before
- [ ] With `--compare`, each successful LA gets one comparison for each of
      the five stages
- [ ] Each comparison picks the latest earlier release date for its own stage
- [ ] A stage with no earlier version writes a JSON with `status: baseline`
      and exits 0
- [ ] The report, plots and JSON for each comparison are on S3 under
      `outputs/data/{la}/{release_date}/comparisons/`
- [ ] `--trigger`, when given, reaches every comparison, and its tolerance
      warnings appear in the report and in the JSON
- [ ] A comparison that exits non-zero is logged, and the LA still counts as
      succeeded
- [ ] `orbit_pipeline.py` passes `--compare` and `--trigger` to every chunk
- [ ] `--old_release_date` reaches every comparison in both launchers; it
      is an argument error without `--compare` or when not earlier than the
      release date; a stage with no output at that date fails its comparison
- [ ] ~~An Orbit run over the sample LAs with `--compare` leaves the expected
      reports and JSONs on S3~~ Not run (2026-10-13). `orbit_pipeline.py`
      only passes the flags on to `pipeline.py`, and a `pipeline.py --compare` run for East Lothian wrote all five stages' reports and
      JSONs to `east_lothian/20261013/comparisons/`. A real Orbit run was
      judged overkill

Tests follow the high-signal rule: the old-date choice (including baseline)
and the JSON content get unit tests; the subprocess wiring in `pipeline.py`
is checked by the acceptance run, not by mock-heavy tests.
