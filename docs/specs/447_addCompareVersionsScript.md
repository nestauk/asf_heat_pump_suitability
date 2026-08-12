---
title: Cross-version comparison script — base checks
status: draft
github_issue: https://github.com/nestauk/asf_heat_pump_suitability/issues/447
pr: https://github.com/nestauk/asf_heat_pump_suitability/pull/451
asana: https://app.asana.com/1/5571817120120/project/1214222223606748/task/1216704619895657
created: 2026-07-23
---

## Problem

No way to compare two versions of any pipeline output. When a methodology
change or an input re-release happens, there is no systematic way to tell
expected drift from silent breakage, or to see which commits are responsible
for a given change.

Second issue in the comparison-script stack (see
`docs/specs/pipeline_validation_checks_issues.md`): the run manifest (#440,
implemented on the branch this one stacks on) records `git_commit` and
`input_versions` per output, which this script reads from day one. The
follow-on issues (cluster geometry checks, distribution checks, thresholding)
each extend the report this issue creates.

## Proposal

A new module, `pipeline/validate/compare_versions.py`, takes a pipeline
stage and a local authority, plus optionally two dated versions
(`release_date` folders) — omitted, it compares the latest two dated
versions found for that stage and LA, so the routine current-vs-previous
comparison runs with no version bookkeeping — and an optional trigger
(`methodology_change` or `input_release`) supplied by whoever runs the
comparison — they know why they're running it, and that's the one piece of
context the script can't infer. The trigger deliberately has no default:
omitted, the report presents raw numbers only, skipping rubric
interpretation (including the rubric-keyed tolerance warnings). It reports:

- row/UPRN count delta
- schema diff
- UPRN churn (added/removed/retained, joined on UPRN)
- tech-assignment transition matrix for the decision-tree stage, plus
  per-tech marginal counts for both the UPRN-level and building-level
  outputs
- a module-scoped commit log between the two versions' recorded commits,
  read from each version's run manifest (`{output_basename}.manifest.json`,
  #440) — so the reader sees which commits are the candidate cause of a
  divergence, without unrelated commits cluttering the list

Tolerances are configurable via `base.yaml`, not hardcoded.

Decisions settled during kickoff interview (2026-07-23):

- **Lives in `pipeline/validate/compare_versions.py`.** The
  `pipeline/validate/` subpackage was created by #434 on a parallel branch
  off `dev` that hasn't merged; this branch creates it independently (the
  only overlap at merge is a trivial `__init__.py`), so validation tooling
  converges in the home the initiative planned for it.
- **Report is a local markdown file plus a console summary**, not an S3
  upload. A comparison is run ad-hoc by a human deciding whether drift is
  expected; a read-only diagnostic shouldn't need bucket write access.
  Follow-on issues keep extending this report (and thresholding will group
  flagged checks), which is why it's a structured file rather than log lines
  only.
- **The tech transition matrix uses the UPRN-level decision-tree output
  only.** UPRNs are stable external identifiers — the same join key the
  churn check uses — so the matrix reads "this property moved from tech A to
  tech B". Building IDs are pipeline-generated and can renumber between
  versions, which would pollute a building-level matrix with fake
  transitions.
- **Follow-on issues are filed when their work starts**, not upfront — their
  bodies reference this issue's branch name, and the thresholding issue
  still carries a confirm-criteria-first flag (drafts doc, issues 4–6).

Decisions added 2026-08-10, reconciling the Asana high-priority metric list
(see the drafts doc's header note); implemented 2026-08-12:

- **Version pair becomes optional**, defaulting to the latest two dated
  versions found for the stage and LA — the "comparison happens
  automatically" steer. Output versions only; auto-detecting _input_
  prefixes stays with #429.
- **Trigger becomes optional with no default.** Omitted, the report is raw
  numbers with no rubric interpretation — a silently-defaulted trigger would
  grade drift against the wrong rubric, which is worse than no rubric.
- **Per-tech marginal counts at both UPRN and building level** join the
  transition matrix in the report. This does not reopen the rejected
  building-level _matrix_: marginal counts are per-version tallies needing
  no cross-version key, so building-ID instability doesn't pollute them.

## Alternatives considered

- **`pipeline/run/compare_versions.py`** — rejected; avoids the trivial
  `__init__.py` merge overlap with #434 but leaves the comparison script
  outside the validation subpackage once #434 lands.
- **Uploading the report to S3 next to the newer version's outputs** —
  deferred, not rejected; can bolt on later if reports prove worth
  archiving. Local-first keeps the diagnostic read-only.
- **Console/log output only** — rejected; issues 4–6 keep extending the
  report and thresholding wants flagged checks grouped, which gets unwieldy
  as pure log lines.
- **Building-level transition matrix as well** — rejected for v1;
  building-ID instability between versions creates spurious transitions.
- **Storing the comparison trigger in the manifest** — rejected during the
  stack's design (drafts doc): the trigger is supplied by whoever runs the
  comparison, who knows at that moment which rubric applies.
- **Defaulting the trigger to `methodology_change`** — rejected 2026-08-10
  when the trigger became optional; the two rubrics have near-opposite
  expectations for the same signals, so a forgotten flag would silently
  grade an input release against the methodology rubric (or vice versa).
  Omitting the trigger degrades to raw numbers instead.

## Out of scope

- Cluster geometry checks (stacked follow-on issue)
- Categorical mix deltas, numeric drift (stacked follow-on issue)
- Turning deltas into an automatic expected/suspicious flag (stacked
  follow-on issue; thresholding criteria still need confirming with
  Aidan/Roisin before filing)
- PSI or other distribution-drift scalars
- Uploading reports to S3 (local markdown + console only for v1)

## Open questions

Both resolved during implementation (2026-07-24):

- **Stage→module mapping for the commit log**: a curated
  `compare_versions.stage_module_paths` dict in `base.yaml`, mirroring the
  stages of `STAGE_INPUT_KEYS` in `utils/manifest_utils.py` (a test pins the
  key sets equal and every path's existence). Each stage lists its entrypoint
  plus the pipeline modules it directly drives; cross-cutting code
  (`getters/`, `utils/`, `config/`) is excluded to keep the log scoped to
  candidate causes.
- **Missing manifest** (pre-#440 outputs): the report degrades gracefully —
  the input-version and commit-log sections are replaced by a note naming
  which version lacks a manifest; all data-level checks still run. A manifest
  recording an `unknown` commit, or commits absent from local git history,
  degrade the commit-log section the same way.

Implementation decisions within the spec's frame:

- **Tolerances** live in `compare_versions.tolerances` in `base.yaml`, keyed
  by trigger rubric, with one v1 tolerance: `max_removed_uprn_share`
  (defaults 0.05 for both rubrics). Exceeding it adds a warning line to the
  report and console — a per-check note, not the grouped expected/suspicious
  flagging the thresholding follow-on owns.
- **Input version changes** recorded in the two manifests are diffed into
  their own report section (the manifest's `input_versions` is read from day
  one, per the Problem section).
- The report is written to `outputs/comparisons/` (gitignored) by default;
  `--report_dir` overrides.

## Verification

- [x] Runs against two dated version folders for one LA and one stage
- [x] Accepts a trigger input (`methodology_change` / `input_release`);
      report states which rubric it was read against
- [x] Reports row/UPRN count delta, schema diff, and UPRN churn
      (added/removed/retained)
- [x] Tech-assignment transition matrix for the decision-tree stage
      (UPRN-level output)
- [x] Reads `git_commit` from each version's manifest and embeds the
      module-scoped commit log between the two recorded commits in the
      report
- [x] Unit tests cover: no drift, expected UPRN churn, unexpected UPRN loss

Added 2026-08-10 (implemented 2026-08-12):

- [x] When versions are omitted, defaults to the latest two dated versions
      found for that stage and LA — S3 glob over the dated output
      directories, validated by the strict release-date parser; passing
      exactly one date is a CLI error naming both options
- [x] Trigger is optional with no default; omitted, the report presents raw
      numbers with no rubric interpretation or tolerance warnings — pinned
      by `test_omitted_trigger_reports_raw_numbers_without_rubric`
- [x] Per-tech marginal counts reported for both the UPRN-level and
      building-level decision-tree outputs (building-level dataset named in
      `compare_versions.decision_tree_buildings_dataset` in `base.yaml`; a
      missing building output degrades its section to a note)
- [x] Unit tests cover: default version selection picks the latest two;
      omitted-trigger report carries no rubric labels
      (`TestGetTupleDefaultReleaseDates`, `TestGenerateListReleaseDates`,
      `TestGenerateDfTechCounts` + report-level tests; 61 tests pass)
