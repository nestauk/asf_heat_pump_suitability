---
title: Cross-version comparison script — base checks
status: draft
github_issue: https://github.com/nestauk/asf_heat_pump_suitability/issues/447
pr:
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
stage, a local authority, two dated versions (`release_date` folders), and an
explicit trigger (`methodology_change` or `input_release`) supplied by
whoever runs the comparison — they know why they're running it, and that's
the one piece of context the script can't infer. It reports:

- row/UPRN count delta
- schema diff
- UPRN churn (added/removed/retained, joined on UPRN)
- tech-assignment transition matrix for the decision-tree stage
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

## Out of scope

- Cluster geometry checks (stacked follow-on issue)
- Categorical mix deltas, numeric drift (stacked follow-on issue)
- Turning deltas into an automatic expected/suspicious flag (stacked
  follow-on issue; thresholding criteria still need confirming with
  Aidan/Roisin before filing)
- PSI or other distribution-drift scalars
- Uploading reports to S3 (local markdown + console only for v1)

## Open questions

- The stage→module mapping for the module-scoped commit log (which repo
  paths count as "the relevant stage's module" for `git log A..B -- <paths>`)
  — expected to be a curated dict mirroring `STAGE_INPUT_KEYS` in
  `utils/run_manifest.py`; exact path lists left to `/implement`.
- How the script behaves when a version's manifest is missing (pre-#440
  outputs have none) — presumably degrade gracefully: skip the commit-log
  section with a note rather than fail the whole report; left to
  `/implement`'s judgment.

## Verification

- [ ] Runs against two dated version folders for one LA and one stage
- [ ] Accepts an explicit trigger input (`methodology_change` /
      `input_release`); report states which rubric it was read against
- [ ] Reports row/UPRN count delta, schema diff, and UPRN churn
      (added/removed/retained)
- [ ] Tech-assignment transition matrix for the decision-tree stage
      (UPRN-level output)
- [ ] Reads `git_commit` from each version's manifest and embeds the
      module-scoped commit log between the two recorded commits in the
      report
- [ ] Unit tests cover: no drift, expected UPRN churn, unexpected UPRN loss
