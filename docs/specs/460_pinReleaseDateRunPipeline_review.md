---
title: Review findings for pinning release date in run_pipeline.sh
status: complete
github_issue: https://github.com/nestauk/asf_heat_pump_suitability/issues/460
spec: docs/specs/460_pinReleaseDateRunPipeline.md
reviewed: 2026-08-13
passes: 2 (review + bounded re-review of fixes; simplify pass in between)
---

## Scope

Five-axis review (correctness, readability, architecture, security,
performance) of `dev...460_pinReleaseDateRunPipeline`, chartered by the
spec's Verification section and repo conventions. Diff under review:
`run_pipeline.sh` (+43/−6) and the spec itself.

## Findings

### Fixed in this cycle

1. **Empty `--release_date` value silently defaulted to today and ran the
   pipeline** (correctness, new code). `[ $# -ge 2 ]` accepted `""`, then the
   `${release_date:+...}` expansion dropped the empty argument, so
   `get_str_release_date` saw no input and returned today. Confirmed live:
   the review probe unintentionally started a real pipeline run (killed at
   ~35s; verified no S3 writes resulted). Fixed by rejecting empty values at
   parse time — commit `dc90e67`; spec criterion added in `2ac7111`.

### Open — Aidan's call

2. **Invalid date prints a full Python traceback** (readability/operator UX).
   `--release_date 2026-08-01` exits 1 before any stage, per spec, but the
   useful one-line ValueError is wrapped in ~20 lines of traceback. Optional
   cleanup: catch ValueError in the one-liner and `sys.exit(str(e))`. Left
   open because it trades one-liner minimalism for output polish.

### FYI — no action

3. `${succeeded:-0}` in the summary line is redundant now the counter is
   always initialised. The line is untouched by this PR; harmless.
4. `create_manifest.py` runs even when zero local authorities succeed —
   pre-existing behaviour, unrelated to this diff.

## Simplify pass

No cleanups warranted: guards are one condition per line, no duplication,
style matches the file. Merging the two `|| usage` guards into a compound
line was considered and rejected as less readable.

## Gates after fixes

- `uv run pytest`: 35 passed.
- `uv run pre-commit run --files run_pipeline.sh`: green.
- `bash -n run_pipeline.sh`: clean.
- All four fail-fast argument paths executed directly (each exits 1 before
  any stage); valid-path pinning re-proven via stubbed dry run
  (35 × one distinct date).

## Follow-up issues

None proposed — findings 3 and 4 are too small to carry an issue each.
