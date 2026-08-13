---
title: Pin a single release date across all stages in run_pipeline.sh
status: in-review
github_issue: https://github.com/nestauk/asf_heat_pump_suitability/issues/460
pr: https://github.com/nestauk/asf_heat_pump_suitability/pull/471
asana:
created: 2026-08-12
---

## Problem

Since [#413](https://github.com/nestauk/asf_heat_pump_suitability/issues/413),
each pipeline stage resolves `--release_date` independently, defaulting to
today. `run_pipeline.sh` invokes all five stages per local authority without
passing a date, so a run that crosses midnight silently splits one release
across two dated S3 directories, and the first stage invoked after midnight
fails with `FileNotFoundError` because the upstream output lives under the
previous day's release directory. Raised by @sofiapinto in the review of #413.

## Proposal

Resolve the release date once at the top of `run_pipeline.sh` and pass the
same `--release_date` to every stage invocation. Decisions (interviewed
2026-08-12):

1. **Interface: a `--release_date` flag** (`./run_pipeline.sh --release_date 20260812`), parsed with a small `while`/`case` loop. Rationale: mirrors the
   Python stages' own interface, so the script and the stages it wraps take
   the date the same way. Unknown arguments error out.
2. **Default + validation via `save_utils.get_str_release_date`**, called in a
   Python one-liner at the top of the script. Rationale: the `%Y%m%d` format
   stays defined only in `config["constant"]["release_date_format"]` (no
   duplication in bash), and an invalid user-supplied date fails fast before
   any stage runs, instead of at stage 1's argparse. Cost is one extra Python
   startup per pipeline run, negligible against the pipeline's runtime.
3. **Fix the pre-existing `succeeded = 0` bug in the same PR**, as its own
   commit. The spaces make bash execute a command named `succeeded` rather
   than assign; the counter only works today because `$((succeeded+1))`
   treats the unset variable as 0. One character, same file, easy to review
   separately.

## Alternatives considered

- **Positional argument** (`./run_pipeline.sh 20260812`): smallest diff, but
  inconsistent with the `--release_date` flag every stage already uses.
  Rejected in interview.
- **Environment variable** (`RELEASE_DATE=... ./run_pipeline.sh`): no parsing
  at all, but less discoverable and easy to leave set in a shell by accident.
  Rejected in interview.
- **Pure bash default** (`${1:-$(date +%Y%m%d)}`): avoids a Python call but
  duplicates the config's date format in bash and defers validation of a
  user-supplied date to stage 1. Rejected in favour of reusing
  `get_str_release_date`.

## Out of scope

- `create_manifest.py` — it intentionally leaves the `{release_date}`
  placeholder unresolved (identity: only affects the directory, not the
  suffix), so it takes no date.
- Any change to the five stage scripts themselves; their `--release_date`
  handling (added in #413) is unchanged.

## Open questions

- How to verify manually: is a dry run against one sampling LA (e.g.
  Plymouth) with a pinned past date acceptable, or is inspection plus the
  fail-fast check on a bad date enough? There is no bash test infrastructure
  in this repo.

## Verification

- [x] `./run_pipeline.sh` with no arguments resolves today's date once and
      passes the identical `--release_date` value to all five stage
      invocations for every local authority.
- [x] `./run_pipeline.sh --release_date 20260801` pins all stages to that
      date.
- [x] An invalid date (e.g. `--release_date 2026-08-01`) exits with
      `get_str_release_date`'s error before any stage runs.
- [x] An unknown argument exits with a usage error.
- [x] An empty flag value (`--release_date ""` or a trailing `--release_date`)
      exits with a usage error rather than silently defaulting to today
      (found in review).
- [x] `succeeded` counts correctly (assignment bug fixed) and the final
      summary line reports the right number.
