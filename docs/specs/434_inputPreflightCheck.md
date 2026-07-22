---
title: Input preflight check before pipeline runs
status: draft
github_issue: https://github.com/nestauk/asf_heat_pump_suitability/issues/434
pr:
asana:
created: 2026-07-22
---

## Problem

Nothing verifies S3 input paths exist before a pipeline run starts. A missing
or misconfigured path only surfaces deep inside a stage once a lazy S3 read
attempts it — as happened when an input dataset was reorganised into new
dated prefixes and the pipeline crashed mid-run on a single missing file,
well after other stages had already completed for that local authority.

## Proposal

A new module, `pipeline/validate/check_inputs.py`, recursively walks
`config["data"]` (every S3 input path used by the five live pipeline
entrypoints), checks each leaf exists in S3, and reports **all** missing
paths in one pass rather than failing on the first one encountered deep in a
stage. `run_pipeline.sh` calls it as its very first step (right after `cd`
to the repo root, before the local-authority loop begins) and hard-exits if
it fails — the same exit-code-check idiom the script already uses for its
`create_manifest.py` step.

Decisions settled during kickoff interview (2026-07-22):

- **Scope is `config["data"]` only** (v2), not the deprecated
  `config["data_source"]` (v1) namespace. `data_source` is being phased out
  per CLAUDE.md and is only read by the standalone legacy script
  `ref_calculate_garden_size.py`, not by any of the five entrypoints
  `run_pipeline.sh` orchestrates — checking it would preflight paths the
  live pipeline never touches.
- **New subpackage `pipeline/validate/`**, not `pipeline/run/`. This is the
  first module in a validation subpackage that the pipeline-validation-checks
  initiative already has more work queued for (cluster geometry checks,
  distribution checks, thresholding — see
  `docs/specs/pipeline_validation_checks_issues.md`), so it gets a shared
  home from the start rather than living alongside unrelated run scripts.
- **Existence check, not a content read.** No `head_object`/`moto`/`Stubber`
  precedent exists in this repo; the closest reusable building block is
  `utils/s3_utils.py`'s `fetch_list_file_paths_from_s3_folder`
  (`list_objects_v2`-based). `/implement` should add a lightweight
  existence-check helper alongside it (`list_objects_v2` with `MaxKeys=1`
  works for both a folder prefix and an exact file key, unlike
  `head_object`, which only works for exact keys).
- **Templated paths are truncated to their prefix**, not resolved per grid
  square/LA — e.g. `.../opmplc_essh_gb/20260708/data/{square}/{square}_{layer}.shp`
  truncates to `.../opmplc_essh_gb/20260708/data/` at the first `{` token.
  Per-square resolution needs the LA→squares mapping and is explicitly out
  of scope (below).

## Alternatives considered

- **Also checking `config["data_source"]`** — rejected; it would preflight
  paths only a legacy standalone script reads, expanding scope for no
  benefit to the five live entrypoints this issue is actually about.
- **`pipeline/run/check_inputs.py`** (flat, alongside other entrypoint
  scripts) — rejected in favour of a new `pipeline/validate/` subpackage,
  since more validation checks are already planned to follow this one.

## Out of scope

- Per-square existence checks (would need the LA→squares mapping;
  prefix-level only for v1)
- Auto-detecting the latest dated prefix (tracked separately in #429)

## Open questions

- ~~Whether the existence-check helper belongs in `utils/s3_utils.py` or
  stays inline in `pipeline/validate/check_inputs.py`~~ — resolved at
  implementation: placed in `utils/s3_utils.py` as
  `get_bool_s3_path_exists`, alongside the existing `list_objects_v2`-based
  helper, reusable by future validation checks.
- Noticed but not in scope: `run_pipeline.sh:19` has a pre-existing bug
  unrelated to this issue (`succeeded = 0` — invalid bash syntax due to
  spaces around `=`). Flagging so it isn't lost, not fixing here.

## Verification

- [x] Checks every configured S3 input path under `config["data"]`,
      resolving templated paths to their prefix (27 paths checked in the
      acceptance run)
- [x] Reports all missing paths in one pass and exits non-zero — the
      acceptance run surfaced a genuinely missing configured input
      (`inputs/geodata/council_tax/PLYMOUTH_CTBANDS_ONSUD_202512.csv`; the
      S3 folder is empty) and exited 1
- [x] Wired into `run_pipeline.sh` as its first step, before the
      local-authority loop
- [x] Unit tests cover at least one missing-path case and one all-present
      case (`pipeline/validate/tests/test_check_inputs.py`)
