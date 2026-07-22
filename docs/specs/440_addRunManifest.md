---
title: Run manifest for pipeline output lineage
status: draft
github_issue: https://github.com/nestauk/asf_heat_pump_suitability/issues/440
pr:
asana:
created: 2026-07-22
---

## Problem

Nothing records which input versions, git commit, or parameters produced a
given pipeline output. Only the final geojson carries any metadata (creation
date + licence — no lineage). The existing `create_manifest.py` is unrelated:
it's a front-end dataset listing (local authority + geojson URL), not run
lineage, and already owns the filename `manifest.json` in the front-end
bucket.

This is the first issue in the cross-version comparison-script stack (see
`docs/specs/pipeline_validation_checks_issues.md`) — the comparison script,
and everything after it in that stack, depends on this landing first.

## Proposal

Each of the five pipeline entrypoints (uprns, add_features, decision_tree,
cluster, compute_contextual_features) writes a companion JSON manifest
alongside its output, gated on `--save` exactly like the existing output
write.

Decisions settled during kickoff interview (2026-07-22):

- **Filename: `{output_basename}.manifest.json`, co-located** with the
  output it describes (e.g. `plymouth_domestic_uprns.parquet` →
  `plymouth_domestic_uprns.manifest.json` in the same S3 directory). Avoids
  any collision with `create_manifest.py`'s `manifest.json` and guarantees
  an unambiguous 1:1 pairing even in a directory holding multiple outputs.
- **`compute_contextual_features`'s double write**: it writes its geojson
  twice per run — once dated to the data-science bucket, once as an undated
  "latest" copy to the front-end bucket. Only the **dated data-science
  copy** gets a manifest; the undated front-end copy is overwritten every
  run and has no version history of its own to attach lineage to.
- **`decision_tree.py`'s non-standard save path is left untouched.** It
  writes its two outputs via raw `gdf.to_parquet(...)` rather than
  `save_utils.save_to_s3` (already flagged as an open question on another
  PR). Manifest writing bolts on alongside that existing write; refactoring
  it to `save_utils` first is out of scope here to avoid scope creep into
  unrelated code.
- **`input_versions` records raw resolved `config["data"...]` path strings**,
  not extracted version tokens. Dated prefixes use inconsistent formats
  today (`YYYYMMDD` folder vs `v{YYYYMM}` prefix, no shared convention), so
  a per-dataset regex would need bespoke rules with no common pattern to
  write once. The raw path string is unambiguous provenance the comparison
  script can diff directly, with zero parsing risk.
- **`git_commit` is captured fresh via `subprocess`** (`git rev-parse HEAD`
  or equivalent) — no existing pattern in this repo to reuse; this is new
  code.
- **`params` = CLI args only**: `local_authorities`, `release_date`, and
  (for `add_features` only) `detail`. `config["constant"]` thresholds
  (radii, `tolerance_m`, etc.) are excluded — they're static per code
  version and already implied by `git_commit`; including them would bloat
  every manifest with the entire config file for no lineage benefit.
- **Manifest-dict construction is a pure function**, tested directly on its
  return value (matching `utils/tests/test_save_utils.py`'s style) — no
  `tmp_path` or mocked-file-write precedent exists anywhere in this repo's
  test suite, so this issue doesn't introduce one.

## Alternatives considered

- **Fixed name `run_manifest.json` per directory** — rejected; ambiguous
  when a directory holds more than one output file.
- **Manifest for both compute_contextual_features writes** — rejected; the
  front-end "latest" copy has no version history to attach lineage to and
  would go stale/misleading immediately on the next run.
- **Refactor `decision_tree.py` to `save_utils.save_to_s3`** — rejected;
  unrelated concern already flagged elsewhere, would expand this issue's
  diff into code it doesn't need to touch.
- **Extracted version token via per-dataset regex** — rejected; no shared
  naming convention exists today to write a single regex against.

## Out of scope

- Auto-detecting input versions from S3 (tracked separately in #429); until
  it lands, `input_versions` is read from the dated prefixes already
  hardcoded in `config["data"]`
- Any reporting or comparison logic on manifest contents (the cross-version
  comparison script's job — first follow-on issue in this stack)

## Open questions

- What `git_commit` should record when `git rev-parse HEAD` isn't available
  (e.g. a deployed environment without a `.git` directory) — no fallback
  behaviour decided yet; deferred to `/implement`'s judgment (e.g. an
  `"unknown"` sentinel rather than a hard failure, since a run should still
  complete even if lineage is incomplete).

## Verification

- [ ] A manifest JSON is written next to each of the five entrypoints'
      outputs (uprns, add_features, decision_tree, cluster,
      compute_contextual_features), gated on `--save`
- [ ] Manifest includes: stage, local_authority, run_at, git_commit,
      input_versions, row_count, params
- [ ] Filename (`{output_basename}.manifest.json`) avoids collision with
      the existing `create_manifest.py` output
- [ ] Unit test covers manifest dict content for at least one entrypoint
