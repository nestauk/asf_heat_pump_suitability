---
title: Run manifest for pipeline output lineage
status: in-review
github_issue: https://github.com/nestauk/asf_heat_pump_suitability/issues/440
pr: https://github.com/nestauk/asf_heat_pump_suitability/pull/449
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
  unrelated code. _Superseded at review (2026-08-10) — see below._
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

Decisions settled during implementation (2026-07-22):

- **Manifest code lives in `utils/manifest_utils.py`** (1:1 tests in
  `utils/tests/test_manifest_utils.py`), keeping `save_utils` focused on
  output writing. Renamed from `run_manifest.py` in review to match the
  `{domain}_utils.py` convention every other `utils/` module follows.
- **`git_commit` falls back to an `"unknown"` sentinel** when
  `git rev-parse HEAD` fails (resolves the open question below) — a run
  still completes with incomplete lineage rather than hard-failing. The
  subprocess runs with `cwd=PROJECT_DIR` so the hash reflects the imported
  package, not the caller's working directory.
- **`params` records the resolved `release_date`** (the actual YYYYMMDD
  dated directory used), not the raw CLI value, which may be `None`.
- **`decision_tree` manifests are written in `__main__`** from the
  function's previously discarded return value — one manifest per output
  file, each with its own `row_count` — initially leaving the non-standard
  `gdf.to_parquet` save path untouched as decided above (later reversed at
  review, 2026-08-10 — the saves moved to `__main__` too).

Decisions settled at review (Aidan's call, 2026-07-22):

- **Manifest writes are non-fatal.** A lineage sidecar must never abort a
  pipeline run that just produced its output: `save_manifest_to_s3` catches
  any exception around the write, logs a warning naming the manifest path,
  and returns — the same degrade-don't-fail philosophy as the `git_commit`
  `"unknown"` fallback. The docstring states the contract. In
  `compute_contextual_features` the manifest write moves after the
  front-end "latest" publish so even the (now merely cosmetic) failure log
  cannot sit between the two output saves; the manifest still describes
  the dated data-science copy only.
- **`input_versions` is a curated per-stage list, not a flat snapshot of
  the whole `config["data"]` tree.** Each stage's manifests record only
  the datasets that stage actually reads (traced through its getters and
  transform modules), so diffs between runs point at inputs that could
  have changed the output rather than burying them in ~27 identical keys.
  The lists live in `base.yaml` under `run_manifest.stage_input_keys`
  (dataset knowledge belongs in config, next to the dataset definitions;
  moved from a hardcoded `manifest_utils.py` dict in review), exposed as
  `manifest_utils.STAGE_INPUT_KEYS`. `generate_dict_run_manifest` defaults
  `input_keys` to the stage's list, so callers cannot pass a mismatched
  stage/keys pair, and `generate_dict_input_versions` raises `KeyError` on
  a key that is missing from `config["data"]` or resolves to a subtree — a
  typo in a curated list fails loudly at run time instead of silently
  omitting lineage. Legacy `config["data_source"]` (v1) reads stay out of
  scope.

Decisions settled at review (Aidan's call, 2026-07-24):

- **Entrypoints call a single `save_run_manifest_to_s3` wrapper** instead of
  nesting `generate_dict_run_manifest` inside `save_manifest_to_s3` at every
  call site — the identical two-call pattern repeated in all five entrypoints.
  The wrapper takes explicit typed parameters (not `**kwargs`, which would
  erase the signature and defer typos to a deeper stack frame) and omits
  `input_keys`, which no entrypoint overrides. The two underlying functions
  stay public: `generate_dict_run_manifest` is pure and unit-testable without
  S3, `save_manifest_to_s3` isolates the non-fatal write. The wrapper stays in
  `manifest_utils`, not `save_utils`, because its swallow-errors contract is
  the opposite of `save_to_s3`'s raise-on-failure contract.

Decisions settled at review (2026-08-10, prompted by crispy-wonton's PR
review agreeing with Aidan's own inline comment):

- **`decision_tree.py`'s saves move to `__main__`, via `save_utils.save_to_s3`**
  — reversing the kickoff decision above.
  `identify_gdf_tuple_most_suitable_tech_uprn_and_building` becomes a pure
  transform (drops its `save`/`release_date`/`local_authorities` params); the
  entrypoint derives each output path **once** and passes it to both
  `save_to_s3` and `save_run_manifest_to_s3`. The clincher was
  manifest correctness, not tidiness: the interim state derived the path
  twice (inside the function for the write, in `__main__` for the manifest),
  so a drift between the two derivations would make the manifest silently
  point at a path the output was never written to. Behaviourally identical
  writes — `save_to_s3` calls the same `gdf.to_parquet(path)` for
  GeoDataFrame+parquet, plus its logging chokepoint. No external callers of
  the function existed (repo-wide grep), so the signature change is safe.
  Covered by new `pipeline/transform/tests/test_decision_tree.py`, including
  a test pinning that the transform performs no path-building or saving.

## Alternatives considered

- **Fixed name `run_manifest.json` per directory** — rejected; ambiguous
  when a directory holds more than one output file.
- **Manifest for both compute_contextual_features writes** — rejected; the
  front-end "latest" copy has no version history to attach lineage to and
  would go stale/misleading immediately on the next run.
- **Refactor `decision_tree.py` to `save_utils.save_to_s3`** — initially
  rejected as scope creep; adopted at review (2026-08-10) once the reviewer
  asked for it and the double path-derivation it removed turned out to be a
  manifest-correctness risk, not just an inconsistency. See the decision
  above.
- **Extracted version token via per-dataset regex** — rejected; no shared
  naming convention exists today to write a single regex against.
- **Flat `input_versions` snapshot of the whole `config["data"]` tree**
  (identical for every stage) — the original implementation choice, on the
  reasoning that stages pull datasets through nested getters so a
  hand-curated per-stage list would be fragile and silently rot as code
  changes which datasets it reads. Superseded at review (2026-07-22) by
  the curated per-stage lists above: per-stage signal beats
  rot-resistance, and the loud-failure key resolution plus a test that
  every curated key resolves in config contain the rot risk.

## Out of scope

- Auto-detecting input versions from S3 (tracked separately in #429); until
  it lands, `input_versions` is read from the dated prefixes already
  hardcoded in `config["data"]`
- Any reporting or comparison logic on manifest contents (the cross-version
  comparison script's job — first follow-on issue in this stack)

## Open questions

- ~~What `git_commit` should record when `git rev-parse HEAD` isn't
  available~~ — resolved during implementation: an `"unknown"` sentinel
  with a logged warning, never a hard failure.

## Verification

- [x] A manifest JSON is written next to each of the five entrypoints'
      outputs (uprns, add_features, decision_tree, cluster,
      compute_contextual_features), gated on `--save` — each write sits in
      the same `if args.save:` block as the output write (decision_tree:
      six manifests total across the five stages, since it has two outputs)
- [x] Manifest includes: stage, local_authority, run_at, git_commit,
      input_versions, row_count, params — pinned by
      `TestGenerateDictRunManifest::test_contains_exactly_the_expected_keys`
- [x] Filename (`{output_basename}.manifest.json`) avoids collision with
      the existing `create_manifest.py` output — pinned by
      `TestGetStrManifestPath` (derived name always keeps the output
      basename, so it can never be the bare `manifest.json`)
- [x] Unit test covers manifest dict content for at least one entrypoint —
      `utils/tests/test_manifest_utils.py` (17 tests) covers dict content,
      curated input-version resolution (exact selection; loud `KeyError`
      on unknown or subtree keys; every `STAGE_INPUT_KEYS` entry resolves),
      the swallowed-and-logged manifest write failure, git-commit fallback
      and path derivation
