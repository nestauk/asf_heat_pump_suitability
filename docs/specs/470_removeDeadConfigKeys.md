---
title: Clean up base.yaml (dead keys, stages block, research split, v1 inputs)
status: draft
github_issue: https://github.com/nestauk/asf_heat_pump_suitability/issues/470
pr:
asana:
created: 2026-10-14
---

## Problem

The issue describes the four problems in `config/base.yaml`, from an audit on
`dev` on 2026-10-14: dead keys, per-stage facts in seven places, research and
production keys in one config, and production inputs in the v1 bucket. This
spec does not repeat that list. Refer to the issue.

The issue has four phases. This spec covers all four. The branch
`470_removeDeadConfigKeys` does Phase 1 only. Each later phase gets its own
branch and PR, and updates this spec.

## Proposal

### Phase 1: remove dead keys (this branch)

1. Remove these keys from `base.yaml`:
   - `webmap_data_source` (all keys)
   - `mapping` (all keys)
   - top-level `features`
   - top-level `tech_types`
   - `constant.ID`
   - `constant.sampling_areas.la_names`
   - `data.geodata.heat_network_zones.desnz_files`
   - the 18 dead `data_source` keys that the issue lists
2. Remove `geodata.gb_code_points` and `geodata.heat_network_zones.desnz_files`
   from `run_manifest.stage_input_keys.add_features`. No stage reads them.
3. Keep `data.geodata.gb_code_points`. `clip_inputs.py` reads it (see decision 2).
4. Fix `pipeline/transform/clip_inputs.py`:
   - `geodata.uk_osopen_uprn` becomes `geodata.uk_osopen_uprn_raw`. The script
     reads the `.zip` CSV, and the raw key points to that file.
   - `geodata.gb_uprn_country_mapping` becomes `geodata.gb_uprn_lookup`
     (renamed in `03c4b9fd`).
   - `__main__` calls `clip_df_and_save_dataset`, but the function is named
     `clip_gdf_and_save_dataset`. Make the call use the function name.
5. Before each removal, do a `git grep` for the key name in `.py`, `.yaml`,
   `Makefile` and `.md` files. If a reader exists, keep the key and record it
   here.

### Phases 2 to 4 (later branches)

- Phase 2: one top-level `stages` block in run order. See the issue for its
  fields. Starts after #534 merges into `dev`.
- Phase 3: the preflight checks the union of `stages.*.inputs`. Remove
  `RESEARCH_ONLY_PATHS`. Starts after Phase 2.
- Phase 4: copy the 7 v1 production inputs to the v2 bucket and move their
  keys into `config["data"]`. Can start after Phase 1.

### Decisions

1. **One spec for all phases, one branch for Phase 1.** Phase 1 has no
   blockers. Phases 2 and 3 wait for #534. One spec keeps the full plan in one
   place. Each phase stays a small PR.
2. **Fix `clip_inputs.py`, do not delete it.** It is a recent testing tool
   (June 2026). It broke when the UPRN keys were renamed in July and August 2026. The fix is three small edits.
3. **No guard test for dead keys.** Code reads many keys dynamically
   (f-strings, dataset-name strings, loops over key names). A grep-based test
   would give false alarms and need an allow-list. Phase 2's structural test
   protects the stage keys.
4. **Ask Seth to close PR #271.** That PR restructures `base.yaml` in a
   different way and has had no activity since March 2026. Comment posted on
   2026-10-14.

## Alternatives considered

- **Four sub-issues, one for each phase.** Rejected. More tracking work, and
  the phases are small.
- **Delete `clip_inputs.py`.** Rejected. Nothing imports it, but it is a
  colleague's recent tool, and the fix is small.
- **Guard test with an allow-list.** Rejected (decision 3).
- **Use the config shape of PR #271** (`data` and `data_source` become
  `inputs`, `output` becomes `outputs`). Not adopted. It renames every key
  that the pipeline reads and does not solve the per-stage problem. Seth can
  add ideas to #470.

## Out of scope

- Phases 2 to 4 on this branch
- The legacy standalone scripts (`grid_capacity.py`, the INSPIRE scripts,
  `ref_calculate_garden_size.py`) and their `data_source` keys
- Hard-coded S3 paths and magic numbers in pipeline code
- Research scripts that read keys that do not exist (Phase 3 lists them in a
  follow-up issue)

## Open questions

- Phase 3: do research-only keys go into the research scripts, or into a
  small research config file? The #434 spec allows both ("script or
  research-local config"). Decide at the Phase 3 kickoff.
- Phase 4: who copies the 7 v1 files to the v2 bucket, and do they keep their
  current release versions? Decide at the Phase 4 kickoff.
- Do we need `clip_inputs.py` to clip `geodata.uk_osopen_uprn_partitioned`
  (which the pipeline reads) in place of the raw `.zip`? Answer when the
  script runs again.

## Verification

Phase 1:

- [ ] Each removed key has no reader (`git grep` for the key name returns no
      `.py` hit)
- [ ] `uv run pytest` passes
- [ ] The input preflight (`check_inputs.py`) passes
- [ ] `run_manifest.stage_input_keys.add_features` does not list
      `gb_code_points` or `desnz_files`
- [ ] `clip_inputs.py --local_authorities Plymouth` runs to the end without
      `--save`

Phases 2 to 4: see the acceptance criteria in the issue.
