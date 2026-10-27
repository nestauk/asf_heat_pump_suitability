---
title: Skip missing non-land grid-square files in the partitioned parquet getters
status: in-review
github_issue: https://github.com/nestauk/asf_heat_pump_suitability/issues/555
pr: https://github.com/nestauk/asf_heat_pump_suitability/pull/560
asana:
created: 2026-10-23
---

## Problem

See the issue for the behaviour and acceptance criteria. In short: `uprns.py` fails for North Yorkshire
because its boundary (+1 km buffer, and also with no buffer) touches BNG square OV, which is sea only.
OV has no file in the OS Open UPRN, POI or NSUL partitions, and their getters give every requested path
to `pl.read_parquet`, which fails on a 404.

Facts found during kickoff (2026-10-23):

- Four datasets are partitioned per square: OS Open UPRN (56 files), POI (56), NSUL (57), domestic EPC (91).
  UPRN, POI and NSUL only have files for squares with data. EPC has a file for every square (OV exists, 0 rows).
- Every square in `config["constant"]["land_grid_squares"]` (54) has a file in all four datasets.
- NP and NQ have UPRN files (1 UPRN each) but are not land squares. No GB LA touches them.
- In GB, only North Yorkshire requests a square with no file. NI LAs request NV/SA/SB, but NI is outside the BNG.
- Affected getters: `load_geodata.load_df_osopen_uprn`, `load_geodata.load_gdf_poi`, `load_data.load_df_uprn_lookup`.
  The OS OpenMap and OpenRoad getters already skip missing files (#423, #424), and the #416 spec records that
  North Yorkshire requests OV for those products.

## Proposal

1. **Fix in the getters, not in grid-square selection.** The three partitioned parquet getters handle a missing
   file themselves. Rationale: it follows the #424 pattern for the OS getters, and it keeps
   `get_set_la_grid_squares` unchanged for its other callers.
2. **Change the UPRN, POI and NSUL getters.** Rationale: all three request OV for North Yorkshire. A fix in the
   UPRN getter only would move the failure to the next getter in the same stage.
3. **Skip only non-land squares.** For each requested square whose file does not exist: if the square is not in
   `land_grid_squares`, log a warning and skip it; if it is a land square, raise an error. Rationale: an expected
   empty square (sea) is skipped, but a real data gap (for example, a failed upload of a land square) still stops
   the run and is not hidden in a long log.
4. **One parametrised test** covers both cases (non-land skipped, land raises) for all three getters, with a mocked
   S3 listing. Rationale: one regression check per getter without three copies of the same mock setup.
5. **Branch off `dev`**, independent of the failure-log PRs. Rationale: the fix can merge on its own.
6. **The shared check is `s3_utils.filter_list_partition_paths`** (decided in `/implement`). Rationale:
   `base_getters.py` must stay free of dataset-specific logic, and other `utils` modules already read `config`.
7. **One `s3fs` listing of the partition folder per getter call**, not one `exists` call per square (decided in
   `/implement`). Rationale: fewer S3 calls. It uses `s3fs` directly, not `s3_utils.list_files_s3_path`: on `dev`
   that helper's glob (`{dir_uri}*`) returns the folder itself, not its files. PR #551 fixes it; this branch does
   not merge #551, so that the two PRs stay independent.

## Alternatives considered

- **Filter grid squares to `land_grid_squares` in `get_set_la_grid_squares`** (one place, fixes all datasets).
  Rejected: it changes the behaviour for every caller (including the OS getters, which already work) and makes
  the land list the only guard.
- **Skip any missing file** (exact #424 behaviour). Rejected: a missing land-square file would only give a
  warning, and the LA would silently lose UPRNs.
- **Write empty partition files to S3** for non-land squares, as EPC does. Rejected: no code change, but every
  future re-partition must remember it and nothing enforces that.
- **UPRN getter only.** Rejected: North Yorkshire probably fails again at the POI or NSUL getter.
- **Also change the EPC getter.** Rejected: EPC has a file for every square, so the skip would never run.
- **Stack on PR #551.** Rejected: the fix would wait for that stack to merge.

## Out of scope

- Turning the S3 input preflight check (`check_inputs.py`) back on in `pipeline.py`.
- The domestic EPC getter.
- Writing empty partition files to S3.
- Northern Ireland LAs.

## Open questions

- Should the OS OpenMap and OpenRoad getters use the same land-square rule later, so that a missing land file
  also raises there? Not in this issue.
- `land_grid_squares` does not contain NP and NQ (1 UPRN each). This does not affect GB LAs today.

## Verification

- [x] The UPRN, POI and NSUL getters skip a missing non-land square file with a log warning, and return the rows
      from the other squares. Evidence: real S3 calls with `SE`, `OV`, `NZ` for all three datasets; the North
      Yorkshire run logs `Skipping missing partition file for non-land square OV` for UPRN and POI.
- [x] The three getters raise an error when a land square's file is missing. Evidence: the
      `test_raises_for_missing_land_square` cases (real S3 has no missing land-square file).
- [x] One parametrised test covers both cases for all three getters; `uv run pytest` passes (214 tests).
- [x] `uprns.py --local_authorities "North Yorkshire"` runs to the end locally (no `--save`): exit 0,
      383,008 domestic UPRNs. Note: the census check used its default range, because the census data has no row
      for "North Yorkshire" (see the follow-up issue on census names).
