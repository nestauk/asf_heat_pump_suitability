---
title: Input preflight check before pipeline runs
status: in-review
github_issue: https://github.com/nestauk/asf_heat_pump_suitability/issues/434
pr: https://github.com/nestauk/asf_heat_pump_suitability/pull/448
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
  of scope (below). **Revised in review (2026-08-12, crispy-wonton's
  suggestion):** `{square}` tokens are now expanded over
  `config["constant"]["sampling_areas"]["grid_squares"]` before checking, so
  a missing square is reported individually. Remaining `{layer}` tokens
  still truncate to a prefix, requiring at least one file per square rather
  than every layer, since some layers are legitimately absent in a square
  (`load_gdf_os_openmap_layer` deliberately skips those).
  **Revised again in review (2026-08-18, crispy-wonton's follow-up):** the
  sampling-squares expansion is replaced by the same derivation the pipeline
  uses. `check_inputs.py` takes `--local_authorities` (`nargs="+"`,
  `default=["GB"]` — the list form; a bare-string default would be iterated
  character-wise downstream, the latent bug at `uprns.py:419`, fixed in this
  PR — see #480). Named LAs
  expand `{square}` over `get_list_la_grid_squares(resolved)` — the BNG grid
  clipped to the union of the LAs' buffered boundaries — and the GB default
  uses `get_list_la_grid_squares(None)` (clipped to all LA boundaries), NOT
  `get_dict_la_data(None)`, whose GB branch returns the raw unclipped grid
  including sea-only squares. `run_pipeline.sh` passes its LA array to the
  preflight, so the production check covers exactly the squares the run
  reads. The GB acceptance run surfaced seven fringe squares (HW, HX, NQ,
  NV, OV, SA, SB — remote islands/sea slivers inside LA boundaries) that
  some OS OpenMap products ship no files for; these are excluded via the new
  `config["constant"]["os_data_absent_grid_squares"]` mapping (same
  expected-absence pattern as `RESEARCH_ONLY_PATHS`), **per product** — keyed
  by the product folder name in the path — so e.g. OpenMap Local is still
  checked for HW/HX/OV, which it does ship (St Kilda has buildings but no
  roads). This also stops island-LA runs (e.g. Western Isles) failing the
  preflight on squares the pipeline itself tolerates by silent skip. The
  absences were verified as genuine OS publication sparsity, not upload gaps
  — see the verification record below.

## Alternatives considered

- **Also checking `config["data_source"]`** — rejected; it would preflight
  paths only a legacy standalone script reads, expanding scope for no
  benefit to the five live entrypoints this issue is actually about.
- **`pipeline/run/check_inputs.py`** (flat, alongside other entrypoint
  scripts) — rejected in favour of a new `pipeline/validate/` subpackage,
  since more validation checks are already planned to follow this one.

## Out of scope

- ~~Per-square existence checks~~ — pulled into scope during PR #448 review
  (see revised decision above). ~~Still out of scope: deriving the required
  squares from the requested LAs or a canonical GB-wide list~~ — also pulled
  into scope (2026-08-18 revision above): squares now derive from
  `--local_authorities` via boundary clipping, defaulting to whole-of-GB.
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
      S3 folder is empty) and exited 1. Resolved (Aidan, 2026-07-22): the
      file was never uploaded and only two research/exploratory scripts read
      the key (already broken in practice), so the stale
      `geodata.council_tax_data` config entry was removed rather than the
      dataset sourced; the preflight then passes (all 26 paths exist, exit
      0). The `config/README.md` citation row is kept as provenance for the
      past domestic-filtering research that used the data. Convention
      established from this (Aidan, 2026-07-22), recorded as comments on the
      `data:` section in `base.yaml` and in the module docstring:
      `config["data"]` is production-pipeline inputs only — every path in it
      is preflight-gated, so datasets read only by research/exploratory
      scripts keep their paths in the script or a research-local config.
      **Revised in review (2026-08-12, crispy-wonton):** the
      `geodata.council_tax_data` entry was restored so research scripts keep
      a config reference until production and research configs are split
      (low-priority follow-up issue to open); the preflight skips it via the
      explicit `RESEARCH_ONLY_PATHS` constant in `check_inputs.py`. The
      script also now exits non-zero if the collector finds zero configured
      paths, so a broken config walk can't pass vacuously. Post-review
      acceptance run: all 44 paths exist (23 plain + 21 per-square), exit 0.
- [x] Wired into `run_pipeline.sh` as its first step, before the
      local-authority loop
- [x] Unit tests cover at least one missing-path case and one all-present
      case (`pipeline/validate/tests/test_check_inputs.py`)
- [x] LA-scoped acceptance runs pass (2026-08-18): Plymouth (1 square, 26
      paths, exit 0) and the two-LA union Glasgow City + Midlothian (29
      paths, exit 0)
- [x] GB-default acceptance run passes (2026-08-18): 59 derived grid
      squares, 182 paths after per-product skips, exit 0. The failure mode
      was demonstrated en route: before the `os_data_absent_grid_squares`
      exclusion existed, the same run correctly reported all 18 absent
      square-paths in one pass and exited 1

### Verification record: `os_data_absent_grid_squares` entries (2026-08-18)

The risk with an exclusion list is circularity: "the file is absent from our
bucket" cannot by itself distinguish OS-publishes-nothing from
we-failed-to-upload, because the bucket is downstream of our own upload. The
entries were therefore verified against the **independently acquired GB-wide
GeoPackage copies** of the same products
(`inputs/geodata/opmplc_gb.gpkg` / `opgrsp_gb.gpkg` / `oproad_gb.gpkg`,
downloaded from OS in a different format ~2 months before the shapefile
delivery was uploaded). If OS publishes no features in a square, the GB-wide
file must be empty there too; an upload gap on our side would not reproduce
in it.

Method: bbox-query each square (bounds from `load_gdf_bng_grid_squares`)
against every layer of each GeoPackage via GDAL's `/vsis3/` ranged reads:

```python
import pyogrio
from asf_heat_pump_suitability.getters import load_geodata
grid = load_geodata.load_gdf_bng_grid_squares().set_index("bng_ref")
bounds = tuple(grid.loc["HW"].geometry.bounds)
pyogrio.read_dataframe(
    "/vsis3/asf-local-heat-planning-tool/inputs/geodata/oproad_gb.gpkg",
    layer="road_link", bbox=bounds,
)
```

Results — feature counts per square across all layers of each product:

| Square          | opmplc (20 layers)  | opgrsp (2 layers) | oproad (road_link) | Excluded for   |
| --------------- | ------------------- | ----------------- | ------------------ | -------------- |
| HW (St Kilda)   | 306                 | 0                 | 0                  | opgrsp, oproad |
| HX              | shipped in delivery | 0                 | 0                  | opgrsp, oproad |
| OV              | shipped in delivery | 0                 | 0                  | opgrsp, oproad |
| NQ              | 0                   | 0                 | 0                  | all three      |
| NV (NI overlap) | 0                   | 0                 | 0                  | all three      |
| SA (NI overlap) | 0                   | 0                 | 0                  | all three      |
| SB (NI overlap) | 0                   | 0                 | 0                  | all three      |
| NA (control)    | —                   | 1                 | 10                 | not excluded   |
| NL (control)    | —                   | —                 | 361                | not excluded   |

Every excluded square/product combination has zero features in the
independent copy; every control has features. The per-product asymmetry is
theme-coherent (remote islets have buildings but no roads or greenspace
sites; NQ/NV/SA/SB's land is Northern Ireland, outside OS GB products and
this tool's GB-only scope). If a future OS release adds data for these
squares, re-verification is one bbox query per square.
