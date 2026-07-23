---
title: Stream OS open data grid squares from the OS Downloads API to S3
status: draft
github_issue: https://github.com/nestauk/asf_heat_pump_suitability/issues/419
pr:
asana: TBD
created: 2026-07-23
---

## Problem

The per-grid-square OS inputs (OpenMap Local, Open Roads, Open Greenspace) are
refreshed by hand: download tiles from the OS portal, upload via the S3
console. The last manual refresh silently dropped 4 road files, and the
greenspace upload initially missed 5 island tiles (HP/HT/HU/HY/HZ) — gaps only
discovered downstream. The public OS Downloads API
(`https://api.os.uk/downloads/v1`, no auth for OpenData products) lists every
per-area file with download URLs, md5s and sizes, so the refresh can be one
reproducible, self-checking command.

Full background in the
[issue](https://github.com/nestauk/asf_heat_pump_suitability/issues/419) and
its kickoff-interview
[comment](https://github.com/nestauk/asf_heat_pump_suitability/issues/419#issuecomment-5059398210).

## Proposal

New script `pipeline/run/stream_os_open_data.py` (pattern precedent:
`pipeline/run/stream_inspire_files.py`, which streams external zips to S3 —
but written to v2 conventions: `config["data"]` paths, `logging`, no
hard-coded bucket).

Decisions, each settled in the kickoff interview (2026-07-23):

1. **CLI** — `--products OpenMapLocal OpenRoads OpenGreenspace`; dry-run by
   default (list tiles, sizes and the release version; no upload) unless
   `--save`. An optional destination-root override flag supports the test
   rehearsal below. Product names are the API's own IDs, so no mapping layer.
2. **Version-named prefixes** — the dated segment of each S3 prefix comes from
   the API product `version` field verbatim (e.g. `2026-04`), not the download
   date. Rationale: names the OS release rather than the download event (ADR
   0002's spirit — pins should identify the data, and two people refreshing
   the same release must land on the same prefix). No ingestion code changes:
   getters read whole path templates from `base.yaml` and only substitute
   `{square}`/`{layer}`.
3. **Keep each product's current internal layout** (no normalisation):

   - OpenMap Local: `opmplc_essh_gb/{version}/data/{square}/{square}_{layer}.*`
   - Open Roads: `oproad_essh_gb/{version}/data/{square}_{layer}.*`
   - Open Greenspace: `opgrsp_essh_gb/{version}/{square}/data/{square}_{layer}.*`

   Rationale: match what is already on S3 so consumers and reviewers see only
   the dated segment change. All shapefile sidecars (`.shp`, `.dbf`, `.prj`,
   `.shx`, `.cpg`, …) are uploaded, not just `.shp`. The zips' `licence.txt`
   and `readme.txt` are also kept in their current on-S3 locations: for
   OpenMap Local and Open Roads, `readme.txt` at the dated prefix root and
   `licence.txt` under `doc/` beside it; for Open Greenspace, both under
   each `{square}/`.

4. **Roads fan-out** — the API offers no per-square downloads for OpenRoads,
   only a single ~606 MB GB shapefile zip whose members are per-square files;
   the script streams that zip and fans its members out to the roads prefix.
   OpenMap Local (55 tiles) and Greenspace (52 tiles) download per-area zips.
5. **Reconciliation in-script** — after upload, list S3 under the new prefix
   and compare against the API's offered areas (for roads: the GB zip's
   members); fail loudly on any mismatch. This validates the upload the script
   just performed; the planned pipeline-run preflight (#434) remains
   complementary, not a duplicate. md5s from the API listing are verified on
   the downloaded zips as a cheap corruption check.
6. **Test rehearsal before production** — first run writes to a clearly-named
   test prefix inside `asf-local-heat-planning-tool` (e.g.
   `inputs/geodata/test_419_os_downloads/…`); getters are pointed at it
   explicitly to verify end-to-end loading. Only then does the production run
   write to the real product prefixes (inert until repoint, per ADR 0002), and
   `base.yaml` templates are repointed as the final reviewed commit on this
   branch. The test prefix is deleted after verification.
7. **Config, not code** — the API base URL and the per-product S3 prefix
   templates live in `base.yaml`.

## Alternatives considered

- **Normalise the three layouts to one convention** (issue proposal, e.g.
  roads-style flat `data/{square}_{layer}.shp` for all products) — rejected in
  interview: matching the existing on-S3 structure means less churn for
  consumers, and the getters are indifferent (they format whole templates).
- **Separate test bucket** — rejected: tests a different config than ships,
  needs new bucket/IAM, uploads ~3 GB twice. A test prefix inside the
  production bucket gives the same isolation (nothing reads a prefix
  `base.yaml` doesn't point at).
- **Upload-date (`YYYYMMDD`) prefix naming** (current convention) — rejected:
  names the download event, not the release; identical data refreshed on two
  days would occupy two prefixes.
- **Defer reconciliation to the #434 preflight** — rejected: a silently
  incomplete refresh would go undetected until the next pipeline run, which is
  exactly the failure mode motivating this issue.
- **Auto-resolving "latest" prefixes** — out of bounds here; settled by ADR
  0002 (explicit pinning). Issue #429 tracks the counter-proposal.

## Out of scope

- GB-wide single files (`opmplc_gb.gpkg`, `opgrsp_gb.gpkg`) — separate undated
  keys, refreshed rarely; `gb_os_openroad` already globs the per-square prefix
  so it is covered by the repoint.
- Cleanup of superseded prefixes on S3 — issue #420 (this work adds to its
  list: the old `20260708`/`20260709` prefixes once repointed away from).
- Pipeline-run-time input validation — issue #434.

## Open questions

- Should `stream_inspire_files.py` later be migrated to the same
  config-driven, reconciling pattern (it still hard-codes the v1 bucket and
  uses deprecated `config["data_source"]`)?
- Does `config/README.md`'s citation table need an access-method note (API vs
  portal) for the three OS datasets, or is the existing citation unchanged?

## Verification

- [x] Dry run (no `--save`) lists per-product tile names, sizes and the API
      `version` without writing to S3. (Run 2026-07-23 with AWS credentials
      nulled: OpenMapLocal 2026-04, 55 zips / 2455.4 MB; OpenRoads 2026-04,
      1 GB zip / 606.1 MB; OpenGreenspace 2026-04, 52 zips / 39.9 MB; island
      tiles HP/HT/HU/HY/HZ all listed; exit 0.)
- [ ] Test-prefix run uploads via the destination override; getters load
      grid-square data from the test prefix end-to-end; test prefix deleted
      afterwards.
- [ ] Production run populates version-named prefixes for all three products
      in their current layouts, all sidecar files included.
- [x] Reconciliation compares S3 against the API's offered areas (roads: GB
      zip members) and exits non-zero with a clear message on mismatch.
      (Diff helper unit-tested; failure path demonstrated 2026-07-23 with
      mocked S3 — a missing island tile plus a stray `.DS_Store` produced a
      clear ERROR naming both keys and `SystemExit(1)`.)
- [ ] `base.yaml` repointed to the new prefixes as the final commit;
      `load_gdf_os_openmap_layer` and `load_gdf_os_openroad` load spot-check
      squares via config alone.
- [x] API base URL and prefix templates read from `base.yaml`; no hard-coded
      S3 paths or magic values in the script (`os_downloads` section:
      `api_url`, `s3_destination_root`, per-product prefix templates).
- [x] Unit tests (fixture JSON, no network) for the pure helpers: API response
      → download list, zip member → S3 key mapping per product, reconciliation
      diff. 1:1 test module `pipeline/run/tests/test_stream_os_open_data.py`.
