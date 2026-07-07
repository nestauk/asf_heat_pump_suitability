# Versioning the output datasets (Asana 1216293268467175, due 2026-07-15)

> **Updated 2026-07-07:** versioning scheme DECIDED — no version tokens; outputs get a
> dated release directory laid out as **`{LA_slug}/{YYYYMMDD}`** (Sofia's proposal, e.g.
> `vale_of_glamorgan/20260708`), confirmed by Aidan 2026-07-07. Former open questions
> 1–2 are closed; remaining questions renumbered below.

## Task

Introduce a versioning scheme for the pipeline's output datasets on S3 (`s3://asf-local-heat-planning-tool/outputs/data/...`), which today are written to fixed, unversioned per-LA paths and silently overwritten on every `--save` run. Raw *inputs* carry version tokens in their filenames (`{pub_date}_v{version}_...`), but outputs (per-LA parquets and the final `clusters_contextual_features_{tolerance_m}m.geojson` consumed by the heat planning tool) carry no release information at all, so there is no way to pin, compare, or roll back a dataset release. Per the task comments and Aidan's confirmation (2026-07-07), the scheme is decided: outputs are written under `{LA_slug}/{YYYYMMDD}` release directories rather than version tokens. The work is to thread that dated path through every write *and* every inter-stage read, and record the date in the geojson metadata.

## Relevant files/functions

- `asf_heat_pump_suitability/config/base.yaml` (`output:` section, lines ~110–125) — the six output path templates (`domestic_uprns`, `domestic_uprns_with_features`, `uprns_most_suitable_tech`, `buildings_most_suitable_tech`, `tech_clusters`, `clusters_tech_contextual_info`); these gain the `{LA_slug}/{YYYYMMDD}` segment — defined here, not hard-coded in scripts.
- `asf_heat_pump_suitability/utils/save_utils.py::save_to_s3` — the single write helper all pipeline saves go through; a natural chokepoint for building the dated release paths.
- `asf_heat_pump_suitability/pipeline/run/compute_contextual_features.py` — writes the tool-facing geojson (line ~384) and *reads* two earlier outputs via `config["output"]` `.format()` calls (lines ~350, ~361); `create_json_contextual_features_metadata` already stamps "Data file date of creation" — the only versioning that exists today, and it lives inside the file, not the path.
- `asf_heat_pump_suitability/pipeline/transform/uprns.py` (~line 542), `pipeline/run/add_features.py` (~line 341), `pipeline/transform/decision_tree.py` (~lines 423–449), `pipeline/cluster/cluster.py` (~lines 845–896) — the remaining write sites and cross-stage reads that must all resolve the same release date consistently.
- `asf_heat_pump_suitability/config/README.md` + `CLAUDE.md` — document the new `{LA_slug}/{YYYYMMDD}` output convention alongside the existing input naming convention (inputs keep their `{pub_date}_v{version}_...` scheme; the two are deliberately different).

## Assumptions

- Scope is the v2 outputs under `config["output"]["dataset"]` (bucket `asf-local-heat-planning-tool`), not the deprecated v1 `config["data_source"]` paths or the pickled model.
- A "version" is a dated release directory shared by all six datasets for an LA (per the task comments), not per-file counters — the datasets are interdependent stages of one run.
- Date directories are the release mechanism (human-browsable, tool can pin a URL); S3 bucket versioning is at most a safety net underneath, not the scheme.
- A "latest" pointer (stable path or manifest) is needed so the front-end tool keeps working without code changes per release.
- Old release directories are retained on S3 (storage cost is acceptable at ~6 LAs now, all GB later).

## Open questions

~~1–2. Path order and date format~~ **CLOSED 2026-07-07: `{LA_slug}/{YYYYMMDD}`** (Sofia's proposal, confirmed by Aidan). Post the confirmation on the Asana thread so it's visible to the team.

1. **Do intermediate parquets also live under the dated release directory, or only the final geojson the tool consumes?**
   **DECIDED (Aidan, 2026-07-07): all six.** Downstream stages read upstream outputs by config path, so mixing release dates mid-pipeline would silently produce inconsistent releases — and the current S3 layout already keeps all six per-LA files together (East Lothian, July 2026):

   ```
   east_lothian_domestic_uprns.parquet                    1.0 MB   2 Jul
   east_lothian_with_features.parquet                     1.7 MB   2 Jul
   east_lothian_uprns_most_suitable_tech.parquet          5.9 MB   3 Jul
   east_lothian_buildings_most_suitable_tech.parquet      2.2 MB   3 Jul
   east_lothian_clustered_tech_polygons.parquet          14.2 MB   3 Jul
   east_lothian_clusters_contextual_features_5m.geojson  50.3 MB   3 Jul
   ```

   Note the real listing also shows one release already spans multiple days (2–3 Jul) — see Q2's override — and that filenames on S3 don't all match config key names (`..._with_features`, `..._clustered_tech_polygons`).
2. **Where does the release date come from at runtime — auto-derived from the run date, or supplied explicitly?**
   **DECIDED (Aidan, 2026-07-07): run-date default with an explicit override option** (`--release_date` CLI arg). Without the override, a single release would scatter across several `{YYYYMMDD}` directories when stages run on different days (as the listing above shows they do).
3. **Does the front-end tool need a stable "latest" path, and who owns updating its pointer?**
   **DECIDED for now (Aidan, 2026-07-07): keep writing the current undated paths as "latest"** alongside the dated copies; confirm downstream dependencies with the team — the frontend dev may be copying the geojson into his own bucket.
4. **Should S3 bucket versioning be enabled underneath as a safety net?**
   **DECIDED (Aidan, 2026-07-07): no — not worth it.** The date directories already give rollback.

<!-- asana-sync: 2026-07-07T09:32:08.044Z -->
