---
title: Remove city centre areas smaller than 0.72 ha
status: draft
github_issue: https://github.com/nestauk/asf_heat_pump_suitability/issues/582
pr:
asana: https://app.asana.com/1/5571817120120/project/1214222223606748/task/1219988772795465
created: 2026-10-29
---

## Problem

The city centre areas come from six Spatial Signatures Framework types (`CITY_CENTRE_TYPES` in `pipeline/transform/city_centres.py`). Since PR #579, the getter clips them to the LA boundary. `compute_contextual_features.py` uses them in two places:

1. The cluster flag `in_city_centre`: "Yes" if the cluster touches any city centre area. The flag selects the "district heat network potential" text in the logic trace.
2. The map layer `areas_of_district_heat_network_potential`, together with the heat network zones (HNZs).

There is no minimum size. The Spatial Signatures Framework gives a type to each enclosed tessellation cell (about one street block), so one block can get a city centre type when the blocks around it do not. The frontend then shows small areas that users see as noise, for example one 317 m² area in Babergh.

GB analysis (city centre types clipped to each LA, touching areas merged, 2026-10-29):

| Measure                                     | Value                                  |
| ------------------------------------------- | -------------------------------------- |
| Separate city centre areas / LAs with areas | 3,459 / 248                            |
| Median area                                 | 1,273 m²                               |
| Areas < 1 ha (10,000 m²)                    | 82% of areas, 0.5% of city centre land |
| Areas < 5 ha                                | 90% of areas, 1.25% of land            |

UPRN effect (2026-10-26 pipeline run, 349 LAs, 33.7 million domestic UPRNs, threshold 7,170 m²):

| Measure                                       | Now       | With minimum area | Change           |
| --------------------------------------------- | --------- | ----------------- | ---------------- |
| Domestic UPRNs inside city centre areas       | 4,956,008 | 4,940,866         | −15,142 (−0.3%)  |
| UPRNs in clusters with `in_city_centre = Yes` | 6,175,699 | 5,950,378         | −225,321 (−3.6%) |
| LAs with any "Yes" cluster                    | 245       | 218               | −27              |

The cluster effect is about 12 times the direct effect, because a small area that touches a cluster gives "Yes" to all UPRNs in that cluster. Largest share of "Yes" UPRNs lost: Dudley 58%, County Durham 47%, Slough 44%, Sutton 42%.

## Proposal

1. **Rule.** Merge all city centre areas in the run (`union_all`), split the result into separate parts, and remove each part smaller than the minimum area. Merge first: a filter on each signature polygon would remove small polygons that are part of a large centre.
2. **Value: 7,200 m² (0.72 ha).** The smallest DESNZ heat network zone is 7,966 m² (Gateshead). 7,200 m² is about 90% of that, rounded, to be conservative in case smaller zones are drawn in the future. The HNZ is not a city centre area; we use only its size as a reference. The result is not sensitive to the exact value: from 0.5 ha to 1 ha the number of areas kept changes by about 6% and the land kept by less than 1%.
3. **Unit: area, not UPRN count.** The Spatial Signature type already shows density; the area adds scale. A domestic UPRN count would remove commercial centres, which have few domestic UPRNs but are good heat loads.
4. **Scope: map layer and cluster flag.** Both use the same filtered areas, so the map and the logic trace text always agree. HNZs are not filtered: they are official zones.
5. **LAs can lose all their city centre areas** (27 LAs in the 2026-10-26 run, for example Babergh). Their areas are only noise. No "keep the largest area" exception: each area passes or fails alone.
6. **Location.** New function in `pipeline/transform/city_centres.py`, for example `filter_gdf_city_centre_min_area(city_centres_gdf, min_area_m2)`. It returns one row per kept part with `source_annotation` and `geometry` (EPSG:27700), and an empty GeoDataFrame with the same columns if no part is kept. `compute_contextual_features.py` calls it once, directly after `load_gdf_spatial_signatures_gb`. Downstream code only calls `union_all()` and `source_annotation.iloc[0]`, so it needs no change. The getter stays "load only".
7. **Config.** `constant.threshold.city_centre_min_area_m2: 7200` in `base.yaml`, next to the other m² thresholds, with a comment that gives the source (smallest DESNZ HNZ, 7,966 m², HNZ inputs as of 2026-10). The `in_city_centre` metadata description states the minimum area.
8. **Evidence lives in this spec only.** The analysis script is not committed.
9. **Tests.** One test in a new `pipeline/transform/tests/test_city_centres.py`: two small touching polygons that pass together are kept, and one isolated small polygon is removed. This catches the most likely regression (filter before merge).

## Alternatives considered

- **Closing step (buffer out, merge, buffer in) before the filter.** Rejected: with a 1 ha floor it keeps the centre layer in only 2 (25 m) to 7 (50 m) more LAs, and it adds land between areas that is not a city centre type.
- **0.8 ha (the smallest HNZ, no margin).** Rejected for a conservative margin. The difference is about 40 areas in GB (1%).
- **1 ha (round number).** Rejected: the HNZ reference is easier to explain than a round number with no source.
- **5–6 ha (near the HNZ 5th percentile).** Rejected: removes real town centres in market-town LAs (Buckinghamshire keeps 1 area in place of 4), for almost no visual gain in cities.
- **Minimum domestic UPRN count or heat demand per area.** Rejected: see Proposal 3.
- **Filter only the map layer.** Rejected: the map and the logic trace text would disagree.
- **Filter in the getter.** Rejected: getters only load data (CLAUDE.md).
- **Commit the analysis script in `research/exploratory/`.** Rejected by Aidan: spec only.

## Out of scope

- Joining areas that are near each other but do not touch.
- Changing which Spatial Signature types count as city centre.
- Filtering HNZs.
- Different frontend colours for HNZ and city centre areas.
- The UPRN-level functions in `city_centres.py` (`extend_df_city_centre_labels`, `label_gdf_city_centre_spatial_signatures_uprns`). Only research scripts use them; this work does not change them.

## Open questions

- If DESNZ publishes a zone smaller than 7,966 m², do we lower the threshold, or keep 7,200 m² as a fixed value?
- After the next GB run, does the compare_versions report for Dudley, County Durham, Slough and Sutton show only the expected `in_city_centre` and logic trace changes?

## Verification

- [ ] `constant.threshold.city_centre_min_area_m2` is 7200, with a source comment.
- [ ] Touching areas are merged before the size test (the new test passes).
- [ ] The `areas_of_district_heat_network_potential` layer has no city centre part smaller than 7,200 m².
- [ ] `in_city_centre` uses the same filtered areas as the layer.
- [ ] HNZ geometries in the layer do not change.
- [ ] The `in_city_centre` metadata description states the minimum area.
- [ ] Local run of `compute_contextual_features.py` (no `--save`): Babergh has no city centre areas; Gateshead has three (about 0.77 ha, 3.2 ha and 374 ha).
- [ ] `uv run pytest` passes.
