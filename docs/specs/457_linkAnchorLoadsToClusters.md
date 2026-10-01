---
title: Link anchor loads to communal clusters by ID so the UI can highlight them
status: draft
github_issue: https://github.com/nestauk/asf_heat_pump_suitability/issues/457
pr:
asana: TBD
created: 2026-08-28
---

## Problem

Buildings within the anchor radius (50 m, `config/base.yaml`) of an anchor load are switched from Networked heat pump to Communal solution, and since #485 their clusters carry `communal_origin = "anchor proximity"`. But the anchor itself is invisible to that story:

- The tool geojson ships anchor loads as a geometry-only layer (`layer: "anchor_loads"`, added in `compute_contextual_features`), so the frontend can draw them but cannot reference one.
- Cluster features carry only a boolean `within_50m_from_anchor_load`; nothing says _which_ anchor.
- `reassign_gdf_near_anchor_properties` finds each building's nearest anchor and discards its identity (`drop(columns="index_right")`).
- The cluster stage and the contextual-features stage each load anchors independently, so there is no shared identity to link on.

The sprint review (2026-08) considered merging the anchor's geometry into its cluster (three rules, explored on branch `457_anchorClusterOptionsExploration`) and rejected all of them in favour of highlighting the anchor in the UI when a cluster is selected. That needs a data link, not a geometry change.

## Proposal

Thread anchor identity from the anchor list to the geojson, exposing both ends of a cluster→anchor link. Decisions settled in the kickoff interview, with rationale:

1. **Link rule: a cluster references every anchor that flipped at least one of its buildings.** Causal and consistent with the logic trace; many-to-many is harmless now that nothing merges geometry. Only anchor-origin communal clusters carry links. _Rejected: all anchors within 50 m (overstates the anchor's role for clusters that were communal anyway — the PR #466 objection); nearest anchor only (hides real multi-anchor cases: 18 contested clusters in Plymouth)._

   **Tension to keep visible — proximity is not causation.** A block of flats within 50 m of a school is communal because it is a block of flats, so it carries `communal_origin = "block of flats"`, `anchor_ids = null` and is _not_ highlighted when selected — even though `within_50m_from_anchor_load` is `True` for it. The link means "this anchor is why the cluster is communal", not "an anchor is nearby". The three states a communal cluster can be in:

   | Cluster                          | `communal_origin` | `anchor_ids` | `within_50m_from_anchor_load` |
   | -------------------------------- | ----------------- | ------------ | ----------------------------- |
   | Houses switched by a school      | anchor proximity  | `["…"]`      | True                          |
   | Flats that happen to be near one | block of flats    | null         | True                          |
   | Flats nowhere near an anchor     | block of flats    | null         | False                         |

   The boolean flag is kept unchanged precisely so the frontend can give the middle row a weaker cue (for example a dotted outline) without a backend change, if the product view wants proximity shown.

2. **Contract: enrich the existing single geojson.** Anchor-load features gain an `anchor_id` property; anchor-origin cluster features gain an anchor-ID list. No new layer or file — matches the frontend's "an ID could work" suggestion and their hesitancy about juggling layers. _Rejected: a standalone anchors file (adds a fetch and a layer)._
3. **Single source of anchor IDs: the cluster stage saves an anchors dataset** (id + geometry: all anchors in the LA, plus any just outside it that a cluster references) to the dated release directory; the contextual-features stage loads it for the `anchor_loads` layer instead of re-deriving anchors. The drawn polygons and the cluster links cannot disagree. _Rejected: deriving IDs independently in both stages (silent divergence if inputs differ)._
4. **ID format: geometry-hash** — a short hex digest of the normalised footprint WKB. Stable across runs, releases and LAs while the footprint is unchanged. _Rejected: sequential per run (shuffles whenever the list changes); OS building IDs (mixed provenance across the two anchor sources; can change between OS releases)._
5. **Layer content: all anchors ship, all with IDs.** The layer keeps its role as general context; clusters reference the subset that caused their buildings to be reassigned; the frontend filters by ID. Anchors just outside the local authority that a cluster references also ship (reassignment searches whole grid squares), so every ID in `anchor_ids` resolves; found in review. _Rejected: shipping only linked anchors (changes the layer's meaning silently)._
6. **Tests: fold in #392.** The reassignment function is rewritten to keep anchor identity, so it gets the tests #392 asked for, including the equidistant-anchor case — which also fixes the latent duplication (`sjoin_nearest` returns one row per tied anchor and the pipeline never deduplicated).
7. **Branch: stacked on `485_splitCommunalClustersByOrigin`.** Links are defined via `communal_origin`, which only exists post-split. Merges after #485.
8. **Issue: re-scope #457** rather than open a new one, so the exploration and the sprint-review decision stay on one thread; #453 was closed by PR #462, which added the geometry-only `anchor_loads` layer; this work adds the IDs to it.

Implementation sketch (pipeline only):

- `load_transform_anchor_property_gdfs` assigns `anchor_id` after its normalise/dedupe step.
- Cluster stage saves the anchors dataset (new `output.dataset` entry in `config/base.yaml`, saved via `save_utils`, manifest recorded) alongside `tech_clusters`.
- `reassign_gdf_near_anchor_properties` keeps the nearest anchor's ID for flipped buildings (deterministic tie-break, one row per building) and returns it as a column.
- `generate_gdf_clusters` aggregates per cluster the sorted unique anchor IDs of its flipped buildings; `tech_clusters` gains the list column (null for non-anchor-origin clusters).
- `compute_contextual_features` loads the anchors dataset for the `anchor_loads` layer (now carrying `anchor_id`) and passes the cluster list column through; metadata descriptions cover both new properties. Existing properties, including `within_50m_from_anchor_load`, are unchanged.

## Alternatives considered

- **Merge the anchor's geometry into its cluster** — three rules explored on Plymouth data (merge all flagged clusters; nearest-anchor tiebreak; contiguity). Rejected at the sprint review: each either crosses barriers or isolates most anchors (172 of 639 under contiguity), and the UI can show the relationship without changing cluster geometry.
- Per-decision rejections are listed inline in the Proposal.

## Out of scope

- Frontend implementation of the outline-on-select behaviour.
- Any change to cluster geometry, block-of-flats clusters, or the anchor category list.
- Retiring `within_50m_from_anchor_load` (it becomes derivable from the ID list, but removing it would break the current frontend contract).
- Making the building-to-cluster join robust to floating-point rounding at cluster edges. A building fractionally outside its cluster's outline is not attached, so it is missing from the cluster's flag and `anchor_ids` (seen on Plymouth as `COM_99_plymouth`). This predates #457 and is tracked in [#524](https://github.com/nestauk/asf_heat_pump_suitability/issues/524).

## Open questions

- Should the UI give a weaker visual cue to communal clusters that are near an anchor without being caused by one (`within_50m_from_anchor_load` true, `anchor_ids` null)? The data supports it; the product call is the frontend's.

- Property name for the cluster-side list (proposed `anchor_ids`) — confirm with the frontend before the PR opens.
- Serialisation of the list column: native list type in parquet and a JSON array in the geojson, versus a delimited string — decide at implementation with the frontend's parsing preference.
- Whether the anchors dataset should carry a category/type column (school, hospital, …) for future UI labelling; the anchor sources expose it unevenly.
- Asana task for this re-scoped work (frontmatter `asana: TBD`).

## Verification

- [ ] A per-LA anchors dataset with stable geometry-derived IDs is saved to the dated release directory and consumed by both the cluster and contextual-features stages
- [ ] Reassignment records the flipping anchor's ID for each switched building, with equidistant ties handled deterministically and no duplicated buildings
- [ ] Each anchor-origin communal cluster lists the anchor IDs that flipped at least one of its buildings; non-anchor-origin clusters carry no list
- [ ] In the tool geojson, anchor-load features carry the anchor ID and anchor-origin cluster features carry the ID list; all existing properties and layers are unchanged
- [ ] All anchors in the local authority ship in the anchor layer, not only linked ones
- [ ] Tests cover the reassignment function, including the equidistant-anchor case (#392)
- [ ] Output metadata descriptions cover the new properties
