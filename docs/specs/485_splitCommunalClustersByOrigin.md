---
title: Separate communal clusters by assignment origin
status: draft
github_issue: https://github.com/nestauk/asf_heat_pump_suitability/issues/485
pr:
asana: https://app.asana.com/1/5571817120120/project/1214222223606748/task/1217818496734712
created: 2026-08-26
---

## Problem

Buildings are assigned "Communal solution" by two different routes:

1. **Block of flats** outside a heat-network zone / city centre — decision tree (`pipeline/transform/decision_tree.py`, `identify_dict_most_suitable_tech`).
2. **Anchor proximity** — a Networked-heat-pump building within `anchor_radius` (50 m, `config/base.yaml`) of an anchor load is reassigned to Communal (`pipeline/cluster/cluster.py`, `reassign_gdf_near_anchor_properties`).

A third route existed on `dev` when this spec was drafted — a footprint whose UPRNs split between "District heat network" and "Networked heat pump" resolved to Communal in `assign_df_unique_solution` — but the base branch chain (#452/#459) removed the DHN decision-tree path and that mixing rule (commit 67de5cc), so it cannot occur here. See decision 4.

Clustering dissolves cells by `assigned_tech` alone (`generate_gdf_clusters`), so touching communal buildings merge into one cluster regardless of route. The per-cluster `logic_trace` added on branch `459_logic_trace` (PR #466) therefore cannot state one truthful reason for a communal cluster and hedges with "There might also be one or multiple blocks of flats in the cluster"; two TODOs in that branch name this separation as the fix. Routes 1 and 3 also describe different physical propositions (a self-contained per-building system vs a shared scheme around an anchor), so merging them misrepresents both.

The origin information exists at UPRN level (`decision_tree_path`, `in_block_of_flats`) but is dropped before clustering: `buildings_most_suitable_tech` carries only `ID`, `assigned_tech`, `geometry`.

## Proposal

Interview decisions (2026-08-26), each with rationale:

1. **Add a building-level communal-origin column** with two values: block of flats, anchor proximity. Set the first in `decision_tree.py` when the building resolves to Communal and keep the column in `buildings_most_suitable_tech`; set the second in `cluster.py` during anchor reassignment. _Rationale: cheapest place to capture each origin is where it is decided; downstream steps then never have to reverse-engineer it._
2. **Decision-tree origin takes precedence** when a building is also within the anchor radius. Anchor reassignment only flips Networked-heat-pump buildings, so already-communal buildings naturally keep their decision-tree origin; proximity stays visible via the existing `within_{radius}m_from_anchor_load` flag. _Rationale: matches the PR #466 review point that a building "could have been assigned a communal solution regardless and happen to be within anchor load radius"; needs no extra logic._
3. **Dissolve on `(assigned_tech, communal_origin)`** instead of `assigned_tech` alone in `generate_gdf_clusters`. Adjacent communal buildings with the same origin still merge; different origins never do. Cluster IDs are built after the dissolve, so `COM_n_{la}` IDs keep working. _Rationale: minimal change that delivers exactly the separation the PR #466 TODOs describe._
4. **Mixed-building origin dropped** (amended 2026-08-26; originally "mixed buildings get their own origin value"). Implementation found the mixed DHN/NHP → Communal rule does not exist on this branch's base — the #452/#459 chain removed the DHN decision-tree path and the mixing rule (commit 67de5cc) — so a third origin value would be dead code. _Rationale: no untriggerable code; if DHN mixing returns after the chain merges, adding a third origin value is a small additive change with real behaviour to test against._
5. **This work includes the logic-trace rewrite** in `pipeline/run/compute_contextual_features.py`: communal trace text keyed on cluster origin, hedging sentences and both TODOs removed, and the two runtime bugs on the communal branches fixed (missing comma making a tuple get called; `assigned_tech == tech_types["communal"] & dhn_potential` operator precedence). _Rationale: the trace is the user-visible payoff; leaving it to a later PR ships the plumbing without the fix._
6. **`communal_origin` stays in the tool-facing output, documented** (review decision, 2026-08-27): the column flows into the cluster geojson and gets a "Variable names and descriptions" metadata row in `base.yaml`. _Rationale: the front end can then show why a cluster is communal — the point of the feature; an undocumented column would be an unexplained contract._
7. **Branch is stacked on `459_logic_trace`** (PR #466) rather than `dev`, diverging from the usual branch-off-`dev` rule. PRs into `459_logic_trace`, or retargets `dev` if #466 merges first. _Rationale: the trace rewrite edits code that only exists on that branch; stacking respects PR #466's ownership instead of superseding it. Needs coordinating with its author._

## Alternatives considered

- **Each block of flats as its own single-building cluster** — semantically truthful (a communal system serves one building) but sharply raises cluster counts and changes what the tool displays; rejected in favour of same-origin merging. Revisit if user testing finds merged flats-blocks confusing.
- **Anchor proximity takes precedence over decision-tree origin** — would grow anchor-scheme clusters but overstates the anchor's role for buildings that were communal anyway; rejected.
- **Fold mixed buildings into another origin** — fewer categories but an untrue trace for those clusters; rejected (then made moot when the mixing rule turned out to be removed upstream).
- **Add a dormant third origin value for mixed buildings** — ready if the DHN mixing rule returns, but ships untriggerable code and a trace sentence no cluster can receive; rejected.
- **Clustering-only scope, trace rewrite left to PR #466** — smallest PR, no coordination risk, but delays the user-visible fix; rejected.
- **Branch off `dev` and supersede PR #466** — avoids building on a known-buggy branch but takes over a colleague's open PR; rejected in favour of stacking.

## Out of scope

- Including the anchor building itself in clusters (issue #457) and a separate anchor map layer (issue #453).
- A "mixed solutions within building" origin — the rule that produced such buildings was removed upstream (see decision 4); revisit only if DHN mixing returns.
- Non-communal trace wording changes beyond what the rewrite requires.
- The disabled island reassignment `reassign_gdf_communal_networked` (`cluster.py`) — but if it is ever re-enabled it must set/preserve the origin column.

## Open questions

- Exact user-facing wording of the communal trace sentences (product/Roisin call; engineering only needs the origin split).
- Naming of the origin values and column (settle in implementation, follow config conventions in `base.yaml` if they become constants).

## Verification

Acceptance criteria from [issue #485](https://github.com/nestauk/asf_heat_pump_suitability/issues/485):

- [ ] Building-level tech output includes a communal-origin value for every communal building
- [ ] No cluster contains communal buildings with different origins; same-origin neighbours still merge
- [ ] A block of flats within the anchor radius keeps origin "block of flats"
- [ ] Communal trace text has no hedging sentence and the two TODOs are removed
- [ ] Logic-trace code runs without error (tuple-call and `&`-precedence bugs fixed)
- [ ] Tests cover origin assignment and origin-aware dissolve
