---
title: Add each Voronoi seed once and bound the duplicate jitter
status: draft
github_issue: https://github.com/nestauk/asf_heat_pump_suitability/issues/561
pr:
asana:
created: 2026-10-26
---

## Problem

See the issue for the full analysis and the agreed plan (comment
https://github.com/nestauk/asf_heat_pump_suitability/issues/561#issuecomment-6420296340).
In short, `cluster.extend_edges_gdf` makes the Voronoi seeds in two steps, and both have a problem:

- Line 325 puts a seed every 1 m along each outline, starting at 0 m, which is the first corner. Line 327 then adds
  all corners from `exterior.coords[:-1]`, which includes the first corner again. So every building adds its first
  corner two times.
- Line 346 moves the k-th duplicate east by k × 0.1 mm, with k counted across the whole LA, not inside each group of
  identical points.

Facts (Windsor and Maidenhead, from #561, `dev` 7f3628f6):

| Duplicate seeds                                                           | Count  |
| ------------------------------------------------------------------------- | ------ |
| First corner of a building, added two times                               | 37,329 |
| Another corner that is exactly a whole number of metres along the outline | 16     |
| A corner that two different buildings share                               | 12     |
| Total                                                                     | 37,357 |

- Largest shift: 3.74 m. 27,357 seeds move more than 1 m, and 235 land inside a different building.
- The seed order follows the grid-square load order, which comes from a Python `set`. In the order TQ, SU one moved
  seed lands exactly on a seed that did not move, and `voronoi_polygons(..., ordered=True)` raises
  `Multiple input coordinates in cell at 492949 166056`. Reproduce with `PYTHONHASHSEED=0` (fails) and `=1` (passes).
- Shapely always closes a ring (last coordinate = first), so `coords[:-1]` removes the closing copy correctly for any
  vertex order. A repeated vertex (`[A, A, ...]`) would add one more duplicate, but Windsor and Maidenhead has none
  (the three counts above add up to the total).
- `voronoi_polygons` with `ordered=False` merges identical seeds (4 seeds give 3 cells); `ordered=True` raises. The
  code needs `ordered=True`, because cell n is paired with seed n and its building ID (line 357).

## Proposal

All changes are in `extend_edges_gdf`.

1. **Line 327 adds the corners with `exterior.coords[1:-1]`.** Rationale: line 325 always adds the first corner as
   `interpolate(0)`, so the corner list does not need it. This removes 37,329 of the 37,357 duplicates. It is the
   smallest change at the cause (Roisin's proposal).
2. **Number the copies of each point separately.** The first copy stays where it is, the second copy moves 0.05 mm
   east, and the third copy moves 0.10 mm east. Today the numbers run across the whole LA (1 to 37,357), so the last
   copies move metres. To find which copy is the first, sort the seeds with a stable sort
   (`np.argsort(..., kind="stable")`). A stable sort keeps equal points in their original order, so the copy that
   comes first in the seed list always stays. Rationale: the largest shift then depends only on how many buildings
   share one corner (usually 2 or 3), not on the number of buildings in the LA.
3. **The jitter step is 0.05 mm (`5e-5` m), as a named module-level constant.** Rationale: Aidan's choice, between a
   smaller step (Roisin) and the risk that points very close together make the GEOS input more nearly degenerate
   (the #556 type of failure). Not tested before this spec, so the acceptance runs cover it.
4. **A check before `voronoi_polygons` raises a `ValueError` if any duplicate seeds remain after the jitter.**
   Rationale: a future regression gives an error that names the cause, not only a GEOS coordinate. About a few
   seconds for 2.4 M seeds.
5. **One new test in `pipeline/cluster/tests/test_cluster.py`:** two buildings that share a corner give two Voronoi
   cells, and each cell contains its building. Rationale: it covers the shared-corner case that the jitter exists
   for (Roisin's proposal), and fits the high-signal testing rule.
6. **Acceptance runs (no `--save`):** Windsor and Maidenhead with `PYTHONHASHSEED=0` and `=1`, then
   `compare_versions --stage cluster --trigger methodology_change` on the sample LAs. Rationale: the fix changes cells
   in every LA, so the team must see the size of the change.

Owner: Aidan builds it; Roisin reviews.

## Alternatives considered

- **De-duplicate per building** (for example a dict keyed by coordinate in the densify loop). It also removes the 16
  whole-metre corners and repeated vertices. Rejected: a larger change for the same practical result, because those
  few remaining duplicates only move 0.05 mm after decision 2.
- **Start the 1 m points at `i = 1`.** It has the same effect as decision 1. Rejected: decision 1 keeps the even
  spacing from the first corner and changes only the corner list.
- **Keep the 0.1 mm step, or use a smaller step (for example 0.01 mm).** Rejected in favour of 0.05 mm (decision 3).
- **No duplicate check.** Rejected: the check is cheap and makes a regression clear.
- **Sort the grid squares (`sorted()`) in this fix.** Rejected: after the fix the result does not depend on the load
  order, and the change touches `local_authority.py`, which every stage uses.

## Out of scope

- A fixed grid-square load order (`sorted()`). Not an issue yet; Aidan decides separately.
- Robustness to GEOS versions (#558) and the shapely cap (#556, merged as #559).
- Other duplicate sources outside `extend_edges_gdf`.

## Open questions

- Does 0.05 mm pass for every sample LA and for Windsor and Maidenhead in both orders? The acceptance runs answer it.
- Does the fix reduce the number of buildings with no cluster (#519, #524)? A moved seed inside a building with a
  different technology could push that building partly outside its cluster. Compare the missing-cluster count before
  and after in the `compare_versions` output.
- Should the jitter become a small private helper, so that a test can check the shift directly? Decide in
  `/implement`, only if the shared-corner test cannot show a regression on its own.

## Verification

- [ ] `extend_edges_gdf` adds the first corner of each building one time only.
- [ ] The n-th copy of a duplicated coordinate moves east by n × 0.05 mm; the first copy does not move.
- [ ] A remaining duplicate seed raises a `ValueError` with a clear message before `voronoi_polygons`.
- [ ] The new shared-corner test passes, and `uv run pytest` passes.
- [ ] Windsor and Maidenhead `cluster.py` runs to the end with `PYTHONHASHSEED=0` and `=1` (no `--save`).
- [ ] `compare_versions --stage cluster --trigger methodology_change` on the sample LAs shows the change, and the team
      accepts it.
