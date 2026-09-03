"""
y-Check algorithm for GMSPP (per-strip feasibility check).

Given a set of items with fixed x-positions on a strip of width W
and a target height H, determine whether y-coordinates can be assigned
so that no two items sharing a column overlap vertically, all within
height H.

Implementation: Côté-style skyline enumeration tree with three
preprocessing steps (merge, lift widths, shrink strip) and five
fathoming criteria.

Reference: Côté, Dell'Amico, and Iori (2014), "Combinatorial Benders'
Cuts for the Strip Packing Problem", Operations Research 62(3), §3.
"""

from typing import List, Tuple, Dict, Optional, Set
from dataclasses import dataclass
import numpy as np


@dataclass
class YCheckItem:
    """Item for y-check: has an x-position and width on a specific strip."""
    id: int           # original item id
    x_pos: int        # x-coordinate (left edge)
    width: int        # item width
    height: int       # item height

    @property
    def x_right(self) -> int:
        return self.x_pos + self.width


def items_overlap_x(a: YCheckItem, b: YCheckItem) -> bool:
    """Check if two items share at least one column (x-overlap)."""
    return a.x_pos < b.x_right and b.x_pos < a.x_right


def compute_column_heights(
    items: List[YCheckItem], strip_width: int
) -> np.ndarray:
    """
    Compute the total item height covering each column.
    column_height[q] = sum of h_j for items covering column q.
    """
    col_h = np.zeros(strip_width, dtype=int)
    for it in items:
        col_h[it.x_pos:it.x_right] += it.height
    return col_h


def _get_conflict_pairs(items: List[YCheckItem]) -> List[Tuple[int, int]]:
    """Return list of (idx_a, idx_b) pairs where items overlap in x."""
    pairs = []
    for a_idx, a in enumerate(items):
        for b_idx, b in enumerate(items):
            if a_idx < b_idx and items_overlap_x(a, b):
                pairs.append((a_idx, b_idx))
    return pairs


# ═══════════════════════════════════════════════════════════════════════
#  Internal data structure for the skyline enumeration tree
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class _SkylineItem:
    """Internal item representation (may be modified by preprocessing)."""
    orig_id: int      # original YCheckItem.id (for mapping back)
    x_pos: int        # left edge (column index, possibly shifted)
    width: int        # width (possibly enlarged by preprocessing)
    height: int       # height (unchanged)

    @property
    def x_right(self) -> int:
        return self.x_pos + self.width


# ═══════════════════════════════════════════════════════════════════════
#  Preprocessing (Côté 2014, §3.2)
# ═══════════════════════════════════════════════════════════════════════

def _preprocessing_merge(items: List[_SkylineItem], strip_width: int,
                         target_height: int) -> List[_SkylineItem]:
    """
    Preprocessing 1: Merge Items.

    For each item j (in nondecreasing x_pos order), identify items that
    can only be packed to its left L(j) or right R(j). Try to merge j
    with L(j) items into a single wider item.  Similarly for R(j).

    Following Côté 2014, §3.2, Preprocessing 1.
    """
    if len(items) <= 1:
        return items

    # Sort by nondecreasing x_pos
    items = sorted(items, key=lambda it: it.x_pos)

    merged = True
    while merged:
        merged = False
        new_items = []
        skip = set()

        for idx, j in enumerate(items):
            if idx in skip:
                continue

            # L(j): items that can only be packed to the left of j
            # i.e., i where p_i + w_i <= p_j  (i ends before j starts)
            L_j = [i_idx for i_idx, i in enumerate(items)
                   if i_idx != idx and i_idx not in skip
                   and i.x_pos + i.width <= j.x_pos]

            # Try to merge: if all L(j) items have h_i <= h_j and they
            # fit in the substrip [0, p_j) of height h_j, merge them
            # with j into a wider item.
            # Simplified: merge j with a single left neighbor if it fits
            # directly adjacent and has same or smaller height.
            did_merge = False
            for l_idx in L_j:
                l_item = items[l_idx]
                if (l_item.x_pos + l_item.width == j.x_pos
                        and l_item.height <= j.height):
                    # Merge: new item spans from l_item.x_pos with
                    # combined width, height = j.height
                    merged_item = _SkylineItem(
                        orig_id=j.orig_id,
                        x_pos=l_item.x_pos,
                        width=l_item.width + j.width,
                        height=j.height,
                    )
                    new_items.append(merged_item)
                    skip.add(l_idx)
                    merged = True
                    did_merge = True
                    break

            if not did_merge:
                # Try R(j): items that can only be packed to the right
                R_j = [i_idx for i_idx, i in enumerate(items)
                       if i_idx != idx and i_idx not in skip
                       and i.x_pos >= j.x_pos + j.width]

                for r_idx in R_j:
                    r_item = items[r_idx]
                    if (j.x_pos + j.width == r_item.x_pos
                            and r_item.height <= j.height):
                        merged_item = _SkylineItem(
                            orig_id=j.orig_id,
                            x_pos=j.x_pos,
                            width=j.width + r_item.width,
                            height=j.height,
                        )
                        new_items.append(merged_item)
                        skip.add(r_idx)
                        merged = True
                        did_merge = True
                        break

                if not did_merge:
                    new_items.append(j)

        items = sorted(new_items, key=lambda it: it.x_pos)

    return items


def _preprocessing_lift_widths(items: List[_SkylineItem],
                               strip_width: int) -> List[_SkylineItem]:
    """
    Preprocessing 2: Lift Item Widths.

    For each item j (nondecreasing width, breaking ties by
    nondecreasing height), compute the leftmost it can be pushed
    (ℓ_j) and rightmost it can extend (r_j), then set
    p_j = ℓ_j and w_j = r_j - ℓ_j.

    L(j) and R(j) are computed against ALL other items (both already-
    processed and not-yet-processed) using their CURRENT positions,
    following Côté 2014, §3.2, Preprocessing 2, eqs. (27)-(28).
    """
    if not items:
        return items

    # Work on a mutable copy; sort by nondecreasing width, then height
    # to determine processing order
    work = [_SkylineItem(orig_id=it.orig_id, x_pos=it.x_pos,
                         width=it.width, height=it.height)
            for it in items]
    order = sorted(range(len(work)), key=lambda k: (work[k].width, work[k].height))

    for idx in order:
        j = work[idx]

        # L(j): ALL other items packed entirely to the left of j
        # R(j): ALL other items packed entirely to the right of j
        # Using current (possibly already-lifted) positions
        ell_j = 0
        r_j = strip_width
        for k, other in enumerate(work):
            if k == idx:
                continue
            if other.x_right <= j.x_pos:
                # other is entirely left of j
                ell_j = max(ell_j, other.x_right)
            if other.x_pos >= j.x_right:
                # other is entirely right of j
                r_j = min(r_j, other.x_pos)

        # Move j as far left as possible, then enlarge as far right
        work[idx] = _SkylineItem(
            orig_id=j.orig_id,
            x_pos=ell_j,
            width=r_j - ell_j,
            height=j.height,
        )

    return work


def _preprocessing_shrink_strip(
    items: List[_SkylineItem], strip_width: int
) -> Tuple[List[_SkylineItem], int]:
    """
    Preprocessing 3: Shrink the Strip.

    Remove columns where no item's left border is placed. Only keep
    columns that are "active" (have some item starting there),
    reducing strip width and adjusting item widths accordingly.

    Following Côté 2014, §3.2, Preprocessing 3.

    Returns (new_items, new_strip_width).
    """
    if not items:
        return items, strip_width

    # Collect all columns where an item's left border is placed
    active_cols = sorted(set(it.x_pos for it in items))

    if not active_cols:
        return items, strip_width

    # Add the strip right edge as a sentinel
    active_cols.append(strip_width)

    # Map old columns to new compressed columns
    col_map = {}
    new_col = 0
    for i, col in enumerate(active_cols[:-1]):
        col_map[col] = new_col
        # The gap from this active col to the next defines the
        # compressed width contribution
        new_col += 1  # each active column becomes one unit...

    # Actually, the correct approach: keep only the active columns,
    # and adjust widths so items span the correct set of active columns.
    # For each item, its new width = number of active columns in
    # [x_pos, x_pos + width).
    # New strip width = number of active columns.

    new_strip_width = len(active_cols) - 1  # excluding sentinel

    # Build mapping: for each active column index, what's its new index
    col_to_new = {}
    for new_idx, col in enumerate(active_cols[:-1]):
        col_to_new[col] = new_idx

    new_items = []
    for it in items:
        new_x = col_to_new[it.x_pos]
        # New width = number of active columns in [x_pos, x_right)
        new_w = sum(1 for c in active_cols[:-1]
                    if it.x_pos <= c < it.x_right)
        new_items.append(_SkylineItem(
            orig_id=it.orig_id,
            x_pos=new_x,
            width=new_w,
            height=it.height,
        ))

    return new_items, new_strip_width


# ═══════════════════════════════════════════════════════════════════════
#  Skyline enumeration tree (Côté 2014, §3.2)
# ═══════════════════════════════════════════════════════════════════════

def _skyline_ycheck(
    items: List[_SkylineItem],
    strip_width: int,
    target_height: int,
    max_iterations: int = 2_000_000,
) -> bool:
    """
    Skyline enumeration tree for y-check (optimized).

    Uses tuple-based skyline (no array copies), precomputed item data,
    and incremental fathoming for performance.

    Returns True if feasible (all items can be packed), False if infeasible.
    """
    n = len(items)
    W = strip_width
    H = target_height

    if n == 0:
        return True

    # ── Precompute item data as plain tuples for speed ──
    # (x_pos, x_right, width, height) indexed by item index
    items = sorted(items, key=lambda it: (it.x_pos, it.width))
    ix = tuple(it.x_pos for it in items)
    ir = tuple(it.x_right for it in items)
    iw = tuple(it.width for it in items)
    ih = tuple(it.height for it in items)
    ia = tuple(iw[j] * ih[j] for j in range(n))  # areas

    # Quick column-load check
    col_load = [0] * W
    for j in range(n):
        for p in range(ix[j], ir[j]):
            col_load[p] += ih[j]
    if max(col_load) > H:
        return False

    # Precompute: for each column p, which items cover it
    col_items = [[] for _ in range(W)]
    for j in range(n):
        for p in range(ix[j], ir[j]):
            col_items[p].append(j)

    # Precompute remaining column loads per item (for incremental update)
    # remaining_load[p] = sum of heights of remaining items covering p
    # We'll track this incrementally.

    # ── DFS with explicit stack ──
    # State: (skyline_tuple, remaining_frozenset, remaining_area,
    #         remaining_col_load_tuple, wasted_space)
    #
    # Using tuples for skyline avoids array copy overhead.
    #
    # Free-space bound (Côté 2014): track cumulative wasted space from
    # niche closures and left-gap fills.  The maximum allowable waste is
    # W*H − Σ(item areas).  If wasted_space exceeds this budget at any
    # node, the branch is infeasible — prune immediately.

    sky0 = (0,) * W
    rem0 = frozenset(range(n))
    total_area = sum(ia)
    rcl0 = tuple(col_load)  # remaining column loads
    allowed_waste = W * H - total_area  # free-space budget

    stack = [(sky0, rem0, total_area, rcl0, 0)]  # last element: wasted_space
    iterations = 0

    while stack:
        iterations += 1
        if iterations > max_iterations:
            return True  # conservative: don't generate invalid cuts

        sky, remaining, rem_area, rcl, wasted = stack.pop()

        if not remaining:
            return True  # all items packed

        # ── Find the niche ──
        min_h = min(sky)
        ell = sky.index(min_h)

        r = ell
        while r < W - 1 and sky[r + 1] == min_h:
            r += 1

        h_left = H if ell == 0 else sky[ell - 1]
        h_right = H if r == W - 1 else sky[r + 1]
        h_wall = min(h_left, h_right)
        niche_width = r - ell + 1

        # ── Find niche items ──
        niche_items = [j for j in remaining
                       if ix[j] >= ell and ir[j] <= r + 1]

        if not niche_items:
            # Close niche — the entire rectangle is wasted
            close_waste = (h_wall - min_h) * niche_width
            new_wasted = wasted + close_waste
            if new_wasted > allowed_waste:
                continue  # free-space bound: prune
            new_sky = list(sky)
            for p in range(ell, r + 1):
                new_sky[p] = h_wall
            stack.append((tuple(new_sky), remaining, rem_area, rcl,
                          new_wasted))
            continue

        # Sort by x_pos (stable, for symmetry breaking)
        niche_items.sort(key=lambda j: (ix[j], ih[j]))

        # ── Closed-niche dominance (Côté criterion 2) ──
        # If any niche item fits entirely below both walls, closing the
        # niche without placing it is dominated — skip the close branch.
        has_dominant = False
        for j in niche_items:
            if min_h + ih[j] <= h_wall:
                has_dominant = True
                break

        if not has_dominant:
            # ── Close niche branch (pack nothing) ──
            close_waste = (h_wall - min_h) * niche_width
            new_wasted = wasted + close_waste
            if new_wasted <= allowed_waste:  # free-space bound check
                close_sky = list(sky)
                for p in range(ell, r + 1):
                    close_sky[p] = h_wall
                stack.append((tuple(close_sky), remaining, rem_area, rcl,
                              new_wasted))

        # ── Branch on each niche item (reverse for DFS order) ──
        seen_sig = set()
        for j in reversed(niche_items):
            # Symmetry breaking: skip items with identical (x, w, h)
            sig = (ix[j], iw[j], ih[j])
            if sig in seen_sig:
                continue
            seen_sig.add(sig)

            # Column-load feasibility for this item
            will_exceed = False
            for p in range(ix[j], ir[j]):
                if sky[p] + ih[j] > H:
                    will_exceed = True
                    break
            if will_exceed:
                continue

            # ── Left-gap column-load check (Côté lines 273–281) ──
            # When the item doesn't start at the niche edge, the left
            # gap [ell, ix[j]) is filled to min(h_left, min_h+ih[j]).
            # Check that remaining items covering each gap column can
            # still fit above the new floor height.
            fill = min(h_left, min_h + ih[j]) if ix[j] > ell else 0
            if ix[j] > ell:
                gap_infeasible = False
                fill_h = fill - min_h  # additional height added to gap cols
                for p in range(ell, ix[j]):
                    if rcl[p] > H - fill:
                        gap_infeasible = True
                        break
                if gap_infeasible:
                    continue

            # ── Left-gap dominance (Côté lines 286–291) ──
            # If another niche item k fits entirely in the left gap
            # (between niche edge ell and item j's x-position), skip
            # this placement — the "twin" node that packs k in the
            # gap first covers this case.
            if ix[j] > ell:
                gap_dominated = False
                for k in niche_items:
                    if k == j:
                        continue
                    # k fits in the gap if: starts at/after ell,
                    # ends at/before ix[j], and height <= fill_h
                    if (ix[k] >= ell and ix[k] + iw[k] <= ix[j]
                            and ih[k] <= fill_h):
                        gap_dominated = True
                        break
                if gap_dominated:
                    continue

            # Build new skyline and compute wasted space from left-gap fill
            new_sky = list(sky)
            item_wasted = 0
            if ix[j] > ell:
                for p in range(ell, ix[j]):
                    item_wasted += fill - min_h  # height gained, no item
                    new_sky[p] = fill
            for p in range(ix[j], ir[j]):
                new_sky[p] = sky[p] + ih[j]

            new_wasted = wasted + item_wasted
            if new_wasted > allowed_waste:
                continue  # free-space bound: prune

            new_rem = remaining - {j}
            new_area = rem_area - ia[j]
            new_rcl = list(rcl)
            for p in range(ix[j], ir[j]):
                new_rcl[p] -= ih[j]

            # Column-load bound: check all columns and compute residual
            fathom = False
            residual = 0
            for p in range(W):
                gap = H - new_sky[p]
                if new_rcl[p] > gap:
                    fathom = True
                    break
                residual += gap
            if fathom:
                continue

            # Area bound
            if new_area > residual:
                continue

            stack.append((tuple(new_sky), new_rem, new_area,
                          tuple(new_rcl), new_wasted))

    return False


# ═══════════════════════════════════════════════════════════════════════
#  Numba-accelerated skyline enumeration (Côté 2014, §3.2)
# ═══════════════════════════════════════════════════════════════════════

try:
    import numba

    @numba.njit(cache=True)
    def _skyline_numba_recurse(
        sky, remaining, rcl, rem_area, wasted,
        ix, ir, iw, ih, ia,
        n, W, H, allowed_waste, max_iter, iter_count,
    ):
        """
        Recursive skyline enumeration with in-place backtracking.

        sky, rcl: modified in-place and restored on backtrack.
        remaining: boolean array, modified in-place and restored.
        iter_count: single-element array used as mutable counter.

        Returns True if feasible.
        """
        iter_count[0] += 1
        if iter_count[0] > max_iter:
            return True  # conservative

        # Check if all items placed
        n_remaining = 0
        for j in range(n):
            if remaining[j]:
                n_remaining += 1
        if n_remaining == 0:
            return True

        # ── Find the niche ──
        min_h = sky[0]
        ell = 0
        for p in range(1, W):
            if sky[p] < min_h:
                min_h = sky[p]
                ell = p

        r = ell
        while r < W - 1 and sky[r + 1] == min_h:
            r += 1

        h_left = H if ell == 0 else sky[ell - 1]
        h_right = H if r == W - 1 else sky[r + 1]
        h_wall = min(h_left, h_right)
        niche_width = r - ell + 1

        # ── Find niche items (sorted by x, then h) ──
        # Collect indices into a local array
        niche_buf = np.empty(n, dtype=numba.int32)
        niche_count = 0
        for j in range(n):
            if remaining[j] and ix[j] >= ell and ir[j] <= r + 1:
                niche_buf[niche_count] = j
                niche_count += 1

        # Sort niche items by (ix, ih) — simple insertion sort (small n)
        for i in range(1, niche_count):
            key = niche_buf[i]
            j2 = i - 1
            while j2 >= 0 and (ix[niche_buf[j2]] > ix[key] or
                               (ix[niche_buf[j2]] == ix[key] and
                                ih[niche_buf[j2]] > ih[key])):
                niche_buf[j2 + 1] = niche_buf[j2]
                j2 -= 1
            niche_buf[j2 + 1] = key

        if niche_count == 0:
            # Close niche — entire rectangle is wasted
            close_waste = (h_wall - min_h) * niche_width
            new_wasted = wasted + close_waste
            if new_wasted > allowed_waste:
                return False
            # Modify skyline in-place
            old_vals = np.empty(niche_width, dtype=numba.int32)
            for p in range(ell, r + 1):
                old_vals[p - ell] = sky[p]
                sky[p] = h_wall
            result = _skyline_numba_recurse(
                sky, remaining, rcl, rem_area, new_wasted,
                ix, ir, iw, ih, ia, n, W, H, allowed_waste,
                max_iter, iter_count)
            # Restore
            for p in range(ell, r + 1):
                sky[p] = old_vals[p - ell]
            return result

        # ── Closed-niche dominance (Côté criterion 2) ──
        has_dominant = False
        for ni in range(niche_count):
            j = niche_buf[ni]
            if min_h + ih[j] <= h_wall:
                has_dominant = True
                break

        # ── Close niche branch (if not dominated) ──
        if not has_dominant:
            close_waste = (h_wall - min_h) * niche_width
            new_wasted = wasted + close_waste
            if new_wasted <= allowed_waste:
                old_vals = np.empty(niche_width, dtype=numba.int32)
                for p in range(ell, r + 1):
                    old_vals[p - ell] = sky[p]
                    sky[p] = h_wall
                result = _skyline_numba_recurse(
                    sky, remaining, rcl, rem_area, new_wasted,
                    ix, ir, iw, ih, ia, n, W, H, allowed_waste,
                    max_iter, iter_count)
                for p in range(ell, r + 1):
                    sky[p] = old_vals[p - ell]
                if result:
                    return True

        # ── Branch on each niche item ──
        # Symmetry: skip items with identical (x, w, h)
        for ni in range(niche_count):
            j = niche_buf[ni]

            # Symmetry breaking: check if a previous niche item has same sig
            skip = False
            for pi in range(ni):
                k = niche_buf[pi]
                if ix[k] == ix[j] and iw[k] == iw[j] and ih[k] == ih[j]:
                    skip = True
                    break
            if skip:
                continue

            # Column-load feasibility for this item
            will_exceed = False
            for p in range(ix[j], ir[j]):
                if sky[p] + ih[j] > H:
                    will_exceed = True
                    break
            if will_exceed:
                continue

            # ── Left-gap column-load check ──
            has_gap = ix[j] > ell
            fill = 0
            fill_h = 0
            if has_gap:
                fill = min(h_left, min_h + ih[j])
                fill_h = fill - min_h
                gap_infeasible = False
                for p in range(ell, ix[j]):
                    if rcl[p] > H - fill:
                        gap_infeasible = True
                        break
                if gap_infeasible:
                    continue

            # ── Left-gap dominance ──
            if has_gap:
                gap_dominated = False
                for ki in range(niche_count):
                    k = niche_buf[ki]
                    if k == j:
                        continue
                    if (ix[k] >= ell and ix[k] + iw[k] <= ix[j]
                            and ih[k] <= fill_h):
                        gap_dominated = True
                        break
                if gap_dominated:
                    continue

            # ── Compute wasted space from left-gap fill ──
            item_wasted = 0
            if has_gap:
                item_wasted = fill_h * (ix[j] - ell)
            new_wasted = wasted + item_wasted
            if new_wasted > allowed_waste:
                continue

            # ── Modify skyline in-place ──
            # Save old values for backtracking
            save_len = ir[j] - ell if has_gap else ir[j] - ix[j]
            save_start = ell if has_gap else ix[j]
            old_sky = np.empty(save_len, dtype=numba.int32)
            for p in range(save_len):
                old_sky[p] = sky[save_start + p]

            if has_gap:
                for p in range(ell, ix[j]):
                    sky[p] = fill
            for p in range(ix[j], ir[j]):
                sky[p] = sky[p] + ih[j]
            # Note: sky[ix[j]:ir[j]] was already saved before += ih[j],
            # so old_sky has the pre-modification values.
            # Fix: we need to save BEFORE modifying. Let me redo this.
            # Actually old_sky was saved before the modification, so it's correct.

            # ── Update remaining and rcl ──
            remaining[j] = False
            new_area = rem_area - ia[j]
            old_rcl = np.empty(iw[j], dtype=numba.int32)
            for p in range(iw[j]):
                old_rcl[p] = rcl[ix[j] + p]
                rcl[ix[j] + p] -= ih[j]

            # ── Column-load bound + area bound ──
            fathom = False
            residual = 0
            for p in range(W):
                gap = H - sky[p]
                if rcl[p] > gap:
                    fathom = True
                    break
                residual += gap

            if not fathom and new_area <= residual:
                result = _skyline_numba_recurse(
                    sky, remaining, rcl, new_area, new_wasted,
                    ix, ir, iw, ih, ia, n, W, H, allowed_waste,
                    max_iter, iter_count)
                if result:
                    # Restore before returning (for clean state)
                    remaining[j] = True
                    for p in range(iw[j]):
                        rcl[ix[j] + p] = old_rcl[p]
                    for p in range(save_len):
                        sky[save_start + p] = old_sky[p]
                    return True

            # ── Restore state (backtrack) ──
            remaining[j] = True
            for p in range(iw[j]):
                rcl[ix[j] + p] = old_rcl[p]
            for p in range(save_len):
                sky[save_start + p] = old_sky[p]

        return False

    def _skyline_ycheck_numba(
        items: list,
        strip_width: int,
        target_height: int,
        max_iterations: int = 2_000_000,
    ) -> bool:
        """
        Numba-accelerated wrapper for skyline y-check.
        Converts Python objects to numpy arrays and calls the JIT core.
        """
        n = len(items)
        W = strip_width
        H = target_height

        if n == 0:
            return True

        items = sorted(items, key=lambda it: (it.x_pos, it.width))
        ix = np.array([it.x_pos for it in items], dtype=np.int32)
        ir = np.array([it.x_right for it in items], dtype=np.int32)
        iw = np.array([it.width for it in items], dtype=np.int32)
        ih = np.array([it.height for it in items], dtype=np.int32)
        ia = np.array([iw[j] * ih[j] for j in range(n)], dtype=np.int32)

        # Quick column-load check
        col_load = np.zeros(W, dtype=np.int32)
        for j in range(n):
            for p in range(ix[j], ir[j]):
                col_load[p] += ih[j]
        if col_load.max() > H:
            return False

        sky = np.zeros(W, dtype=np.int32)
        remaining = np.ones(n, dtype=np.bool_)
        rcl = col_load.copy()
        total_area = int(ia.sum())
        allowed_waste = W * H - total_area
        iter_count = np.zeros(1, dtype=np.int64)

        return _skyline_numba_recurse(
            sky, remaining, rcl, total_area, 0,
            ix, ir, iw, ih, ia,
            n, W, H, allowed_waste, max_iterations, iter_count,
        )

    _NUMBA_AVAILABLE = True

except ImportError:
    _NUMBA_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════
#  Main y-check: preprocessing + skyline enumeration
# ═══════════════════════════════════════════════════════════════════════

def y_check(
    items: List[YCheckItem],
    strip_width: int,
    target_height: int,
    time_limit: float = 30.0,
    threads: int = 4,
) -> Tuple[bool, Optional[Dict[int, int]]]:
    """
    Determine if items can be assigned y-coordinates within target_height,
    with no vertical overlap among items sharing columns.

    Uses Côté-style skyline enumeration tree with preprocessing.

    Always returns a conclusive answer (True/False), though may return
    True (conservative) if the iteration limit is exceeded.

    Args:
        items: list of YCheckItem with x-positions fixed
        strip_width: width of the strip
        target_height: maximum allowed height (makespan)
        time_limit: (kept for interface compatibility; iteration limit
                     is used internally instead)
        threads: (kept for interface compatibility; not used)

    Returns:
        (feasible, y_coords) where:
          feasible = True  -> feasible, y_coords maps item_id -> y
          feasible = False -> infeasible, y_coords is None
    """
    if not items:
        return True, {}

    # Quick check: column height bound
    col_h = compute_column_heights(items, strip_width)
    if col_h.max() > target_height:
        return False, None

    # Find conflicting pairs (items sharing at least one column)
    conflict_pairs = _get_conflict_pairs(items)

    # If no conflicts, all items can be placed at y=0
    if not conflict_pairs:
        return True, {it.id: 0 for it in items}

    # ── Convert to internal representation ──
    internal_items = [
        _SkylineItem(
            orig_id=it.id,
            x_pos=it.x_pos,
            width=it.width,
            height=it.height,
        )
        for it in items
    ]

    # ── Preprocessing ──
    # Step 1: Merge items — DISABLED
    # The merge preprocessing (Côté 2014, Preprocessing 1) requires
    # recursive y-check calls to verify that L(j) items fit within the
    # substrip, and that no other item can enter the merged region.
    # A simplified merge that only checks heights can incorrectly
    # enlarge items, increasing column loads and producing false
    # infeasibility.  Enabling this requires a full implementation
    # with the sub-problem feasibility check.  For now, skip it.
    # internal_items = _preprocessing_merge(
    #     internal_items, strip_width, target_height
    # )

    # Step 2: Lift item widths
    internal_items = _preprocessing_lift_widths(internal_items, strip_width)

    # Step 3: Shrink the strip
    internal_items, new_strip_width = _preprocessing_shrink_strip(
        internal_items, strip_width
    )

    # ── Skyline enumeration tree ──
    # Scale iteration limit based roughly on time_limit
    max_iter = max(500_000, int(2_000_000 * (time_limit / 30.0)))

    if _NUMBA_AVAILABLE:
        feasible = _skyline_ycheck_numba(
            internal_items, new_strip_width, target_height,
            max_iterations=max_iter,
        )
    else:
        feasible = _skyline_ycheck(
            internal_items, new_strip_width, target_height,
            max_iterations=max_iter,
        )

    if feasible:
        # We know it's feasible but we don't reconstruct y-coordinates
        # from the skyline search (would require tracking the solution
        # path).  Return None for y_coords — the Benders solver only
        # needs feasibility, not the actual y-assignment.
        return True, None
    else:
        return False, None


# ═══════════════════════════════════════════════════════════════════════
#  Fallback: Gurobi-based y-check for cases needing y-coordinates
#  or when the skyline tree is inconclusive
# ═══════════════════════════════════════════════════════════════════════

def _y_check_gurobi(
    items: List[YCheckItem],
    strip_width: int,
    target_height: int,
    time_limit: float = 30.0,
    threads: int = 4,
) -> Tuple[bool, Optional[Dict[int, int]]]:
    """
    Gurobi-based y-check (fallback). Uses big-M disjunctive MIP.
    Kept for reference and for MIS detection where we need a fast
    single-call feasibility oracle.
    """
    if not items:
        return True, {}

    col_h = compute_column_heights(items, strip_width)
    if col_h.max() > target_height:
        return False, None

    conflict_pairs = _get_conflict_pairs(items)
    if not conflict_pairs:
        return True, {it.id: 0 for it in items}

    import gurobipy as gp
    from gurobipy import GRB

    n = len(items)
    H = target_height

    model = gp.Model("y_check")
    model.Params.OutputFlag = 0
    model.Params.TimeLimit = time_limit
    model.Params.Threads = threads
    model.Params.SolutionLimit = 1

    y = {}
    for idx, it in enumerate(items):
        y[idx] = model.addVar(lb=0, ub=H - it.height, name=f"y_{it.id}")

    b = {}
    for (a_idx, b_idx) in conflict_pairs:
        b[a_idx, b_idx] = model.addVar(vtype=GRB.BINARY,
                                        name=f"b_{a_idx}_{b_idx}")
    model.update()
    model.setObjective(0, GRB.MINIMIZE)

    for (a_idx, b_idx) in conflict_pairs:
        a_item = items[a_idx]
        b_item = items[b_idx]
        bij = b[a_idx, b_idx]
        model.addConstr(
            y[a_idx] + a_item.height <= y[b_idx] + H * (1 - bij))
        model.addConstr(
            y[b_idx] + b_item.height <= y[a_idx] + H * bij)

    model.optimize()

    if model.SolCount > 0:
        y_coords = {}
        for idx, it in enumerate(items):
            y_coords[it.id] = int(round(y[idx].X))
        return True, y_coords
    elif model.Status in (GRB.OPTIMAL, GRB.INFEASIBLE):
        return False, None
    else:
        return True, None


# ═══════════════════════════════════════════════════════════════════════
#  Minimal Infeasible Subset (MIS) detection
# ═══════════════════════════════════════════════════════════════════════

def find_minimal_infeasible_subset(
    items: List[YCheckItem],
    strip_width: int,
    target_height: int,
    time_limit_per_check: float = 10.0,
    max_attempts: int = 0,
    threads: int = 4,
) -> Optional[List[int]]:
    """
    Find a (near-)minimal infeasible subset of items for y-check.

    Greedy removal: try removing each item; if the remainder becomes
    feasible, the item is essential for infeasibility.

    Args:
        max_attempts: maximum number of removal attempts (y-check calls).
            0 means unlimited (true MIS). A positive value caps the number
            of trials, yielding a near-minimal but still valid infeasible
            subset. This controls the trade-off between cut strength and
            computation time.

    Returns list of item IDs in the (near-)MIS, or None if feasible.
    """
    # First verify the full set is infeasible
    feasible, _ = y_check(items, strip_width, target_height,
                          time_limit=time_limit_per_check, threads=threads)
    if feasible:
        return None

    id_to_item = {it.id: it for it in items}
    remaining = set(it.id for it in items)

    # Sort by area descending — try removing large items first
    sorted_by_area = sorted(
        [it.id for it in items],
        key=lambda cid: id_to_item[cid].width * id_to_item[cid].height,
        reverse=True,
    )

    attempts = 0
    for item_id in sorted_by_area:
        if item_id not in remaining:
            continue

        if max_attempts > 0 and attempts >= max_attempts:
            break  # cap reached — return current (near-minimal) subset

        # Try without this item
        test_items = [id_to_item[cid] for cid in remaining if cid != item_id]
        if not test_items:
            break

        attempts += 1
        feas, _ = y_check(test_items, strip_width, target_height,
                          time_limit=time_limit_per_check, threads=threads)
        if feas:
            # Removing makes it feasible → item is essential, keep it
            pass
        else:
            # Still infeasible without it → remove it
            remaining.discard(item_id)

    return list(remaining)


# ═══════════════════════════════════════════════════════════════════════
#  Lifted Combinatorial Benders' Cut
# ═══════════════════════════════════════════════════════════════════════

def verify_lifted_intervals(
    mis_items: List[YCheckItem],
    intervals: Dict[int, Tuple[int, int]],
    strip_width: int,
    target_height: int,
    time_limit: float = 10.0,
) -> bool:
    """
    Verify that lifted intervals are jointly valid: no feasible y-assignment
    exists for ANY combination of x-positions within the intervals at
    target_height.

    Returns True if intervals are valid (all combinations infeasible).
    Returns False if some combination is feasible (intervals too wide).
    """
    if not mis_items:
        return True

    all_single = all(
        intervals[it.id][0] == intervals[it.id][1] for it in mis_items
    )
    if all_single:
        return True

    import gurobipy as gp
    from gurobipy import GRB

    n = len(mis_items)
    H = target_height

    model = gp.Model("verify_lift")
    model.Params.OutputFlag = 0
    model.Params.TimeLimit = time_limit
    model.Params.SolutionLimit = 1

    p = {}
    for idx, it in enumerate(mis_items):
        lb, ub = intervals[it.id]
        p[idx] = model.addVar(lb=lb, ub=ub, vtype=GRB.INTEGER,
                              name=f"p_{it.id}")

    y = {}
    for idx, it in enumerate(mis_items):
        y[idx] = model.addVar(lb=0, ub=max(0, H - it.height),
                              name=f"y_{it.id}")

    model.update()
    model.setObjective(0, GRB.MINIMIZE)

    for a_idx in range(n):
        for b_idx in range(a_idx + 1, n):
            a_it = mis_items[a_idx]
            b_it = mis_items[b_idx]

            a_lb, a_ub = intervals[a_it.id]
            b_lb, b_ub = intervals[b_it.id]
            if a_lb >= b_ub + b_it.width or b_lb >= a_ub + a_it.width:
                continue

            d = {}
            for k in range(4):
                d[k] = model.addVar(vtype=GRB.BINARY,
                                    name=f"d_{a_idx}_{b_idx}_{k}")
            model.addConstr(d[0] + d[1] + d[2] + d[3] >= 1)

            M_pos = strip_width
            M_y = H

            model.addConstr(
                p[a_idx] >= p[b_idx] + b_it.width - M_pos * (1 - d[0]))
            model.addConstr(
                p[b_idx] >= p[a_idx] + a_it.width - M_pos * (1 - d[1]))
            model.addConstr(
                y[a_idx] >= y[b_idx] + b_it.height - M_y * (1 - d[2]))
            model.addConstr(
                y[b_idx] >= y[a_idx] + a_it.height - M_y * (1 - d[3]))

    model.optimize()

    if model.SolCount > 0:
        return False
    else:
        return True


def compute_lifted_intervals(
    mis_items: List[YCheckItem],
    strip_width: int,
    target_height: int,
    all_items: List[YCheckItem],
    time_limit_per_check: float = 5.0,
    verify_time_limit: float = 10.0,
) -> Optional[Dict[int, Tuple[int, int]]]:
    """
    Compute lifted intervals for the MIS using column-load analysis.

    If a critical column q* has total item height > target_height, then
    any x-assignment where the same items still cover q* remains
    infeasible by column load.

    Returns intervals dict if column-load lifting applies, or None.
    """
    col_loads = compute_column_heights(mis_items, strip_width)
    max_load = int(col_loads.max())

    if max_load <= target_height:
        return None

    critical_cols = [q for q in range(strip_width)
                     if col_loads[q] > target_height]

    best_q = None
    best_total_width = -1
    for q in critical_cols:
        total_width = 0
        for item in mis_items:
            if item.x_pos <= q < item.x_right:
                l = max(0, q - item.width + 1)
                r = min(strip_width - item.width, q)
                total_width += (r - l + 1)
        if total_width > best_total_width:
            best_total_width = total_width
            best_q = q

    intervals = {}
    for item in mis_items:
        if item.x_pos <= best_q < item.x_right:
            l = max(0, best_q - item.width + 1)
            r = min(strip_width - item.width, best_q)
            intervals[item.id] = (l, r)
        else:
            intervals[item.id] = (item.x_pos, item.x_pos)

    return intervals


def compute_lifted_intervals_lp(
    mis_items: List[YCheckItem],
    strip_width: int,
    target_height: int,
    all_items: List[YCheckItem],
    time_limit: float = 10.0,
) -> Optional[Dict[int, Tuple[int, int]]]:
    """
    Compute lifted intervals via LP, following Section 4.2 of
    Côté et al. (2014).

    LP formulation (eqs. 29-32):
        max  Σ_{j ∈ C*}  (r_j - l_j)
        s.t. l_j + w_j  ≥  r_i + 1     ∀ j ∈ C*, i ∈ K(j)
             0  ≤  l_j  ≤  p_j          ∀ j ∈ C*
             p_j  ≤  r_j  ≤  W - w_j    ∀ j ∈ C*

    Returns intervals dict, or None if the LP is infeasible.
    """
    if not mis_items:
        return None

    import gurobipy as gp
    from gurobipy import GRB

    n = len(mis_items)
    W = strip_width

    conflict = {it.id: [] for it in mis_items}
    for a_idx in range(n):
        for b_idx in range(a_idx + 1, n):
            a = mis_items[a_idx]
            b = mis_items[b_idx]
            if items_overlap_x(a, b):
                conflict[a.id].append(b.id)
                conflict[b.id].append(a.id)

    has_conflicts = any(len(v) > 0 for v in conflict.values())
    if not has_conflicts:
        return None

    model = gp.Model("lift_lp")
    model.Params.OutputFlag = 0
    model.Params.Threads = 1
    model.Params.TimeLimit = time_limit

    l = {}
    r = {}
    id_to_item = {it.id: it for it in mis_items}
    for it in mis_items:
        j = it.id
        l[j] = model.addVar(lb=0, ub=it.x_pos, name=f"l_{j}")
        r[j] = model.addVar(lb=it.x_pos, ub=W - it.width, name=f"r_{j}")

    model.update()

    model.setObjective(
        gp.quicksum(r[it.id] - l[it.id] for it in mis_items),
        GRB.MAXIMIZE,
    )

    added_pairs = set()
    for it in mis_items:
        j = it.id
        for i in conflict[j]:
            pair = (min(j, i), max(j, i))
            if pair in added_pairs:
                continue
            added_pairs.add(pair)
            j_item = id_to_item[j]
            i_item = id_to_item[i]
            model.addConstr(l[j] + j_item.width >= r[i] + 1,
                            name=f"conf_{j}_{i}")
            model.addConstr(l[i] + i_item.width >= r[j] + 1,
                            name=f"conf_{i}_{j}")

    model.optimize()

    if model.Status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
        return None

    intervals = {}
    for it in mis_items:
        j = it.id
        lv = int(round(l[j].X))
        rv = int(round(r[j].X))
        intervals[j] = (lv, rv)

    return intervals


# ═══════════════════════════════════════════════════════════════════════
#  Convenience: run full y-check + cut generation for a strip
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class YCheckResult:
    """Result of y-check on one strip."""
    feasible: bool
    y_coords: Optional[Dict[int, int]] = None
    mis_item_ids: Optional[List[int]] = None
    lifted_intervals: Optional[Dict[int, Tuple[int, int]]] = None


def run_ycheck_and_cuts(
    items: List[YCheckItem],
    strip_width: int,
    target_height: int,
    compute_mis: bool = True,
    compute_lift: bool = True,
    lifting_method: str = 'lp',
    ycheck_time_limit: float = 30.0,
    mis_time_limit: float = 10.0,
    mis_max_attempts: int = 0,
    lift_time_limit: float = 5.0,
    threads: int = 4,
) -> YCheckResult:
    """
    Run y-check on a strip. If infeasible, compute MIS and lifted intervals.

    Always returns a conclusive answer (feasible or infeasible).

    Args:
        lifting_method: 'column_load' (lift1) uses critical-column analysis;
                        'lp' (lift2) uses Côté et al.'s LP-based conflict-
                        graph preservation.
        mis_max_attempts: cap on MIS removal trials (0 = unlimited).
            A positive value yields a near-minimal (but still valid)
            infeasible subset, reducing callback time.
        threads: number of Gurobi threads for subproblem solves.
    """
    feasible, y_coords = y_check(items, strip_width, target_height,
                                 time_limit=ycheck_time_limit, threads=threads)

    if feasible:
        return YCheckResult(feasible=True, y_coords=y_coords)

    result = YCheckResult(feasible=False)

    if compute_mis:
        mis_ids = find_minimal_infeasible_subset(
            items, strip_width, target_height,
            time_limit_per_check=mis_time_limit,
            max_attempts=mis_max_attempts,
            threads=threads,
        )
        result.mis_item_ids = mis_ids

        if compute_lift and mis_ids is not None:
            id_to_item = {it.id: it for it in items}
            mis_items = [id_to_item[cid] for cid in mis_ids]

            if lifting_method == 'lp':
                intervals = compute_lifted_intervals_lp(
                    mis_items, strip_width, target_height, items,
                    time_limit=lift_time_limit,
                )
            else:  # 'column_load' (default)
                intervals = compute_lifted_intervals(
                    mis_items, strip_width, target_height, items,
                    time_limit_per_check=lift_time_limit,
                )
            result.lifted_intervals = intervals

    return result
