"""
Normal position computation for GMSPP (v2).

For each strip i and item j, compute the set of normal x-positions
W_i(j) via dynamic programming on item widths.  All widths are
evaluated with the dimensions the items take ON THAT STRIP
(w_kj = instance.w(k, i)), so strip-dependent instances are handled
transparently.

Normal positions are the set of x-coordinates reachable by sums of
subsets of item widths, restricted to [0, W_i - w_ij].

Following Herz (1972), Christofides and Whitlock (1977).
"""

import numpy as np
from typing import List, Dict, Tuple
from .data_structures import Instance


def compute_normal_positions_for_strip(
    strip_width: int,
    item_widths: List[int],
    item_id: int,
) -> List[int]:
    """
    Normal x-positions for item *item_id* on a strip of width
    *strip_width*, where *item_widths* are the widths all items take on
    this strip.

    W_i(j) = {p = sum of subset of widths of other items :
              0 <= p <= W_i - w_j}

    Uses DP (subset-sum reachability).
    """
    w_j = item_widths[item_id]
    max_pos = strip_width - w_j

    if max_pos < 0:
        return []  # item doesn't fit

    reachable = np.zeros(max_pos + 1, dtype=bool)
    reachable[0] = True

    for k, w_k in enumerate(item_widths):
        if k == item_id:
            continue
        if w_k > max_pos:
            continue
        # Process in reverse to avoid using same item twice
        for p in range(max_pos, w_k - 1, -1):
            if reachable[p - w_k]:
                reachable[p] = True

    return list(np.where(reachable)[0])


def compute_all_normal_positions(
    instance: Instance,
) -> Dict[Tuple[int, int], List[int]]:
    """
    Normal x-positions for all feasible (strip, item) pairs.

    Returns
    -------
    dict mapping (strip_id, item_id) -> list of normal positions
    """
    positions = {}

    for strip in instance.strips:
        i = strip.id
        # Widths every item takes on THIS strip.  Items that do not fit
        # never enter another item's subset sums either -- exclude them
        # by giving them a width larger than the strip (skipped by DP).
        item_widths = [
            instance.w(k, i) if instance.fits(k, i) else strip.width + 1
            for k in range(instance.n)
        ]
        for j in range(instance.n):
            if instance.fits(j, i):
                positions[(i, j)] = compute_normal_positions_for_strip(
                    strip.width, item_widths, j
                )
            # else: item doesn't fit, no entry

    return positions


def compute_coverage_sets(
    instance: Instance,
    positions: Dict[Tuple[int, int], List[int]],
) -> Dict[Tuple[int, int, int], List[int]]:
    """
    Coverage sets W_i(j, q): positions at which item j covers column q
    on strip i.

    W_i(j, q) = {p in W_i(j) : q - w_ij + 1 <= p <= q}

    Returns
    -------
    dict mapping (strip_id, item_id, column_q) -> list of positions
    """
    coverage = {}

    for (i, j), pos_list in positions.items():
        w_ij = instance.w(j, i)
        W_i = instance.strips[i].width

        for p in pos_list:
            # Item j at position p covers columns p .. p + w_ij - 1
            for q in range(p, min(p + w_ij, W_i)):
                key = (i, j, q)
                if key not in coverage:
                    coverage[key] = []
                coverage[key].append(p)

    return coverage


def get_columns_per_strip(instance: Instance) -> Dict[int, List[int]]:
    """For each strip, the list of column indices [0, ..., W_i - 1]."""
    return {s.id: list(range(s.width)) for s in instance.strips}
