"""
Instance generator for GMSPP experiments.

Two types following Vasilyev et al. (2023):

Type I: Derived from standard SPP benchmarks (Martello-Vigo classes).
  - Generate items for a single strip of width W_ref.
  - Split into m strips of specified widths.
  - Items have fixed dimensions (w_j, h_j) independent of strip.

Type II: Synthetic scheduling instances.
  - For simplicity we generate fixed-dimension items directly
    (not strip-dependent, since we study the packing version).

We also support generating items from MV classes 1-4 (Lodi et al. 2004)
as used in the CX and ZDF benchmark collections.
"""

import numpy as np
from typing import List, Tuple, Optional

from .data_structures import Item, Strip, Instance


# ─── MV item type definitions (Lodi et al. 2004) ──────────────────────────

ITEM_TYPES = {
    1: (67, 100, 1, 50),     # wide-short
    2: (1, 50, 67, 100),     # narrow-tall
    3: (50, 100, 50, 100),   # large
    4: (1, 50, 1, 50),       # small
}

CLASS_TYPE_ORDER = {
    1: [1, 2, 3, 4],
    2: [2, 1, 3, 4],
    3: [3, 1, 2, 4],
    4: [4, 1, 2, 3],
}

CLASS_PROBS = [0.70, 0.10, 0.10, 0.10]


def generate_spp_items_MV(
    class_id: int,
    n: int,
    W_ref: int = 100,
    seed: Optional[int] = None,
) -> List[Tuple[int, int]]:
    """
    Generate n items from MV Class class_id (1-4).
    Reference frame: W_ref x W_ref (default 100x100).

    Returns list of (width, height) tuples.
    """
    rng = np.random.default_rng(seed)
    type_order = CLASS_TYPE_ORDER[class_id]
    type_indices = rng.choice(4, size=n, p=CLASS_PROBS)

    items = []
    for idx in type_indices:
        item_type = type_order[idx]
        w_lo, w_hi, h_lo, h_hi = ITEM_TYPES[item_type]
        w = int(rng.integers(w_lo, w_hi, endpoint=True))
        h = int(rng.integers(h_lo, h_hi, endpoint=True))
        items.append((w, h))

    return items


def compute_strip_costs(
    widths: List[int],
    cost_type: str = 'proportional',
) -> List[float]:
    """
    Compute per-unit-area costs C_i for strips.

    The objective is sum_i C_i * W_i * H_i (area-cost weighted).
    C_i = 1 gives uniform per-area cost (total area objective).

    Cost structures use a simple additive step Delta = 0.1 per strip
    rank, keeping the spread realistic (at most (m-1)*10 % difference).
    Strips must be sorted by width ascending before calling.

    Args:
        widths: list of strip widths (ascending order)
        cost_type: 'proportional' (C_i=1), 'economies', 'diseconomies'

    Returns:
        list of per-unit-area costs C_i
    """
    m = len(widths)
    DELTA = 0.1
    if cost_type == 'uniform' or cost_type == 'proportional':
        return [1.0 for _ in widths]
    elif cost_type == 'economies':
        # Wider strips cheaper per area: widest = 1.0, each narrower +0.1
        return [1.0 + DELTA * (m - 1 - k) for k in range(m)]
    elif cost_type == 'diseconomies':
        # Wider strips more expensive per area: narrowest = 1.0, each wider +0.1
        return [1.0 + DELTA * k for k in range(m)]
    else:
        raise ValueError(f"Unknown cost_type: {cost_type}")


def generate_type1_instance(
    class_id: int,
    n: int,
    strip_widths: List[int],
    cost_type: str = 'uniform',
    seed: Optional[int] = None,
) -> Instance:
    """
    Generate a Type I GMSPP instance (packing-derived).

    Items are generated from MV Class class_id with W_ref = max(strip_widths).
    All items have fixed dimensions.
    Items must fit on at least one strip (w_j <= max(W_i)).

    Args:
        class_id: MV class (1-4)
        n: number of items
        strip_widths: list of strip widths [W_1, ..., W_m]
        cost_type: 'proportional' (C_i=1, uniform per-area cost),
                   'economies' (narrower +0.1 each, wider cheaper),
                   'diseconomies' (wider +0.1 each, wider more expensive)
        seed: random seed

    Returns:
        Instance object
    """
    W_max = max(strip_widths)

    rng = np.random.default_rng(seed)
    # Generate items with W_ref = W_max so all items fit on the widest strip
    raw_items = generate_spp_items_MV(class_id, n, W_ref=W_max, seed=seed)

    # Ensure every item fits on at least one strip
    items = []
    W_min = min(strip_widths)
    for i, (w, h) in enumerate(raw_items):
        w = min(w, W_max)
        items.append(Item(id=i, width=w, height=h))

    costs = compute_strip_costs(strip_widths, cost_type)
    strips = [Strip(id=i, width=W, cost=c)
              for i, (W, c) in enumerate(zip(strip_widths, costs))]

    return Instance(items=items, strips=strips)


def generate_type2_instance(
    n: int,
    strip_widths: List[int],
    cost_type: str = 'uniform',
    seed: Optional[int] = None,
) -> Instance:
    """
    Generate a Type II GMSPP instance (scheduling-inspired, fixed dimensions).

    For the fixed-dimension packing version:
    - Width w_j ~ U[1, W_max]
    - Height h_j ~ U[1, W_max]
    - Each item must fit on at least one strip.

    Args:
        n: number of items
        strip_widths: list of strip widths [W_1, ..., W_m]
        seed: random seed

    Returns:
        Instance object
    """
    rng = np.random.default_rng(seed)
    W_max = max(strip_widths)
    W_min = min(strip_widths)

    items = []
    for i in range(n):
        # Width: U[1, W_max] but ensure fits on at least one strip
        w = int(rng.integers(1, W_max, endpoint=True))
        w = min(w, W_max)  # ensure feasibility on widest strip
        h = int(rng.integers(1, W_max, endpoint=True))
        items.append(Item(id=i, width=w, height=h))

    costs = compute_strip_costs(strip_widths, cost_type)
    strips = [Strip(id=i, width=W, cost=c)
              for i, (W, c) in enumerate(zip(strip_widths, costs))]

    return Instance(items=items, strips=strips)


# ─── Predefined strip configurations ─────────────────────────────────────

STRIP_CONFIGS = {
    # (name, widths) — some standard configurations
    '2-equal':    [50, 50],
    '2-hetero':   [40, 60],
    '3-equal':    [34, 33, 33],
    '3-hetero':   [25, 35, 50],
    '4-hetero':   [20, 30, 40, 50],
    '5-hetero':   [15, 20, 30, 40, 50],
    # Vasilyev-style: derived from W_ref = 100
    '2-vasil':    [50, 100],
    '3-vasil':    [30, 60, 100],
    '4-vasil':    [25, 50, 75, 100],
    '5-vasil':    [20, 40, 60, 80, 100],
}
