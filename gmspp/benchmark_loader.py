"""
Benchmark loader and GMSPP adapter.

Loads standard SPP benchmark instances from 2DPackLib .ins2D format
and converts them to GMSPP instances with multiple heterogeneous strips.

Benchmarks used in our experiments:
  - N (30 instances, Hopper-Turton 2002, n = 17..97, W = 200)
  - T (30 instances, Hopper-Turton 2002, n = 17..199, W = 200)

Strip generation strategies for GMSPP:
  Given a single-strip SPP instance with width W and items {(w_j, h_j)},
  create m strips by splitting the original width into heterogeneous strips,
  following Vasilyev et al.'s approach of adapting SPP benchmarks to GMSPP.
"""

import os
import glob
from typing import List, Tuple, Dict, Optional
import numpy as np

from .data_structures import Item, Strip, Instance

# ═══════════════════════════════════════════════════════════════════════
#  Parse .ins2D format (2DPackLib standard)
# ═══════════════════════════════════════════════════════════════════════

def parse_ins2d(filepath: str) -> Dict:
    """
    Parse a 2DPackLib .ins2D file.

    Format:
        m                          (number of items)
        W H                        (bin/strip width and height; H=-1 for SPP)
        i w_i h_i d_i b_i p_i     (for each item)

    Returns dict with keys: 'n', 'W', 'H', 'items' (list of (w, h) tuples).
    For knapsack instances (NGCUT/CGCUT), d_i/b_i > 1 means multiple copies;
    we expand them.
    """
    with open(filepath, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    n = int(lines[0])
    W, H = map(int, lines[1].split())

    items = []
    for k in range(2, 2 + n):
        parts = lines[k].split()
        # id, w, h, demand, copies, profit
        w = int(parts[1])
        h = int(parts[2])
        demand = int(parts[3])
        copies = int(parts[4])
        # For packing: demand=1, copies=1 → 1 item
        # For knapsack: copies > 1 → up to 'copies' items (use demand)
        # For SPP adaptation: each item appears once
        num = max(1, demand) if demand > 0 else 1
        for _ in range(num):
            items.append((w, h))

    return {
        'n': len(items),
        'W': W,
        'H': H,  # -1 for SPP instances
        'items': items,
        'source_file': os.path.basename(filepath),
    }


def load_benchmark_set(benchmark_dir: str, pattern: str = "*.ins2D") -> List[Dict]:
    """Load all instances from a benchmark directory."""
    files = sorted(glob.glob(os.path.join(benchmark_dir, pattern)))
    instances = []
    for f in files:
        try:
            inst = parse_ins2d(f)
            instances.append(inst)
        except Exception as e:
            print(f"Warning: failed to parse {f}: {e}")
    return instances


# ═══════════════════════════════════════════════════════════════════════
#  GMSPP strip generation strategies
# ═══════════════════════════════════════════════════════════════════════

# Deterministic strip width ratios (relative to original W).
# Strips straddle the original SPP width W, keeping the assignment
# problem non-trivial while guaranteeing feasibility (1.2W ≥ w_j
# for every item from an SPP instance of width W).
STRIP_RATIOS = {
    2: [1.0, 1.2],              # 2 strips: W, 1.2W
    3: [0.8, 1.0, 1.2],         # 3 strips: 0.8W, W, 1.2W
}


def make_strips_deterministic(
    W: int,
    m: int = 2,
    cost_type: str = 'proportional',
) -> List[Tuple[int, float]]:
    """
    Generate m heterogeneous strips from an SPP strip of width W.

    Deterministic (no randomness): uses fixed ratios from STRIP_RATIOS.
    Strips straddle the original SPP width W (e.g. 0.8W, 1.2W),
    ensuring reproducibility and feasibility (1.2W >= w_j for all items).

    The cost field C_i is the per-unit-AREA cost.  The objective is
    sum_i C_i * W_i * H_i, so that height on wider strips is weighted
    more heavily (reflecting greater material consumption).

    Cost structures use a simple additive step Delta = 0.1 per strip
    rank, keeping the spread realistic (at most (m-1)*10 % difference).

    Args:
        W: original SPP strip width
        m: number of strips (2, 3, or 4)
        cost_type:
            'proportional'   → C_i = 1  (uniform per-area cost; baseline)
            'economies'      → widest strip C=1.0, each narrower +0.1
                               (wider cheaper per area)
            'diseconomies'   → narrowest strip C=1.0, each wider +0.1
                               (wider more expensive per area)

    Returns:
        list of (width, cost) tuples, sorted by width ascending.
    """
    ratios = STRIP_RATIOS.get(m)
    if ratios is None:
        # Fallback: linearly spaced from 1.0 to 0.3
        ratios = [1.0 - 0.7 * k / (m - 1) for k in range(m)]

    widths = [max(1, int(round(W * r))) for r in ratios]
    widths.sort()  # ascending: narrowest first

    DELTA = 0.1  # cost step per strip rank
    if cost_type == 'proportional':
        # Uniform per-area cost (baseline): C_i = 1.
        # Objective becomes sum W_i * H_i = total area used.
        costs = [1.0 for _ in widths]
    elif cost_type == 'economies':
        # Economies of scale: narrower strips more expensive per area.
        # Widest strip (last): C = 1.0.  Each narrower strip: +Delta.
        # e.g. m=3: [1.2, 1.1, 1.0]
        costs = [1.0 + DELTA * (len(widths) - 1 - k) for k in range(len(widths))]
    elif cost_type == 'diseconomies':
        # Diseconomies of scale: wider strips more expensive per area.
        # Narrowest strip (first): C = 1.0.  Each wider strip: +Delta.
        # e.g. m=3: [1.0, 1.1, 1.2]
        costs = [1.0 + DELTA * k for k in range(len(widths))]
    else:  # 'uniform' — alias for proportional
        costs = [1.0 for _ in widths]

    return list(zip(widths, costs))


# ═══════════════════════════════════════════════════════════════════════
#  Convert SPP benchmark to GMSPP instance
# ═══════════════════════════════════════════════════════════════════════

def spp_to_gmspp(
    spp: Dict,
    m: int = 2,
    cost_type: str = 'uniform',
) -> Instance:
    """
    Convert an SPP benchmark instance to a GMSPP instance.

    Uses deterministic strip generation (no randomness).
    Strip 0 always has the original SPP width W, ensuring feasibility.

    Args:
        spp: parsed SPP instance dict from parse_ins2d
        m: number of strips (2, 3, or 4)
        cost_type: 'uniform', 'economies', or 'diseconomies'

    Returns:
        GMSPP Instance
    """
    W = spp['W']
    strip_specs = make_strips_deterministic(W, m, cost_type)

    items = [Item(j, w, h) for j, (w, h) in enumerate(spp['items'])]
    strips = [Strip(i, sw, sc) for i, (sw, sc) in enumerate(strip_specs)]

    return Instance(items=items, strips=strips)


def load_and_convert_benchmark(
    benchmark_dir: str,
    m: int = 2,
    cost_type: str = 'uniform',
    max_items: Optional[int] = None,
) -> Dict[str, Instance]:
    """
    Load all .ins2D files from a benchmark directory and convert to GMSPP.

    Uses deterministic strip generation (no randomness, no seed).

    Args:
        benchmark_dir: path to benchmark directory
        m: number of strips
        cost_type: 'uniform' or 'proportional'
        max_items: if set, skip instances with n > max_items

    Returns:
        dict mapping instance_name -> GMSPP Instance
    """
    spp_instances = load_benchmark_set(benchmark_dir)
    gmspp_instances = {}

    for spp in spp_instances:
        name = spp['source_file'].replace('.ins2D', '')
        if max_items is not None and spp['n'] > max_items:
            continue
        inst = spp_to_gmspp(spp, m=m, cost_type=cost_type)
        gmspp_instances[name] = inst

    return gmspp_instances
