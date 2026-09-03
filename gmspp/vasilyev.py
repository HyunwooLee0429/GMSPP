"""
Vasilyev et al. (2023) comparison infrastructure.

Three components, following their Section 6:

1.  SPP-derived GMSPP instances (CX / ZDF protocol):
    an SPP instance with n0 rectangles is split into m subsets of size
    n = n0 // m; GMSPP item i takes, on strip k, the dimensions of
    source rectangle (k * n + i).  CX: sequential split
    (deterministic).  ZDF: rectangles are randomly shuffled first.
    Strip widths: the original strip width W for every strip (default;
    parameter `strip_widths` overrides).  Objective: makespan.

    REPRODUCIBILITY NOTE (2026-08): with default widths, neither the
    sequential nor the shuffled protocol reproduces the published
    MMSP LBs of Vasilyev et al. (e.g. zdf1_2: ours 26-27 vs their
    44.68; no simple area quantity matches either).  Consistent with
    their remark that they "set relatively large strip widths ... to
    ensure feasibility", the actual strip widths (and the shuffle
    seed) are NOT specified in the paper.  Exact per-instance
    reproduction therefore requires the authors' files; until then,
    use these converters with a documented width protocol and compare
    relaxation strength (LP-PC vs MMSP) on OUR instances.

2.  Synthetic Type I / Type II generators (their 6.1):
    Type I  -- each rectangle has the same area on all strips
               (up to rounding) but different integer width/height.
    Type II -- integer widths/heights drawn independently per strip;
               no consistency, >= 1 feasible strip per item.

3.  MMSP lower bound (their Prop. 1 + eq. (21)-(26), (29)-(33)):
    "liquified" min-makespan scheduling relaxation
    s_ij = w_ij * h_ij / W_i, solved as a MIP.  Valid lower bound for
    the makespan GMSPP.
"""

import time
from typing import Dict, List, Optional, Sequence

import numpy as np

from .data_structures import Item, Strip, Instance, OBJ_MAKESPAN


# =====================================================================
#  1. SPP-derived instances (CX / ZDF protocol)
# =====================================================================

def spp_split_to_gmspp(
    spp: Dict,
    m: int,
    shuffle_seed: Optional[int] = None,
    strip_widths: Optional[Sequence[int]] = None,
) -> Instance:
    """Convert a parsed SPP instance (dict from parse_ins2d) into a
    makespan GMSPP instance with strip-dependent dims by the
    Vasilyev split protocol.

    Parameters
    ----------
    spp : dict with keys 'W', 'items' ([(w, h), ...])
    m : number of strips
    shuffle_seed : if given, shuffle rectangles first (ZDF protocol);
        None = sequential split (CX protocol, deterministic)
    strip_widths : optional explicit widths; default: original W for
        every strip
    """
    rects = list(spp['items'])
    if shuffle_seed is not None:
        rng = np.random.default_rng(shuffle_seed)
        rects = [rects[k] for k in rng.permutation(len(rects))]

    n = len(rects) // m
    if n == 0:
        raise ValueError(f"too few rectangles ({len(rects)}) for m={m}")

    W = spp['W']
    widths = list(strip_widths) if strip_widths is not None else [W] * m
    if len(widths) != m:
        raise ValueError("strip_widths must have length m")

    w_mat = np.empty((n, m), dtype=np.int64)
    h_mat = np.empty((n, m), dtype=np.int64)
    for k in range(m):
        for i in range(n):
            w, h = rects[k * n + i]
            w_mat[i, k] = w
            h_mat[i, k] = h

    # Feasibility: forbid (i, k) pairs with w_ij > W_k implicitly; the
    # Instance validator requires >= 1 feasible strip per item.
    items = [Item(i, int(w_mat[i, 0]), int(h_mat[i, 0])) for i in range(n)]
    strips = [Strip(k, int(widths[k]), 1.0) for k in range(m)]
    return Instance(items=items, strips=strips, objective=OBJ_MAKESPAN,
                    w_mat=w_mat, h_mat=h_mat)


def load_zdf_gmspp(zdf_dir: str, name: str, m: int,
                   shuffle_seed: int = 0) -> Instance:
    """Load e.g. name='zdf1', m=2 -> the 'zdf1_2' instance."""
    from .benchmark_loader import parse_ins2d
    import os
    spp = parse_ins2d(os.path.join(zdf_dir, f"{name}.ins2D"))
    return spp_split_to_gmspp(spp, m, shuffle_seed=shuffle_seed)


# =====================================================================
#  2. Synthetic Type I / Type II generators
# =====================================================================

def generate_type1(
    n: int, m: int, seed: int = 0,
    strip_width_range=(50, 150),
    area_range=(50, 2000),
) -> Instance:
    """Type I: same area on all strips (up to rounding), different
    integer width/height per strip."""
    rng = np.random.default_rng(seed)
    widths = sorted(int(rng.integers(*strip_width_range)) for _ in range(m))
    w_mat = np.empty((n, m), dtype=np.int64)
    h_mat = np.empty((n, m), dtype=np.int64)
    for j in range(n):
        a = int(rng.integers(*area_range))
        for k in range(m):
            w = int(rng.integers(1, widths[k] + 1))
            w_mat[j, k] = w
            h_mat[j, k] = max(1, round(a / w))
    items = [Item(j, int(w_mat[j, 0]), int(h_mat[j, 0])) for j in range(n)]
    strips = [Strip(k, widths[k], 1.0) for k in range(m)]
    return Instance(items=items, strips=strips, objective=OBJ_MAKESPAN,
                    w_mat=w_mat, h_mat=h_mat)


def generate_type2(
    n: int, m: int, seed: int = 0,
    strip_width_range=(50, 150),
    width_factor: float = 1.5,
    height_range=(1, 100),
) -> Instance:
    """Type II: widths/heights drawn independently per strip; w_ij may
    exceed W_i (pair infeasible); >= 1 feasible strip guaranteed."""
    rng = np.random.default_rng(seed)
    widths = sorted(int(rng.integers(*strip_width_range)) for _ in range(m))
    w_mat = np.empty((n, m), dtype=np.int64)
    h_mat = np.empty((n, m), dtype=np.int64)
    for j in range(n):
        while True:
            for k in range(m):
                w_mat[j, k] = int(rng.integers(
                    1, int(width_factor * widths[k]) + 1))
                h_mat[j, k] = int(rng.integers(*height_range))
            if any(w_mat[j, k] <= widths[k] for k in range(m)):
                break
    items = [Item(j, int(w_mat[j, 0]), int(h_mat[j, 0])) for j in range(n)]
    strips = [Strip(k, widths[k], 1.0) for k in range(m)]
    return Instance(items=items, strips=strips, objective=OBJ_MAKESPAN,
                    w_mat=w_mat, h_mat=h_mat)


# =====================================================================
#  3. MMSP lower bound (liquified scheduling relaxation)
# =====================================================================

def mmsp_lower_bound(
    instance: Instance,
    time_limit: float = 300.0,
    threads: int = 4,
    ub: Optional[float] = None,
    mip_gap: float = 0.005,
    relax: bool = False,
) -> Dict:
    """Vasilyev's MMSP lower bound for the makespan GMSPP.

    Item j "liquified" on strip i occupies s_ij = w_ij*h_ij / W_i
    height units.  min H s.t. per-strip liquified load <= H is a valid
    relaxation (their Prop. 1).  If *ub* is given, assignments with
    h_ij > ub are additionally forbidden (their eq. (32); valid since
    any solution with value <= ub cannot use them).

    Returns dict: lb (valid GMSPP LB), objective (MMSP incumbent),
    optimal, solve_time, n_vars.
    """
    if instance.objective != OBJ_MAKESPAN:
        raise ValueError("MMSP bound applies to the makespan objective")
    import gurobipy as gp
    from gurobipy import GRB

    t0 = time.time()
    model = gp.Model("MMSP")
    model.Params.OutputFlag = 0
    model.Params.TimeLimit = time_limit
    model.Params.Threads = threads
    model.Params.MIPGap = mip_gap

    n, m = instance.n, instance.m
    v = {}
    for j in range(n):
        feas = [i for i in instance.feasible_strips(j)
                if ub is None or instance.h(j, i) <= ub]
        if not feas:                       # keep model feasible
            feas = instance.feasible_strips(j)
        vtype = GRB.CONTINUOUS if relax else GRB.BINARY
        for i in feas:
            v[i, j] = model.addVar(vtype=vtype, lb=0.0, ub=1.0,
                                   name=f"v_{i}_{j}")
    H = model.addVar(lb=0.0, name="H")
    model.update()

    model.setObjective(H, GRB.MINIMIZE)
    for j in range(n):
        model.addConstr(
            gp.quicksum(v[i, j] for i in range(m) if (i, j) in v) == 1)
    for i in range(m):
        terms = [(instance.w(j, i) * instance.h(j, i)
                  / instance.strips[i].width) * v[i, j]
                 for j in range(n) if (i, j) in v]
        if terms:
            model.addConstr(gp.quicksum(terms) <= H)

    model.optimize()
    elapsed = time.time() - t0

    obj = model.ObjVal if model.SolCount > 0 else float("inf")
    # Any LOWER bound on the MMSP optimum is a valid GMSPP LB.
    lb = model.ObjBound if model.Status != GRB.INFEASIBLE else 0.0
    return {
        "lb": lb,
        "objective": obj,
        "optimal": model.Status == GRB.OPTIMAL,
        "solve_time": elapsed,
        "n_vars": model.NumVars,
    }
