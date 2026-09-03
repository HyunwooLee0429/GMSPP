"""
CP-SAT baseline solver for GMSPP (off-the-shelf comparison).

Direct constraint-programming model using OR-Tools CP-SAT:
optional interval variables per (item, strip) pair with a 2D
no-overlap constraint per strip.  Supports strip-dependent dims and
both objectives.  Non-integer strip costs are scaled to integers.
"""

import time
from typing import Dict

from .data_structures import Instance, OBJ_MAKESPAN


def _cost_scale(costs, cap: int = 10 ** 6) -> int:
    """Smallest power-of-ten scale making all costs integral."""
    s = 1
    while s <= cap:
        if all(abs(c * s - round(c * s)) < 1e-9 for c in costs):
            return s
        s *= 10
    return cap


def solve_cpsat(
    instance: Instance,
    time_limit: float = 900.0,
    threads: int = 16,
    verbose: bool = False,
) -> Dict:
    """Solve GMSPP with CP-SAT.

    Returns dict: objective, lower_bound, gap_pct, optimal, solve_time,
    solution ({strip: [(item, x, y), ...]}), objective_mode.
    """
    from ortools.sat.python import cp_model

    t0 = time.time()
    inst = instance
    n, m = inst.n, inst.m

    # Height cap per strip: stack of all fitting items
    Hcap = {i: max(1, sum(inst.h(j, i) for j in range(n)
                          if inst.fits(j, i)))
            for i in range(m)}

    mdl = cp_model.CpModel()
    H = {i: mdl.NewIntVar(0, Hcap[i], f"H{i}") for i in range(m)}
    z, xv, yv = {}, {}, {}
    ivx, ivy = {}, {}
    for j in range(n):
        feas = inst.feasible_strips(j)
        for i in feas:
            w, h = inst.dims(j, i)
            W = inst.strips[i].width
            z[j, i] = mdl.NewBoolVar(f"z{j}_{i}")
            xv[j, i] = mdl.NewIntVar(0, W - w, f"x{j}_{i}")
            yv[j, i] = mdl.NewIntVar(0, Hcap[i] - h, f"y{j}_{i}")
            ivx[j, i] = mdl.NewOptionalFixedSizeIntervalVar(
                xv[j, i], w, z[j, i], f"ix{j}_{i}")
            ivy[j, i] = mdl.NewOptionalFixedSizeIntervalVar(
                yv[j, i], h, z[j, i], f"iy{j}_{i}")
            mdl.Add(yv[j, i] + h <= H[i]).OnlyEnforceIf(z[j, i])
        mdl.AddExactlyOne(z[j, i] for i in feas)

    for i in range(m):
        xs = [ivx[j, i] for j in range(n) if (j, i) in ivx]
        ys = [ivy[j, i] for j in range(n) if (j, i) in ivy]
        if xs:
            mdl.AddNoOverlap2D(xs, ys)

    # Symmetry breaking on interchangeable strips
    for grp in inst.identical_strip_groups():
        for a, c in zip(grp, grp[1:]):
            mdl.Add(H[a] >= H[c])

    scale = 1
    if inst.objective == OBJ_MAKESPAN:
        Hm = mdl.NewIntVar(0, max(Hcap.values()), "Hm")
        mdl.AddMaxEquality(Hm, [H[i] for i in range(m)])
        mdl.Minimize(Hm)
    else:
        costs = [s.cost for s in inst.strips]
        scale = _cost_scale(costs)
        mdl.Minimize(sum(int(round(inst.strips[i].cost * scale))
                         * inst.strips[i].width * H[i]
                         for i in range(m)))

    sv = cp_model.CpSolver()
    sv.parameters.max_time_in_seconds = time_limit
    sv.parameters.num_workers = threads
    sv.parameters.log_search_progress = bool(verbose)
    status = sv.Solve(mdl)
    elapsed = time.time() - t0

    has_sol = status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
    obj = sv.ObjectiveValue() / scale if has_sol else float("inf")
    lb = sv.BestObjectiveBound() / scale
    gap = ((obj - lb) / obj * 100) if has_sol and obj > 0 else 100.0

    solution = None
    if has_sol:
        solution = {i: [] for i in range(m)}
        for (j, i), var in z.items():
            if sv.Value(var):
                solution[i].append(
                    (j, int(sv.Value(xv[j, i])), int(sv.Value(yv[j, i]))))

    return {
        "objective": obj,
        "lower_bound": lb,
        "gap_pct": gap,
        "optimal": status == cp_model.OPTIMAL,
        "solve_time": elapsed,
        "solution": solution,
        "objective_mode": inst.objective,
        "status": sv.StatusName(status),
    }
