"""
Normal-position GMSPP-Master formulation (v2).

This is the P|cont|C_max analogue extended to multiple heterogeneous
strips, using normal positions to restrict item placement.  Supports
strip-dependent item dimensions (w_ij, h_ij) and both objectives
('total_cost' and 'makespan').

Variables:
  x_{ijp} in {0,1} : item j packed at normal x-position p on strip i
  H_i >= 0         : height used on strip i
  Hmax >= 0        : makespan (only for objective='makespan')

Constraints:
  (assign)  sum_{i in F_j} sum_{p in W_i(j)} x_{ijp} = 1,   for all j
  (colload) sum_{j} sum_{p in W_i(j,q)} h_ij * x_{ijp} <= H_i, all i, q

This is the MASTER problem only (no y-variables, no non-overlap).
It provides a lower bound via LP relaxation (LP-PC).
The full algorithm adds Benders' cuts from y-check.
"""

import time
from typing import Dict, Optional
from .data_structures import Instance, OBJ_MAKESPAN
from .normal_positions import (
    compute_all_normal_positions,
    compute_coverage_sets,
)


def _build_normal_model(instance: Instance, relax: bool = False):
    """
    Build the normal-position GMSPP-Master formulation.

    Args:
        instance: GMSPP instance
        relax: if True, relax integrality (LP relaxation)

    Returns:
        (model, variables_dict, build_info)
    """
    import gurobipy as gp
    from gurobipy import GRB

    n = instance.n
    m = instance.m
    items = instance.items
    strips = instance.strips

    t_build = time.time()

    # ── Compute normal positions ──
    positions = compute_all_normal_positions(instance)
    coverage = compute_coverage_sets(instance, positions)

    t_pos = time.time() - t_build

    # ── Build model ──
    model = gp.Model("GMSPP_Normal")
    model.Params.OutputFlag = 0

    vtype = GRB.CONTINUOUS if relax else GRB.BINARY

    # x[i,j,p] = 1 if item j at position p on strip i
    x = {}
    n_xvars = 0
    for (i, j), pos_list in positions.items():
        for p in pos_list:
            x[i, j, p] = model.addVar(
                vtype=vtype, name=f"x_{i}_{j}_{p}"
            )
            n_xvars += 1

    # H_i = height used on strip i (per-strip)
    H = {}
    for strip in strips:
        H[strip.id] = model.addVar(lb=0, name=f"H_{strip.id}")

    Hmax = None
    if instance.objective == OBJ_MAKESPAN:
        Hmax = model.addVar(lb=0, name="Hmax")

    model.update()

    # ── Objective ──
    if instance.objective == OBJ_MAKESPAN:
        for strip in strips:
            model.addConstr(Hmax >= H[strip.id], name=f"mk_{strip.id}")
        model.setObjective(Hmax, GRB.MINIMIZE)
    else:
        model.setObjective(
            gp.quicksum(strip.cost * strip.width * H[strip.id]
                        for strip in strips),
            GRB.MINIMIZE,
        )

    # ── Constraint (assign): each item packed exactly once ──
    for j in range(n):
        feasible_strips = instance.feasible_strips(j)
        terms = []
        for i in feasible_strips:
            if (i, j) in positions:
                for p in positions[(i, j)]:
                    terms.append(x[i, j, p])
        if terms:
            model.addConstr(
                gp.quicksum(terms) == 1,
                name=f"assign_{j}"
            )

    # ── Constraint (colload): column load on strip i <= H_i ──
    n_colload = 0
    for strip in strips:
        i = strip.id
        W_i = strip.width
        for q in range(W_i):
            terms = []
            for j in range(n):
                key = (i, j, q)
                if key in coverage:
                    h_ij = instance.h(j, i)
                    for p in coverage[key]:
                        terms.append(h_ij * x[i, j, p])
            if terms:
                model.addConstr(
                    gp.quicksum(terms) <= H[i],
                    name=f"colload_{i}_{q}"
                )
                n_colload += 1

    # ── Height-linking valid inequalities: H_i >= h_ij * sum_p x_ijp ──
    # Implied by colload at integer points (any covered column already
    # carries load h_ij), so the IP optimum is unchanged; strictly
    # tightens the LP for items that spread fractionally across
    # positions (their column load dilutes below h_ij).
    n_link = 0
    for (i, j), pos_list in positions.items():
        if pos_list:
            model.addConstr(
                instance.h(j, i) * gp.quicksum(x[i, j, p] for p in pos_list)
                <= H[i],
                name=f"link_{i}_{j}"
            )
            n_link += 1

    # ── Symmetry breaking on interchangeable strips ──
    for grp in instance.identical_strip_groups():
        for a, c in zip(grp, grp[1:]):
            model.addConstr(H[a] >= H[c], name=f"sym_{a}_{c}")

    model.update()

    build_info = {
        'n_positions': sum(len(v) for v in positions.values()),
        'n_xvars': n_xvars,
        'n_colload_constrs': n_colload,
        'position_time': t_pos,
    }

    return model, {'x': x, 'H': H, 'Hmax': Hmax,
                   'positions': positions}, build_info


def solve_normal_lp(instance: Instance, time_limit: float = 300.0) -> Dict:
    """
    Solve the LP relaxation of the normal-position master.

    Returns dict with:
      lp_bound, solve_time, build_time, status,
      n_vars, n_constrs, n_positions, n_colload_constrs
    """
    t0 = time.time()
    model, variables, build_info = _build_normal_model(instance, relax=True)
    t_build = time.time() - t0

    model.Params.TimeLimit = max(time_limit - t_build, 10)
    model.optimize()

    elapsed = time.time() - t0

    from gurobipy import GRB
    result = {
        'lp_bound': model.ObjVal if model.Status in (GRB.OPTIMAL, GRB.SUBOPTIMAL) else (model.ObjBound if hasattr(model, 'ObjBound') and model.SolCount > 0 else 0.0),
        'solve_time': elapsed,
        'build_time': t_build,
        'status': model.Status,
        'n_vars': model.NumVars,
        'n_constrs': model.NumConstrs,
        'n_positions': build_info['n_positions'],
        'n_colload_constrs': build_info['n_colload_constrs'],
        'position_time': build_info['position_time'],
    }
    return result


def solve_normal_mip(
    instance: Instance,
    time_limit: float = 900.0,
    threads: int = 16,
) -> Dict:
    """
    Solve the normal-position master as MIP (no y-check / Benders).

    Note: Without y-check, feasible MIP solutions may not correspond to
    feasible GMSPP solutions (items might overlap vertically).
    The LP relaxation bound IS valid.

    Returns dict with:
      objective, lower_bound, gap_pct, optimal, solve_time, build_time,
      nodes, n_vars, n_constrs, n_positions
    """
    t0 = time.time()
    model, variables, build_info = _build_normal_model(instance, relax=False)
    t_build = time.time() - t0

    model.Params.TimeLimit = max(time_limit - t_build, 10)
    model.Params.Threads = threads
    model.optimize()

    elapsed = time.time() - t0

    from gurobipy import GRB
    obj = model.ObjVal if model.SolCount > 0 else float('inf')
    lb = model.ObjBound if model.Status != GRB.INFEASIBLE else 0.0
    gap = model.MIPGap * 100 if model.SolCount > 0 else 100.0
    opt = (model.Status == GRB.OPTIMAL)

    result = {
        'objective': obj,
        'lower_bound': lb,
        'gap_pct': gap,
        'optimal': opt,
        'solve_time': elapsed,
        'build_time': t_build,
        'nodes': int(model.NodeCount),
        'n_vars': model.NumVars,
        'n_constrs': model.NumConstrs,
        'n_positions': build_info['n_positions'],
    }
    return result
