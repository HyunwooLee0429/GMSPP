"""
Big-M formulation for GMSPP (v2).

Vasilyev et al. (2023) exact linearization (eqs. 1-12) in full
generality: strip-dependent item dimensions (w_ij, h_ij) and both
objectives ('total_cost' and 'makespan').  Fixed dimensions are the
special case w_ij = w_j, h_ij = h_j handled transparently by
``Instance.w(j, i)`` / ``Instance.h(j, i)``.

Variables:
  z_{ij} in {0,1}  : item j assigned to strip i
  x_{ij} >= 0      : x-coordinate of item j on strip i
  y_{ij} >= 0      : y-coordinate of item j on strip i
  l_{jk} in {0,1}  : item j is to the left of item k
  b_{jk} in {0,1}  : item j is below item k
  H_i    >= 0      : height used on strip i
  Hmax   >= 0      : makespan (only for objective='makespan')

Symmetry breaking: within each group of interchangeable strips
(identical width, cost, and dimension columns), heights are forced
non-increasing.
"""

import time
from typing import Dict
from .data_structures import Instance, OBJ_MAKESPAN


def _build_vasilyev_model(instance: Instance, relax: bool = False):
    """Build Vasilyev et al.'s linearized GMSPP formulation.

    Objective mode is taken from ``instance.objective``.

    Returns
    -------
    (model, variables_dict)
    """
    import gurobipy as gp
    from gurobipy import GRB

    n = instance.n
    m = instance.m
    strips = instance.strips

    # Per-strip big-M for vertical positions: total height of all items
    # that fit on the strip, evaluated with that strip's dimensions.
    M_y = {
        s.id: sum(instance.h(j, s.id)
                  for j in instance.items_fitting_strip(s.id))
        for s in strips
    }

    model = gp.Model("GMSPP_BigM")
    model.Params.OutputFlag = 0

    vtype = GRB.CONTINUOUS if relax else GRB.BINARY

    feas = {j: instance.feasible_strips(j) for j in range(n)}

    # -- Variables ----------------------------------------------------
    z, x, y = {}, {}, {}
    for j in range(n):
        for i in feas[j]:
            z[i, j] = model.addVar(vtype=vtype, name=f"z_{i}_{j}")
            x[i, j] = model.addVar(lb=0, name=f"x_{i}_{j}")
            y[i, j] = model.addVar(lb=0, name=f"y_{i}_{j}")

    H = {s.id: model.addVar(lb=0, name=f"H_{s.id}") for s in strips}

    l, b = {}, {}
    for j in range(n):
        for k in range(n):
            if k != j:
                l[j, k] = model.addVar(vtype=vtype, name=f"l_{j}_{k}")
                b[j, k] = model.addVar(vtype=vtype, name=f"b_{j}_{k}")

    Hmax = None
    if instance.objective == OBJ_MAKESPAN:
        Hmax = model.addVar(lb=0, name="Hmax")

    model.update()

    # -- Objective ----------------------------------------------------
    if instance.objective == OBJ_MAKESPAN:
        for s in strips:
            model.addConstr(Hmax >= H[s.id], name=f"mk_{s.id}")
        model.setObjective(Hmax, GRB.MINIMIZE)
    else:
        model.setObjective(
            gp.quicksum(s.cost * s.width * H[s.id] for s in strips),
            GRB.MINIMIZE,
        )

    # -- Constraints --------------------------------------------------

    # Eq (1): x_ij + w_ij <= W_i
    for j in range(n):
        for i in feas[j]:
            model.addConstr(
                x[i, j] + instance.w(j, i) <= strips[i].width,
                name=f"eq1_{i}_{j}")

    # Eq (2): y_ij + h_ij <= H_i + h_ij*(1 - z_ij)
    # Tight big-M = h_ij: when z_ij = 0, eq (6) forces y_ij = 0 and the
    # constraint is vacuous; when z_ij = 1 it reads y_ij + h_ij <= H_i.
    # In the LP it still yields H_i >= h_ij * z_ij.
    for j in range(n):
        for i in feas[j]:
            hij = instance.h(j, i)
            model.addConstr(
                y[i, j] + hij <= H[i] + hij * (1 - z[i, j]),
                name=f"eq2_{i}_{j}")

    # Eq (4): each item on exactly one strip
    for j in range(n):
        model.addConstr(
            gp.quicksum(z[i, j] for i in feas[j]) == 1,
            name=f"eq4_{j}")

    # Eq (5)/(6): coordinate-assignment linking
    for j in range(n):
        for i in feas[j]:
            model.addConstr(
                x[i, j] <= (strips[i].width - instance.w(j, i)) * z[i, j],
                name=f"eq5_{i}_{j}")
            model.addConstr(
                y[i, j] <= (M_y[i] - instance.h(j, i)) * z[i, j],
                name=f"eq6_{i}_{j}")

    # Eq (9): linearized non-overlap per common strip
    for j in range(n):
        fj = set(feas[j])
        for k in range(n):
            if k == j:
                continue
            common = fj & set(feas[k])
            for i in common:
                W_i = strips[i].width
                model.addConstr(
                    x[i, j] + instance.w(j, i)
                    <= x[i, k] + W_i * (3 - l[j, k] - z[i, j] - z[i, k]),
                    name=f"eq9x_{j}_{k}_{i}")
                model.addConstr(
                    y[i, j] + instance.h(j, i)
                    <= y[i, k] + M_y[i] * (3 - b[j, k] - z[i, j] - z[i, k]),
                    name=f"eq9y_{j}_{k}_{i}")

    # Eq (11): exactly one relative position per unordered pair
    for j in range(n):
        for k in range(j + 1, n):
            model.addConstr(
                l[j, k] + l[k, j] + b[j, k] + b[k, j] == 1,
                name=f"eq11_{j}_{k}")

    # Symmetry breaking on interchangeable strips
    for grp in instance.identical_strip_groups():
        for a, c in zip(grp, grp[1:]):
            model.addConstr(H[a] >= H[c], name=f"sym_{a}_{c}")

    model.update()
    return model, {'z': z, 'x': x, 'y': y, 'H': H, 'l': l, 'b': b,
                   'Hmax': Hmax}


def _extract_solution(model, variables, instance: Instance):
    """Placement dict {strip_id: [(item, x, y), ...]} from a MIP solution."""
    if model.SolCount == 0:
        return None
    z, x, y = variables['z'], variables['x'], variables['y']
    sol = {s.id: [] for s in instance.strips}
    for (i, j), var in z.items():
        if var.X > 0.5:
            sol[i].append((j, int(round(x[i, j].X)), int(round(y[i, j].X))))
    return sol


def solve_bigm_lp(instance: Instance, time_limit: float = 300.0) -> Dict:
    """LP relaxation of the big-M formulation.

    Returns dict with: lp_bound, solve_time, status, n_vars, n_constrs.
    """
    t0 = time.time()
    model, _ = _build_vasilyev_model(instance, relax=True)
    model.Params.TimeLimit = time_limit
    model.optimize()

    elapsed = time.time() - t0

    from gurobipy import GRB
    lp_bound = 0.0
    if model.Status == GRB.OPTIMAL:
        lp_bound = model.ObjVal
    elif model.Status in (GRB.TIME_LIMIT, GRB.INTERRUPTED):
        try:
            lp_bound = model.ObjVal
        except Exception:
            try:
                lp_bound = model.ObjBound
            except Exception:
                lp_bound = 0.0

    return {
        'lp_bound': lp_bound,
        'solve_time': elapsed,
        'status': model.Status,
        'n_vars': model.NumVars,
        'n_constrs': model.NumConstrs,
        'objective_mode': instance.objective,
    }


def solve_bigm_mip(
    instance: Instance,
    time_limit: float = 900.0,
    threads: int = 16,
) -> Dict:
    """Solve the big-M formulation as MIP.

    Returns dict with: objective, lower_bound, gap_pct, optimal,
    solve_time, nodes, n_vars, n_constrs, solution, objective_mode.
    """
    t0 = time.time()
    model, variables = _build_vasilyev_model(instance, relax=False)
    model.Params.TimeLimit = time_limit
    model.Params.Threads = threads
    model.optimize()

    elapsed = time.time() - t0

    from gurobipy import GRB
    obj = model.ObjVal if model.SolCount > 0 else float('inf')
    lb = model.ObjBound if model.Status != GRB.INFEASIBLE else 0.0
    gap = model.MIPGap * 100 if model.SolCount > 0 else 100.0
    opt = (model.Status == GRB.OPTIMAL)

    return {
        'objective': obj,
        'lower_bound': lb,
        'gap_pct': gap,
        'optimal': opt,
        'solve_time': elapsed,
        'nodes': int(model.NodeCount),
        'n_vars': model.NumVars,
        'n_constrs': model.NumConstrs,
        'solution': _extract_solution(model, variables, instance),
        'objective_mode': instance.objective,
    }


def solve_bigm_mip_le(
    instance: Instance,
    time_limit: float = 900.0,
    threads: int = 16,
    lp_pc_time_limit: float = 300.0,
) -> Dict:
    """BigM-LE: big-M MIP with Lower-bound Enhancement from LP-PC.

    Solves the LP relaxation of the normal-position formulation first
    and injects that dual bound as a constraint (objective >= bound).
    Total wall-clock (LP-PC + MIP) capped at *time_limit*.

    Returns dict with: objective, lower_bound, gap_pct, optimal,
    solve_time, nodes, n_vars, n_constrs, lp_pc_bound, lp_pc_time,
    solution, objective_mode.
    """
    from .formulation_normal import solve_normal_lp
    from gurobipy import GRB

    t_total = time.time()

    # -- Step 1: LP-PC dual bound ------------------------------------
    lp_pc_result = solve_normal_lp(instance, time_limit=lp_pc_time_limit)
    lp_pc_bound = lp_pc_result['lp_bound']
    lp_pc_time = lp_pc_result['solve_time']

    # -- Step 2: big-M MIP with remaining budget ---------------------
    remaining_time = max(time_limit - (time.time() - t_total), 1.0)

    model, variables = _build_vasilyev_model(instance, relax=False)
    model.Params.TimeLimit = remaining_time
    model.Params.Threads = threads

    if lp_pc_bound > 0:
        model.addConstr(model.getObjective() >= lp_pc_bound, name='lp_pc_lb')
        model.update()

    model.optimize()

    total_elapsed = time.time() - t_total

    obj = model.ObjVal if model.SolCount > 0 else float('inf')
    lb = model.ObjBound if model.Status != GRB.INFEASIBLE else 0.0
    lb = max(lb, lp_pc_bound)
    gap = (((obj - lb) / max(abs(obj), 1e-10)) * 100
           if (model.SolCount > 0 and obj < float('inf')) else 100.0)
    opt = (model.Status == GRB.OPTIMAL)

    return {
        'objective': obj,
        'lower_bound': lb,
        'gap_pct': gap,
        'optimal': opt,
        'solve_time': total_elapsed,
        'nodes': int(model.NodeCount),
        'n_vars': model.NumVars,
        'n_constrs': model.NumConstrs,
        'lp_pc_bound': lp_pc_bound,
        'lp_pc_time': lp_pc_time,
        'solution': _extract_solution(model, variables, instance),
        'objective_mode': instance.objective,
    }
