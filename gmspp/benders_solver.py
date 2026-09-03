"""
BendM: Benders' Method for Multiple strips (v2).

Benders' decomposition solver for GMSPP, extending the BLUE algorithm
of Cote et al. (2014) to multiple heterogeneous strips.  Supports
strip-dependent item dimensions (w_ij, h_ij) and both objectives
('total_cost' and 'makespan') -- the y-check subproblem and all
Benders' cuts are pure feasibility devices and are objective-agnostic.

Master problem: normal-position formulation (GMSPP-Master)
  - x_{ijp} binary: item j at position p on strip i
  - H_i: height used on strip i
  - Objective: min sum C_i * W_i * H_i   or   min max_i H_i
  - Column-load constraints per strip per column
  - Benders' cuts added via lazy constraint callback

Subproblem: y-check per strip
  - Given item assignments and x-positions, check vertical feasibility
  - If infeasible: compute MIS, lift cut via LP, add as lazy constraint
"""

import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np

from .data_structures import Instance, OBJ_MAKESPAN
from .normal_positions import (
    compute_all_normal_positions,
    compute_coverage_sets,
)
from .ycheck import YCheckItem, run_ycheck_and_cuts, YCheckResult

# Optional: skyline heuristic for warm-starting the MIP.
# If not available, the solver runs without a primal heuristic.
try:
    from .skyline_heuristic import (
        skyline_heuristic, verify_solution as verify_heuristic,
    )
    _HAS_HEURISTIC = True
except ImportError:
    _HAS_HEURISTIC = False


@dataclass
class BendersResult:
    """Results from the BendM (Benders' Method) solver."""
    objective: float = float('inf')
    lower_bound: float = 0.0
    gap_percent: float = 100.0
    optimal: bool = False
    feasible_solution: Optional[Dict] = None  # strip_id -> [(item_id, x, y)]

    # Statistics
    n_benders_cuts: int = 0
    n_mis_found: int = 0
    n_ycheck_calls: int = 0
    n_ycheck_feasible: int = 0
    n_ycheck_infeasible: int = 0
    total_time: float = 0.0
    master_time: float = 0.0
    ycheck_time: float = 0.0

    # Heuristic info
    heuristic_obj: float = float('inf')
    heuristic_time: float = 0.0
    heuristic_used: bool = False

    # Model info
    n_vars: int = 0
    n_constrs: int = 0
    nodes: int = 0


def solve_benders(
    instance: Instance,
    time_limit: float = 900.0,
    threads: int = 16,
    use_lifted_cuts: bool = True,
    use_combinatorial_cuts: bool = True,
    lifting_method: str = 'lp',
    ycheck_time_limit: float = 10.0,
    mis_time_limit: float = 5.0,
    mis_max_attempts: int = 0,
    lift_time_limit: float = 5.0,
    subproblem_threads: int = 16,
    use_skyline_heuristic: bool = True,
    skyline_iterations: int = 50,
    skyline_time_limit: float = 10.0,
    mip_start: Optional[Dict[int, List[Tuple[int, int, int]]]] = None,
    mip_start_obj: Optional[float] = None,
    lower_bound_inject: float = 0.0,
    verbose: bool = False,
) -> BendersResult:
    """
    Solve GMSPP using Benders' decomposition with Gurobi lazy callbacks.

    Args:
        instance: GMSPP instance
        time_limit: total time limit in seconds
        threads: number of Gurobi threads
        use_lifted_cuts: compute lifted combinatorial Benders' cuts
        use_combinatorial_cuts: compute minimal infeasible subsets
        lifting_method: 'column_load' (lift1) or 'lp' (lift2, Côté LP)
        ycheck_time_limit: time limit for each y-check Gurobi solve
        mis_time_limit: time limit for each MIS y-check solve
        mis_max_attempts: cap on MIS removal trials per cut (0 = unlimited).
            A positive value (e.g. 15) yields a near-minimal but valid
            infeasible subset, reducing callback time and spillover.
        lift_time_limit: time limit for each lifting solve
        verbose: print progress

    Returns:
        BendersResult with solution and statistics
    """
    import gurobipy as gp
    from gurobipy import GRB

    t_start = time.time()
    result = BendersResult()

    n = instance.n
    m = instance.m
    items = instance.items
    strips = instance.strips

    if verbose:
        print("=" * 60)
        print("  BendM: Benders' Solver for GMSPP")
        print(f"  n={n} items, m={m} strips")
        print(f"  Strips: {[s.width for s in strips]}")
        print("=" * 60)

    # ── Compute normal positions ──
    t_pos = time.time()
    positions = compute_all_normal_positions(instance)
    coverage = compute_coverage_sets(instance, positions)
    if verbose:
        total_pos = sum(len(v) for v in positions.values())
        print(f"  Normal positions: {total_pos} total ({time.time()-t_pos:.2f}s)")

    # ── Build master model ──
    model = gp.Model("GMSPP_Benders")
    model.Params.OutputFlag = 1 if verbose else 0
    model.Params.Threads = threads  # master threads (default 4, user can set higher)
    model.Params.TimeLimit = time_limit
    model.Params.LazyConstraints = 1

    # Variables: x[i,j,p] binary
    x = {}
    for (i, j), pos_list in positions.items():
        for p in pos_list:
            x[i, j, p] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}_{p}")

    # H[i] = height used on strip i (per-strip)
    H = {}
    for strip in strips:
        H[strip.id] = model.addVar(lb=0, name=f"H_{strip.id}")

    Hmax = None
    if instance.objective == OBJ_MAKESPAN:
        Hmax = model.addVar(lb=0, name="Hmax")

    model.update()

    # Objective: min sum C_i * W_i * H_i   or   min max_i H_i
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

    # Assignment constraints: each item packed exactly once
    for j in range(n):
        feasible_strips = instance.feasible_strips(j)
        terms = []
        for i in feasible_strips:
            if (i, j) in positions:
                for p in positions[(i, j)]:
                    terms.append(x[i, j, p])
        if terms:
            model.addConstr(gp.quicksum(terms) == 1, name=f"assign_{j}")

    # Column-load constraints: per strip, per column, <= H_i
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
                model.addConstr(gp.quicksum(terms) <= H[i], name=f"cl_{i}_{q}")

    # Height-linking valid inequalities: H_i >= h_ij * sum_p x_ijp
    # (implied at integer points by colload; tightens LP / node bounds)
    for (i, j), pos_list in positions.items():
        if pos_list:
            model.addConstr(
                instance.h(j, i) * gp.quicksum(x[i, j, p] for p in pos_list)
                <= H[i],
                name=f"link_{i}_{j}")

    # Symmetry breaking on interchangeable strips
    for grp in instance.identical_strip_groups():
        for a, c in zip(grp, grp[1:]):
            model.addConstr(H[a] >= H[c], name=f"sym_{a}_{c}")

    model.update()

    result.n_vars = model.NumVars
    result.n_constrs = model.NumConstrs

    # ── Skyline heuristic: primal bound + MIP start ──
    heur_obj = float('inf')
    if use_skyline_heuristic and _HAS_HEURISTIC:
        t_heur = time.time()
        heur_time_budget = min(skyline_time_limit,
                               max(1.0, time_limit * 0.05))
        heur_obj, heur_sol = skyline_heuristic(
            instance,
            n_random_iterations=skyline_iterations,
            n_swaps_no_improve=30,
            seed=42,
            time_limit=heur_time_budget,
            verbose=verbose,
        )
        heur_elapsed = time.time() - t_heur
        result.heuristic_obj = heur_obj
        result.heuristic_time = heur_elapsed
        result.heuristic_used = True

        if verbose:
            print(f"  Skyline heuristic: obj = {heur_obj:.2f} "
                  f"({heur_elapsed:.2f}s)")

        # Verify heuristic solution
        heur_ok, _ = verify_heuristic(instance, heur_sol)

        if heur_ok and heur_obj < float('inf'):
            # Set cutoff: Gurobi will prune nodes with LB >= cutoff
            model.Params.Cutoff = heur_obj + 0.5

            # Convert heuristic solution to MIP start
            # For each item placed at (strip_id, x_local, y):
            #   find the nearest normal position p and set x[i,j,p].Start = 1
            for strip_id, placements in heur_sol.items():
                for (item_id, x_local, y_pos) in placements:
                    key = (strip_id, item_id)
                    if key in positions:
                        pos_list = positions[key]
                        # Find the exact or nearest normal position
                        if x_local in pos_list:
                            var = x[strip_id, item_id, x_local]
                            var.Start = 1.0
                        else:
                            # Snap to nearest normal position
                            nearest = min(pos_list,
                                          key=lambda p: abs(p - x_local))
                            var = x[strip_id, item_id, nearest]
                            var.Start = 1.0

            # Set H[i] start values from heuristic
            for strip in strips:
                i = strip.id
                if i in heur_sol and heur_sol[i]:
                    max_h = max(y + instance.h(item_id, i)
                                for (item_id, _, y) in heur_sol[i])
                    H[i].Start = max_h
                else:
                    H[i].Start = 0.0

            model.update()

            if verbose:
                print(f"  MIP start set from heuristic (cutoff = {heur_obj:.2f})")

    # ── External warm start (e.g., from ALNS) and dual-bound injection ──
    ws_obj = float('inf')
    ws_sol = None
    if lower_bound_inject > 0:
        model.addConstr(model.getObjective() >= lower_bound_inject,
                        name='lb_inject')
        model.update()
    if mip_start is not None and mip_start_obj is not None:
        ws_obj = mip_start_obj
        ws_sol = mip_start
        for sid, placements in mip_start.items():
            hmax_s = 0
            for (item_id, x_local, y_pos) in placements:
                key = (sid, item_id)
                if key in positions and positions[key]:
                    pos_list = positions[key]
                    p_star = (x_local if x_local in pos_list else
                              min(pos_list, key=lambda p: abs(p - x_local)))
                    x[sid, item_id, p_star].Start = 1.0
                hmax_s = max(hmax_s, y_pos + instance.h(item_id, sid))
            H[sid].Start = hmax_s
        model.Params.Cutoff = mip_start_obj + 1e-4
        model.update()
        if verbose:
            print(f"  Warm start injected: obj = {mip_start_obj:.2f}")

    # ── Statistics tracking ──
    init_best = heur_obj if use_skyline_heuristic else float('inf')
    init_best = min(init_best, ws_obj)
    stats = {
        'n_benders_cuts': 0,
        'n_ycheck_calls': 0,
        'n_ycheck_feasible': 0,
        'n_ycheck_infeasible': 0,
        'ycheck_time': 0.0,
        'best_feasible': init_best,
        'best_solution': ws_sol,
        'best_bound': lower_bound_inject,  # track best LB from callbacks
    }

    # ── Combined callback: MIPSOL (Benders) + MIPNODE (F2 separation) ──
    def benders_callback(model, where):
        past_limit = (time.time() - t_start) > time_limit

        # ── Wall-clock guard for non-essential callbacks ──
        # Skip F2 separation and bound tracking when past the time
        # limit, but NEVER skip MIPSOL: we must still run y-check and
        # add a cut if the solution is infeasible, otherwise Gurobi
        # may report an invalid incumbent.
        if past_limit and where != GRB.Callback.MIPSOL:
            model.terminate()
            return

        # ── Track best bound from B&B tree ──
        if where == GRB.Callback.MIPSOL:
            try:
                bound = model.cbGet(GRB.Callback.MIPSOL_OBJBND)
                if bound > stats['best_bound']:
                    stats['best_bound'] = bound
            except Exception:
                pass
        elif where == GRB.Callback.MIPNODE:
            try:
                bound = model.cbGet(GRB.Callback.MIPNODE_OBJBND)
                if bound > stats['best_bound']:
                    stats['best_bound'] = bound
            except Exception:
                pass

        # ── Benders' lazy cuts at integer solutions ──
        if where != GRB.Callback.MIPSOL:
            return

        # Get current per-strip heights
        H_vals = {si: model.cbGetSolution(H[si]) for si in H}

        # Extract assignment: for each strip, which items at which positions
        strip_assignments = {i: [] for i in range(m)}
        for (i, j, p), var in x.items():
            val = model.cbGetSolution(var)
            if val > 0.5:
                strip_assignments[i].append((j, p))

        # Run y-check on each strip (early stop: cut and return on first infeasible)
        # Following Côté et al. (2014): generate one cut per callback,
        # then return to the solver.  This avoids expensive MIS/lifting
        # calls on remaining strips when one cut suffices.
        all_feasible = True
        for strip in strips:
            i = strip.id
            W_i = strip.width
            assigned = strip_assignments[i]

            if not assigned:
                continue

            # Build YCheckItem list (dims as taken on THIS strip)
            ycheck_items = []
            for (j, p) in assigned:
                ycheck_items.append(YCheckItem(
                    id=j, x_pos=p,
                    width=instance.w(j, i), height=instance.h(j, i),
                ))

            t_yc = time.time()
            stats['n_ycheck_calls'] += 1

            # Wall-clock guard: if past the overall time limit, skip
            # expensive MIS/lifting and emit a standard (weaker but
            # valid) cut instead.  This directly caps spillover.
            cb_past_limit = past_limit or (time.time() - t_start) > time_limit
            cb_compute_mis = use_combinatorial_cuts and not cb_past_limit
            cb_compute_lift = use_lifted_cuts and not cb_past_limit

            yc_result = run_ycheck_and_cuts(
                ycheck_items, W_i, int(np.ceil(H_vals[i])),
                compute_mis=cb_compute_mis,
                compute_lift=cb_compute_lift,
                lifting_method=lifting_method,
                ycheck_time_limit=ycheck_time_limit,
                mis_time_limit=mis_time_limit,
                mis_max_attempts=mis_max_attempts,
                lift_time_limit=lift_time_limit,
                threads=subproblem_threads,
            )

            stats['ycheck_time'] += time.time() - t_yc

            if yc_result.feasible:
                stats['n_ycheck_feasible'] += 1
            else:
                stats['n_ycheck_infeasible'] += 1
                all_feasible = False

                # Generate H_i-aware Benders' cut
                #
                # For the area-cost formulation (min Σ C_i W_i H_i), the
                # cut must account for the fact that a future solution
                # may use a larger H_i on this strip. Instead of
                # unconditionally forbidding the item-position combination:
                #   sum x <= |C| - 1                    (old, invalid)
                # we use:
                #   H[i] >= h_min * (sum x - |C| + 1)   (H_i-aware)
                # which says: IF all items are at these positions,
                # THEN H_i >= target_height + 1.
                #
                # Rearranged to standard form for cbLazy:
                #   h_min * sum_x - H[i] <= h_min * (|C| - 1)

                target_h = int(np.ceil(H_vals[i]))
                h_min = target_h + 1

                if yc_result.mis_item_ids is not None and use_combinatorial_cuts:
                    mis_ids = yc_result.mis_item_ids

                    if yc_result.lifted_intervals is not None and use_lifted_cuts:
                        # Lifted combinatorial Benders' cut
                        # (intervals verified jointly in ycheck.py)
                        cut_terms = []
                        for j_id in mis_ids:
                            l_bound, r_bound = yc_result.lifted_intervals[j_id]
                            if (i, j_id) in positions:
                                for p in positions[(i, j_id)]:
                                    if l_bound <= p <= r_bound:
                                        cut_terms.append(x[i, j_id, p])
                        if cut_terms:
                            model.cbLazy(
                                h_min * gp.quicksum(cut_terms) - H[i]
                                <= h_min * (len(mis_ids) - 1)
                            )
                            stats['n_benders_cuts'] += 1
                    else:
                        # Combinatorial Benders' cut (MIS, exact positions)
                        cut_terms = []
                        assigned_dict = {j: p for (j, p) in assigned}
                        for j_id in mis_ids:
                            p_star = assigned_dict.get(j_id)
                            if p_star is not None and (i, j_id, p_star) in x:
                                cut_terms.append(x[i, j_id, p_star])
                        if cut_terms:
                            model.cbLazy(
                                h_min * gp.quicksum(cut_terms) - H[i]
                                <= h_min * (len(mis_ids) - 1)
                            )
                            stats['n_benders_cuts'] += 1
                else:
                    # Standard Benders' cut (all assigned items, exact pos)
                    cut_terms = []
                    for (j, p) in assigned:
                        if (i, j, p) in x:
                            cut_terms.append(x[i, j, p])
                    if cut_terms:
                        model.cbLazy(
                            h_min * gp.quicksum(cut_terms) - H[i]
                            <= h_min * (len(assigned) - 1)
                        )
                        stats['n_benders_cuts'] += 1

                # Early stop: one cut per callback, return to solver
                break

        # Signal Gurobi to stop after this callback if past time limit
        if past_limit:
            model.terminate()

        # If all strips feasible, we have a valid GMSPP solution
        if all_feasible:
            total_cost = instance.objective_value(
                [H_vals[i] for i in range(m)]
            )
            if total_cost < stats['best_feasible']:
                stats['best_feasible'] = total_cost
                solution = {}
                for strip in strips:
                    i = strip.id
                    assigned = strip_assignments[i]
                    solution[i] = [(j, p, 0) for (j, p) in assigned]
                stats['best_solution'] = solution

    # ── Solve ──
    if verbose:
        print(f"\n  Solving master with Benders' callback...")
        print(f"  Model: {model.NumVars} vars, {model.NumConstrs} constrs")

    model.optimize(benders_callback)

    elapsed = time.time() - t_start

    # ── Collect results ──
    result.total_time = elapsed
    result.master_time = elapsed - stats['ycheck_time']
    result.ycheck_time = stats['ycheck_time']
    result.n_benders_cuts = stats['n_benders_cuts']
    result.n_ycheck_calls = stats['n_ycheck_calls']
    result.n_ycheck_feasible = stats['n_ycheck_feasible']
    result.n_ycheck_infeasible = stats['n_ycheck_infeasible']
    result.nodes = int(model.NodeCount) if model.SolCount > 0 else 0

    if stats['best_feasible'] < float('inf'):
        result.objective = stats['best_feasible']
        result.feasible_solution = stats['best_solution']

    if model.Status in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.NODE_LIMIT,
                         GRB.INTERRUPTED):
        try:
            result.lower_bound = model.ObjBound
        except Exception:
            result.lower_bound = 0.0
        # Use callback-tracked bound as fallback (more reliable after terminate())
        if result.lower_bound <= 0 and stats['best_bound'] > 0:
            result.lower_bound = stats['best_bound']
    # CUTOFF: Gurobi proved no solution better than the injected warm
    # start exists -- the warm start is optimal.
    if model.Status == GRB.CUTOFF and ws_obj < float('inf'):
        result.lower_bound = max(result.lower_bound, ws_obj)
    result.optimal = ((model.Status == GRB.OPTIMAL or
                       (model.Status == GRB.CUTOFF and
                        ws_obj < float('inf'))) and
                      stats['best_feasible'] < float('inf'))

    # Compute optimality gap: (UB - LB) / UB * 100
    # UB = best y-check-verified objective
    # LB = Gurobi's best B&B bound (valid for Benders with lazy constraints)
    if result.objective < float('inf') and result.lower_bound > 0:
        result.gap_percent = (
            (result.objective - result.lower_bound) / result.objective * 100
        )
    else:
        result.gap_percent = 100.0

    if verbose:
        print(f"\n  === BendM Results ===")
        print(f"  Objective:    {result.objective}")
        print(f"  Lower bound:  {result.lower_bound:.2f}")
        print(f"  Gap:          {result.gap_percent:.2f}%")
        print(f"  Optimal:      {result.optimal}")
        print(f"  Benders cuts: {result.n_benders_cuts}")
        print(f"  y-checks:     {result.n_ycheck_calls} "
              f"({result.n_ycheck_feasible} feas, "
              f"{result.n_ycheck_infeasible} infeas)")
        if result.heuristic_used:
            print(f"  Heuristic:    obj={result.heuristic_obj:.2f} "
                  f"({result.heuristic_time:.2f}s)")
        print(f"  Time:         {result.total_time:.1f}s "
              f"(master {result.master_time:.1f}s, "
              f"y-check {result.ycheck_time:.1f}s)")
        print(f"  B&B nodes:    {result.nodes}")

    return result
