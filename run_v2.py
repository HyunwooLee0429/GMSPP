"""GMSPP v2 experiment driver (script version -- OneDrive-safe).

Run from Anaconda Prompt:
    python run_v2.py C          # Block C only
    python run_v2.py B          # Block B only
    python run_v2.py C B        # both, in order
    python run_v2.py A          # main campaign
All blocks resume from results/v2_*.csv. Analysis stays in the
notebook (which only READS the CSVs, so autosave clashes are harmless).
"""
REV = "r9-2026-09-01"

import os, sys, time
import numpy as np, pandas as pd
import multiprocessing as mp

from gmspp.data_structures import Instance, OBJ_MAKESPAN
from gmspp import (solve_alns_parallel, solve_benders, solve_bigm_mip_le,
                   mmsp_lower_bound, generate_type1, generate_type2,
                   load_zdf_gmspp)
from gmspp.formulation_normal import solve_normal_lp
from gmspp.formulation_bigm import solve_bigm_lp, solve_bigm_mip
from gmspp.alns_solver import solve_alns
from gmspp.benchmark_loader import (load_and_convert_benchmark,
                                    parse_ins2d, spp_to_gmspp)

# ============ CONFIG ============
# T_MAIN = 1200 adopts the per-instance budget of Cote et al. (2014)
T_MAIN, T_SWEEP, THREADS = 1200, 300, 16
ENDGAME_TIME, ALNS_RUNS, ALNS_WORKERS = 30, 10, None
BIGM_LP_MAX_N, EXACT_MAX_N = 120, 100
OBJECTIVES, MS = ['total_cost', 'makespan'], [2, 3]
TYPE1 = [(150, 2), (150, 3), (250, 4), (400, 4)]
TYPE2 = [(150, 2), (150, 3), (250, 4), (400, 4)]
ZDF_SMALL = [('zdf1', 2), ('zdf2', 2), ('zdf3', 2), ('zdf1', 4)]
# Stress rows (scale coverage; package-only in Tier 2/3, bounds in
# Tier 1 without LP-PC beyond LP_MAX_N): n ~ 1266-2532, m up to 8,
# all from Vasilyev's own source collection.
ZDF_STRESS = [('zdf8', 2), ('zdf10', 4), ('zdf12', 8), ('zdf10', 2)]
# Mid band (localize the exact/matheuristic frontier between n=400
# and n=1258): generator sizes + intermediate ZDF splits.
MID_T = [(600, 4), (800, 4)]            # for both Type I and Type II
ZDF_MID = [('zdf4', 2), ('zdf6', 2)]    # n = 410, 766
LP_MAX_N = 900          # skip LP-PC beyond this n (model too large;
                        # W<=150 keeps LP-PC tractable through n~800)
ZDF_DIR, SEED = 'benchmarks/ZDF/ZDF/ZDF', 0
PILOT_KEYS_SRC = [(3, 60, 1), (3, 100, 1), (8, 60, 1),
                  (10, 60, 1), (10, 100, 1)]
# CLASS (Berkey-Wang / Martello-Vigo): THE primary fixed-dims family.
# N dropped entirely: zero-waste construction makes every dual bound
# collapse to the area bound (Cote 2014, p.657).
CLASS_SPEC = {c: ([20, 60, 100], [1, 2]) for c in range(1, 11)}
A_CLASSES = (1, 3, 8, 10)   # difficulty spectrum for the hierarchy table
# Table A1 (hierarchy, small sizes): plain big-M collapses; LE injects
# the missing bound; BendM wins.  Establishes why solver-on-model is
# not "an exact method" and BendM is.
A1_NS = (20, 60)
METHODS_A1 = ['bigm', 'bigmle', 'bendm', 'package']
# Table A2 (frontier, full range): BendM vs package everywhere -->
# where does exact stop, where does the matheuristic take over.
METHODS_A2 = ['bendm', 'package']


def build_registry():
    registry = {}
    for n, m in TYPE1:
        registry[f'T1_{n}_{m}'] = generate_type1(n, m, seed=SEED)
    for n, m in TYPE2:
        registry[f'T2_{n}_{m}'] = generate_type2(n, m, seed=SEED)
    for name, m in ZDF_SMALL:
        registry[f'ZDF_{name}_{m}'] = load_zdf_gmspp(ZDF_DIR, name, m,
                                                     shuffle_seed=SEED)
    for name, m in ZDF_STRESS:
        registry[f'ZDFS_{name}_{m}'] = load_zdf_gmspp(ZDF_DIR, name, m,
                                                      shuffle_seed=SEED)
    for n, m in MID_T:
        registry[f'T1_{n}_{m}'] = generate_type1(n, m, seed=SEED)
        registry[f'T2_{n}_{m}'] = generate_type2(n, m, seed=SEED)
    for name, m in ZDF_MID:
        registry[f'ZDF_{name}_{m}'] = load_zdf_gmspp(ZDF_DIR, name, m,
                                                     shuffle_seed=SEED)
    for cls, (ns, idxs) in CLASS_SPEC.items():
        for n0 in ns:
            for idx in idxs:
                f = (f'benchmarks/CLASS/'
                     f'cl_{cls:02d}_{n0:03d}_{idx:02d}.ins2D')
                spp = parse_ins2d(f)
                for m in MS:
                    base = spp_to_gmspp(spp, m=m,
                                        cost_type='proportional')
                    for obj in OBJECTIVES:
                        key = (f'CL{cls:02d}_{n0:03d}_{idx:02d}'
                               f'_m{m}_{obj}')
                        registry[key] = Instance(
                            items=base.items, strips=base.strips,
                            objective=obj)
    return registry


def load_done(path):
    if os.path.exists(path):
        df = pd.read_csv(path)
        return df, set(df['key'])
    return pd.DataFrame(), set()


def append_row(path, df, row):
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(path, index=False)
    return df


def dual_bound(inst, lp_tl=300):
    """max(LP-PC, MMSP, simple combinatorial bound).  LP-PC is
    skipped beyond LP_MAX_N (stress rows: portfolio = MMSP + h*)."""
    if inst.n <= LP_MAX_N:
        lp = solve_normal_lp(inst, time_limit=lp_tl)['lp_bound']
    else:
        lp = float('nan')
    simple = inst.simple_lower_bound()
    cands = [simple] + ([lp] if lp == lp else [])
    if inst.objective == OBJ_MAKESPAN:
        try:
            mm = mmsp_lower_bound(inst, time_limit=120)['lb']
        except Exception:
            mm = float('nan')
        cands += [mm] if mm == mm else []
        return max(cands), lp, mm
    return max(cands), lp, float('nan')


def block_C(registry):
    path = 'results/v2_relax.csv'
    df, done = load_done(path)
    keys = [k for k in registry
            if registry[k].objective == OBJ_MAKESPAN and k not in done]
    print(f'C: {len(keys)} to run, {len(done)} done')
    for i, key in enumerate(keys):
        inst = registry[key]
        row = {'key': key, 'n': inst.n, 'm': inst.m}
        mm = mmsp_lower_bound(inst, time_limit=120, threads=THREADS)
        row['mmsp_lb'] = mm['lb']; row['mmsp_time'] = mm['solve_time']
        ml = mmsp_lower_bound(inst, time_limit=120, threads=THREADS,
                              relax=True)
        row['mmsp_lp'] = ml['lb']
        if inst.n <= LP_MAX_N:
            lp = solve_normal_lp(inst, time_limit=300)
            row['lp_pc'] = lp['lp_bound']
            row['lp_pc_time'] = lp['solve_time']
        else:
            row['lp_pc'] = float('nan')
        if inst.n <= BIGM_LP_MAX_N:
            bl = solve_bigm_lp(inst, time_limit=300)
            row['bigm_lp'] = bl['lp_bound']
            row['bigm_lp_time'] = bl['solve_time']
        else:
            row['bigm_lp'] = float('nan')
        row['simple_lb'] = inst.simple_lower_bound()
        row['best_lb'] = max(v for v in (row['lp_pc'], row['mmsp_lb'],
                                         row['simple_lb']) if v == v)
        df = append_row(path, df, row)
        print(f"[{i+1}/{len(keys)}] {key}: MMSP={row['mmsp_lb']:.1f} "
              f"LP-PC={row['lp_pc']:.1f} simple={row['simple_lb']:.1f}",
              flush=True)


def verify_result(inst, r, lb=None, tol=1e-6):
    """Independent feasibility check of a solver result, OUTSIDE the
    solver: re-validates the raw placements from scratch (each item
    packed exactly once, in bounds, pairwise non-overlap) and
    recomputes the objective.  Raises on any violation, so an invalid
    solution can never reach the CSV; the returned (re-verified)
    objective is the only value ever recorded."""
    seen, heights = [], []
    for s in inst.strips:
        plc = r.solution.get(s.id, [])
        H, rects = 0, []
        for j, x, y in plc:
            w, h = inst.dims(j, s.id)
            if x < 0 or y < 0 or x + w > s.width:
                raise RuntimeError(
                    f'VERIFY: strip {s.id}, item {j} out of bounds')
            rects.append((x, y, x + w, y + h))
            seen.append(j)
            H = max(H, y + h)
        for a in range(len(rects)):
            x1, y1, x2, y2 = rects[a]
            for b in range(a + 1, len(rects)):
                u1, v1, u2, v2 = rects[b]
                if x1 < u2 and u1 < x2 and y1 < v2 and v1 < y2:
                    raise RuntimeError(
                        f'VERIFY: overlap on strip {s.id}: items '
                        f'{rects[a]} / {rects[b]}')
        heights.append(H)
    if sorted(seen) != list(range(inst.n)):
        raise RuntimeError('VERIFY: item multiset mismatch '
                           f'(got {len(seen)} placements)')
    obj = inst.objective_value(heights)
    scale = max(1.0, abs(obj))
    if abs(obj - r.objective) > tol * scale:
        raise RuntimeError(f'VERIFY: objective mismatch: reported '
                           f'{r.objective}, recomputed {obj}')
    if lb is not None and obj < lb - tol * scale:
        raise RuntimeError(f'VERIFY: solution beats dual bound: '
                           f'obj={obj}, lb={lb} -- bound bug?')
    return obj


def block_B(registry):
    path = 'results/v2_alns.csv'
    df, done = load_done(path)
    keys = [k for k in registry if k not in done]
    print(f'B: {len(keys)} to run at {T_SWEEP}s, {len(done)} done')
    for i, key in enumerate(keys):
        inst = registry[key]
        lb, lp, mm = dual_bound(inst)
        r = solve_alns_parallel(inst, n_runs=ALNS_RUNS,
                                n_workers=ALNS_WORKERS,
                                time_limit=T_SWEEP, compute_dual=False,
                                endgame_time=ENDGAME_TIME, verbose=False)
        obj_v = verify_result(inst, r, lb=lb)   # raises if infeasible
        gap = ((obj_v - lb) / max(obj_v, 1e-9) * 100
               if lb > 0 else float('nan'))
        row = {'key': key, 'n': inst.n, 'm': inst.m,
               'objective': inst.objective,
               'obj': obj_v, 'lb': lb, 'lp_pc': lp, 'mmsp': mm,
               'gap_pct': gap, 'time': r.total_time,
               'obj_after_alns': r.obj_after_alns,
               'ls_moves': r.ls_moves, 'endgame_strips': r.endgame_strips,
               'obj_mean': r.obj_mean, 'obj_std': r.obj_std,
               'repairs': getattr(r, 'state_repairs', 0)}
        df = append_row(path, df, row)
        if row['repairs']:
            print(f"  WARNING {key}: {row['repairs']} internal state "
                  f"repairs -- investigate", flush=True)
        print(f"[{i+1}/{len(keys)}] {key}: obj={obj_v:.0f} "
              f"lb={lb:.1f} gap={gap:.1f}% [verified]", flush=True)


def block_P(registry):
    path = 'results/v2_pilot.csv'
    df, done = load_done(path)
    keys = [f'CL{c:02d}_{n0:03d}_{i0:02d}_m{m}_{obj}'
            for (c, n0, i0) in PILOT_KEYS_SRC
            for m in MS for obj in OBJECTIVES]
    keys = [k for k in keys if k in registry and k not in done]
    print(f'P: {len(keys)} at {T_MAIN}s x 2 methods')
    for i, key in enumerate(keys):
        inst = registry[key]
        row = {'key': key, 'n': inst.n, 'm': inst.m,
               'objective': inst.objective}
        r0 = solve_benders(inst, time_limit=T_MAIN, threads=THREADS,
                           use_skyline_heuristic=False, verbose=False)
        row.update(cold_obj=r0.objective, cold_lb=r0.lower_bound,
                   cold_gap=r0.gap_percent, cold_opt=r0.optimal)
        t0 = time.time()
        lb, _, _ = dual_bound(inst, lp_tl=60)
        a = solve_alns(inst, time_limit=60, seed=42, verbose=False,
                       ls_time=10, endgame_time=15)
        a_obj = verify_result(inst, a, lb=lb)    # raises if infeasible
        rem = max(T_MAIN - (time.time() - t0), 10)
        r1 = solve_benders(inst, time_limit=rem, threads=THREADS,
                           use_skyline_heuristic=False,
                           mip_start=a.solution, mip_start_obj=a_obj,
                           lower_bound_inject=lb, verbose=False)
        row.update(warm_alns=a_obj,
                   warm_obj=min(r1.objective, a_obj),
                   warm_lb=r1.lower_bound, warm_gap=r1.gap_percent,
                   warm_opt=r1.optimal)
        df = append_row(path, df, row)
        print(f"[{i+1}/{len(keys)}] {key}: cold={r0.objective:.0f} "
              f"warm={row['warm_obj']:.0f}", flush=True)


def block_A(registry):
    """Equal-budget campaign (T_MAIN each), one CSV, resume by
    (key, method).

    A1 hierarchy: CLASS classes A_CLASSES, n in A1_NS, all four
       methods (bigm, bigmle, bendm, package).
    A2 frontier: ALL CLASS keys (n <= EXACT_MAX_N) + every sched/ZDF
       key, bendm + package only.  Overlapping (key, method) pairs
       are computed once.
    """
    path = 'results/v2_main.csv'
    df, _ = load_done(path)
    done = set(zip(df['key'], df['method'])) if len(df) else set()
    a1_keys = [k for k in registry
               if k.startswith('CL')
               and int(k[2:4]) in A_CLASSES
               and registry[k].n in A1_NS]
    a2_keys = [k for k in registry
               if (k.startswith('CL') and registry[k].n <= EXACT_MAX_N)
               or not k.startswith('CL')]   # incl. stress: BendM frontier
                                            # must be measured, not assumed
    todo = [(k, meth) for k in a1_keys for meth in METHODS_A1]
    todo += [(k, meth) for k in a2_keys for meth in METHODS_A2]
    todo = [(k, meth) for (k, meth) in dict.fromkeys(todo)
            if (k, meth) not in done]
    todo.sort(key=lambda km: registry[km[0]].n)   # small first
    print(f'A: {len(todo)} pairs at {T_MAIN}s '
          f'(A1 {len(a1_keys)} keys x 4, A2 {len(a2_keys)} keys x 2)')
    for i, (key, meth) in enumerate(todo):
        inst = registry[key]
        row = {'key': key, 'n': inst.n, 'm': inst.m,
               'objective': inst.objective, 'method': meth}
        t0 = time.time()
        if meth == 'bigm':
            r = solve_bigm_mip(inst, time_limit=T_MAIN, threads=THREADS)
            row.update(obj=r['objective'], lb=r['lower_bound'],
                       gap=r['gap_pct'], opt=r['optimal'])
        elif meth == 'bigmle':
            r = solve_bigm_mip_le(inst, time_limit=T_MAIN, threads=THREADS,
                                  lp_pc_time_limit=120)
            row.update(obj=r['objective'], lb=r['lower_bound'],
                       gap=r['gap_pct'], opt=r['optimal'])
        elif meth == 'bendm':
            r = solve_benders(inst, time_limit=T_MAIN, threads=THREADS,
                              use_skyline_heuristic=False)
            row.update(obj=r.objective, lb=r.lower_bound,
                       gap=r.gap_percent, opt=r.optimal)
        elif meth == 'package':
            lb0, _, _ = dual_bound(inst)
            r = solve_alns_parallel(inst, n_runs=ALNS_RUNS,
                                    n_workers=ALNS_WORKERS,
                                    time_limit=T_MAIN - 60,
                                    compute_dual=False,
                                    endgame_time=ENDGAME_TIME,
                                    verbose=False)
            obj_v = verify_result(inst, r, lb=lb0)  # raises if infeasible
            g = ((obj_v - lb0) / max(obj_v, 1e-9) * 100
                 if lb0 > 0 else float('nan'))
            row.update(obj=obj_v, lb=lb0, gap=g, opt=abs(g) < 1e-6)
        row['time'] = time.time() - t0
        df = append_row(path, df, row)
        print(f"[{i+1}/{len(todo)}] {key} {meth}: obj={row['obj']:.0f} "
              f"gap={row['gap']:.1f}%", flush=True)


def main():
    print(f"run_v2 {REV} | cores: {mp.cpu_count()}")
    os.makedirs('results', exist_ok=True)
    registry = build_registry()
    print(f"{len(registry)} instances registered")
    blocks = [b.upper() for b in sys.argv[1:]] or ['C', 'B']
    for b in blocks:
        {'C': block_C, 'B': block_B, 'P': block_P, 'A': block_A}[b](registry)


if __name__ == '__main__':          # REQUIRED on Windows (multiprocessing)
    main()
