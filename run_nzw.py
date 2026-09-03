"""Zero-waste (N) appendix block: endgame on/off ablation.

Justifies Phase 3 empirically: on jigsaw-like zero-waste instances
the exact endgame recovers packing slack that the heuristic packer
leaves; on CLASS it is inactive (Section 6.6).  Writes
results/v2_nzw.csv (resume-safe).

Run:  python run_nzw.py
"""
REV = "nzw-r1-2026-09-01"

import os, time
import pandas as pd

from gmspp.benchmark_loader import load_and_convert_benchmark
from gmspp.data_structures import Instance
from gmspp.alns_solver import solve_alns

import sys
T = int(sys.argv[1]) if len(sys.argv) > 1 else 300
SEED = 42
NAMES = [f'N{i}{v}' for i in range(1, 8) for v in ('a', 'c', 'e')]
MS, OBJS = (2, 3), ('total_cost', 'makespan')
PATH = f'results/v2_nzw{T}.csv' if T != 300 else 'results/v2_nzw.csv'


def main():
    print(f'run_nzw {REV}')
    os.makedirs('results', exist_ok=True)
    df = (pd.read_csv(PATH) if os.path.exists(PATH)
          else pd.DataFrame())
    done = set(df['key']) if len(df) else set()
    todo = []
    for m in MS:
        insts = load_and_convert_benchmark('benchmarks/N', m=m,
                                           cost_type='proportional')
        for nm in NAMES:
            if nm not in insts:
                continue
            for obj in OBJS:
                key = f'{nm}_m{m}_{obj}'
                if key not in done:
                    todo.append((key, insts[nm], obj))
    print(f'{len(todo)} to run at {T}s x 2 configs, {len(done)} done')
    for i, (key, base, obj) in enumerate(todo):
        inst = Instance(items=base.items, strips=base.strips,
                        objective=obj)
        ls = min(15, max(5, T // 8))      # T=300 -> 15 (unchanged)
        eg = max(15, T // 10)             # T=300 -> 30 (unchanged)
        r0 = solve_alns(inst, time_limit=T, seed=SEED, verbose=False,
                        ls_time=ls, endgame_time=0.0, use_endgame=False)
        r1 = solve_alns(inst, time_limit=T, seed=SEED, verbose=False,
                        ls_time=ls, endgame_time=eg)
        gain = (r0.objective - r1.objective) / r0.objective * 100
        row = {'key': key, 'n': inst.n, 'm': inst.m, 'objective': obj,
               'obj_off': r0.objective, 'obj_on': r1.objective,
               'gain_pct': gain, 'eg_strips': r1.endgame_strips,
               'repairs': r0.state_repairs + r1.state_repairs}
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        df.to_csv(PATH, index=False)
        print(f'[{i+1}/{len(todo)}] {key}: off={r0.objective:.0f} '
              f'on={r1.objective:.0f} gain={gain:.2f}%', flush=True)


if __name__ == '__main__':
    main()
