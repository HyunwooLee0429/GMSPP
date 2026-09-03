"""Fuzz harness for ALNS state invariants (OneDrive-safe script).

Runs many short searches over randomized instances (both objectives,
strip-dependent dims via Type I/II generators) and asserts, per trial:
  (1) state_repairs == 0  -- no internal drift was ever detected
      (the search self-heals, but a nonzero count means a bug exists);
  (2) the returned solution passes an independent from-scratch
      feasibility check and its objective matches the reported value.

Any failure prints the full recipe (generator, n, m, objective, seeds)
so the trial is exactly reproducible.

Usage:
    python fuzz_alns.py            # 40 trials, base seed 0
    python fuzz_alns.py 200 7      # 200 trials, base seed 7
"""
import sys
import random

from gmspp.data_structures import Instance
from gmspp.vasilyev import generate_type1, generate_type2
from gmspp.alns_solver import solve_alns


def verify(inst, r, tol=1e-6):
    """Independent from-scratch feasibility check (mirrors
    run_v2.verify_result, duplicated to stay import-light)."""
    seen, heights = [], []
    for s in inst.strips:
        plc = r.solution.get(s.id, [])
        H, rects = 0, []
        for j, x, y in plc:
            w, h = inst.dims(j, s.id)
            assert x >= 0 and y >= 0 and x + w <= s.width, \
                f'strip {s.id}, item {j} out of bounds'
            rects.append((x, y, x + w, y + h))
            seen.append(j)
            H = max(H, y + h)
        for a in range(len(rects)):
            x1, y1, x2, y2 = rects[a]
            for b in range(a + 1, len(rects)):
                u1, v1, u2, v2 = rects[b]
                assert not (x1 < u2 and u1 < x2
                            and y1 < v2 and v1 < y2), \
                    f'overlap on strip {s.id}'
        heights.append(H)
    assert sorted(seen) == list(range(inst.n)), 'item multiset mismatch'
    obj = inst.objective_value(heights)
    assert abs(obj - r.objective) <= tol * max(1.0, abs(obj)), \
        f'objective mismatch: reported {r.objective}, recomputed {obj}'
    return obj


def main():
    n_trials = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    base = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    rng = random.Random(base)
    fails = 0
    for t in range(n_trials):
        gen = rng.choice([generate_type1, generate_type2])
        n = rng.randint(10, 45)
        m = rng.randint(2, 4)
        gseed = rng.randint(0, 10**6)
        obj_name = rng.choice(['total_cost', 'makespan'])
        sseed = rng.randint(0, 10**6)
        recipe = (f'{gen.__name__}(n={n}, m={m}, seed={gseed}) '
                  f'obj={obj_name} alns_seed={sseed}')
        b = gen(n, m, seed=gseed)
        inst = Instance(items=b.items, strips=b.strips,
                        objective=obj_name,
                        w_mat=b.w_mat, h_mat=b.h_mat)
        try:
            r = solve_alns(inst, time_limit=2.5, seed=sseed,
                           verbose=False, ls_time=0.5, endgame_time=1.0)
            verify(inst, r)
            assert r.state_repairs == 0, \
                f'{r.state_repairs} internal state repairs (drift!)'
            print(f'[{t + 1}/{n_trials}] ok  obj={r.objective:.0f}  '
                  f'{recipe}', flush=True)
        except (AssertionError, RuntimeError) as e:
            fails += 1
            print(f'[{t + 1}/{n_trials}] FAIL  {recipe}\n    -> {e}',
                  flush=True)
    print(f'\n{n_trials - fails}/{n_trials} passed'
          + ('' if fails == 0 else f'  ({fails} FAILURES)'))
    sys.exit(1 if fails else 0)


if __name__ == '__main__':
    main()
