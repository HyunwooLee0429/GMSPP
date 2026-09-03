"""
ALNS (Adaptive Large Neighborhood Search) solver for GMSPP (v2).

Heuristic solver for the Generalized Multiple Strip Packing Problem.
Supports strip-dependent item dimensions (w_ij, h_ij) and both
objectives ('total_cost': min sum C_i*W_i*H_i, 'makespan': min max H_i).

Three-phase matheuristic:
    Phase 1 -- ALNS over strip assignments (destroy/repair + SA),
               per-strip packing by skyline best-fit / multi-order BLF.
    Phase 2 -- local search: single-item relocation + pairwise swap.
    Phase 3 -- CP-SAT endgame: freeze the assignment, re-optimize each
               strip's packing exactly (time-boxed).

Destroy operators (4):
    random   -- remove k random items
    worst    -- remove items with highest objective-contribution proxy
    related  -- Shaw removal: geometrically similar items
    strip    -- empty an entire strip (chosen by contribution)

Repair operators (4):
    greedy   -- insert each item at strip with minimum surrogate cost
    regret-2 -- insert item with maximum (2nd_best - best) regret first
    random   -- uniform random feasible strip (diversity)
    spread   -- round-robin distribution across strips

Acceptance: simulated annealing with geometric cooling.
Weights: adaptive roulette-wheel per Ropke & Pisinger (2006).

Usage::

    from gmspp.alns_solver import solve_alns, solve_alns_parallel
    result = solve_alns(instance, time_limit=300, seed=42)
    result = solve_alns_parallel(instance, n_runs=10, time_limit=300)

References:
    Ropke, S. & Pisinger, D. (2006). An ALNS heuristic for the PDPTW.
    Burke, E., Kendall, G. & Whitwell, G. (2004). A new placement
        heuristic for the orthogonal stock-cutting problem.
"""

import math
import multiprocessing as _mp
import random
import time as _time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .data_structures import Instance, OBJ_MAKESPAN

# Surrogate scaling: makespan increase dominates load-balance term
_MK_SCALE = 10 ** 7


# =====================================================================
#  Packing core (numpy height-map)
# =====================================================================

def _best_pos(hmap: np.ndarray, w: int) -> Tuple[int, int]:
    """Leftmost x minimizing the max height over window of width *w*.

    Returns (x, y) or (-1, -1) if w > len(hmap).
    """
    W = len(hmap)
    if w > W:
        return -1, -1
    if w == W:
        return 0, int(hmap.max())
    n_pos = W - w + 1
    try:
        from numpy.lib.stride_tricks import sliding_window_view
        maxh = sliding_window_view(hmap, w).max(axis=1)
    except Exception:
        maxh = np.array([int(hmap[x:x + w].max()) for x in range(n_pos)])
    bx = int(np.argmin(maxh))
    return bx, int(maxh[bx])


def _pack_order(W: int, seq: List[Tuple[int, int, int]],
                ) -> Tuple[np.ndarray, int, List[Tuple[int, int, int]]]:
    """BLF-pack items in the given sequence of (id, w, h).

    Returns (height_map, total_height, placements[(id, x, y)]).
    """
    hmap = np.zeros(W, dtype=np.int64)
    placements: List[Tuple[int, int, int]] = []
    for j, w, h in seq:
        bx, by = _best_pos(hmap, w)
        if bx < 0:
            continue                                # wider than strip
        placements.append((j, bx, by))
        hmap[bx:bx + w] = by + h
    return hmap, int(hmap.max()) if len(hmap) else 0, placements


def _pack_bestfit(W: int, dims: List[Tuple[int, int, int]],
                  ) -> Tuple[np.ndarray, int, List[Tuple[int, int, int]]]:
    """Burke-style best-fit: repeatedly find the lowest gap and place
    the widest unpacked item that fits it (ties: taller first).
    Items that fit nowhere raise the gap to the neighbour level.
    """
    hmap = np.zeros(W, dtype=np.int64)
    placements: List[Tuple[int, int, int]] = []
    # unpacked sorted by (width desc, height desc) for gap matching
    unpacked = sorted(dims, key=lambda t: (t[1], t[2]), reverse=True)

    while unpacked:
        # lowest gap: leftmost run of columns at min height
        y0 = int(hmap.min())
        xs = np.where(hmap == y0)[0]
        gx = int(xs[0])
        gw = 1
        while gx + gw < W and hmap[gx + gw] == y0:
            gw += 1
        # widest item fitting the gap
        pick = -1
        for idx, (j, w, h) in enumerate(unpacked):
            if w <= gw:
                pick = idx
                break
        if pick < 0:
            # nothing fits: raise gap to lowest neighbouring level
            left = hmap[gx - 1] if gx > 0 else None
            right = hmap[gx + gw] if gx + gw < W else None
            if left is None and right is None:
                break
            lvl = min(v for v in (left, right) if v is not None)
            hmap[gx:gx + gw] = lvl
            continue
        j, w, h = unpacked.pop(pick)
        placements.append((j, gx, y0))
        hmap[gx:gx + w] = y0 + h
    return hmap, int(hmap.max()) if len(hmap) else 0, placements


_ORDERS = (
    lambda t: (t[2], t[1]),          # height desc
    lambda t: (t[1], t[2]),          # width desc
    lambda t: (t[1] * t[2],),        # area desc
)


def _pack_strip(W: int, dims: List[Tuple[int, int, int]],
                strong: bool = False,
                rng: Optional[random.Random] = None,
                ) -> Tuple[np.ndarray, int, List[Tuple[int, int, int]]]:
    """Pack items (id, w, h) on strip of width W; best of several
    strategies.

    light (default): best-fit + height-desc BLF        (2 packs)
    strong:          + width-desc, area-desc BLF and 2 shuffles
    """
    if not dims:
        return np.zeros(W, dtype=np.int64), 0, []

    best = _pack_bestfit(W, dims)
    cands = [sorted(dims, key=_ORDERS[0], reverse=True)]
    if strong:
        cands.append(sorted(dims, key=_ORDERS[1], reverse=True))
        cands.append(sorted(dims, key=_ORDERS[2], reverse=True))
        if rng is not None:
            for _ in range(2):
                seq = list(dims)
                rng.shuffle(seq)
                cands.append(seq)
    for seq in cands:
        r = _pack_order(W, seq)
        if r[1] < best[1]:
            best = r
    return best


# =====================================================================
#  Mutable solution state
# =====================================================================

class _St:
    """Mutable GMSPP solution used inside the ALNS loop.

    Per-strip numpy height maps allow fast incremental insertion
    evaluation during repair.
    """

    __slots__ = ("inst", "assign", "sitems", "hmaps", "sh", "obj", "plc",
                 "repairs")

    def __init__(self, inst: Instance, assign: List[int],
                 rng: Optional[random.Random] = None):
        self.inst = inst
        self.repairs = 0
        self.assign = list(assign)
        self.sitems: Dict[int, List[int]] = {s.id: [] for s in inst.strips}
        for j, sid in enumerate(self.assign):
            self.sitems[sid].append(j)
        self.hmaps: Dict[int, np.ndarray] = {}
        self.sh: Dict[int, int] = {}
        self.plc: Dict[int, List[Tuple[int, int, int]]] = {}
        for s in inst.strips:
            self._repack(s.id, strong=False, rng=rng)
        self._calc_obj()

    def _strip_dims(self, sid: int) -> List[Tuple[int, int, int]]:
        return [(j, self.inst.w(j, sid), self.inst.h(j, sid))
                for j in self.sitems[sid]]

    def _repack(self, sid: int, strong: bool = False,
                rng: Optional[random.Random] = None):
        W = self.inst.strips[sid].width
        hm, h, p = _pack_strip(W, self._strip_dims(sid), strong=strong,
                               rng=rng)
        self.hmaps[sid] = hm
        self.sh[sid] = h
        self.plc[sid] = p

    def _calc_obj(self):
        self.obj = self.inst.objective_value(
            [self.sh[s.id] for s in self.inst.strips])

    def copy(self):
        new = _St.__new__(_St)
        new.inst = self.inst
        new.assign = list(self.assign)
        new.sitems = {i: list(v) for i, v in self.sitems.items()}
        new.hmaps = {i: v.copy() for i, v in self.hmaps.items()}
        new.sh = dict(self.sh)
        new.plc = {i: list(v) for i, v in self.plc.items()}
        new.obj = self.obj
        new.repairs = self.repairs
        return new

    def check(self, repair: bool = True) -> int:
        """Enforce state invariants; returns the number of violations.

        Verified: (i) sitems is exactly the partition induced by
        assign; (ii) each strip's tracked placements plc pack exactly
        its items feasibly (bounds, no overlap); (iii) sh equals the
        placement-derived height; (iv) obj matches sh.  With
        repair=True, a violated strip is rebuilt from the assignment
        by a strong repack (the packer is ground truth) and the
        violation counted in self.repairs; with repair=False the
        first violation raises.  Called whenever a state may enter
        the elite pool, so no inconsistent state can propagate."""
        bad = 0
        part = sorted(j for it_ in self.sitems.values() for j in it_)
        if (part != list(range(self.inst.n))
                or any(self.assign[j] != sid
                       for sid, it_ in self.sitems.items() for j in it_)):
            if not repair:
                raise RuntimeError("_St.check: assign/sitems mismatch")
            bad += 1
            self.sitems = {s.id: [] for s in self.inst.strips}
            for j, sid in enumerate(self.assign):
                self.sitems[sid].append(j)
            for s in self.inst.strips:
                self._repack(s.id, strong=True)
        else:
            for s in self.inst.strips:
                sid = s.id
                p = self.plc.get(sid, [])
                if not _packing_ok(self.inst, sid, p, self.sitems[sid]):
                    if not repair:
                        raise RuntimeError(
                            f"_St.check: invalid packing on strip {sid}")
                    bad += 1
                    self._repack(sid, strong=True)
                    continue
                h_true = max((y + self.inst.h(j, sid)
                              for j, _, y in p), default=0)
                if self.sh[sid] != h_true:
                    if not repair:
                        raise RuntimeError(
                            f"_St.check: height drift on strip {sid}: "
                            f"tracked {self.sh[sid]}, true {h_true}")
                    bad += 1
                    self.sh[sid] = h_true  # placements valid; height heals
        obj_true = self.inst.objective_value(
            [self.sh[s.id] for s in self.inst.strips])
        if abs(self.obj - obj_true) > 1e-9:
            if not repair and not bad:
                raise RuntimeError("_St.check: objective drift")
            bad += 1 if not bad else 0
            self.obj = obj_true
        elif bad:
            self.obj = obj_true
        self.repairs += bad
        return bad

    # -- surrogate insertion cost -------------------------------------
    def eval_insert(self, j: int, sid: int) -> float:
        """Surrogate objective increase for BF-inserting item j on sid."""
        if not self.inst.fits(j, sid):
            return float("inf")
        s = self.inst.strips[sid]
        w, h = self.inst.dims(j, sid)
        _, by = _best_pos(self.hmaps[sid], w)
        new_h = max(self.sh[sid], by + h)
        dH = new_h - self.sh[sid]
        if self.inst.objective == OBJ_MAKESPAN:
            cur_mk = max(self.sh.values())
            d_mk = max(new_h - cur_mk, 0)
            return d_mk * _MK_SCALE + s.width * dH
        return s.cost * s.width * dH

    def insert(self, j: int, sid: int):
        w, h = self.inst.dims(j, sid)
        bx, by = _best_pos(self.hmaps[sid], w)
        self.assign[j] = sid
        self.sitems[sid].append(j)
        self.hmaps[sid][bx:bx + w] = by + h
        self.sh[sid] = max(self.sh[sid], by + h)
        self.plc[sid].append((j, bx, by))

    def remove(self, ids: List[int]):
        affected = set()
        for j in ids:
            sid = self.assign[j]
            self.sitems[sid].remove(j)
            affected.add(sid)
        for sid in affected:
            self._repack(sid)

    def strong_repack(self, rng: Optional[random.Random] = None):
        """Re-pack every strip with the strong packer, update obj."""
        for s in self.inst.strips:
            self._repack(s.id, strong=True, rng=rng)
        self._calc_obj()

    def placements(self) -> Dict[int, List[Tuple[int, int, int]]]:
        out: Dict[int, List[Tuple[int, int, int]]] = {}
        for s in self.inst.strips:
            _, _, p = _pack_strip(s.width, self._strip_dims(s.id),
                                  strong=True)
            out[s.id] = p
        return out


def _key(st: "_St") -> Tuple[float, float]:
    """Lexicographic solution quality key.  For makespan, ties on the
    max height are broken by total used area (sum W_i * H_i), so moves
    that flatten non-bottleneck strips register as improvements
    (min-max degeneracy fix).  For total_cost the key is the objective
    alone."""
    if st.inst.objective == OBJ_MAKESPAN:
        tot = float(sum(s.width * st.sh[s.id] for s in st.inst.strips))
        return (st.obj, tot)
    return (st.obj, 0.0)


def _improves(ka: Tuple[float, float], kb: Tuple[float, float]) -> bool:
    """True if key *ka* is lexicographically better than *kb*."""
    if ka[0] < kb[0] - 1e-9:
        return True
    if ka[0] < kb[0] + 1e-9 and ka[1] < kb[1] - 1e-9:
        return True
    return False



def _packing_ok(inst: Instance, sid: int,
                plc: List[Tuple[int, int, int]],
                expect: List[int]) -> bool:
    """Verify a per-strip packing: exact item set, in bounds, no
    overlap.  O(k^2), used to guard _finalize candidates."""
    if sorted(j for j, _, _ in plc) != sorted(expect):
        return False
    W = inst.strips[sid].width
    rects = []
    for j, x, y in plc:
        w, h = inst.dims(j, sid)
        if x < 0 or y < 0 or x + w > W:
            return False
        rects.append((x, y, x + w, y + h))
    for a in range(len(rects)):
        x1, y1, x2, y2 = rects[a]
        for b in range(a + 1, len(rects)):
            u1, v1, u2, v2 = rects[b]
            if x1 < u2 and u1 < x2 and y1 < v2 and v1 < y2:
                return False
    return True


def _finalize(st: "_St",
              extra_plc: Optional[Dict[int, List[Tuple[int, int, int]]]]
              = None,
              ) -> Tuple[Dict[int, List[Tuple[int, int, int]]],
                         Dict[int, int], float]:
    """Build the final (solution, strip_heights, objective) triple
    CONSISTENTLY: per strip take the better of a strong heuristic
    repack and an optional externally supplied packing (e.g. CP-SAT
    endgame); heights are recomputed from the chosen placements."""
    inst = st.inst
    solution: Dict[int, List[Tuple[int, int, int]]] = {}
    heights: Dict[int, int] = {}
    for s in inst.strips:
        sid = s.id
        dims = st._strip_dims(sid)
        expect = [j for j, _, _ in dims]
        _, h_heur, p_heur = _pack_strip(s.width, dims, strong=True)
        best_p, best_h = p_heur, h_heur
        # the incremental packing actually held during the search
        p_trk = st.plc.get(sid)
        if (p_trk is not None
                and _packing_ok(inst, sid, p_trk, expect)):
            h_trk = max((y + inst.h(j, sid) for j, _, y in p_trk),
                        default=0)
            if h_trk < best_h:
                best_p, best_h = p_trk, h_trk
        if extra_plc is not None and sid in extra_plc:
            p_ext = extra_plc[sid]
            if _packing_ok(inst, sid, p_ext, expect):
                h_ext = max((y + inst.h(j, sid) for j, _, y in p_ext),
                            default=0)
                if h_ext < best_h:
                    best_p, best_h = p_ext, h_ext
        if not _packing_ok(inst, sid, best_p, expect):
            # the only unguarded candidate is the heuristic repack;
            # fail loudly rather than report an invalid solution
            raise RuntimeError(
                f"_finalize: invalid packing selected on strip {sid} "
                f"(heuristic repack bug?)")
        solution[sid] = best_p
        heights[sid] = best_h
    obj = inst.objective_value([heights[s.id] for s in inst.strips])
    return solution, heights, obj


# =====================================================================
#  Initial solution
# =====================================================================

def _greedy_build(inst: Instance, order: List[int]) -> List[int]:
    """Assignment built by inserting items in *order* at min surrogate."""
    hmaps = {s.id: np.zeros(s.width, dtype=np.int64) for s in inst.strips}
    sh: Dict[int, int] = {s.id: 0 for s in inst.strips}
    assign = [0] * inst.n

    for j in order:
        best_sid, best_c = -1, float("inf")
        for s in inst.strips:
            if not inst.fits(j, s.id):
                continue
            w, h = inst.dims(j, s.id)
            _, by = _best_pos(hmaps[s.id], w)
            new_h = max(sh[s.id], by + h)
            dH = new_h - sh[s.id]
            if inst.objective == OBJ_MAKESPAN:
                cur_mk = max(sh.values())
                c = max(new_h - cur_mk, 0) * _MK_SCALE + s.width * dH
            else:
                c = s.cost * s.width * dH
            if c < best_c:
                best_c, best_sid = c, s.id
        if best_sid < 0:
            best_sid = inst.feasible_strips(j)[-1]
        assign[j] = best_sid
        w, h = inst.dims(j, best_sid)
        bx, by = _best_pos(hmaps[best_sid], w)
        hmaps[best_sid][bx:bx + w] = by + h
        sh[best_sid] = max(sh[best_sid], by + h)
    return assign


def _init_sol(inst: Instance, rng: random.Random) -> _St:
    """Several greedy orderings + random permutations; return best."""
    n = inst.n

    def ref_w(j):    # reference dims for ordering (avg over feasible)
        return max(inst.w(j, i) for i in inst.feasible_strips(j))

    def ref_h(j):
        return max(inst.h(j, i) for i in inst.feasible_strips(j))

    keys = [
        lambda j: ref_w(j) * ref_h(j),
        lambda j: ref_h(j),
        lambda j: ref_w(j),
    ]
    best: Optional[_St] = None
    for key in keys:
        sol = _St(inst, _greedy_build(
            inst, sorted(range(n), key=key, reverse=True)), rng=rng)
        if best is None or sol.obj < best.obj:
            best = sol
    for _ in range(3):
        perm = list(range(n))
        rng.shuffle(perm)
        sol = _St(inst, _greedy_build(inst, perm), rng=rng)
        if sol.obj < best.obj:
            best = sol
    assert best is not None
    return best


# =====================================================================
#  Destroy operators
# =====================================================================

def _d_random(st: _St, deg: int, rng: random.Random) -> List[int]:
    n = len(st.assign)
    return rng.sample(range(n), min(deg, n))


def _d_worst(st: _St, deg: int, rng: random.Random) -> List[int]:
    """Highest objective-contribution proxy, 10% noise."""
    inst = st.inst
    n = len(st.assign)
    mk = inst.objective == OBJ_MAKESPAN
    proxies = []
    for j in range(n):
        sid = st.assign[j]
        s = inst.strips[sid]
        h = inst.h(j, sid)
        p = (h if mk else s.cost * s.width * h)
        proxies.append(p * (1.0 + 0.1 * rng.random()))
    order = sorted(range(n), key=lambda j: proxies[j], reverse=True)
    return order[:deg]


def _d_related(st: _St, deg: int, rng: random.Random) -> List[int]:
    """Shaw removal on reference dims + same-strip bonus."""
    inst, n = st.inst, len(st.assign)
    ref = rng.randint(0, n - 1)
    sr = st.assign[ref]
    wr, hr = inst.w(ref, sr), inst.h(ref, sr)

    def dist(j: int) -> float:
        sj = st.assign[j]
        d = float(abs(inst.w(j, sj) - wr) + abs(inst.h(j, sj) - hr))
        if sj == sr:
            d *= 0.5
        return d + 1e-9 * rng.random()

    return sorted(range(n), key=dist)[:deg]


def _d_strip(st: _St, deg: int, rng: random.Random) -> List[int]:
    """Empty a strip chosen probabilistically by contribution."""
    inst = st.inst
    mk = inst.objective == OBJ_MAKESPAN
    wts = [(st.sh.get(s.id, 0) if mk
            else s.cost * s.width * st.sh.get(s.id, 0)) + 1e-6
           for s in inst.strips]
    total = sum(wts)
    r = rng.random() * total
    c = 0.0
    chosen = inst.strips[-1].id
    for s, w in zip(inst.strips, wts):
        c += w
        if c >= r:
            chosen = s.id
            break
    removed = list(st.sitems.get(chosen, []))
    return removed if removed else _d_random(st, deg, rng)


# =====================================================================
#  Repair operators
# =====================================================================

def _r_greedy(st: _St, removed: List[int], rng: random.Random) -> None:
    inst = st.inst
    order = sorted(removed,
                   key=lambda j: inst.w(j, st.inst.feasible_strips(j)[0])
                   * inst.h(j, st.inst.feasible_strips(j)[0]),
                   reverse=True)
    for j in order:
        best_sid, best_c = -1, float("inf")
        for i in inst.feasible_strips(j):
            c = st.eval_insert(j, i)
            if c < best_c:
                best_c, best_sid = c, i
        if best_sid < 0:
            best_sid = inst.feasible_strips(j)[-1]
        st.insert(j, best_sid)


def _r_regret(st: _St, removed: List[int], rng: random.Random) -> None:
    inst = st.inst
    unplaced = set(removed)
    while unplaced:
        best_j, best_reg = -1, -1.0
        best_sid, best_c1 = -1, float("inf")
        for j in unplaced:
            costs = []
            for i in inst.feasible_strips(j):
                c = st.eval_insert(j, i)
                if c < float("inf"):
                    costs.append((c, i))
            if not costs:
                continue
            costs.sort()
            c1 = costs[0][0]
            c2 = costs[1][0] if len(costs) > 1 else c1
            reg = c2 - c1
            if reg > best_reg or (reg == best_reg and c1 < best_c1):
                best_reg, best_j = reg, j
                best_sid, best_c1 = costs[0][1], c1
        if best_j < 0:
            best_j = next(iter(unplaced))
            best_sid = inst.feasible_strips(best_j)[-1]
        st.insert(best_j, best_sid)
        unplaced.discard(best_j)


def _r_random(st: _St, removed: List[int], rng: random.Random) -> None:
    for j in removed:
        st.insert(j, rng.choice(st.inst.feasible_strips(j)))


def _r_spread(st: _St, removed: List[int], rng: random.Random) -> None:
    inst = st.inst
    order = sorted(removed,
                   key=lambda j: max(inst.area(j, i)
                                     for i in inst.feasible_strips(j)),
                   reverse=True)
    m = inst.m
    cycle = list(range(m))
    rng.shuffle(cycle)
    idx = 0
    for j in order:
        feas = set(inst.feasible_strips(j))
        for _ in range(m):
            sid = cycle[idx % m]
            idx += 1
            if sid in feas:
                st.insert(j, sid)
                break
        else:
            st.insert(j, inst.feasible_strips(j)[-1])


# =====================================================================
#  Phase 2: local search (relocation + swap)
# =====================================================================

def _local_search(st: _St, time_budget: float,
                  rng: random.Random) -> int:
    """First-improvement relocation + pairwise swap.  Returns number of
    improving moves applied.  Modifies *st* in place."""
    inst = st.inst
    t0 = _time.time()
    n_moves = 0
    improved = True

    def _adopt(trial):
        st.assign = trial.assign
        st.sitems = trial.sitems
        st.hmaps = trial.hmaps
        st.sh = trial.sh
        st.obj = trial.obj
        st.plc = trial.plc      # placements move WITH the state; this
                                # omission was the root of the stale-
                                # packing (negative-gap) bug class

    while improved and (_time.time() - t0) < time_budget:
        improved = False
        # -- single-item relocation --
        for j in range(inst.n):
            if (_time.time() - t0) > time_budget:
                break
            a = st.assign[j]
            for b in inst.feasible_strips(j):
                if b == a:
                    continue
                trial = st.copy()
                trial.sitems[a].remove(j)
                trial.sitems[b].append(j)
                trial.assign[j] = b
                trial._repack(a)
                trial._repack(b)
                trial._calc_obj()
                if _improves(_key(trial), _key(st)):
                    _adopt(trial)
                    improved = True
                    n_moves += 1
                    break
        # -- pairwise swap (sampled) --
        pairs = [(j, k) for j in range(inst.n) for k in range(j + 1, inst.n)
                 if st.assign[j] != st.assign[k]]
        rng.shuffle(pairs)
        for j, k in pairs[:200]:
            if (_time.time() - t0) > time_budget:
                break
            a, b = st.assign[j], st.assign[k]
            if not (inst.fits(j, b) and inst.fits(k, a)):
                continue
            trial = st.copy()
            trial.sitems[a].remove(j)
            trial.sitems[b].remove(k)
            trial.sitems[b].append(j)
            trial.sitems[a].append(k)
            trial.assign[j], trial.assign[k] = b, a
            trial._repack(a)
            trial._repack(b)
            trial._calc_obj()
            if _improves(_key(trial), _key(st)):
                _adopt(trial)
                improved = True
                n_moves += 1
    return n_moves


# =====================================================================
#  Phase 3: CP-SAT endgame (exact per-strip repack)
# =====================================================================

def _cpsat_pack_strip(dims: List[Tuple[int, int, int]], W: int,
                      ub: int, time_limit: float, threads: int = 4,
                      ) -> Optional[Tuple[int, List[Tuple[int, int, int]]]]:
    """Exact min-height packing of (id, w, h) on width W, given upper
    bound *ub*.  Returns (height, placements) or None on failure."""
    try:
        from ortools.sat.python import cp_model
    except ImportError:
        return None
    if not dims:
        return 0, []
    area_lb = -(-sum(w * h for _, w, h in dims) // W)
    lb = max(area_lb, max(h for _, _, h in dims))
    if lb >= ub:                                    # already optimal
        return None
    mdl = cp_model.CpModel()
    H = mdl.NewIntVar(lb, ub, 'H')
    xs, ivx, ivy = [], [], []
    for k, (j, w, h) in enumerate(dims):
        x = mdl.NewIntVar(0, W - w, f'x{k}')
        y = mdl.NewIntVar(0, ub - h, f'y{k}')
        xs.append((x, y))
        ivx.append(mdl.NewFixedSizeIntervalVar(x, w, f'ix{k}'))
        ivy.append(mdl.NewFixedSizeIntervalVar(y, h, f'iy{k}'))
        mdl.Add(y + h <= H)
    mdl.AddNoOverlap2D(ivx, ivy)
    mdl.Minimize(H)
    sv = cp_model.CpSolver()
    sv.parameters.max_time_in_seconds = max(time_limit, 0.5)
    sv.parameters.num_workers = threads
    st = sv.Solve(mdl)
    if st not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None
    hgt = int(sv.Value(H))
    if hgt >= ub:
        return None
    plc = [(j, int(sv.Value(x)), int(sv.Value(y)))
           for (x, y), (j, w, h) in zip(xs, dims)]
    return hgt, plc


def _endgame(inst: Instance, st: _St, time_budget: float,
             threads: int = 4, verbose: bool = False,
             ) -> Tuple[int, Dict[int, List[Tuple[int, int, int]]]]:
    """Freeze assignment; CP-SAT-repack strips in order of potential
    gain.  Updates *st* heights/objective and returns
    (#improved strips, {strip_id: placements})."""
    t0 = _time.time()
    mk_mode = inst.objective == OBJ_MAKESPAN

    def strip_lb(sid: int) -> int:
        dims = st._strip_dims(sid)
        if not dims:
            return 0
        area_lb = -(-sum(w * h for _, w, h in dims)
                    // inst.strips[sid].width)
        return max(area_lb, max(h for _, _, h in dims))

    n_improved = 0
    plc_out: Dict[int, List[Tuple[int, int, int]]] = {}
    attempted = set()

    while True:
        remaining = time_budget - (_time.time() - t0)
        if remaining < 0.5:
            break
        # Candidates: strips with provable slack, not yet attempted.
        cands = [s.id for s in inst.strips
                 if s.id not in attempted
                 and st.sitems[s.id]
                 and st.sh[s.id] > strip_lb(s.id)]
        if not cands:
            break
        if mk_mode:
            # bottleneck first: only the tallest strip moves the
            # makespan; re-evaluated each round as heights change
            sid = max(cands, key=lambda i: st.sh[i])
        else:
            sid = max(cands, key=lambda i: st.sh[i] - strip_lb(i))
        attempted.add(sid)

        res = _cpsat_pack_strip(st._strip_dims(sid),
                                inst.strips[sid].width,
                                ub=st.sh[sid], time_limit=remaining,
                                threads=threads)
        if res is not None:
            new_h, plc = res
            if verbose:
                print(f"[endgame] strip {sid}: {st.sh[sid]} -> {new_h}")
            hm = np.zeros(inst.strips[sid].width, dtype=np.int64)
            for j, x, y in plc:
                w, h = inst.dims(j, sid)
                hm[x:x + w] = np.maximum(hm[x:x + w], y + h)
            st.hmaps[sid] = hm
            st.sh[sid] = new_h
            plc_out[sid] = plc
            n_improved += 1
    if n_improved:
        st._calc_obj()
    return n_improved, plc_out


# =====================================================================
#  Result container
# =====================================================================

@dataclass
class ALNSResult:
    objective: float
    lower_bound: float
    gap_pct: float
    solution: Dict[int, List[Tuple[int, int, int]]]
    strip_heights: Dict[int, int]
    total_time: float
    n_iterations: int
    n_improvements: int
    initial_objective: float
    history: List[float]
    # Phase gains
    obj_after_alns: float = 0.0
    ls_moves: int = 0
    endgame_strips: int = 0
    # internal state repairs (invariant violations healed during the
    # search; 0 in a healthy run -- nonzero values flag solver drift)
    state_repairs: int = 0
    # Parallel multi-start stats (filled by solve_alns_parallel)
    n_runs: int = 1
    obj_mean: float = 0.0
    obj_std: float = 0.0
    obj_all: Optional[List[float]] = field(default=None, repr=False)


# =====================================================================
#  Main ALNS
# =====================================================================

def solve_alns(
    instance: Instance,
    *,
    time_limit: float = 300.0,
    max_iter: int = 10_000_000,   # safety cap only; stopping is by time
    min_destroy: float = 0.10,
    max_destroy: float = 0.40,
    sa_start: float = 0.05,
    sa_end: float = 0.0001,
    segment: int = 100,
    compute_dual: bool = False,
    dual_time_limit: float = 60.0,
    use_local_search: bool = True,
    ls_time: float = 10.0,
    use_endgame: bool = True,
    endgame_time: float = 30.0,
    endgame_threads: int = 4,
    seed: Optional[int] = None,
    verbose: bool = True,
) -> ALNSResult:
    """Solve GMSPP with the three-phase ALNS matheuristic.

    *time_limit* covers all phases (LP-PC + ALNS + LS + endgame).
    """
    t0 = _time.time()
    rng = random.Random(seed)
    n = instance.n

    # -- Optional dual bound ------------------------------------------
    lb = 0.0
    if compute_dual:
        try:
            from .formulation_normal import solve_normal_lp
            if verbose:
                print("[ALNS] Computing LP-PC dual bound ...")
            lp_budget = min(dual_time_limit, time_limit * 0.2)
            res = solve_normal_lp(instance, time_limit=lp_budget)
            lb = res["lp_bound"]
            if verbose:
                print(f"[ALNS] LP-PC = {lb:.2f}  ({res['solve_time']:.1f}s)")
        except Exception as exc:
            if verbose:
                print(f"[ALNS] LP-PC failed: {exc}")

    # Phase budgets (LS/endgame carved out of the tail)
    ls_budget = ls_time if use_local_search else 0.0
    eg_budget = endgame_time if use_endgame else 0.0
    alns_deadline = t0 + max(time_limit - ls_budget - eg_budget, 1.0)

    # -- Initial solution ---------------------------------------------
    if verbose:
        print("[ALNS] Building initial solution ...")
    cur = _init_sol(instance, rng)
    best = cur.copy()
    init_obj = cur.obj
    if verbose:
        print(f"[ALNS] Initial obj = {init_obj:.2f}   "
              f"heights = {dict(cur.sh)}")

    # -- SA schedule (geometric cooling in TIME, so the schedule
    #    adapts to any time_limit; max_iter is only a safety cap) ----
    T0 = max(sa_start * init_obj, 1e-6)
    Tf = max(sa_end * init_obj, 1e-9)
    T = T0
    sa_t0 = _time.time()
    sa_span = max(alns_deadline - sa_t0, 1e-9)

    D_ops = [_d_random, _d_worst, _d_related, _d_strip]
    R_ops = [_r_greedy, _r_regret, _r_random, _r_spread]
    nd, nr = len(D_ops), len(R_ops)
    dw = [1.0] * nd;  rw = [1.0] * nr
    ds = [0.0] * nd;  rs = [0.0] * nr
    dc = [0] * nd;    rc = [0] * nr

    history: List[float] = [init_obj]
    n_imp = 0
    it = 0

    # -- Phase 1: ALNS loop -------------------------------------------
    while it < max_iter and _time.time() < alns_deadline:
        it += 1

        di = rng.choices(range(nd), weights=dw)[0]
        ri = rng.choices(range(nr), weights=rw)[0]
        dc[di] += 1
        rc[ri] += 1

        deg = max(1, int(n * rng.uniform(min_destroy, max_destroy)))
        cand = cur.copy()
        removed = D_ops[di](cand, deg, rng)
        cand.remove(removed)
        R_ops[ri](cand, removed, rng)
        cand._calc_obj()

        score = 0.0
        if _improves(_key(cand), _key(best)):
            # gate to the elite pool: enforce state invariants first
            # (drifted strips are rebuilt; the packer is ground truth)
            cand.check(repair=True)
            inc = cand.copy()          # verified incremental packing
            cand.strong_repack(rng)    # polishing repack
            winner = cand if _improves(_key(cand), _key(inc)) else inc
            if _improves(_key(winner), _key(best)):
                best = winner.copy()
                n_imp += 1
                score = 33.0
                history.append(best.obj)
        elif _improves(_key(cand), _key(cur)):
            score = 9.0
        elif T > 0 and rng.random() < math.exp(
                min((cur.obj - cand.obj) / T, 0.0)):
            score = 3.0

        if score > 0:
            cur = cand

        ds[di] += score
        rs[ri] += score
        frac = min(1.0, (_time.time() - sa_t0) / sa_span)
        T = T0 * (Tf / T0) ** frac

        if it % segment == 0:
            for k in range(nd):
                if dc[k]:
                    dw[k] = 0.8 * dw[k] + 0.2 * (ds[k] / dc[k])
                    dw[k] = max(dw[k], 0.05)
            for k in range(nr):
                if rc[k]:
                    rw[k] = 0.8 * rw[k] + 0.2 * (rs[k] / rc[k])
                    rw[k] = max(rw[k], 0.05)
            ds = [0.0] * nd;  rs = [0.0] * nr
            dc = [0] * nd;    rc = [0] * nr

    obj_after_alns = best.obj

    # -- Phase 2: local search ----------------------------------------
    ls_moves = 0
    if use_local_search and _time.time() - t0 < time_limit:
        remaining = min(ls_budget + max(alns_deadline - _time.time(), 0),
                        time_limit - (_time.time() - t0) - eg_budget)
        if remaining > 0.2:
            ls_moves = _local_search(best, remaining, rng)
            if verbose and ls_moves:
                print(f"[ALNS] Local search: {ls_moves} moves, "
                      f"obj {obj_after_alns:.2f} -> {best.obj:.2f}")

    # phase boundary: re-assert invariants before the exact endgame
    best.check(repair=True)

    # -- Phase 3: CP-SAT endgame --------------------------------------
    endgame_strips = 0
    eg_plc: Dict[int, List[Tuple[int, int, int]]] = {}
    if use_endgame and _time.time() - t0 < time_limit:
        remaining = min(eg_budget, time_limit - (_time.time() - t0))
        if remaining > 0.5:
            endgame_strips, eg_plc = _endgame(
                instance, best, remaining,
                threads=endgame_threads, verbose=verbose)

    # -- Consistent final solution ------------------------------------
    solution, heights, obj_final = _finalize(best, eg_plc)

    total = _time.time() - t0
    gap = ((obj_final - lb) / max(obj_final, 1e-10) * 100
           if lb > 0 else float("nan"))

    if verbose:
        print(f"[ALNS] Done: obj={obj_final:.2f} "
              f"(alns {obj_after_alns:.2f}, ls {ls_moves} moves, "
              f"endgame {endgame_strips} strips)  "
              f"iters={it}  time={total:.1f}s")

    return ALNSResult(
        objective=obj_final,
        lower_bound=lb,
        gap_pct=gap,
        solution=solution,
        strip_heights=heights,
        total_time=total,
        n_iterations=it,
        n_improvements=n_imp,
        initial_objective=init_obj,
        history=history,
        obj_after_alns=obj_after_alns,
        ls_moves=ls_moves,
        endgame_strips=endgame_strips,
        state_repairs=best.repairs,
    )


# =====================================================================
#  Parallel multi-start
# =====================================================================

def _alns_worker(args):
    instance, seed, time_limit, kwargs = args
    kwargs = {k: v for k, v in kwargs.items()
              if k not in ('time_limit', 'compute_dual', 'seed',
                           'verbose', 'use_endgame', 'endgame_time')}
    return solve_alns(instance, time_limit=time_limit, compute_dual=False,
                      seed=seed, verbose=False, use_endgame=False,
                      **kwargs)


def solve_alns_parallel(
    instance: Instance,
    *,
    n_runs: int = 5,
    n_workers: Optional[int] = None,
    time_limit: float = 300.0,
    compute_dual: bool = False,
    dual_time_limit: float = 60.0,
    endgame_time: float = 30.0,
    seeds: Optional[List[int]] = None,
    verbose: bool = True,
    **alns_kwargs,
) -> ALNSResult:
    """Parallel multi-start ALNS: *n_runs* independent runs on separate
    cores, then ONE CP-SAT endgame on the overall best solution.

    LP-PC (optional) and the endgame run in the main process; workers
    run ALNS + local search only.
    """
    t0 = _time.time()

    if seeds is not None:
        n_runs = len(seeds)
    else:
        _r = random.Random(42)
        seeds = [_r.randint(0, 2 ** 31) for _ in range(n_runs)]

    if n_workers is None:
        n_workers = min(n_runs, _mp.cpu_count())

    # -- LP-PC once in main process -----------------------------------
    lb = 0.0
    if compute_dual:
        try:
            from .formulation_normal import solve_normal_lp
            if verbose:
                print("[ALNS-P] Computing LP-PC dual bound ...")
            lp_budget = min(dual_time_limit, time_limit * 0.2)
            res = solve_normal_lp(instance, time_limit=lp_budget)
            lb = res["lp_bound"]
            if verbose:
                print(f"[ALNS-P] LP-PC = {lb:.2f} ({res['solve_time']:.1f}s)")
        except Exception as exc:
            if verbose:
                print(f"[ALNS-P] LP-PC failed: {exc}")

    worker_time = max(time_limit - (_time.time() - t0) - endgame_time, 1.0)

    if verbose:
        print(f"[ALNS-P] {n_runs} runs on {n_workers} workers "
              f"({worker_time:.0f}s each) ...")

    worker_args = [(instance, seed, worker_time, alns_kwargs)
                   for seed in seeds]

    if n_workers <= 1 or n_runs <= 1:
        results = [_alns_worker(a) for a in worker_args]
    else:
        with _mp.Pool(n_workers) as pool:
            results = pool.map(_alns_worker, worker_args)

    objs = [r.objective for r in results]
    best = min(results, key=lambda r: r.objective)

    # -- Endgame on top-K distinct assignments ------------------------
    # The best-of-runs assignment may already be optimally packed while
    # a slightly worse assignment repacks strictly better; spreading
    # the endgame over distinct candidates recovers such cases.
    endgame_strips = best.endgame_strips
    obj_final = best.objective
    strip_heights = dict(best.strip_heights)
    solution = best.solution
    remaining = time_limit - (_time.time() - t0)
    if endgame_time > 0.5 and remaining > 0.5:
        K = 3
        cands, seen = [], set()
        for r in sorted(results, key=lambda r: r.objective):
            sig = tuple(sorted((sid, tuple(sorted(j for j, _, _ in plc)))
                               for sid, plc in r.solution.items()))
            if sig not in seen:
                seen.add(sig)
                cands.append(r)
            if len(cands) >= K:
                break
        share = min(endgame_time, remaining) / max(len(cands), 1)
        for r in cands:
            assign = [0] * instance.n
            for sid, plc in r.solution.items():
                for j, _, _ in plc:
                    assign[j] = sid
            st = _St(instance, assign)
            for sid, h in r.strip_heights.items():
                st.sh[sid] = min(st.sh[sid], h)
            st._calc_obj()
            eg, eg_plc = _endgame(instance, st, share,
                                  threads=_mp.cpu_count(), verbose=verbose)
            for sid, plc in r.solution.items():
                eg_plc.setdefault(sid, plc)
            sol2, hts2, obj2 = _finalize(st, eg_plc)
            if obj2 < obj_final:
                endgame_strips += eg
                obj_final = obj2
                strip_heights = hts2
                solution = sol2

    total = _time.time() - t0
    gap = ((obj_final - lb) / max(obj_final, 1e-10) * 100
           if lb > 0 else float("nan"))

    out = ALNSResult(
        objective=obj_final,
        lower_bound=lb,
        gap_pct=gap,
        solution=solution,
        strip_heights=strip_heights,
        total_time=total,
        n_iterations=best.n_iterations,
        n_improvements=best.n_improvements,
        initial_objective=best.initial_objective,
        history=best.history,
        obj_after_alns=best.obj_after_alns,
        ls_moves=best.ls_moves,
        endgame_strips=endgame_strips,
        state_repairs=sum(r.state_repairs for r in results),
        n_runs=n_runs,
        obj_mean=float(np.mean(objs)),
        obj_std=float(np.std(objs)),
        obj_all=objs,
    )

    if verbose:
        print(f"\n{'=' * 55}")
        print(f"[ALNS-P] {n_runs} runs in {total:.1f}s")
        print(f"[ALNS-P] Best={obj_final:.2f}  mean={out.obj_mean:.2f}  "
              f"std={out.obj_std:.2f}")
        if lb > 0:
            print(f"[ALNS-P] LB={lb:.2f}  gap={gap:.2f}%")
    return out
