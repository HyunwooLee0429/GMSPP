"""
Data structures for GMSPP instances (v2).

Two modeling traditions are supported by a single Instance class:

  Cutting & packing view (fixed dims):
      Item j has fixed (width, height); strips differ in width W_i and
      optionally in per-unit-area cost C_i.  This is the classical
      setting: "multiple strip packing with heterogeneous widths".

  Scheduling view (strip-dependent dims):
      Item j has dimensions (w_ij, h_ij) depending on the strip i it is
      assigned to (Vasilyev et al., 2023).  Strips model heterogeneous
      computing nodes / quays.  Fixed dims are the special case
      w_ij = w_j, h_ij = h_j.

Two objectives:
      'total_cost' : min sum_i C_i * W_i * H_i   (trim loss when C_i = 1)
      'makespan'   : min max_i H_i

Dimension access inside solvers MUST go through Instance.w(j, i) /
Instance.h(j, i) so both settings are handled transparently.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple
import numpy as np

OBJ_TOTAL_COST = 'total_cost'
OBJ_MAKESPAN = 'makespan'
_VALID_OBJECTIVES = (OBJ_TOTAL_COST, OBJ_MAKESPAN)


@dataclass(frozen=True)
class Item:
    """A rectangular item.

    ``width`` / ``height`` are the *reference* dimensions.  For
    fixed-dimension instances they are THE dimensions on every strip.
    For strip-dependent instances the per-strip dimensions live in
    ``Instance.w_mat`` / ``Instance.h_mat`` and the reference dims are
    only used for reporting.
    """
    id: int
    width: int
    height: int


@dataclass(frozen=True)
class Strip:
    """A strip with a given width, per-unit-area cost, and infinite height."""
    id: int
    width: int
    cost: float = 1.0  # per-unit-area cost C_i (1.0 -> total-area objective)


@dataclass
class Instance:
    """
    A GMSPP instance: pack items into heterogeneous strips.

    Parameters
    ----------
    items : list of Item
    strips : list of Strip
        Re-sorted by non-decreasing width and re-indexed 0..m-1.
    objective : str
        'total_cost' (min sum C_i*W_i*H_i) or 'makespan' (min max H_i).
    w_mat, h_mat : (n, m) int arrays, optional
        Strip-dependent dimensions: w_mat[j, i] = width of item j on
        strip i.  Columns must be given in the SAME order as the
        ``strips`` argument; they are permuted together with the strips
        during sorting.  An item is infeasible on strip i iff
        w_mat[j, i] > W_i (use e.g. W_i + 1 to forbid a pair).
        If omitted, dims are fixed: w_ij = items[j].width.
    """
    items: List[Item]
    strips: List[Strip]
    objective: str = OBJ_TOTAL_COST
    w_mat: Optional[np.ndarray] = None
    h_mat: Optional[np.ndarray] = None
    n: int = field(default=0)
    m: int = field(default=0)

    # ------------------------------------------------------------------
    def __post_init__(self):
        if self.objective not in _VALID_OBJECTIVES:
            raise ValueError(
                f"objective must be one of {_VALID_OBJECTIVES}, "
                f"got {self.objective!r}")

        self.n = len(self.items)
        self.m = len(self.strips)

        # Sort strips by non-decreasing width, remember the permutation
        # so dimension-matrix columns stay aligned.
        order = sorted(range(self.m), key=lambda k: self.strips[k].width)
        self.strips = [
            Strip(id=i, width=self.strips[k].width, cost=self.strips[k].cost)
            for i, k in enumerate(order)
        ]
        if self.w_mat is not None:
            self.w_mat = np.asarray(self.w_mat, dtype=np.int64)[:, order]
        if self.h_mat is not None:
            self.h_mat = np.asarray(self.h_mat, dtype=np.int64)[:, order]

        # Validate matrices
        for name, mat in (('w_mat', self.w_mat), ('h_mat', self.h_mat)):
            if mat is not None and mat.shape != (self.n, self.m):
                raise ValueError(
                    f"{name} has shape {mat.shape}, "
                    f"expected ({self.n}, {self.m})")
        if (self.w_mat is None) != (self.h_mat is None):
            raise ValueError("w_mat and h_mat must be given together")

        # Every item must fit on at least one strip.
        for j in range(self.n):
            if not self.feasible_strips(j):
                raise ValueError(f"item {j} fits on no strip")

    # ------------------------------------------------------------------
    #  Dimension access (THE interface all solvers must use)
    # ------------------------------------------------------------------
    @property
    def strip_dependent(self) -> bool:
        """True if item dimensions depend on the strip."""
        return self.w_mat is not None

    def w(self, j: int, i: int) -> int:
        """Width of item j on strip i."""
        if self.w_mat is not None:
            return int(self.w_mat[j, i])
        return self.items[j].width

    def h(self, j: int, i: int) -> int:
        """Height of item j on strip i."""
        if self.h_mat is not None:
            return int(self.h_mat[j, i])
        return self.items[j].height

    def dims(self, j: int, i: int) -> Tuple[int, int]:
        """(width, height) of item j on strip i."""
        return self.w(j, i), self.h(j, i)

    def area(self, j: int, i: int) -> int:
        """Area of item j when placed on strip i."""
        return self.w(j, i) * self.h(j, i)

    def fits(self, j: int, i: int) -> bool:
        """True if item j fits on strip i (w_ij <= W_i)."""
        return self.w(j, i) <= self.strips[i].width

    def feasible_strips(self, item_id: int) -> List[int]:
        """Strip IDs that can hold item j."""
        return [s.id for s in self.strips if self.fits(item_id, s.id)]

    def items_fitting_strip(self, strip_id: int) -> List[int]:
        """Item IDs that fit on strip i."""
        return [j for j in range(self.n) if self.fits(j, strip_id)]

    # ------------------------------------------------------------------
    #  Objective evaluation (central definition used by every solver)
    # ------------------------------------------------------------------
    def objective_value(self, strip_heights: Sequence[float]) -> float:
        """Objective for given per-strip heights H_i (indexed by strip id)."""
        if self.objective == OBJ_MAKESPAN:
            return max(strip_heights)
        return sum(s.cost * s.width * strip_heights[s.id]
                   for s in self.strips)

    # ------------------------------------------------------------------
    #  Convenience arrays (fixed-dims reference values)
    # ------------------------------------------------------------------
    def widths(self) -> np.ndarray:
        """Reference item widths (fixed-dims value; reporting only)."""
        return np.array([it.width for it in self.items])

    def heights(self) -> np.ndarray:
        """Reference item heights (fixed-dims value; reporting only)."""
        return np.array([it.height for it in self.items])

    def strip_widths(self) -> List[int]:
        return [s.width for s in self.strips]

    def strip_costs(self) -> List[float]:
        return [s.cost for s in self.strips]

    # ------------------------------------------------------------------
    #  Simple lower bounds (valid for both settings)
    # ------------------------------------------------------------------
    def min_item_area(self, j: int) -> int:
        """Smallest possible area of item j over its feasible strips."""
        return min(self.area(j, i) for i in self.feasible_strips(j))

    def min_item_height(self, j: int) -> int:
        """Smallest possible height of item j over its feasible strips."""
        return min(self.h(j, i) for i in self.feasible_strips(j))

    def total_area(self) -> int:
        """Total item area; for strip-dependent dims, the best case
        (each item on its area-minimizing strip)."""
        return sum(self.min_item_area(j) for j in range(self.n))

    def simple_lower_bound(self) -> float:
        """Valid lower bound on the configured objective."""
        if self.objective == OBJ_MAKESPAN:
            return self._makespan_lb()
        return self._total_cost_lb()

    def _total_cost_lb(self) -> float:
        # Cost of item j on strip i is at least C_i * w_ij * h_ij
        # (it occupies that much weighted area).  Sum of per-item minima.
        area_lb = sum(
            min(self.strips[i].cost * self.area(j, i)
                for i in self.feasible_strips(j))
            for j in range(self.n)
        )
        # At least one strip reaches the height of the tallest
        # mandatory item; cheapest way to pay for any height h is
        # min_i C_i * W_i * h over strips that item fits on.
        height_lb = max(
            min(self.strips[i].cost * self.strips[i].width * self.h(j, i)
                for i in self.feasible_strips(j))
            for j in range(self.n)
        )
        return float(max(area_lb, height_lb))

    def _makespan_lb(self) -> float:
        # (1) every item forces height >= min over feasible strips of h_ij
        h_lb = max(self.min_item_height(j) for j in range(self.n))
        # (2) work conservation: sum of minimal item areas spread over
        #     the total width of all strips
        w_total = sum(s.width for s in self.strips)
        area_lb = self.total_area() / w_total
        return float(max(h_lb, area_lb))

    # ------------------------------------------------------------------
    def identical_strip_groups(self) -> List[List[int]]:
        """Groups of >= 2 mutually interchangeable strips (same width,
        same cost, and -- for strip-dependent instances -- identical
        dimension columns).  Used for symmetry breaking: any solution
        can be permuted so heights are non-increasing within a group,
        so constraints H_a >= H_b (a before b in group) are valid.
        """
        groups: List[List[int]] = []
        used = set()
        for a in range(self.m):
            if a in used:
                continue
            grp = [a]
            for b in range(a + 1, self.m):
                if b in used:
                    continue
                sa, sb = self.strips[a], self.strips[b]
                if sa.width != sb.width or sa.cost != sb.cost:
                    continue
                if self.strip_dependent and not (
                    np.array_equal(self.w_mat[:, a], self.w_mat[:, b])
                    and np.array_equal(self.h_mat[:, a], self.h_mat[:, b])
                ):
                    continue
                grp.append(b)
                used.add(b)
            if len(grp) >= 2:
                groups.append(grp)
        return groups

    # ------------------------------------------------------------------
    def describe(self) -> str:
        kind = ("strip-dependent dims" if self.strip_dependent
                else "fixed dims")
        lines = [
            f"GMSPP Instance: n={self.n} items, m={self.m} strips "
            f"[{kind}, objective={self.objective}]",
            f"  Strips: widths={self.strip_widths()}, "
            f"costs={self.strip_costs()}",
            f"  Total (min) item area: {self.total_area()}",
            f"  Simple LB: {self.simple_lower_bound():.1f}",
        ]
        if not self.strip_dependent:
            w, h = self.widths(), self.heights()
            lines.insert(1, f"  Items: w in [{w.min()}, {w.max()}], "
                            f"h in [{h.min()}, {h.max()}]")
        return '\n'.join(lines)
