"""
GMSPP: Generalized Multiple Strip Packing Problem solver.

Exact methods for the cost-weighted GMSPP:
  min sum_i C_i * W_i * H_i

Two exact IP formulations:
  1. Big-M formulation (adapted from Vasilyev et al. 2023)
  2. Normal-position formulation (extending Cote et al. 2014)

BendM (Benders' Method for Multiple strips):
  Benders' decomposition on the normal-position formulation with
  lifted combinatorial cuts via MIS computation and LP-based lifting.
"""

from .data_structures import Item, Strip, Instance
from .instance_generator import (
    generate_type1_instance,
    generate_type2_instance,
    generate_spp_items_MV,
    compute_strip_costs,
)
from .formulation_bigm import solve_bigm_lp, solve_bigm_mip, solve_bigm_mip_le
from .formulation_normal import solve_normal_lp, solve_normal_mip
from .ycheck import (
    YCheckItem, y_check, find_minimal_infeasible_subset,
    compute_lifted_intervals, compute_lifted_intervals_lp,
    run_ycheck_and_cuts,
)
from .benders_solver import solve_benders, BendersResult
from .alns_solver import solve_alns, solve_alns_parallel, ALNSResult
from .dataset import (
    generate_dataset, save_dataset, load_dataset,
    generate_and_save, get_or_create_dataset,
)
from .benchmark_loader import (
    parse_ins2d, load_benchmark_set,
    spp_to_gmspp, load_and_convert_benchmark,
)
from .vasilyev import (
    spp_split_to_gmspp, load_zdf_gmspp,
    generate_type1, generate_type2, mmsp_lower_bound,
)
from .cpsat_solver import solve_cpsat
