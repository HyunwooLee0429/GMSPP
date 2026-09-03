"""Generate the online supplement (per-instance tables) from the
results CSVs.  Output: paper_v2/supplement.tex (standalone).

    python gen_supplement.py && cd paper_v2 && pdflatex supplement
"""
import numpy as np
import pandas as pd

HEAD = r"""\documentclass[10pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage[T1]{fontenc}
\usepackage{times}
\usepackage{booktabs,longtable}
\usepackage{caption}
\title{Online Supplement:\\ Per-Instance Results for
``Exact and Matheuristic Methods for the\\ Generalized Multiple
Strip Packing Problem''}
\date{}
\begin{document}
\maketitle
\noindent This supplement reports per-instance results underlying
every summary table of the main paper. All values are produced by
the released experiment drivers from the released instances;
column conventions follow the main paper.
"""

def esc(k):
    return k.replace('_', r'\_')

out = [HEAD]

# ---------- S1: Tier 1 per-instance bounds ----------
df = pd.read_csv('results/v2_relax.csv')
df = df[~df['key'].str.startswith('N_')].sort_values('key')
out.append(r"""
\section*{S1. Tier 1: relaxation values per instance}
\begin{longtable}{lrrrrrrr}
\caption{Bound values on every makespan instance (cf.\ Table~1 of
the main paper). Missing entries: not computed (size gates).}\\
\toprule
Instance & $n$ & $m$ & big-$M$ LP & liq.\ LP & $Z^{\mathrm{ms}}$ &
LP-PC & $h^*$\\
\midrule\endfirsthead
\toprule
Instance & $n$ & $m$ & big-$M$ LP & liq.\ LP & $Z^{\mathrm{ms}}$ &
LP-PC & $h^*$\\
\midrule\endhead
\bottomrule\endlastfoot
""")
def v(x, fmt='{:.2f}'):
    return '--' if x != x or x <= 0 else fmt.format(x)
for _, r in df.iterrows():
    out.append(f"{esc(r['key'])} & {int(r['n'])} & {int(r['m'])} & "
               f"{v(r['bigm_lp'])} & {v(r['mmsp_lp'])} & "
               f"{v(r['mmsp_lb'])} & {v(r['lp_pc'])} & "
               f"{v(r['simple_lb'])} \\\\\n")
out.append("\\end{longtable}\n")

# ---------- S2: Tier 2 per-instance certified gaps ----------
db = pd.read_csv('results/v2_alns.csv').sort_values('key')
db['lb_int'] = np.ceil(db['lb'] - 1e-9)
db['gap'] = (db['obj'] - db['lb_int']) / db['obj'].clip(lower=1e-9) * 100
db['cv'] = db['obj_std'] / db['obj_mean'].clip(lower=1e-9) * 100
out.append(r"""
\section*{S2. Tier 2: certified gaps per instance}
\begin{longtable}{llrrrrrr}
\caption{Package results on every instance at $300$\,s (cf.\
Tables~2--3).}\\
\toprule
Instance & objective & $n$ & $m$ & obj &
$\lceil\mathrm{LB}\rceil$ & gap\,\% & CV\,\%\\
\midrule\endfirsthead
\toprule
Instance & objective & $n$ & $m$ & obj &
$\lceil\mathrm{LB}\rceil$ & gap\,\% & CV\,\%\\
\midrule\endhead
\bottomrule\endlastfoot
""")
for _, r in db.iterrows():
    obj_s = 'area' if r['objective'] == 'total_cost' else 'makespan'
    out.append(f"{esc(r['key'])} & {obj_s} & {int(r['n'])} & "
               f"{int(r['m'])} & {r['obj']:.0f} & {r['lb_int']:.0f} & "
               f"{r['gap']:.2f} & {r['cv']:.2f} \\\\\n")
out.append("\\end{longtable}\n")

# ---------- S3: Tier 3 per-instance, per-method ----------
da = pd.read_csv('results/v2_main.csv').sort_values(['key', 'method'])
out.append(r"""
\section*{S3. Tier 3: equal-budget results per instance and method}
\begin{longtable}{llrrrr}
\caption{All equal-budget runs at $T=1200$\,s (cf.\ Tables~4--5).
`inf' marks runs that returned no feasible solution; negative or
missing bounds are reported as `--'.}\\
\toprule
Instance & method & obj & LB & gap\,\% & time (s)\\
\midrule\endfirsthead
\toprule
Instance & method & obj & LB & gap\,\% & time (s)\\
\midrule\endhead
\bottomrule\endlastfoot
""")
for _, r in da.iterrows():
    obj_s = 'inf' if not np.isfinite(r['obj']) else f"{r['obj']:.0f}"
    lb_s = '--' if not np.isfinite(r['lb']) or r['lb'] < 0 \
        else f"{r['lb']:.2f}"
    out.append(f"{esc(r['key'])} & {r['method']} & {obj_s} & {lb_s} & "
               f"{min(r['gap'], 100):.2f} & {r['time']:.0f} \\\\\n")
out.append("\\end{longtable}\n\\end{document}\n")

with open('paper_v2/supplement.tex', 'w') as f:
    f.write(''.join(out))
print('wrote paper_v2/supplement.tex')
