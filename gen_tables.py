"""Generate paper tables (paper_v2/tables/*.tex) from results CSVs.

Reproducible pipeline: results/v2_relax.csv, v2_alns.csv, v2_main.csv
-> booktabs tables.  Run after any campaign update:
    python gen_tables.py
"""
import os
import numpy as np
import pandas as pd

OUT = 'paper_v2/tables'
os.makedirs(OUT, exist_ok=True)


def fam(k):
    if k.startswith('CL'):
        return f"C{int(k[2:4]):02d}"
    return k.split('_')[0]


def write(name, txt):
    with open(f'{OUT}/{name}', 'w') as f:
        f.write(txt)
    print('wrote', f'{OUT}/{name}')


def sched_hstar():
    """Recompute h* for sched keys (missing in early relax rows)."""
    from gmspp import (generate_type1, generate_type2, load_zdf_gmspp)
    out = {}
    for n, m in [(150, 2), (150, 3), (250, 4), (400, 4)]:
        out[f'T1_{n}_{m}'] = generate_type1(n, m, seed=0)
        out[f'T2_{n}_{m}'] = generate_type2(n, m, seed=0)
    for z, m in [('zdf1', 2), ('zdf2', 2), ('zdf3', 2), ('zdf1', 4)]:
        out[f'ZDF_{z}_{m}'] = load_zdf_gmspp('benchmarks/ZDF/ZDF/ZDF',
                                             z, m, shuffle_seed=0)
    return {k: v.simple_lower_bound() for k, v in out.items()}


# ================= Table C: relaxation strength =================
def table_C():
    df = pd.read_csv('results/v2_relax.csv')
    df = df[~df['key'].str.startswith('N_')].copy()
    # LP time-outs are recorded as 0 -> treat as unavailable
    df.loc[df['lp_pc'] <= 0, 'lp_pc'] = np.nan
    hs = sched_hstar()
    df['simple_lb'] = df.apply(
        lambda r: hs.get(r['key'], r['simple_lb']), axis=1)
    df['fam'] = df['key'].map(fam)
    B = ['bigm_lp', 'mmsp_lp', 'mmsp_lb', 'lp_pc', 'simple_lb']
    best = df[B].max(axis=1)
    for b in B:
        df[b + '_d'] = (best - df[b]) / best * 100
    df['geo'] = (df['lp_pc'] - df['mmsp_lb']) / df['mmsp_lb'] * 100

    rows = []
    order = [f'C{c:02d}' for c in range(1, 11)] + ['T1', 'T2', 'ZDF', 'ZDFS']
    for f_ in order:
        g = df[df['fam'] == f_]
        if not len(g):
            continue

        def m(col):
            v = g[col].mean()
            return '--' if v != v else f'{v:.1f}'
        geo = g['geo'].max()
        rows.append(
            f"{f_.replace('C0', 'CLASS ').replace('C10', 'CLASS 10')} & "
            f"{len(g)} & {m('bigm_lp_d')} & {m('mmsp_lp_d')} & "
            f"{m('mmsp_lb_d')} & {m('lp_pc_d')} & {m('simple_lb_d')} & "
            f"{'--' if geo != geo else f'{geo:.1f}'} \\\\")
    body = '\n'.join(rows)
    write('table_relax.tex', r"""\begin{table}[h!]
\centering\small
\caption{Tier 1 --- relaxation strength on all makespan instances.
Columns 3--7: mean percentage below the best bound of the portfolio
(0.0 = this bound is the best on every instance of the row).
Last column: largest gain of LP-PC over $Z^{\mathrm{ms}}$ in the row.
LP-PC is not computed beyond $n=500$ (stress rows); the big-$M$ LP is
not computed beyond $n=120$.}
\label{tab:relax}
\begin{tabular}{lrrrrrrr}
\toprule
 & & \multicolumn{5}{c}{mean \% below best bound} &
   max gain (\%)\\
\cmidrule(lr){3-7}
Family & \# & big-$M$ LP & liq.\ LP & $Z^{\mathrm{ms}}$ & LP-PC &
$h^*$ & LP-PC vs $Z^{\mathrm{ms}}$\\
\midrule
""" + body + r"""
\bottomrule
\end{tabular}
\end{table}
""")


# ================= Table B: certified-gap sweep =================
def table_B():
    db = pd.read_csv('results/v2_alns.csv')
    db['lb_int'] = np.ceil(db['lb'] - 1e-9)
    db['gap'] = (db['obj'] - db['lb_int']) / db['obj'].clip(lower=1e-9) * 100
    db['cv'] = db['obj_std'] / db['obj_mean'].clip(lower=1e-9) * 100
    db['fam'] = db['key'].map(fam)
    cl = db[db['fam'].str.startswith('C')]
    rows = []
    for c in range(1, 11):
        g = cl[cl['fam'] == f'C{c:02d}']
        line = [f'CLASS {c}']
        for obj in ('total_cost', 'makespan'):
            h = g[g['objective'] == obj]
            line += [f"{h['gap'].mean():.1f}", f"{h['gap'].max():.1f}",
                     f"{int((h['gap'] < 1e-9).sum())}",
                     f"{h['cv'].mean():.2f}"]
        rows.append(' & '.join(line) + r' \\')
    body = '\n'.join(rows)
    write('table_sweep_class.tex', r"""\begin{table}[h!]
\centering\small
\caption{Tier 2 --- certified optimality gaps of the matheuristic
package on the CLASS families ($300$\,s per instance, $10$ parallel
runs with distinct seeds; $12$ instances per cell).  Certified
gaps (in \%) are against the rounded-up bound portfolio; CV is the
coefficient of variation---std/mean, in \%---of the ten runs'
final objective values, averaged over the cell.}
\label{tab:sweep-class}
\begin{tabular}{lrrrrrrrr}
\toprule
 & \multicolumn{4}{c}{total area} & \multicolumn{4}{c}{makespan}\\
\cmidrule(lr){2-5}\cmidrule(lr){6-9}
Class & mean\,\% & max\,\% & \#opt & CV\,\% & mean\,\% & max\,\% & \#opt & CV\,\%\\
\midrule
""" + body + r"""
\bottomrule
\end{tabular}
\end{table}
""")
    sc = db[~db['fam'].str.startswith('C')].sort_values('n')
    rows = []
    for _, r in sc.iterrows():
        star = r['key'].startswith('ZDFS')
        name = r['key'].replace('ZDFS_', '').replace('ZDF_', '') \
                       .replace('_', r'\_')
        rows.append(
            f"{name}{'$^*$' if star else ''} & {int(r['n'])} & "
            f"{int(r['m'])} & {r['obj']:.0f} & {r['lb_int']:.0f} & "
            f"{r['gap']:.2f} & {r['cv']:.2f} \\\\")
    body = '\n'.join(rows)
    write('table_sweep_sched.tex', r"""\begin{table}[h!]
\centering\small
\caption{Tier 2 --- certified gaps on the strip-dependent
(scheduling-lens) instances, $300$\,s each.  Starred rows are the
stress instances (LP-PC skipped; portfolio $= Z^{\mathrm{ms}},
h^*$).  A gap of $0.00$ is a proven optimum; CV as in
Table~\ref{tab:sweep-class}.}
\label{tab:sweep-sched}
\begin{tabular}{lrrrrrr}
\toprule
Instance & $n$ & $m$ & obj & $\lceil\mathrm{LB}\rceil$ &
gap\,\% & CV\,\%\\
\midrule
""" + body + r"""
\bottomrule
\end{tabular}
\end{table}
""")


# ================= Tables A1 / A2 =================
def table_A():
    da = pd.read_csv('results/v2_main.csv')
    cl = da[da['key'].str.startswith('CL')].copy()
    cl['cls'] = cl['key'].str[2:4].astype(int)
    a1 = cl[cl['cls'].isin([1, 3, 8, 10]) & cl['n'].isin([20, 60])]
    NAME = {'bigm': 'BigM', 'bigmle': 'BigM-LE',
            'bendm': 'BendM', 'package': 'Package'}
    piv = a1.pivot_table(index='key', columns='method', values='obj')
    lpiv = a1.pivot_table(index='key', columns='method', values='lb')
    best = piv.min(axis=1)
    bestlb = lpiv.max(axis=1)
    rows = []
    for n in (20, 60):
        keys = [k for k in piv.index if f'_{n:03d}_' in k]
        p, b = piv.loc[keys], best.loc[keys]
        q, c = lpiv.loc[keys], bestlb.loc[keys]
        for meth in ('bigm', 'bigmle', 'bendm', 'package'):
            g = a1[(a1['n'] == n) & (a1['method'] == meth)]
            nb = int((p[meth] <= b * (1 + 1e-6)).sum())
            dev = ((p[meth] - b) / b * 100).mean()
            nbd = int((q[meth] >= c * (1 - 1e-6)).sum())
            devd = ((c - q[meth]) / c * 100).mean()
            rows.append(
                f"{n if meth == 'bigm' else ''} & {NAME[meth]} & "
                f"{int(g['opt'].sum())}/{len(g)} & "
                f"{g['gap'].mean():.2f} & "
                f"{nb}/{len(keys)} & {dev:.2f} & "
                f"{nbd}/{len(keys)} & {devd:.2f} & "
                f"{g['time'].mean():.0f} \\\\")
        if n == 20:
            rows.append(r'\midrule')
    body = '\n'.join(rows)
    write('table_hierarchy.tex', r"""\begin{table}[h!]
\centering\small
\caption{Tier 3 --- elimination experiment: the method hierarchy
at small size (classes
1, 3, 8, 10; both objectives; $32$ instances per row; $T=1200$\,s
each).  The terminal gap decomposes into its two sides: the primal
pair compares each method's incumbent, and the dual pair its
terminal lower bound, with the best of the four methods; \#best
counts instances at that best, $\Delta$ is the mean relative
deviation from it.}
\label{tab:hierarchy}
\begin{tabular}{llrrrrrrr}
\toprule
 & & & & \multicolumn{2}{c}{primal (incumbent)} &
\multicolumn{2}{c}{dual (bound)} & \\
\cmidrule(lr){5-6}\cmidrule(lr){7-8}
$n$ & Method & opt & gap\,\% & \#best & $\Delta$\,\% &
\#best & $\Delta$\,\% & time (s)\\
\midrule
""" + body + r"""
\bottomrule
\end{tabular}
\end{table}
""")

    a2 = da[da['method'].isin(['bendm', 'package'])].copy()

    def band(r):
        k = r['key']
        if k.startswith('ZDFS'):
            return '5_stress'
        if not k.startswith('CL'):
            return '4_sched'
        return {20: '1', 60: '2', 100: '3'}[r['n']] + f"_n={r['n']}"
    a2['band'] = a2.apply(band, axis=1)
    piv = a2.pivot_table(index=['band', 'key'], columns='method',
                         values='obj').dropna()
    duel = piv.groupby('band').apply(lambda x: pd.Series({
        'bw': int((x['bendm'] < x['package'] * (1 - 1e-6)).sum()),
        'tie': int((abs(x['bendm'] - x['package'])
                    <= 1e-6 * x['package']).sum()),
        'pw': int((x['package'] < x['bendm'] * (1 - 1e-6)).sum())}))
    LBL = {'1_n=20': 'CLASS $n=20$', '2_n=60': 'CLASS $n=60$',
           '3_n=100': 'CLASS $n=100$',
           '4_sched': 'sched.\ $n=145$--$800$',
           '5_stress': 'stress $n=1258$--$2532$'}
    rows = []
    for b in sorted(a2['band'].unique()):
        line = [LBL[b]]
        for meth in ('bendm', 'package'):
            g = a2[(a2['band'] == b) & (a2['method'] == meth)]
            if len(g):
                line += [f"{int(g['opt'].sum())}/{len(g)}",
                         f"{g['gap'].mean():.2f}"]
            else:
                line += ['--', '--']
        if b in duel.index:
            d = duel.loc[b]
            line += [f"{d['bw']}/{d['tie']}/{d['pw']}"]
        else:
            line += ['--']
        rows.append(' & '.join(line) + r' \\')
    body = '\n'.join(rows)
    write('table_frontier.tex', r"""\begin{table}[h!]
\centering\small
\caption{Tier 3 --- the exact/matheuristic frontier: BendM vs the
package under a common budget of $1200$\,s.  ``Duel'' counts
instances whose best solution is strictly better under
BendM\,/\,tied\,/\,strictly better under the package.  On the stress
band BendM returned no feasible solution on three of the four
instances and no valid lower bound on any (gap reported as $100$);
building its master alone exceeded the budget several-fold.}
\label{tab:frontier}
\begin{tabular}{lrrrrr}
\toprule
 & \multicolumn{2}{c}{BendM} & \multicolumn{2}{c}{Package} & \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}
Band & opt & gap\,\% & opt & gap\,\% & Duel (B/t/P)\\
\midrule
""" + body + r"""
\bottomrule
\end{tabular}
\end{table}
""")


# ============ Appendix: zero-waste ablation + calibration ============
def table_nzw():
    from gmspp.benchmark_loader import load_and_convert_benchmark
    insts = load_and_convert_benchmark('benchmarks/N', m=2,
                                       cost_type='proportional')
    area = {nm: sum(it.width * it.height for it in inst.items)
            for nm, inst in insts.items()}
    d60 = pd.read_csv('results/v2_nzw60.csv')
    d300 = pd.read_csv('results/v2_nzw.csv')
    for d in (d60, d300):
        d['ncls'] = d['key'].str.extract(r'N(\d)')[0].astype(int)
        d['base'] = d['key'].str.split('_').str[0]
    # true gap on total_cost at T=300, endgame on (zero waste: OPT = area)
    tc = d300[d300['objective'] == 'total_cost'].copy()
    tc['true_gap'] = (tc['obj_on'] - tc['base'].map(area)) \
        / tc['obj_on'] * 100
    rows = []
    for c in range(1, 8):
        g60 = d60[d60['ncls'] == c]['gain_pct']
        g300 = d300[d300['ncls'] == c]['gain_pct']
        tg = tc[tc['ncls'] == c]['true_gap']
        n = d300[d300['ncls'] == c]['n'].iloc[0]
        rows.append(f"N{c} & {n} & {g60.mean():+.2f} & "
                    f"{g60.abs().max():.2f} & {g300.mean():+.2f} & "
                    f"{g300.abs().max():.2f} & {tg.mean():.2f} \\\\")
    body = '\n'.join(rows)
    write('table_nzw.tex', r"""\begin{table}[h!]
\centering\small
\caption{Zero-waste family: endgame on/off ablation (paired runs,
identical seed) and true-gap calibration.  $\Delta$ is the objective
change from disabling the endgame, averaged over both objectives and
$m\in\{2,3\}$ ($12$ pairs per row); positive would favor the
endgame.  All differences are within seed noise, and the endgame
improved no strip in any run.  The last column is the mean
\emph{true} optimality gap of the package under total area
($T=300$\,s), computable exactly here because the area bound equals
the optimum on zero-waste instances.}
\label{tab:nzw}
\begin{tabular}{lrrrrrr}
\toprule
 & & \multicolumn{2}{c}{$T=60$\,s} & \multicolumn{2}{c}{$T=300$\,s}
 & true gap\,\%\\
\cmidrule(lr){3-4}\cmidrule(lr){5-6}
Class & $n$ & mean $\Delta\%$ & max $|\Delta|\%$ &
mean $\Delta\%$ & max $|\Delta|\%$ & (total area)\\
\midrule
""" + body + r"""
\bottomrule
\end{tabular}
\end{table}
""")


if __name__ == '__main__':
    import sys, types
    try:
        import gurobipy                              # noqa
    except ImportError:                              # sandbox stub
        sys.modules['gurobipy'] = types.ModuleType('gurobipy')
        sys.modules['gurobipy'].GRB = types.SimpleNamespace()
    table_C()
    table_B()
    table_A()
    table_nzw()
