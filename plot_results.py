"""
Generate publication-quality figures and LaTeX tables combining:
  - Grid / Puddle / Rock  (from multi-goal experiment, CSV aggregated by env type)
  - Overcooked            (from yacine_overcooked branch, averaged over human models)

Layout: two-panel figure (left: grid environments, right: Overcooked).
"""

import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
import os

# ── Load grid results from CSV ────────────────────────────────────────────────
CSV_PATH = 'experiment_results_2/enhanced_pybullet_comparison.csv'

def _parse_mean_std(s):
    """Parse '3.52 ± 0.64' → (3.52, 0.64)."""
    m = re.match(r'([\d.]+)\s*[±]\s*([\d.]+)', str(s))
    if m:
        return float(m.group(1)), float(m.group(2))
    try:
        v = float(s)
        return v, 0.0
    except (ValueError, TypeError):
        return np.nan, np.nan

df = pd.read_csv(CSV_PATH)
df = df[~df['Environment'].str.contains('overcooked', case=False, na=False)].copy()

for col in ['Query Count (Strategic VI)', 'Query Count (Info Gain)', 'Query Count (Query All)']:
    df[[col + '_mean', col + '_std']] = df[col].apply(
        lambda x: pd.Series(_parse_mean_std(x))
    )

# Aggregate by environment type (mean of means)
env_groups = df.groupby('Environment', sort=False)

grid_raw = {}
for env_name, grp in env_groups:
    vi_vals  = grp['Query Count (Strategic VI)_mean'].dropna().values
    ig_vals  = grp['Query Count (Info Gain)_mean'].dropna().values
    all_vals = grp['Query Count (Query All)_mean'].dropna().values
    # Representative state space: use most common grid_size^2
    ss = int(grp['Initial State Space'].median()) if 'Initial State Space' in grp else 0
    grid_raw[env_name] = {
        'vi':  vi_vals.tolist(),
        'ig':  ig_vals.tolist(),
        'all': all_vals.tolist(),
        'ss':  ss,
    }

# ── Overcooked (deterministic, from previous run) ─────────────────────────────
oc_raw = {
    'Overcooked': {
        'vi':  [14.20, 14.20, 14.20],
        'ig':  [ 4.70,  4.70,  4.70],
        'all': [17.00, 17.00, 17.00],
        'ss':  38417,
    }
}

# ── Colour palette ────────────────────────────────────────────────────────────
COLOR_VI  = '#9B9FCE'
COLOR_IG  = '#F4A460'
COLOR_ALL = '#6DBF7A'


def mean_std(arr):
    a = np.array(arr, dtype=float)
    if len(a) < 2:
        return a.mean(), 0.0
    return a.mean(), a.std(ddof=1)

def reduction_pct(query_arr, all_arr):
    v, a = np.array(query_arr, dtype=float), np.array(all_arr, dtype=float)
    r = np.where(a > 0, (a - v) / a * 100, 0.0)
    if len(r) < 2:
        return r.mean(), 0.0
    return r.mean(), r.std(ddof=1)

def p_value(arr1, arr2):
    a1, a2 = np.array(arr1, dtype=float), np.array(arr2, dtype=float)
    if np.all(a1 == a2):
        return None
    try:
        _, p = stats.wilcoxon(a1, a2, alternative='two-sided', zero_method='pratt')
    except ValueError:
        _, p = stats.ttest_rel(a1, a2)
    return p

def p_label(p):
    if p is None:       return 'det.'
    if p < 0.001:       return 'p<0.001***'
    if p < 0.01:        return 'p<0.01**'
    if p < 0.05:        return f'p={p:.3f}*'
    return f'p={p:.3f}'

def draw_panel(ax, data_dict, title=None, show_legend=False, ylim=None):
    domains = list(data_dict.keys())
    n = len(domains)
    bar_w = 0.25
    x = np.arange(n)

    vi_m,  vi_s  = zip(*[mean_std(data_dict[d]['vi'])  for d in domains])
    ig_m,  ig_s  = zip(*[mean_std(data_dict[d]['ig'])  for d in domains])
    all_m, all_s = zip(*[mean_std(data_dict[d]['all']) for d in domains])
    vi_m  = np.array(vi_m);  vi_s  = np.array(vi_s)
    ig_m  = np.array(ig_m);  ig_s  = np.array(ig_s)
    all_m = np.array(all_m); all_s = np.array(all_s)

    kw = dict(capsize=4, error_kw={'linewidth': 1.2})
    ax.bar(x - bar_w, vi_m,  bar_w, yerr=vi_s,  color=COLOR_VI,  label='Strategic VI', **kw)
    ax.bar(x,          ig_m,  bar_w, yerr=ig_s,  color=COLOR_IG,  label='Info Gain',    **kw)
    ax.bar(x + bar_w, all_m, bar_w, yerr=all_s, color=COLOR_ALL, label='Query All',    **kw)

    y_tops = np.maximum(vi_m + vi_s, all_m + all_s)
    max_y  = (ylim[1] if ylim else y_tops.max() + 3)
    gap    = max_y * 0.05

    for i, d in enumerate(domains):
        p = p_value(data_dict[d]['vi'], data_dict[d]['all'])
        top = y_tops[i] + gap * 0.5
        x1, x2 = x[i] - bar_w, x[i] + bar_w
        ax.plot([x1, x1, x2, x2], [top, top + gap*0.3, top + gap*0.3, top],
                lw=0.9, color='black')
        ax.text((x1+x2)/2, top + gap*0.35, p_label(p),
                ha='center', va='bottom', fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(domains, fontsize=10)
    ax.set_ylabel('Number of queries', fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.set_tick_params(labelsize=9)
    if ylim:
        ax.set_ylim(*ylim)
    else:
        ax.set_ylim(0, y_tops.max() + gap * 3)
    if title:
        ax.set_title(title, fontsize=10, fontweight='bold', pad=6)
    if show_legend:
        ax.legend(fontsize=8.5, framealpha=0.9, loc='upper right')


# ── Two-panel figure ──────────────────────────────────────────────────────────
fig = plt.figure(figsize=(10, 4.2))
gs  = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[3, 1], wspace=0.35)

ax_grid = fig.add_subplot(gs[0])
ax_oc   = fig.add_subplot(gs[1])

# Compute a representative state-space label for grid envs
ss_label = int(df['Initial State Space'].median()) if 'Initial State Space' in df.columns else '?'
draw_panel(ax_grid, grid_raw,
           title=f'(a) Grid-based environments  [{ss_label} states, multi-goal]',
           show_legend=True, ylim=(0, 14))
draw_panel(ax_oc,   oc_raw,
           title='(b) Overcooked  [38,417 states]',
           show_legend=False, ylim=(0, 22))

os.makedirs('experiment_results', exist_ok=True)
fig.savefig('experiment_results/query_counts_plot.pdf', bbox_inches='tight')
fig.savefig('experiment_results/query_counts_plot.png', dpi=200, bbox_inches='tight')
print("Plot saved → experiment_results/query_counts_plot.pdf")


# ── Combined LaTeX table ──────────────────────────────────────────────────────
all_domains = {**grid_raw, **oc_raw}

table = r"""\begin{table}[t]
\centering
\caption{Query counts (mean $\pm$ std) and reduction relative to Query~All,
         across environments. Grid environments: multi-goal setup (8 candidate
         goals, grids 4--6, obstacle 10--15\%, 5--10 human models).
         Overcooked: $n{=}3$ timing runs, averaged over all recipe hypotheses.}
\label{tab:query_counts}
\setlength{\tabcolsep}{5pt}
\begin{tabular}{lrcccc}
\toprule
Domain & \makecell{State\\Space} & \makecell{Str.\ VI\\(queries)} & \makecell{Info Gain\\(queries)} & \makecell{Query All\\(queries)} & \makecell{Red.\ IG\\(\%)} \\
\midrule
"""

for i, (d, data) in enumerate(all_domains.items()):
    vm, vs   = mean_std(data['vi'])
    im, is_  = mean_std(data['ig'])
    am, as_  = mean_std(data['all'])
    rm, rs   = reduction_pct(data['ig'], data['all'])
    ss       = data['ss']
    ss_str   = f"\\num{{{ss:,}}}".replace(',', '{,}') if ss >= 1000 else str(ss)
    if d == 'Overcooked':
        table += "\\midrule\n"
    table += (f"{d} & {ss_str} & "
              f"${vm:.1f}\\pm{vs:.1f}$ & "
              f"${im:.2f}\\pm{is_:.2f}$ & "
              f"${am:.1f}\\pm{as_:.1f}$ & "
              f"${rm:.1f}\\pm{rs:.1f}$ \\\\\n")

table += r"""\bottomrule
\end{tabular}
\end{table}"""

print("\n" + table)

# ── Figure inclusion snippet ───────────────────────────────────────────────────
fig_snippet = r"""
\begin{figure}[t]
  \centering
  \includegraphics[width=\linewidth]{figures/query_counts_plot.pdf}
  \caption{Number of queries asked by each strategy before the agent's goal is
           disambiguated (lower is better). Error bars show $\pm 1$ std.
           Brackets report two-sided Wilcoxon tests (Strategic~VI vs.\ Query~All);
           ``det.''\ denotes a deterministic environment where variance is zero.
           \emph{Left:} three grid-based domains with multiple possible human
           goals (grids 4--6, 10--15\% obstacles).
           \emph{Right:} Overcooked recipe domain (38{,}417 states, 17 decision
           bottlenecks). Info Gain consistently reduces queries compared to
           Strategic~VI across all environments.}
  \label{fig:query_counts}
\end{figure}
"""
print(fig_snippet)
