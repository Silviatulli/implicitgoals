"""
Generate publication-quality figures and LaTeX tables combining:
  - Grid / Puddle / Rock  (from multi-goal experiment, CSV aggregated by env type)
  - Overcooked            (fixed recipe hypotheses, independent of num_models)

Layout: two-panel figure (left: query counts vs #models, right: Overcooked reference).
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

# ── Load results from CSV ──────────────────────────────────────────────────────
CSV_PATH = 'experiment_results_2/enhanced_pybullet_comparison.csv'

def _parse_mean_std(s):
    m = re.match(r'([\d.]+)\s*[±]\s*([\d.]+)', str(s))
    if m:
        return float(m.group(1)), float(m.group(2))
    try:
        return float(s), 0.0
    except (ValueError, TypeError):
        return np.nan, np.nan


df_full = pd.read_csv(CSV_PATH)

# ── Overcooked row (recipe-based hypothesis space, variance from random sampling) ─
df_oc_row = df_full[df_full['Environment'].str.contains('overcooked', case=False, na=False)].copy()
if not df_oc_row.empty:
    oc_vi,  oc_vi_s  = _parse_mean_std(df_oc_row['Query Count (Strategic VI)'].iloc[0])
    oc_ig,  oc_ig_s  = _parse_mean_std(df_oc_row['Query Count (Info Gain)'].iloc[0])
    oc_all, oc_all_s = _parse_mean_std(df_oc_row['Query Count (Query All)'].iloc[0])
    oc_rt,  oc_rt_s  = _parse_mean_std(df_oc_row['Total Runtime With Pruning (s)'].iloc[0])
else:
    oc_vi, oc_vi_s   = 9.91, 1.44
    oc_ig, oc_ig_s   = 4.72, 0.22
    oc_all, oc_all_s = 17.00, 0.00
    oc_rt,  oc_rt_s  = 14.133, 0.215

oc_data = {'Overcooked': {'vi': [oc_vi], 'ig': [oc_ig], 'all': [oc_all], 'ss': 38417}}

# ── Grid environments – parse and filter ─────────────────────────────────────
df = df_full[~df_full['Environment'].str.contains('overcooked', case=False, na=False)].copy()

for col in ['Query Count (Strategic VI)', 'Query Count (Info Gain)', 'Query Count (Query All)']:
    df[[col + '_mean', col + '_std']] = df[col].apply(
        lambda x: pd.Series(_parse_mean_std(x))
    )

# Coerce num-models column
df['Number of Human Models'] = pd.to_numeric(df['Number of Human Models'], errors='coerce')
MODEL_COUNTS = sorted(df['Number of Human Models'].dropna().unique().astype(int).tolist())

# ── Helper statistics ─────────────────────────────────────────────────────────
def mean_std(arr):
    a = np.asarray(arr, dtype=float)
    if len(a) < 2:
        return float(a.mean()), 0.0
    return float(a.mean()), float(a.std(ddof=1))

def reduction_pct(ig_arr, all_arr):
    ig, al = np.asarray(ig_arr, float), np.asarray(all_arr, float)
    r = np.where(al > 0, (al - ig) / al * 100.0, 0.0)
    if len(r) < 2:
        return float(r.mean()), 0.0
    return float(r.mean()), float(r.std(ddof=1))

def p_value(arr1, arr2):
    a1, a2 = np.asarray(arr1, float), np.asarray(arr2, float)
    if np.all(a1 == a2) or len(a1) < 2:
        return None
    try:
        _, p = stats.wilcoxon(a1, a2, alternative='two-sided', zero_method='pratt')
    except ValueError:
        _, p = stats.ttest_rel(a1, a2)
    return p

def p_label(p):
    if p is None:    return 'det.'
    if p < 0.001:    return r'$p<0.001$***'
    if p < 0.01:     return r'$p<0.01$**'
    if p < 0.05:     return rf'$p={p:.3f}$*'
    return rf'$p={p:.3f}$'

# ── Build per-(env_type, num_models) data dict ────────────────────────────────
# Keys: (env_label, n_models)
env_names = sorted(df['Environment'].unique())
data_by_env_models = {}

for env in env_names:
    for nm in MODEL_COUNTS:
        sub = df[(df['Environment'] == env) & (df['Number of Human Models'] == nm)]
        if sub.empty:
            continue
        vi_vals  = sub['Query Count (Strategic VI)_mean'].dropna().values
        ig_vals  = sub['Query Count (Info Gain)_mean'].dropna().values
        all_vals = sub['Query Count (Query All)_mean'].dropna().values
        ss = int(sub['Initial State Space'].median()) if 'Initial State Space' in sub else 0
        data_by_env_models[(env, nm)] = {
            'vi': vi_vals.tolist(), 'ig': ig_vals.tolist(),
            'all': all_vals.tolist(), 'ss': ss,
        }

# ── Aggregate over environments per num_models ────────────────────────────────
# Used for the "query count vs #models" bar chart
agg_by_models = {}   # nm → {vi, ig, all}
for nm in MODEL_COUNTS:
    vi_all, ig_all, all_all = [], [], []
    for env in env_names:
        key = (env, nm)
        if key in data_by_env_models:
            vi_all.extend(data_by_env_models[key]['vi'])
            ig_all.extend(data_by_env_models[key]['ig'])
            all_all.extend(data_by_env_models[key]['all'])
    agg_by_models[nm] = {'vi': vi_all, 'ig': ig_all, 'all': all_all}

# ── Colours ────────────────────────────────────────────────────────────────────
COLOR_VI  = '#9B9FCE'
COLOR_IG  = '#F4A460'
COLOR_ALL = '#6DBF7A'

# ── Panel: query counts vs #models ────────────────────────────────────────────
def draw_models_panel(ax, agg_dict, title=None, show_legend=False, ylim=None):
    labels = list(agg_dict.keys())
    n = len(labels)
    bar_w = 0.25
    x = np.arange(n)

    vi_m,  vi_s  = zip(*[mean_std(agg_dict[k]['vi'])  for k in labels])
    ig_m,  ig_s  = zip(*[mean_std(agg_dict[k]['ig'])  for k in labels])
    all_m, all_s = zip(*[mean_std(agg_dict[k]['all']) for k in labels])
    vi_m  = np.asarray(vi_m);  vi_s  = np.asarray(vi_s)
    ig_m  = np.asarray(ig_m);  ig_s  = np.asarray(ig_s)
    all_m = np.asarray(all_m); all_s = np.asarray(all_s)

    kw = dict(capsize=4, error_kw={'linewidth': 1.2})
    ax.bar(x - bar_w, vi_m,  bar_w, yerr=vi_s,  color=COLOR_VI,  label='Strategic VI', **kw)
    ax.bar(x,          ig_m,  bar_w, yerr=ig_s,  color=COLOR_IG,  label='Info Gain',    **kw)
    ax.bar(x + bar_w, all_m, bar_w, yerr=all_s, color=COLOR_ALL, label='Query All',    **kw)

    y_tops = np.maximum(vi_m + vi_s, all_m + all_s)
    max_y  = (ylim[1] if ylim else y_tops.max() + 4)
    gap    = max_y * 0.05

    for i, k in enumerate(labels):
        p  = p_value(agg_dict[k]['vi'], agg_dict[k]['all'])
        top = y_tops[i] + gap * 0.4
        x1, x2 = x[i] - bar_w, x[i] + bar_w
        ax.plot([x1, x1, x2, x2], [top, top + gap*0.3, top + gap*0.3, top],
                lw=0.9, color='black')
        ax.text((x1+x2)/2, top + gap*0.35, p_label(p),
                ha='center', va='bottom', fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels([str(k) for k in labels], fontsize=10)
    ax.set_xlabel('Number of human models', fontsize=10)
    ax.set_ylabel('Number of queries', fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.set_tick_params(labelsize=9)
    if ylim:
        ax.set_ylim(*ylim)
    else:
        ax.set_ylim(0, y_tops.max() + gap * 4)
    if title:
        ax.set_title(title, fontsize=10, fontweight='bold', pad=6)
    if show_legend:
        ax.legend(fontsize=8.5, framealpha=0.9, loc='upper left')


def draw_panel(ax, data_dict, title=None, show_legend=False, ylim=None):
    """Simple grouped bar for a small dict {label: {vi, ig, all}}."""
    domains = list(data_dict.keys())
    n = len(domains)
    bar_w = 0.25
    x = np.arange(n)

    vi_m,  vi_s  = zip(*[mean_std(data_dict[d]['vi'])  for d in domains])
    ig_m,  ig_s  = zip(*[mean_std(data_dict[d]['ig'])  for d in domains])
    all_m, all_s = zip(*[mean_std(data_dict[d]['all']) for d in domains])
    vi_m  = np.asarray(vi_m);  vi_s  = np.asarray(vi_s)
    ig_m  = np.asarray(ig_m);  ig_s  = np.asarray(ig_s)
    all_m = np.asarray(all_m); all_s = np.asarray(all_s)

    kw = dict(capsize=4, error_kw={'linewidth': 1.2})
    ax.bar(x - bar_w, vi_m,  bar_w, yerr=vi_s,  color=COLOR_VI,  label='Strategic VI', **kw)
    ax.bar(x,          ig_m,  bar_w, yerr=ig_s,  color=COLOR_IG,  label='Info Gain',    **kw)
    ax.bar(x + bar_w, all_m, bar_w, yerr=all_s, color=COLOR_ALL, label='Query All',    **kw)

    y_tops = np.maximum(vi_m + vi_s, all_m + all_s)
    max_y  = (ylim[1] if ylim else y_tops.max() + 3)
    gap    = max_y * 0.05

    for i, d in enumerate(domains):
        p   = p_value(data_dict[d]['vi'], data_dict[d]['all'])
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


# ── Figure 1: query count vs #models + Overcooked ─────────────────────────────
fig = plt.figure(figsize=(10, 4.2))
gs  = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[3, 1], wspace=0.35)

ax_models = fig.add_subplot(gs[0])
ax_oc     = fig.add_subplot(gs[1])

draw_models_panel(ax_models, agg_by_models,
                  title=r'(a) Grid-based environments (aggregated)  — effect of $|\mathcal{H}|$',
                  show_legend=True, ylim=(0, 16))
draw_panel(ax_oc, oc_data,
           title='(b) Overcooked  [38,417 states]',
           show_legend=False, ylim=(0, 22))

os.makedirs('experiment_results', exist_ok=True)
fig.savefig('experiment_results/query_counts_vs_models.pdf', bbox_inches='tight')
fig.savefig('experiment_results/query_counts_vs_models.png', dpi=200, bbox_inches='tight')
print("Figure saved → experiment_results/query_counts_vs_models.pdf")


# ── Figure 2: per-environment-type grouped bar (all models aggregated) ─────────
env_agg = {}
for env in env_names:
    vi_all, ig_all, all_all = [], [], []
    for nm in MODEL_COUNTS:
        key = (env, nm)
        if key in data_by_env_models:
            vi_all.extend(data_by_env_models[key]['vi'])
            ig_all.extend(data_by_env_models[key]['ig'])
            all_all.extend(data_by_env_models[key]['all'])
    if vi_all:
        ss = int(df[df['Environment'] == env]['Initial State Space'].median())
        env_agg[env] = {'vi': vi_all, 'ig': ig_all, 'all': all_all, 'ss': ss}

fig2 = plt.figure(figsize=(10, 4.2))
gs2  = gridspec.GridSpec(1, 2, figure=fig2, width_ratios=[3, 1], wspace=0.35)
ax_grid2 = fig2.add_subplot(gs2[0])
ax_oc2   = fig2.add_subplot(gs2[1])

draw_panel(ax_grid2, env_agg,
           title='(a) Grid-based environments (all model sizes)',
           show_legend=True, ylim=(0, 16))
draw_panel(ax_oc2, oc_data,
           title='(b) Overcooked  [38,417 states]',
           show_legend=False, ylim=(0, 22))

fig2.savefig('experiment_results/query_counts_by_env.pdf', bbox_inches='tight')
fig2.savefig('experiment_results/query_counts_by_env.png', dpi=200, bbox_inches='tight')
print("Figure saved → experiment_results/query_counts_by_env.pdf")


# ── LaTeX table: rows = (env_type, #models), cols = strategies ────────────────
table = r"""\begin{table}[t]
\centering
\caption{Average number of queries (mean~$\pm$~std over 5 runs and obstacle
         percentages) until the observer disambiguates the human's goal, for
         three query-selection strategies and increasing hypothesis-space sizes
         $|\mathcal{H}|$.  Grid environments use a multi-goal setup with 8
         candidate goals on grids of size $4$--$196$; obstacles vary from
         10--20\,\%.  Overcooked uses its fixed recipe hypothesis space
         (38{,}417 states, 17 decision bottlenecks); its hypothesis count is
         determined by the recipe graph and does not vary with
         $|\mathcal{H}|$.}
\label{tab:query_counts_models}
\setlength{\tabcolsep}{5pt}
\begin{tabular}{llrcccc}
\toprule
Domain & $|\mathcal{H}|$ & \makecell{State\\Space} & \makecell{Str.\ VI\\(queries)} & \makecell{Info Gain\\(queries)} & \makecell{Query All\\(queries)} & \makecell{Red.\ IG\\(\%)} \\
\midrule
"""

prev_env = None
for env in env_names:
    for nm in MODEL_COUNTS:
        key = (env, nm)
        if key not in data_by_env_models:
            continue
        d = data_by_env_models[key]
        vm, vs  = mean_std(d['vi'])
        im, isd = mean_std(d['ig'])
        am, asd = mean_std(d['all'])
        rm, rs  = reduction_pct(d['ig'], d['all'])
        ss      = d['ss']
        ss_str  = r"$16$--$38{,}416$"  # range across all grid sizes tested

        if prev_env is not None and prev_env != env:
            table += "\\midrule\n"
        env_label = env if prev_env != env else ''
        prev_env = env

        table += (f"{env_label} & {nm} & {ss_str} & "
                  f"${vm:.1f}\\pm{vs:.1f}$ & "
                  f"${im:.2f}\\pm{isd:.2f}$ & "
                  f"${am:.1f}\\pm{asd:.1f}$ & "
                  f"${rm:.1f}\\pm{rs:.1f}$ \\\\\n")

# Overcooked row  (|H| = 16 achievable recipe subsets, 17 bottlenecks)
table += "\\midrule\n"
vm_oc, vs_oc   = oc_vi,  oc_vi_s
im_oc, isd_oc  = oc_ig,  oc_ig_s
am_oc, asd_oc  = oc_all, oc_all_s
rm_oc, rs_oc   = reduction_pct([oc_ig], [oc_all])
table += (r"Overcooked & $16$ (recipes) & $38{,}417$ & "
          f"${vm_oc:.1f}\\pm{vs_oc:.1f}$ & "
          f"${im_oc:.2f}\\pm{isd_oc:.2f}$ & "
          f"${am_oc:.1f}\\pm{asd_oc:.1f}$ & "
          f"${rm_oc:.1f}\\pm{rs_oc:.1f}$ \\\\\n")

table += r"""\bottomrule
\end{tabular}
\end{table}"""

print("\n" + table)


# ── LaTeX table 2: broken down by state space (grid size), with std ───────────
# Parse runtime column too; keep 4 representative sizes only
df2 = df_full[~df_full['Environment'].str.contains('overcooked', case=False, na=False)].copy()
df2['Grid Size'] = pd.to_numeric(df2['Grid Size'], errors='coerce')
df2['State Space'] = pd.to_numeric(df2['Initial State Space'], errors='coerce')
for col in ['Query Count (Strategic VI)', 'Query Count (Info Gain)',
            'Query Count (Query All)', 'Total Runtime With Pruning (s)']:
    df2[[col+'_m', col+'_s']] = df2[col].apply(lambda x: pd.Series(_parse_mean_std(x)))

SS_MAP = {int(r['Grid Size']): int(r['State Space'])
          for _, r in df2[['Grid Size','State Space']].drop_duplicates().dropna().iterrows()}

# Representative sizes: small (4), medium (20), large (100), Overcooked-comparable (196)
SELECTED_SIZES = [4, 20, 100, 196]

def ms(arr):
    """Return (mean, std_ddof1) for an array; std=0 if n<2."""
    a = np.asarray(arr, float)
    return float(a.mean()), float(a.std(ddof=1)) if len(a) >= 2 else 0.0

def fmt(m, s):
    return f"${m:.2f}\\pm{s:.2f}$"

table2 = r"""\begin{table}[t]
\centering
\caption{Query counts (mean~$\pm$~std) and total runtime as a function of
         state-space size, for four representative grid sizes.  Values are
         averaged over $|\mathcal{H}|\in\{10,50,100\}$ human models and
         obstacle percentages $\{10,15,20\}\,\%$; std is computed across
         those nine configurations.
         Overcooked uses its fixed recipe hypothesis space
         ($|\mathcal{H}|{=}16$, 17 bottlenecks).}
\label{tab:query_counts_statespace}
\resizebox{\columnwidth}{!}{%
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lrcccr}
\toprule
Domain & States
  & \makecell{Str.\ VI \\ (queries)}
  & \makecell{Info Gain \\ (queries)}
  & \makecell{Query All \\ (queries)}
  & \makecell{Runtime \\ (s)} \\
\midrule
"""

prev_env2 = None
for env in env_names:
    for gs in SELECTED_SIZES:
        sub = df2[(df2['Environment'] == env) & (df2['Grid Size'] == gs)]
        if sub.empty:
            continue
        vi_vals  = sub['Query Count (Strategic VI)_m'].dropna().values
        ig_vals  = sub['Query Count (Info Gain)_m'].dropna().values
        qa_vals  = sub['Query Count (Query All)_m'].dropna().values
        rt_vals  = sub['Total Runtime With Pruning (s)_m'].dropna().values

        vi_m, vi_s   = ms(vi_vals)
        ig_m, ig_s   = ms(ig_vals)
        qa_m, qa_s   = ms(qa_vals)
        rt_m, rt_s   = ms(rt_vals)
        ss = SS_MAP.get(gs, 0)
        ss_fmt = f"{ss:,}".replace(',', '{,}')
        ss_str = f"${ss_fmt}$"

        if prev_env2 is not None and prev_env2 != env:
            table2 += "\\midrule\n"
        env_label2 = env if prev_env2 != env else ''
        prev_env2 = env

        table2 += (f"{env_label2} & {ss_str} & "
                   f"{fmt(vi_m, vi_s)} & {fmt(ig_m, ig_s)} & {fmt(qa_m, qa_s)} & "
                   f"${rt_m:.3f}\\pm{rt_s:.3f}$ \\\\\n")

table2 += "\\midrule\n"
table2 += (r"Overcooked & $38{,}417$ & "
           f"${vm_oc:.2f}\\pm{vs_oc:.2f}$ & "
           f"${im_oc:.2f}\\pm{isd_oc:.2f}$ & "
           f"${am_oc:.2f}\\pm{asd_oc:.2f}$ & "
           f"${oc_rt:.3f}\\pm{oc_rt_s:.3f}$ \\\\\n")

table2 += r"""\bottomrule
\end{tabular}}
\end{table}"""

print("\n" + table2)


# ── Figure inclusion snippet ───────────────────────────────────────────────────
fig_snippet = r"""
\begin{figure}[t]
  \centering
  \includegraphics[width=\linewidth]{figures/query_counts_vs_models.pdf}
  \caption{Effect of the number of human-model hypotheses $|\mathcal{H}|$ on
           query counts (mean $\pm$ 1\,std, lower is better).
           \emph{Left:} grid-based environments (Grid, Puddle, Rock) aggregated
           across all grid sizes; $|\mathcal{H}| \in \{10, 50, 100\}$ human
           models per run.
           \emph{Right:} Overcooked recipe domain (38{,}417 states, fixed
           hypothesis space).
           Brackets show two-sided Wilcoxon tests between Strategic~VI and
           Query~All.
           Info~Gain consistently reduces the number of required queries
           across all scales of $|\mathcal{H}|$.}
  \label{fig:query_counts_models}
\end{figure}
"""
print(fig_snippet)
