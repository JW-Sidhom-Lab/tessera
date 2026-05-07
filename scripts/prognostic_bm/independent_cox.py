"""
Test whether the foundation-model continuous risk score adds prognostic
value INDEPENDENTLY of the categorical clinical labels (PAM50 for
BRCA, histologic subtype for BLCA, Decipher tertiles for PRAD).

Fair head-to-head: both predictors enter the joint Cox as 1-DOF
continuous log-hazards.

  - subtype_score    : linear predictor of a Cox model fit on subtype
                       dummies -- the "composite categorical risk
                       score" derived from the categorical labels.
  - foundation_score : log(Risk_Score), already saved per-sample by
                       cohort_analysis.py at run time.

Joint model:   hazard ~ subtype_score + foundation_score

For each cohort we report HR (95% CI, two-sided p, one-sided p_HR>1)
for both predictors.

Visualisation: 2 panels per cohort
  a -- forest plot of joint-model HRs
  b -- KM curves by (subtype × within-subtype median split of
        foundation Risk_Score). Within each subtype, samples above
        the subtype's median Risk_Score are "high" (dotted), below
        are "low" (solid). Demonstrates the score separates patients
        within every subtype.

Reads:
    analysis_results_{cohort}_infonce_per_sample_loss/data/fig5_cache.csv
Writes:
    independent_cox_results/{cohort}_independent_cox.csv
    independent_cox_results/{cohort}_independent_cox.{png,pdf}

Usage:
    cd methods/prognostic_bm
    python3 independent_cox.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from scipy.stats import norm

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / 'independent_cox_results'
OUT_DIR.mkdir(exist_ok=True)

COHORTS = [
    ('brca',
     HERE / 'analysis_results_brca_infonce_per_sample_loss',
     'Subtype_Oncotype', 'DSS.time', 'DSS',
     'BRCA -- OncotypeDX tertiles (DLRS reconstruction)'),
    # ('blca',
    #  HERE / 'analysis_results_blca_infonce_per_sample_loss',
    #  'Histologic_Subtype', 'DSS.time', 'DSS', 'BLCA -- histologic subtype'),
    ('prad',
     HERE / 'analysis_results_prad_infonce_per_sample_loss',
     'Subtype_Decipher', 'PFI.time.1', 'PFI.1',
     'PRAD -- Decipher tertiles (curatedPCaData)'),
]

MAX_DAYS = 1825      # 5-yr censoring (matches cohort_analysis.py)
MAX_MONTHS = MAX_DAYS / 30.44
MAX_SUBTYPES_IN_KM = 4


# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------

def load_cohort(results_dir, subtype_col, time_col, event_col):
    df = pd.read_csv(results_dir / 'data' / 'fig5_cache.csv')
    df = df.dropna(subset=[time_col, event_col, 'Risk_Score', subtype_col]).copy()
    df[time_col]  = pd.to_numeric(df[time_col],  errors='coerce')
    df[event_col] = pd.to_numeric(df[event_col], errors='coerce')
    df = df.dropna(subset=[time_col, event_col]).copy()
    late = df[time_col] > MAX_DAYS
    df.loc[late, event_col] = 0
    df.loc[late, time_col]  = MAX_DAYS
    df['time_months']      = df[time_col].astype(float) / 30.44
    df['event']            = df[event_col].astype(int)
    df['foundation_score'] = np.log(df['Risk_Score'].astype(float).clip(lower=1e-3))
    return df.reset_index(drop=True)


# -----------------------------------------------------------------------------
# Composite categorical risk score (subtype linear predictor)
# -----------------------------------------------------------------------------

def subtype_composite(df, subtype_col):
    """Linear predictor of a Cox(survival ~ subtype dummies) model.
    Reference level = most-frequent subtype."""
    counts = df[subtype_col].value_counts()
    ref    = counts.index[0]
    levels = [l for l in counts.index if l != ref]
    X = pd.DataFrame(index=df.index)
    for lvl in levels:
        X[f'subtype_{lvl}'] = (df[subtype_col].values == lvl).astype(int)
    fit = X.copy()
    fit['time_months'] = df['time_months'].values
    fit['event']       = df['event'].values
    cph = CoxPHFitter(penalizer=1e-3).fit(fit, duration_col='time_months',
                                          event_col='event')
    return cph.predict_log_partial_hazard(X).values, ref


# -----------------------------------------------------------------------------
# Joint Cox + summary table
# -----------------------------------------------------------------------------

def joint_cox(df, subtype_col):
    sub_lp, ref = subtype_composite(df, subtype_col)
    joint = pd.DataFrame({
        'subtype_score':    sub_lp,
        'foundation_score': df['foundation_score'].values,
        'time_months':      df['time_months'].values,
        'event':            df['event'].values,
    })
    cph = CoxPHFitter(penalizer=0.0).fit(joint, duration_col='time_months',
                                         event_col='event')
    s = cph.summary

    rows = []
    for var in ['subtype_score', 'foundation_score']:
        coef = float(s.loc[var, 'coef']); se = float(s.loc[var, 'se(coef)'])
        z = coef / se if se > 0 else 0.0
        rows.append(dict(
            predictor=var,
            HR=float(np.exp(coef)),
            CI_lo=float(np.exp(s.loc[var, 'coef lower 95%'])),
            CI_hi=float(np.exp(s.loc[var, 'coef upper 95%'])),
            p_two_sided=float(s.loc[var, 'p']),
            p_one_sided_HR_gt_1=float(1.0 - norm.cdf(z)),
            coef=coef, se=se, z=z,
        ))
    return pd.DataFrame(rows), float(cph.concordance_index_), sub_lp, ref


# -----------------------------------------------------------------------------
# Within-subtype median split + KM
# -----------------------------------------------------------------------------

def assign_within_subtype_risk(df, subtype_col, score_col='foundation_score'):
    """Add a 'risk_class' column = 'high' / 'low', median split within
    each subtype (so each subtype contributes ~50/50)."""
    out = df.copy()
    out['risk_class'] = ''
    for s, g in out.groupby(subtype_col):
        med = g[score_col].median()
        out.loc[g.index, 'risk_class'] = np.where(
            g[score_col] >= med, 'high', 'low')
    return out


# -----------------------------------------------------------------------------
# Plot helpers
# -----------------------------------------------------------------------------

_PRED_COLORS = {'subtype_score': '#1f77b4', 'foundation_score': '#d62728'}
_PRED_LABELS = {'subtype_score':    'Subtype composite',
                'foundation_score': 'Foundation\ncomposite'}


def panel_forest(ax, summary, n, events, cindex):
    """Two-row forest plot of joint-Cox HRs with one-sided + two-sided p."""
    ys = np.arange(len(summary))[::-1]
    xlim = (max(summary['CI_lo'].min() * 0.6, 0.05),
            summary['CI_hi'].max() * 1.3)
    for i, (_, row) in enumerate(summary.iterrows()):
        c = _PRED_COLORS[row['predictor']]
        y = ys[i]
        lo = max(row['CI_lo'], xlim[0]); hi = min(row['CI_hi'], xlim[1])
        ax.plot([lo, hi], [y, y], color=c, linewidth=1.4)
        sig = row['p_one_sided_HR_gt_1'] < 0.05
        mfc = c if sig else 'white'
        ax.plot([row['HR']], [y], marker='o', markersize=6,
                markerfacecolor=mfc, markeredgecolor=c, markeredgewidth=1.0)
        ax.text(xlim[1] * 1.05, y,
                f'HR {row["HR"]:.2f}   '
                f'[{row["CI_lo"]:.2f}, {row["CI_hi"]:.2f}]   '
                f'p = {row["p_one_sided_HR_gt_1"]:.3g}',
                ha='left', va='center', fontsize=8,
                fontweight='bold' if sig else 'normal')
    ax.axvline(1.0, color='#888888', linestyle='--', linewidth=0.5)
    ax.set_xscale('log'); ax.set_xlim(*xlim)
    ax.set_yticks(ys)
    ax.set_yticklabels([_PRED_LABELS[p] for p in summary['predictor']],
                       fontsize=9)
    ax.set_xlabel('Hazard ratio  (95% CI)', fontsize=9)
    ax.set_title(f'Joint Cox  (n={n}, events={events})    '
                 f'C-index = {cindex:.3f}    '
                 f'p = one-sided (HR > 1)',
                 fontsize=8, pad=5)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)


def panel_subtype_risk_km(ax, df, subtype_col):
    """Per-(subtype × high/low) Kaplan-Meier overlaid on one axis.

    Within each subtype, foundation Risk_Score is median-split into
    'high' / 'low' so the curves directly visualise within-subtype
    separation. Color = subtype, linestyle = risk class."""
    df = assign_within_subtype_risk(df, subtype_col)
    counts = df[subtype_col].value_counts()
    keep = counts.index[:MAX_SUBTYPES_IN_KM].tolist()
    sub = df[df[subtype_col].isin(keep)]

    cmap    = plt.get_cmap('tab10')
    palette = {s: cmap(i) for i, s in enumerate(keep)}
    line_st = {'low': '-', 'high': (0, (3, 1.5))}

    kmf = KaplanMeierFitter()
    for subtype in keep:
        for cls in ['low', 'high']:
            mask = (sub[subtype_col] == subtype) & (sub['risk_class'] == cls)
            n_g  = int(mask.sum())
            if n_g < 5:
                continue
            kmf.fit(sub.loc[mask, 'time_months'].values,
                    event_observed=sub.loc[mask, 'event'].values)
            sf = kmf.survival_function_
            ax.step(sf.index, sf.iloc[:, 0], where='post',
                    color=palette[subtype], linestyle=line_st[cls],
                    linewidth=1.0,
                    label=f'{subtype}  {cls}-risk (n={n_g})')

    ax.set_xlim(0, MAX_MONTHS)
    ax.set_xlabel('DSS (months)', fontsize=9)
    ax.set_ylabel('Survival probability', fontsize=9)
    ax.set_title('Within-subtype median split of foundation Risk_Score\n'
                 'solid = below-median, dashed = above-median',
                 fontsize=8, pad=5)
    ax.legend(loc='lower left', fontsize=6.5, frameon=False,
              handlelength=1.6, handletextpad=0.4, labelspacing=0.20,
              borderpad=0.0, borderaxespad=0.3, ncols=2,
              columnspacing=0.6)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)


# -----------------------------------------------------------------------------
# Per-cohort run
# -----------------------------------------------------------------------------

def run_cohort(cohort, results_dir, subtype_col, time_col, event_col, label):
    print('\n' + '=' * 78)
    print(label)
    print('=' * 78)

    df = load_cohort(results_dir, subtype_col, time_col, event_col)
    n  = len(df); ev = int(df['event'].sum())
    summary, cindex, _, ref = joint_cox(df, subtype_col)
    summary.insert(0, 'cohort', cohort)
    summary.insert(2, 'subtype_reference', ref)
    summary['n']      = n
    summary['events'] = ev
    summary['cindex_joint'] = cindex

    print(f'  n={n}, events={ev}, subtype reference={ref!r}')
    print(f'  Joint Cox  hazard ~ subtype_score + foundation_score   '
          f'C-index = {cindex:.4f}')
    print('  ' + '-' * 70)
    print(f'  {"predictor":<22} {"HR":>5} {"95% CI":>18} '
          f'{"p (1-sided HR>1)":>20}')
    for _, r in summary.iterrows():
        ci = f'[{r["CI_lo"]:.2f}, {r["CI_hi"]:.2f}]'
        print(f'  {r["predictor"]:<22} {r["HR"]:>5.2f} {ci:>18} '
              f'{r["p_one_sided_HR_gt_1"]:>20.4g}')

    csv_path = OUT_DIR / f'{cohort}_independent_cox.csv'
    summary.to_csv(csv_path, index=False)
    print(f'  → wrote {csv_path}')

    # ---- Two-panel figure ----
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(11.5, 4.0),
                                     gridspec_kw=dict(width_ratios=[1.5, 1.0]))
    panel_forest(ax_a, summary, n, ev, cindex)
    panel_subtype_risk_km(ax_b, df, subtype_col)
    for ax, lbl in [(ax_a, 'a'), (ax_b, 'b')]:
        ax.text(-0.05, 1.06, lbl, transform=ax.transAxes,
                fontsize=14, fontweight='bold', va='bottom', ha='right')
    fig.suptitle(label, fontsize=11, fontweight='bold', y=0.99)
    fig.subplots_adjust(left=0.105, right=0.985, top=0.83, bottom=0.16,
                        wspace=0.40)
    pos_a = ax_a.get_position()
    ax_a.set_position([pos_a.x0, pos_a.y0,
                       pos_a.width * 0.62, pos_a.height])

    out_base = OUT_DIR / f'{cohort}_independent_cox'
    fig.savefig(f'{out_base}.png', dpi=300, facecolor='white', bbox_inches='tight')
    fig.savefig(f'{out_base}.pdf', facecolor='white', bbox_inches='tight')
    plt.close(fig)
    print(f'  → wrote {out_base}.png')
    print(f'  → wrote {out_base}.pdf')


def main():
    for args in COHORTS:
        run_cohort(*args)
    print('\nDone.')


if __name__ == '__main__':
    main()
