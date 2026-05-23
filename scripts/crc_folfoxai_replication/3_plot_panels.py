"""
Evaluation panels for the exact Abraham 2021 FOLFOXai replication,
mirroring the paper's KM style (IB-predicted vs DB-predicted per arm).

Panels
------
  folfoxai_histogram   — p_IB distribution with 3% no-call band shaded
  folfoxai_km_folfox   — FOLFOX arm KM: predicted IB vs DB  (no-calls excluded)
  folfoxai_km_folfiri  — FOLFIRI arm KM: predicted IB vs DB (no-calls excluded)
  folfoxai_forest      — per-arm unadjusted HR + pooled tx × class interaction

The analysis is **entirely self-contained** to the FOLFOXai model — it
does not reference the latent-feature prognostic Cox from the CATE
pipeline.

Usage:
    python3 3_plot_panels.py
"""

import os
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test
from matplotlib.ticker import FixedLocator, NullLocator, ScalarFormatter
from scipy.stats import chi2

from _plot_style import (
    DPI, FIG_H_IN, FIG_W_IN, apply_nature_style, save_panel,
)

warnings.filterwarnings('ignore')


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))

PRED_CSV    = os.path.join(_THIS_DIR, 'causal_inference_results',
                            'predictions', 'folfoxai_predictions.csv')
FIGURES_DIR = os.path.join(_THIS_DIR, 'causal_inference_results',
                            'paper_figures')

NO_CALL_LO, NO_CALL_HI = 0.47, 0.53
COLOR_IB        = '#2ca02c'      # green: predicted Increased Benefit
COLOR_DB        = '#d62728'      # red:   predicted Decreased Benefit
COLOR_FOLFOX    = '#1f77b4'
COLOR_FOLFIRI   = '#ff7f0e'
SAVE_PDF        = True

# TESSERA-canonical endpoint construction
# (mirrors scripts/predictive_bm/core/cohorts.py:_pfs_time_event and _is_dead).
DAYS_PER_MO      = 30.4375
PFS_HORIZON_MO   = 36.0
OS_HORIZON_MO    = 60.0
GROUND_TRUTH_CSV = os.path.abspath(os.path.join(
    _THIS_DIR, '..', '..', 'data', 'msk_chord_2024',
    'GROUND_TRUTH_CRC_FOLFOX_FOLFIRI_STAGE4_TTNTD.csv'))


def _build_endpoints(pred_df):
    """Re-derive PFS (36mo) and OS (60mo) per TESSERA-canonical convention.

    Reads ``GROUND_TRUTH_CSV``, restricts to 1L FOLFOX/FOLFIRI, applies the
    admin-censoring rules used by ``scripts/predictive_bm/core/cohorts.py``,
    and left-joins ``pfs_t``, ``pfs_e``, ``os_t``, ``os_e`` onto
    ``pred_df`` by PATIENT_ID.

    PFS: from DAYS_TO_PROGRESSION, 36mo horizon. NaN DAYS_TO_PROGRESSION
    rows get NaN pfs_t/pfs_e and are excluded by downstream filters.
    OS:  from OS_MONTHS_FROM_TREATMENT_START + OS_STATUS, 60mo horizon.
    """
    g = pd.read_csv(GROUND_TRUTH_CSV, low_memory=False)
    g = g[g['LINE_OF_THERAPY'] == 1].copy()
    g = g[g['REGIMEN'].isin(['FOLFOX', 'FOLFIRI'])].copy()

    pfs_days = pd.to_numeric(g['DAYS_TO_PROGRESSION'], errors='coerce')
    horizon_d = PFS_HORIZON_MO * DAYS_PER_MO
    pfs_t = np.minimum(pfs_days, horizon_d) / DAYS_PER_MO
    pfs_e = ((pfs_days.notna()) & (pfs_days <= horizon_d)).astype('Int64')
    pfs_t = pfs_t.where(pfs_days.notna(), np.nan)
    pfs_e = pfs_e.where(pfs_days.notna(), pd.NA)
    g['pfs_t'], g['pfs_e'] = pfs_t.values, pfs_e.values

    os_mo = pd.to_numeric(g['OS_MONTHS_FROM_TREATMENT_START'], errors='coerce')
    def _is_dead(s):
        up = str(s).upper()
        return int('DECEASED' in up or up.startswith('1:') or up == '1')
    os_dead = g['OS_STATUS'].apply(_is_dead)
    os_t = np.minimum(os_mo, OS_HORIZON_MO)
    os_e = ((os_mo.notna()) & (os_mo <= OS_HORIZON_MO) & (os_dead == 1)).astype('Int64')
    os_t = os_t.where(os_mo.notna(), np.nan)
    os_e = os_e.where(os_mo.notna(), pd.NA)
    g['os_t'], g['os_e'] = os_t.values, os_e.values

    # Mirror script 1's per-patient dedup: earliest survival, keep first.
    g = g.sort_values('OS_MONTHS_FROM_TREATMENT_START').drop_duplicates(
        'PATIENT_ID', keep='first')
    endpoints = g[['PATIENT_ID', 'pfs_t', 'pfs_e', 'os_t', 'os_e']]
    merged = pred_df.merge(endpoints, on='PATIENT_ID', how='left')
    return merged


# ---------------------------------------------------------------------------

def plot_histogram(df, panel_dir):
    os.makedirs(panel_dir, exist_ok=True)

    fox = df.loc[df['REGIMEN'] == 'FOLFOX',  'p_IB'].dropna()
    fir = df.loc[df['REGIMEN'] == 'FOLFIRI', 'p_IB'].dropna()

    p_all  = df['p_IB'].dropna()
    xlo    = max(0.0, float(p_all.min()) - 0.02)
    xhi    = min(1.0, float(p_all.max()) + 0.02)
    bins   = np.linspace(xlo, xhi, 41)

    fig, ax = plt.subplots(figsize=(FIG_W_IN, FIG_H_IN))
    ax.hist(fox, bins=bins, color=COLOR_FOLFOX, alpha=0.55,
            edgecolor='black', linewidth=0.3)
    ax.hist(fir, bins=bins, color=COLOR_FOLFIRI, alpha=0.55,
            edgecolor='black', linewidth=0.3)
    ax.axvspan(NO_CALL_LO, NO_CALL_HI, color='gray', alpha=0.22)
    ax.axvline(0.5, color='black', lw=0.6, ls='--')

    ax.set_xlim(xlo, xhi)
    ax.set_xlabel('p_IB (FOLFOXai probability of IB)')
    ax.set_ylabel('Patients')
    ax.set_title('FOLFOXai p_IB distribution by arm')
    ax.set_box_aspect(1)

    # In-axes arm labels (color-coded text, no legend box — avoids overlap
    # with the histogram peaks around 0.5-0.55).
    ax.text(0.98, 0.97, f'FOLFOX  n={len(fox):,}',
            transform=ax.transAxes, fontsize=7, ha='right', va='top',
            color=COLOR_FOLFOX, fontweight='bold')
    ax.text(0.98, 0.90, f'FOLFIRI n={len(fir):,}',
            transform=ax.transAxes, fontsize=7, ha='right', va='top',
            color=COLOR_FOLFIRI, fontweight='bold')

    save_panel(fig, os.path.join(panel_dir, 'panel'), save_pdf=SAVE_PDF)
    plt.close(fig)

    rows = []
    for arm, s in [('FOLFOX', fox), ('FOLFIRI', fir), ('POOLED', df['p_IB'].dropna())]:
        n_ib = int((s >= NO_CALL_HI).sum())
        n_db = int((s <= NO_CALL_LO).sum())
        n_nc = int(((s > NO_CALL_LO) & (s < NO_CALL_HI)).sum())
        rows.append({
            'arm':  arm,
            'n':    int(len(s)),
            'mean': float(s.mean()),   'median': float(s.median()),
            'std':  float(s.std()),
            'n_predicted_IB': n_ib, 'n_predicted_DB': n_db,
            'n_no_call':      n_nc,
        })
    pd.DataFrame(rows).to_csv(os.path.join(panel_dir, 'histogram_stats.csv'),
                              index=False)


def plot_km_class(df, panel_dir, subtitle, arm):
    """KM on predicted-IB vs predicted-DB patients (no-calls excluded)."""
    os.makedirs(panel_dir, exist_ok=True)
    sub = df[(df['REGIMEN'] == arm) & df['predicted_class'].notna()].copy()
    sub['event_cens'] = sub['event_cens'].astype(int)

    ib = sub[sub['predicted_class'] == 1]
    db = sub[sub['predicted_class'] == 0]
    lr = logrank_test(ib['T'], db['T'], ib['event_cens'], db['event_cens'])

    cph = CoxPHFitter()
    cph.fit(sub[['T', 'event_cens', 'predicted_class']]
            .assign(predicted_class=sub['predicted_class'].astype(int)),
            duration_col='T', event_col='event_cens')
    hr = float(np.exp(cph.params_['predicted_class']))
    ci = cph.confidence_intervals_
    hr_lo = float(np.exp(ci.loc['predicted_class', '95% lower-bound']))
    hr_hi = float(np.exp(ci.loc['predicted_class', '95% upper-bound']))
    p_cox = float(cph.summary.loc['predicted_class', 'p'])

    fig, ax = plt.subplots(figsize=(FIG_W_IN, FIG_H_IN))
    for grp, color, lab in [
        (ib, COLOR_IB, f'predicted IB  n={len(ib)}'),
        (db, COLOR_DB, f'predicted DB  n={len(db)}'),
    ]:
        kmf = KaplanMeierFitter().fit(grp['T'], grp['event_cens'], label=lab)
        kmf.plot_survival_function(ax=ax, ci_show=True, ci_alpha=0.15,
                                   color=color, linewidth=1.3)

    ax.set_xlim(0, 60)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel('Overall survival (months)')
    ax.set_ylabel('Survival probability')
    ax.set_box_aspect(1)
    p_txt = (f'logrank P = {lr.p_value:.1e}' if lr.p_value < 1e-3
             else f'logrank P = {lr.p_value:.3f}')
    ax.text(0.03, 0.05,
            f'{p_txt}\nHR (IB vs DB) = {hr:.2f} ({hr_lo:.2f}-{hr_hi:.2f})',
            transform=ax.transAxes, fontsize=7, ha='left', va='bottom')
    ax.set_title(subtitle)
    ax.legend(loc='upper right', handlelength=1.2, borderpad=0.3,
              labelspacing=0.3)
    save_panel(fig, os.path.join(panel_dir, 'panel'), save_pdf=SAVE_PDF)
    plt.close(fig)

    pd.DataFrame([{
        'arm': arm, 'subtitle': subtitle,
        'n_total': int(len(sub)),
        'n_ib': int(len(ib)), 'n_db': int(len(db)),
        'events_ib': int(ib['event_cens'].sum()),
        'events_db': int(db['event_cens'].sum()),
        'hr_ib_vs_db': hr, 'hr_ci_lo': hr_lo, 'hr_ci_hi': hr_hi,
        'cox_p': p_cox, 'logrank_p': float(lr.p_value),
    }]).to_csv(os.path.join(panel_dir, 'km_stats.csv'), index=False)

    return {'n_ib': len(ib), 'n_db': len(db),
            'hr': hr, 'p_cox': p_cox, 'p_logrank': float(lr.p_value)}


def plot_forest(df, panel_dir):
    os.makedirs(panel_dir, exist_ok=True)
    sub = df[df['predicted_class'].notna()].copy()
    sub['event_cens']      = sub['event_cens'].astype(int)
    sub['predicted_class'] = sub['predicted_class'].astype(int)
    sub['treatment']       = (sub['REGIMEN'] == 'FOLFOX').astype(int)

    per_arm = {}
    forest_rows = []
    for arm in ['FOLFOX', 'FOLFIRI']:
        arm_df = sub[sub['REGIMEN'] == arm].copy()
        n, ev = len(arm_df), int(arm_df['event_cens'].sum())
        cph = CoxPHFitter()
        cph.fit(arm_df[['T', 'event_cens', 'predicted_class']],
                duration_col='T', event_col='event_cens')
        b  = float(cph.params_['predicted_class'])
        ci = cph.confidence_intervals_
        hr    = float(np.exp(b))
        hr_lo = float(np.exp(ci.loc['predicted_class', '95% lower-bound']))
        hr_hi = float(np.exp(ci.loc['predicted_class', '95% upper-bound']))
        p     = float(cph.summary.loc['predicted_class', 'p'])
        per_arm[arm] = dict(n=n, events=ev,
                            hr=hr, hr_lo=hr_lo, hr_hi=hr_hi, p=p)
        forest_rows.append({
            'arm': arm, 'n': n, 'events': ev,
            'hr_ib_vs_db': hr, 'hr_ci_lo': hr_lo, 'hr_ci_hi': hr_hi, 'p_wald': p,
        })

    # Pooled interaction (no prog covariate — self-contained)
    cox_a = CoxPHFitter()
    cox_a.fit(sub[['T', 'event_cens', 'treatment', 'predicted_class']],
              duration_col='T', event_col='event_cens')
    sub2 = sub.copy()
    sub2['tx_class'] = sub2['treatment'] * sub2['predicted_class']
    cox_b = CoxPHFitter()
    cox_b.fit(sub2[['T', 'event_cens', 'treatment', 'predicted_class', 'tx_class']],
              duration_col='T', event_col='event_cens')
    lrt_chi2 = float(2 * (cox_b.log_likelihood_ - cox_a.log_likelihood_))
    lrt_p    = float(1 - chi2.cdf(lrt_chi2, df=1))
    b_int    = float(cox_b.params_['tx_class'])
    ci       = cox_b.confidence_intervals_
    hr_int    = float(np.exp(b_int))
    hr_int_lo = float(np.exp(ci.loc['tx_class', '95% lower-bound']))
    hr_int_hi = float(np.exp(ci.loc['tx_class', '95% upper-bound']))
    p_int     = float(cox_b.summary.loc['tx_class', 'p'])

    fig, ax = plt.subplots(figsize=(FIG_W_IN, FIG_H_IN))
    for arm, y_pos, color in [('FOLFOX', 1, COLOR_FOLFOX),
                              ('FOLFIRI', 0, COLOR_FOLFIRI)]:
        a = per_arm[arm]
        ax.errorbar(
            a['hr'], y_pos,
            xerr=[[a['hr'] - a['hr_lo']], [a['hr_hi'] - a['hr']]],
            fmt='o', color=color, ecolor=color, capsize=3, markersize=6,
            markerfacecolor=color, markeredgecolor='black',
            markeredgewidth=0.5, elinewidth=1.0,
        )
        ax.text(0.02, y_pos + 0.32,
                f"{arm}  (n={a['n']})",
                transform=ax.get_yaxis_transform(),
                fontsize=7, ha='left', va='bottom',
                color=color, fontweight='bold')
        ax.text(0.02, y_pos + 0.16,
                f"HR {a['hr']:.2f}  ({a['hr_lo']:.2f}-{a['hr_hi']:.2f})  "
                f"p={a['p']:.2g}",
                transform=ax.get_yaxis_transform(),
                fontsize=7, ha='left', va='bottom')

    ax.axvline(1.0, color='black', linestyle='--', linewidth=0.5)
    ax.set_xscale('log')
    hi_hrs = [per_arm[a]['hr_hi'] for a in ['FOLFOX', 'FOLFIRI']]
    lo_hrs = [per_arm[a]['hr_lo'] for a in ['FOLFOX', 'FOLFIRI']]
    xmin = min(0.4, min(lo_hrs) * 0.8)
    xmax = max(2.0, max(hi_hrs) * 1.2)
    ax.set_xlim(xmin, xmax)
    ticks = [t for t in [0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0] if xmin <= t <= xmax]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_minor_locator(NullLocator())
    fmt = ScalarFormatter()
    fmt.set_scientific(False)
    ax.xaxis.set_major_formatter(fmt)
    ax.set_ylim(-0.7, 1.7)
    ax.set_yticks([])
    ax.set_xlabel('HR (predicted IB vs DB)')
    ax.spines['left'].set_visible(False)
    ax.set_title('FOLFOXai class effect on OS by arm')
    ax.set_box_aspect(1)

    fig.savefig(os.path.join(panel_dir, 'panel.png'),
                dpi=DPI, bbox_inches=None, pad_inches=0.1)
    if SAVE_PDF:
        fig.savefig(os.path.join(panel_dir, 'panel.pdf'),
                    bbox_inches=None, pad_inches=0.1)
    plt.close(fig)

    pd.DataFrame(forest_rows).to_csv(
        os.path.join(panel_dir, 'forest_rows.csv'), index=False)
    pd.DataFrame([{
        'n': int(len(sub)), 'events': int(sub['event_cens'].sum()),
        'n_folfox':  int((sub['treatment'] == 1).sum()),
        'n_folfiri': int((sub['treatment'] == 0).sum()),
        'hr_interaction':     hr_int,
        'hr_int_ci_lo':       hr_int_lo,
        'hr_int_ci_hi':       hr_int_hi,
        'p_interaction_wald': p_int,
        'lrt_chi2':           lrt_chi2, 'lrt_df': 1, 'lrt_p': lrt_p,
        'hr_folfox':    per_arm['FOLFOX']['hr'],
        'hr_folfox_lo': per_arm['FOLFOX']['hr_lo'],
        'hr_folfox_hi': per_arm['FOLFOX']['hr_hi'],
        'p_folfox':     per_arm['FOLFOX']['p'],
        'hr_folfiri':    per_arm['FOLFIRI']['hr'],
        'hr_folfiri_lo': per_arm['FOLFIRI']['hr_lo'],
        'hr_folfiri_hi': per_arm['FOLFIRI']['hr_hi'],
        'p_folfiri':     per_arm['FOLFIRI']['p'],
    }]).to_csv(os.path.join(panel_dir, 'interaction_stats.csv'), index=False)

    return per_arm, (hr_int, p_int, lrt_p)


def plot_km_two_groups(
    df, panel_dir, subtitle,
    *,
    group_col, group_pos_val, group_neg_val,
    group_pos_label, group_neg_label,
    color_pos, color_neg,
    time_col, event_col,
    xlabel, xlim_hi,
    ax=None,
):
    """Generic two-group KM with Cox HR + 95% CI + logrank P.

    Standalone mode (``ax is None``): writes ``panel.png``, ``panel.pdf``,
    and ``km_stats.csv`` into ``panel_dir``.
    Grid mode (``ax`` provided): renders into the supplied axis only and
    returns the stats dict.

    Manuscript Fig 6c uses ``group_col='predicted_class'`` (1 vs 0);
    Fig 6d/e use ``group_col='REGIMEN'`` (FOLFOX vs FOLFIRI) after the
    cohort is filtered to one predicted stratum.
    """
    sub = df[
        (df[group_col] == group_pos_val) | (df[group_col] == group_neg_val)
    ].copy()
    sub = sub.dropna(subset=[time_col, event_col]).copy()
    sub[event_col] = sub[event_col].astype(int)
    sub['_group'] = (sub[group_col] == group_pos_val).astype(int)

    pos = sub[sub['_group'] == 1]
    neg = sub[sub['_group'] == 0]

    lr = logrank_test(pos[time_col], neg[time_col],
                      pos[event_col], neg[event_col])
    cph = CoxPHFitter().fit(
        sub[[time_col, event_col, '_group']],
        duration_col=time_col, event_col=event_col,
    )
    hr = float(np.exp(cph.params_['_group']))
    ci = cph.confidence_intervals_
    hr_lo = float(np.exp(ci.loc['_group', '95% lower-bound']))
    hr_hi = float(np.exp(ci.loc['_group', '95% upper-bound']))
    p_cox = float(cph.summary.loc['_group', 'p'])

    standalone = ax is None
    if standalone:
        os.makedirs(panel_dir, exist_ok=True)
        fig, ax = plt.subplots(figsize=(FIG_W_IN, FIG_H_IN))

    for grp, color, lab in [
        (pos, color_pos, f'{group_pos_label}  n={len(pos)}'),
        (neg, color_neg, f'{group_neg_label}  n={len(neg)}'),
    ]:
        if len(grp) == 0:
            continue
        kmf = KaplanMeierFitter().fit(grp[time_col], grp[event_col], label=lab)
        kmf.plot_survival_function(
            ax=ax, ci_show=True, ci_alpha=0.15,
            color=color, linewidth=1.3,
        )

    ax.set_xlim(0, xlim_hi)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Survival probability')
    ax.set_box_aspect(1)
    p_txt = (f'logrank P = {lr.p_value:.1e}' if lr.p_value < 1e-3
             else f'logrank P = {lr.p_value:.3f}')
    hr_txt = (f'HR ({group_pos_label} vs {group_neg_label}) = '
              f'{hr:.2f} ({hr_lo:.2f}-{hr_hi:.2f})')
    ax.text(0.03, 0.05, f'{p_txt}\n{hr_txt}',
            transform=ax.transAxes, fontsize=7, ha='left', va='bottom')
    ax.set_title(subtitle)
    ax.legend(loc='upper right', handlelength=1.2, borderpad=0.3,
              labelspacing=0.3)

    stats = {
        'subtitle': subtitle,
        'n_total': int(len(sub)),
        'n_pos': int(len(pos)), 'n_neg': int(len(neg)),
        'events_pos': int(pos[event_col].sum()),
        'events_neg': int(neg[event_col].sum()),
        'hr_pos_vs_neg': hr,
        'hr_ci_lo': hr_lo, 'hr_ci_hi': hr_hi,
        'cox_p': p_cox,
        'logrank_p': float(lr.p_value),
        'time_col': time_col, 'event_col': event_col,
        'group_col': group_col,
        'group_pos': str(group_pos_val),
        'group_neg': str(group_neg_val),
    }

    if standalone:
        save_panel(fig, os.path.join(panel_dir, 'panel'), save_pdf=SAVE_PDF)
        plt.close(fig)
        pd.DataFrame([stats]).to_csv(
            os.path.join(panel_dir, 'km_stats.csv'), index=False)

    return stats


# ---------------------------------------------------------------------------

def main():
    apply_nature_style()
    os.makedirs(FIGURES_DIR, exist_ok=True)

    pred = pd.read_csv(PRED_CSV)
    print('=' * 72)
    print(f'merged n={len(pred)}  (no-call total={int(pred["no_call"].sum())})')
    print('=' * 72)

    # --- histogram ---
    panel_dir = os.path.join(FIGURES_DIR, 'folfoxai_histogram')
    plot_histogram(pred, panel_dir)
    print(f'  folfoxai_histogram   saved (n={len(pred)})')

    # --- FOLFOX KM (IB vs DB) ---
    for arm, fid, subtitle in [
        ('FOLFOX',  'folfoxai_km_folfox',  'FOLFOX arm: OS - predicted IB vs DB'),
        ('FOLFIRI', 'folfoxai_km_folfiri', 'FOLFIRI arm: OS - predicted IB vs DB'),
    ]:
        panel_dir = os.path.join(FIGURES_DIR, fid)
        stats = plot_km_class(pred, panel_dir, subtitle, arm)
        print(f'  {fid:28s}  IB n={stats["n_ib"]:4d}  DB n={stats["n_db"]:4d}  '
              f'HR={stats["hr"]:.2f}  logrank p={stats["p_logrank"]:.3g}')

    # --- Forest + interaction (unadjusted, self-contained) ---
    panel_dir = os.path.join(FIGURES_DIR, 'folfoxai_forest')
    per_arm, (hr_int, p_int, lrt_p) = plot_forest(pred, panel_dir)
    print(f'  folfoxai_forest          '
          f"FOLFOX HR={per_arm['FOLFOX']['hr']:.3f} p={per_arm['FOLFOX']['p']:.3g}   "
          f"FOLFIRI HR={per_arm['FOLFIRI']['hr']:.3f} p={per_arm['FOLFIRI']['p']:.3g}   "
          f'interaction HR={hr_int:.3f}  Wald p={p_int:.3g}  LRT p={lrt_p:.3g}')

    # --- Manuscript Fig 6 c/d/e analogues (PFS row + OS row) ---
    print('-' * 72)
    print('Building manuscript Fig 6 c/d/e analogues (PFS + OS)')
    pred_ep = _build_endpoints(pred)
    n_pfs = int(pred_ep['pfs_t'].notna().sum())
    n_os  = int(pred_ep['os_t'].notna().sum())
    print(f'  endpoints attached: PFS-available={n_pfs:,}  '
          f'OS-available={n_os:,}  total={len(pred_ep):,}')

    sub = pred_ep[pred_ep['predicted_class'].notna()].copy()
    sub['predicted_class'] = sub['predicted_class'].astype(int)

    panel_specs = [
        # (subdir, filter, group_col, pos_val, neg_val, pos_lab, neg_lab,
        #  color_pos, color_neg, time_col, event_col, xlabel, xlim_hi,
        #  subtitle, description)
        ('folfoxai_pooled_prognostic_pfs',
         lambda d: d[d['pfs_t'].notna()],
         'predicted_class', 1, 0, 'predicted IB', 'predicted DB',
         COLOR_IB, COLOR_DB, 'pfs_t', 'pfs_e', 'PFS (months)', 36,
         'Pooled cohort PFS: predicted IB vs DB',
         'Manuscript Fig 6c analogue (PFS). Pooled FOLFOX+FOLFIRI KM split '
         'by FOLFOXai predicted class (no-call excluded), PFS 36mo admin '
         'censor. Tests whether the FOLFOXai score carries prognostic '
         'signal independent of treatment arm.'),
        ('folfoxai_ib_stratum_by_arm_pfs',
         lambda d: d[(d['predicted_class'] == 1) & d['pfs_t'].notna()],
         'REGIMEN', 'FOLFOX', 'FOLFIRI', 'FOLFOX', 'FOLFIRI',
         COLOR_FOLFOX, COLOR_FOLFIRI, 'pfs_t', 'pfs_e', 'PFS (months)', 36,
         'Predicted-IB stratum PFS: FOLFOX vs FOLFIRI',
         'Manuscript Fig 6d analogue (PFS). Within FOLFOXai-predicted-IB '
         'patients, KM of FOLFOX arm vs FOLFIRI arm. Tests whether '
         'IB-predicted patients fare better on the FOLFOX arm.'),
        ('folfoxai_db_stratum_by_arm_pfs',
         lambda d: d[(d['predicted_class'] == 0) & d['pfs_t'].notna()],
         'REGIMEN', 'FOLFOX', 'FOLFIRI', 'FOLFOX', 'FOLFIRI',
         COLOR_FOLFOX, COLOR_FOLFIRI, 'pfs_t', 'pfs_e', 'PFS (months)', 36,
         'Predicted-DB stratum PFS: FOLFOX vs FOLFIRI',
         'Manuscript Fig 6e analogue (PFS). Within FOLFOXai-predicted-DB '
         'patients, KM of FOLFOX arm vs FOLFIRI arm. Tests direction '
         'reversal — whether DB-predicted patients fare better on FOLFIRI.'),
        ('folfoxai_pooled_prognostic_os',
         lambda d: d[d['os_t'].notna()],
         'predicted_class', 1, 0, 'predicted IB', 'predicted DB',
         COLOR_IB, COLOR_DB, 'os_t', 'os_e', 'OS (months)', 60,
         'Pooled cohort OS: predicted IB vs DB',
         'Manuscript Fig 6c analogue (OS). Pooled cohort KM split by '
         'FOLFOXai predicted class, OS 60mo admin censor.'),
        ('folfoxai_ib_stratum_by_arm_os',
         lambda d: d[(d['predicted_class'] == 1) & d['os_t'].notna()],
         'REGIMEN', 'FOLFOX', 'FOLFIRI', 'FOLFOX', 'FOLFIRI',
         COLOR_FOLFOX, COLOR_FOLFIRI, 'os_t', 'os_e', 'OS (months)', 60,
         'Predicted-IB stratum OS: FOLFOX vs FOLFIRI',
         'Manuscript Fig 6d analogue (OS).'),
        ('folfoxai_db_stratum_by_arm_os',
         lambda d: d[(d['predicted_class'] == 0) & d['os_t'].notna()],
         'REGIMEN', 'FOLFOX', 'FOLFIRI', 'FOLFOX', 'FOLFIRI',
         COLOR_FOLFOX, COLOR_FOLFIRI, 'os_t', 'os_e', 'OS (months)', 60,
         'Predicted-DB stratum OS: FOLFOX vs FOLFIRI',
         'Manuscript Fig 6e analogue (OS).'),
    ]

    for spec in panel_specs:
        (subdir, filt, group_col, pos_v, neg_v, pos_l, neg_l,
         c_pos, c_neg, t_col, e_col, xlabel, xlim_hi,
         subtitle, _desc) = spec
        panel_dir = os.path.join(FIGURES_DIR, subdir)
        stats = plot_km_two_groups(
            filt(sub), panel_dir, subtitle,
            group_col=group_col,
            group_pos_val=pos_v, group_neg_val=neg_v,
            group_pos_label=pos_l, group_neg_label=neg_l,
            color_pos=c_pos, color_neg=c_neg,
            time_col=t_col, event_col=e_col,
            xlabel=xlabel, xlim_hi=xlim_hi,
        )
        print(f"  {subdir:42s} n_pos={stats['n_pos']:4d}  "
              f"n_neg={stats['n_neg']:4d}  "
              f"HR={stats['hr_pos_vs_neg']:.2f} "
              f"({stats['hr_ci_lo']:.2f}-{stats['hr_ci_hi']:.2f})  "
              f"cox_p={stats['cox_p']:.3g}")

    # Composite 2x3 grid (row 1 PFS, row 2 OS)
    grid_dir = os.path.join(FIGURES_DIR, 'folfoxai_fig6_grid')
    os.makedirs(grid_dir, exist_ok=True)
    fig_grid, axes = plt.subplots(
        2, 3, figsize=(FIG_W_IN * 3.2, FIG_H_IN * 2.2),
        constrained_layout=True)
    for i, spec in enumerate(panel_specs):
        row, col = divmod(i, 3)
        (_, filt, group_col, pos_v, neg_v, pos_l, neg_l,
         c_pos, c_neg, t_col, e_col, xlabel, xlim_hi,
         subtitle, _) = spec
        plot_km_two_groups(
            filt(sub), panel_dir=None, subtitle=subtitle,
            group_col=group_col,
            group_pos_val=pos_v, group_neg_val=neg_v,
            group_pos_label=pos_l, group_neg_label=neg_l,
            color_pos=c_pos, color_neg=c_neg,
            time_col=t_col, event_col=e_col,
            xlabel=xlabel, xlim_hi=xlim_hi,
            ax=axes[row, col],
        )
    fig_grid.suptitle(
        'FOLFOXai — manuscript Fig 6 c/d/e analogues '
        '(top: PFS 36mo;  bottom: OS 60mo)', fontsize=9)
    fig_grid.savefig(os.path.join(grid_dir, 'panel.png'),
                     dpi=DPI, bbox_inches=None, pad_inches=0.1)
    if SAVE_PDF:
        fig_grid.savefig(os.path.join(grid_dir, 'panel.pdf'),
                         bbox_inches=None, pad_inches=0.1)
    plt.close(fig_grid)
    print('  folfoxai_fig6_grid                       composite 2x3 saved')

    print('=' * 72)
    print(f'outputs: {FIGURES_DIR}/')


if __name__ == '__main__':
    main()
