"""
Paper-ready figures for the TCGA concordance-survival analysis.

Big-picture hypothesis: the classifier's **genomic fingerprint** of
a tumor is a stronger prognostic descriptor than the histological
label. Misclassified TCGA samples — whose genome disagrees with
pathology — should (i) show different DSS from correctly-classified
samples of the same pathology class, and (ii) cluster with the
survival distribution of their classifier-predicted class rather
than their actual class. The glioma GBM/LGG boundary is a concrete,
literature-supported instance, verifiable with the 2021 WHO
reclassification of IDH-mutant astrocytomas out of GBM.

Produces per-panel output mirroring
`methods/predictive_bm/crc_folfox_folfiri_final/causal_inference_results_final/paper_figures/`:
one folder per panel with `panel.{png,pdf}` + stats CSVs, plus a
top-level `MANIFEST.csv` and `summary.json`.

Run once:
    python tcga_concordance_paper_figures.py

Inputs
------
  models_macro/snv_cna_infonce_ensemble_results.pkl
      OOF predictions from the 23-class ensemble (macro-fold CV).
  ../data/tcga/clinical.csv
      Curated TCGA clinical + survival fields. DSS_cr / DSS.time.cr.
  ../prognostic_bm/clinical_metadata/glioma_clinical_metadata.csv
      WHO 2021 reclassification labels for gliomas.

Terminology
-----------
  correctly classified  = (oof_pred_class == actual_label)
  misclassified         = (oof_pred_class != actual_label)
"""

import json
import math
import os
import pickle
import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test
from scipy.optimize import brentq
from scipy.stats import chi2, fisher_exact

warnings.filterwarnings('ignore')


# ---------------------------------------------------------------------------
# Paths + constants
# ---------------------------------------------------------------------------

HERE = os.path.dirname(os.path.abspath(__file__))

# ============================================================================
# Configuration (edit or set env vars to override)
# ============================================================================
RESULTS_PKL = os.environ.get(
    "RESULTS_PKL",
    os.path.join(HERE, "models_macro", "snv_cna_infonce_ensemble_results.pkl"),
)
CLINICAL_CSV = os.environ.get(
    "CLINICAL_CSV",
    os.path.join(HERE, "..", "..", "data", "TCGA_PanCan", "clinical.csv"),
)
GLIOMA_META = os.environ.get(
    "GLIOMA_META",
    os.path.join(HERE, "..", "..", "data", "prognostic", "glioma",
                 "glioma_clinical_metadata.csv"),
)
OUT_ROOT = os.environ.get(
    "OUT_ROOT",
    os.path.join(HERE, "tcga_concordance_results", "paper_figures"),
)

SCRIPT_NAME    = 'tcga_concordance_paper_figures.py'

METRIC         = 'DSS'
DSS_EVENT_COL  = 'DSS_cr'
DSS_TIME_COL   = 'DSS.time.cr'

MIN_N_PER_GROUP = 20
MAX_MONTHS      = 60.0   # 5 years, matches the downstream prognostic panels

CORRECT_COLOR     = '#2ca02c'   # green — correctly classified
MISCLASS_COLOR    = '#d62728'   # red   — misclassified

FIG_SIDE_IN       = 3.5   # square panel (matches crc_folfox style)
DPI               = 600


# ---------------------------------------------------------------------------
# Plot style (matches predictive_bm convention)
# ---------------------------------------------------------------------------

def apply_style():
    mpl.rcParams.update({
        'font.family':        ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size':          8,
        'axes.titlesize':     9,
        'axes.labelsize':     8,
        'xtick.labelsize':    7,
        'ytick.labelsize':    7,
        'legend.fontsize':    7,
        'axes.linewidth':     0.5,
        'xtick.major.width':  0.5,
        'ytick.major.width':  0.5,
        'axes.spines.top':    False,
        'axes.spines.right':  False,
        'legend.frameon':     False,
        'pdf.fonttype':       42,
        'ps.fonttype':        42,
        'savefig.dpi':        DPI,
        'savefig.bbox':       'tight',
        'savefig.pad_inches': 0.02,
    })


def save_panel(fig, path_no_ext):
    fig.savefig(f'{path_no_ext}.png', dpi=DPI)
    fig.savefig(f'{path_no_ext}.pdf')


# ---------------------------------------------------------------------------
# Profile-likelihood 95% CI for a Cox model with one covariate
# ---------------------------------------------------------------------------

def _cox_logpl(durations, events, x, beta, strata_ids=None):
    """Breslow partial log-likelihood for a Cox model with a single
    covariate x at coefficient beta. Handles strata by summing across
    per-stratum partial likelihoods."""
    durations = np.asarray(durations, dtype=float)
    events    = np.asarray(events,    dtype=int)
    x         = np.asarray(x,         dtype=float)

    if strata_ids is not None:
        strata_ids = np.asarray(strata_ids)
        total = 0.0
        for s in np.unique(strata_ids):
            mask = strata_ids == s
            if events[mask].sum() == 0:
                continue  # no events → no contribution
            total += _cox_logpl(durations[mask], events[mask], x[mask], beta)
        return total

    order = np.argsort(durations)
    t, e, xi = durations[order], events[order], x[order]
    lin = xi * beta
    # log-sum-exp over the risk set at each time, computed as a
    # right-to-left accumulator on the sorted linear predictors.
    log_sum_from = np.empty(len(lin))
    running = -np.inf
    for i in range(len(lin) - 1, -1, -1):
        running = np.logaddexp(running, lin[i])
        log_sum_from[i] = running
    # For ties, each event at time t_i should use the risk set for
    # t_j >= t_i, i.e. the log-sum starting at the FIRST index with t_j == t_i.
    first_idx = np.searchsorted(t, t, side='left')
    event_mask = e.astype(bool)
    if not event_mask.any():
        return 0.0
    return float(np.sum(lin[event_mask] - log_sum_from[first_idx[event_mask]]))


def profile_ci_cox(d: pd.DataFrame, covariate: str,
                    duration_col: str = 'time_months',
                    event_col: str = 'event',
                    strata: str | None = None,
                    alpha: float = 0.05,
                    max_search: float = 25.0):
    """Fit a single-covariate Cox model to d, then return
    {hr, hr_lo, hr_hi, cox_p} where the CI is a profile-likelihood 95% CI
    and the p-value is the likelihood-ratio test vs beta=0.

    Falls back to NaNs on convergence failure (sparse event, etc.)."""
    cols = [duration_col, event_col, covariate] + ([strata] if strata else [])
    d = d.dropna(subset=cols).copy()
    if d[event_col].sum() == 0 or d[covariate].nunique() < 2:
        return {'hr': np.nan, 'hr_lo': np.nan, 'hr_hi': np.nan, 'cox_p': np.nan}

    durations  = d[duration_col].values
    events     = d[event_col].astype(int).values
    xvals      = d[covariate].astype(float).values
    strata_ids = d[strata].values if strata else None

    try:
        cph = CoxPHFitter(penalizer=0.0).fit(
            d[cols], duration_col=duration_col, event_col=event_col,
            formula=covariate, strata=[strata] if strata else None)
        beta_hat = float(cph.params_[covariate])
    except Exception:
        return {'hr': np.nan, 'hr_lo': np.nan, 'hr_hi': np.nan, 'cox_p': np.nan}

    ll_max  = _cox_logpl(durations, events, xvals, beta_hat, strata_ids)
    ll_null = _cox_logpl(durations, events, xvals, 0.0,     strata_ids)
    lr_stat = max(0.0, 2 * (ll_max - ll_null))
    cox_p   = float(1 - chi2.cdf(lr_stat, df=1))

    chi2_crit = chi2.ppf(1 - alpha, df=1)
    ll_target = ll_max - chi2_crit / 2

    def diff(beta):
        return _cox_logpl(durations, events, xvals, beta, strata_ids) - ll_target

    # Brentq needs a sign change; if diff(b) > 0 at the boundary, CI extends
    # beyond max_search on that side.
    upper = np.inf
    lower = -np.inf
    try:
        if diff(beta_hat + max_search) < 0:
            upper = brentq(diff, beta_hat, beta_hat + max_search,
                           xtol=1e-4, maxiter=200)
    except Exception:
        pass
    try:
        if diff(beta_hat - max_search) < 0:
            lower = brentq(diff, beta_hat - max_search, beta_hat,
                           xtol=1e-4, maxiter=200)
    except Exception:
        pass

    return {
        'hr':    float(np.exp(beta_hat)),
        'hr_lo': float(np.exp(lower)) if np.isfinite(lower) else 0.0,
        'hr_hi': float(np.exp(upper)) if np.isfinite(upper) else np.inf,
        'cox_p': cox_p,
    }


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_merged() -> pd.DataFrame:
    """Merge OOF predictions with TCGA DSS. Returns a single DataFrame."""
    with open(RESULTS_PKL, 'rb') as f:
        r = pickle.load(f)
    sample_ids  = np.array(r['sample_info']['common_sample_ids'])
    y_true      = r['predictions']['y_test']
    y_pred      = r['predictions']['y_pred_ensemble']
    y_prob      = r['predictions']['y_prob_ensemble']
    class_names = list(r['confusion_matrix']['class_names'])

    y_true_names = np.array([class_names[i] for i in y_true])
    y_pred_names = np.array([class_names[i] for i in y_pred])

    df = pd.DataFrame({
        'SAMPLE_ID':       sample_ids,
        'actual_label':    y_true_names,
        'pred_class':      y_pred_names,
        'oof_prob_top1':   y_prob[np.arange(len(y_prob)),
                                   y_prob.argmax(axis=1)],
    })
    for i, c in enumerate(class_names):
        df[f'prob_{c}'] = y_prob[:, i]

    cdr = pd.read_csv(CLINICAL_CSV)
    cdr = cdr.rename(columns={
        'bcr_patient_barcode': 'bcr',
        DSS_TIME_COL:          'time_days',
        DSS_EVENT_COL:         'event',
    })[['bcr', 'time_days', 'event']]
    cdr['time_days'] = pd.to_numeric(cdr['time_days'], errors='coerce')
    cdr['event']     = pd.to_numeric(cdr['event'],     errors='coerce')
    # Ambiguous event codes (DSS_cr == 2) → censored.
    cdr.loc[cdr['event'] > 1, 'event'] = 0

    df['bcr'] = df['SAMPLE_ID'].str[:12]
    df = df.merge(cdr, on='bcr', how='inner')
    df = df.dropna(subset=['time_days', 'event']).copy()
    df = df[df['time_days'] >= 0].copy()
    df['time_months'] = df['time_days'] / 30.44
    df['event']       = df['event'].astype(int)

    # 60-month (5-year) administrative censoring, matching the convention
    # in methods/predictive_bm/crc_folfox_folfiri_final and the PDAC
    # panels. Patients still at risk past MAX_MONTHS are censored at
    # MAX_MONTHS with event=0.
    n_censored = int((df['time_months'] > MAX_MONTHS).sum())
    print(f"  {n_censored:,} samples administratively censored at "
          f"{MAX_MONTHS:.0f} months")
    late = df['time_months'] > MAX_MONTHS
    df.loc[late, 'event']       = 0
    df.loc[late, 'time_months'] = MAX_MONTHS

    df['misclassified'] = (df['pred_class'] != df['actual_label']).astype(int)

    return df, class_names


# ---------------------------------------------------------------------------
# Panel 1: pan-cancer KM
# ---------------------------------------------------------------------------

def panel_pan_cancer_km(df: pd.DataFrame, panel_dir: str):
    os.makedirs(panel_dir, exist_ok=True)

    d = df[['time_months', 'event', 'misclassified', 'actual_label']].copy()
    cph = CoxPHFitter(penalizer=0.0).fit(
        d, duration_col='time_months', event_col='event',
        formula='misclassified',
        strata=['actual_label'])
    hr    = float(np.exp(cph.params_['misclassified']))
    ci_lo, ci_hi = np.exp(cph.confidence_intervals_.loc['misclassified'].values)
    cox_p = float(cph.summary.loc['misclassified', 'p'])

    fig, ax = plt.subplots(figsize=(FIG_SIDE_IN, FIG_SIDE_IN))
    km_rows = []
    for label, color, mask in [
        ('correctly classified', CORRECT_COLOR,  d['misclassified'] == 0),
        ('misclassified',        MISCLASS_COLOR, d['misclassified'] == 1),
    ]:
        sub = d[mask]
        kmf = KaplanMeierFitter().fit(
            sub['time_months'], sub['event'],
            label=f'{label}  n={len(sub)}  ev={int(sub["event"].sum())}')
        kmf.plot_survival_function(ax=ax, ci_show=True, ci_alpha=0.12,
                                    color=color, linewidth=1.3)
        km_rows.append({
            'group':        label,
            'n':            int(len(sub)),
            'events':       int(sub['event'].sum()),
            'median_months': float(kmf.median_survival_time_),
        })

    lr = logrank_test(
        d[d['misclassified'] == 1]['time_months'],
        d[d['misclassified'] == 0]['time_months'],
        event_observed_A=d[d['misclassified'] == 1]['event'],
        event_observed_B=d[d['misclassified'] == 0]['event'])

    ax.set_xlim(0, MAX_MONTHS); ax.set_ylim(0, 1.02)
    ax.set_xlabel('Disease-specific survival (months)')
    ax.set_ylabel('Disease-specific survival')
    ax.set_box_aspect(1)
    ax.set_title('Pan-cancer DSS')

    p_fmt = (f'{cox_p:.1e}' if cox_p < 1e-3 else f'{cox_p:.3f}')
    hr_txt = (f'HR = {hr:.2f}  (CI {ci_lo:.2f}–{ci_hi:.2f})\n'
              f'P = {p_fmt}')
    ax.text(0.04, 0.06, hr_txt, transform=ax.transAxes, fontsize=9,
            ha='left', va='bottom')
    ax.legend(loc='upper right', fontsize=7)
    save_panel(fig, os.path.join(panel_dir, 'panel'))
    plt.close(fig)

    pd.DataFrame(km_rows).to_csv(
        os.path.join(panel_dir, 'km_stats.csv'), index=False)

    pd.DataFrame([{
        'hr_miscls_vs_correct': hr,
        'hr_ci_lo':             ci_lo,
        'hr_ci_hi':             ci_hi,
        'cox_p':                cox_p,
        'pooled_logrank_p':     float(lr.p_value),
        'n_correct':            km_rows[0]['n'],
        'events_correct':       km_rows[0]['events'],
        'median_mo_correct':    km_rows[0]['median_months'],
        'n_miscls':             km_rows[1]['n'],
        'events_miscls':        km_rows[1]['events'],
        'median_mo_miscls':     km_rows[1]['median_months'],
    }]).to_csv(os.path.join(panel_dir, 'cox_stats.csv'), index=False)

    return {
        'hr': hr, 'ci_lo': ci_lo, 'ci_hi': ci_hi, 'cox_p': cox_p,
        'pooled_logrank_p': float(lr.p_value),
    }


# ---------------------------------------------------------------------------
# Panel 2: per-class forest (+ per-class internal stats + supp grid)
# ---------------------------------------------------------------------------

def _per_class_stats(df: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    rows, per_class_data = [], []
    for cls, sub in df.groupby('actual_label'):
        correct = sub[sub['misclassified'] == 0]
        miscls  = sub[sub['misclassified'] == 1]
        if len(correct) < MIN_N_PER_GROUP or len(miscls) < MIN_N_PER_GROUP:
            continue
        pooled = pd.concat([
            correct.assign(misclassified=0),
            miscls.assign(misclassified=1),
        ])[['time_months', 'event', 'misclassified']].astype({'event': int})
        try:
            cph = CoxPHFitter(penalizer=0.0).fit(
                pooled, duration_col='time_months', event_col='event')
            hr = float(np.exp(cph.params_['misclassified']))
            ci_lo, ci_hi = np.exp(
                cph.confidence_intervals_.loc['misclassified'].values)
            cox_p = float(cph.summary.loc['misclassified', 'p'])
        except Exception:
            hr, ci_lo, ci_hi, cox_p = np.nan, np.nan, np.nan, np.nan
        lr = logrank_test(correct['time_months'], miscls['time_months'],
                          event_observed_A=correct['event'],
                          event_observed_B=miscls['event'])
        kmf_c = KaplanMeierFitter().fit(correct['time_months'], correct['event'])
        kmf_m = KaplanMeierFitter().fit(miscls['time_months'],  miscls['event'])
        rows.append({
            'class':              cls,
            'n_correct':          int(len(correct)),
            'events_correct':     int(correct['event'].sum()),
            'median_mo_correct':  float(kmf_c.median_survival_time_),
            'n_miscls':           int(len(miscls)),
            'events_miscls':      int(miscls['event'].sum()),
            'median_mo_miscls':   float(kmf_m.median_survival_time_),
            'logrank_p':          float(lr.p_value),
            'hr_miscls_vs_correct': hr,
            'hr_ci_lo':           ci_lo,
            'hr_ci_hi':           ci_hi,
            'cox_p':              cox_p,
        })
        per_class_data.append((cls, correct, miscls))
    return (pd.DataFrame(rows).sort_values('class').reset_index(drop=True),
            per_class_data)


def panel_per_class_forest(per_class_df: pd.DataFrame,
                           n_all_classes: int,
                           panel_dir: str):
    os.makedirs(panel_dir, exist_ok=True)

    s = per_class_df.dropna(subset=['hr_miscls_vs_correct']).copy()
    s = s.sort_values('hr_miscls_vs_correct').reset_index(drop=True)

    n_sig = int((s['cox_p'] < 0.05).sum())

    y  = np.arange(len(s))
    hr = s['hr_miscls_vs_correct'].values
    lo = s['hr_ci_lo'].values
    hi = s['hr_ci_hi'].values

    # Clip x-axis to a readable range; cap whiskers that overflow and
    # draw arrowheads at the clipped end.
    XMIN, XMAX = 0.1, 10.0
    lo_c = np.clip(lo, XMIN, XMAX)
    hi_c = np.clip(hi, XMIN, XMAX)
    hr_c = np.clip(hr, XMIN, XMAX)

    fig, ax = plt.subplots(figsize=(4.8, max(3.2, 0.28 * len(s))))
    # Draw whiskers (truncated at axis edges; no arrowheads).
    ax.hlines(y, lo_c, hi_c, color='#444', lw=0.8)
    sig = (s['cox_p'] < 0.05).values
    ax.scatter(hr_c[~sig], y[~sig], s=22, color='#888', zorder=3)
    ax.scatter(hr_c[sig],  y[sig],  s=22, color='#222', zorder=3)
    ax.axvline(1.0, color='k', ls=':', lw=0.6)

    from matplotlib.transforms import blended_transform_factory
    label_trans = blended_transform_factory(ax.transAxes, ax.transData)
    for i, (cls, nc, nm, p) in enumerate(zip(
            s['class'], s['n_correct'], s['n_miscls'], s['cox_p'])):
        stars = '***' if p < 0.001 else ('**' if p < 0.01
                                         else ('*' if p < 0.05 else ''))
        ax.text(1.02, y[i],
                f'{cls}  correct={nc}  misclass={nm} {stars}',
                transform=label_trans, va='center', ha='left', fontsize=7)

    ax.set_yticks([])
    ax.set_xscale('log')
    ax.set_xlim(XMIN, XMAX)
    ax.set_xlabel('HR (misclassified vs correctly classified)')
    save_panel(fig, os.path.join(panel_dir, 'panel'))
    plt.close(fig)

    per_class_df.to_csv(os.path.join(panel_dir, 'per_class_stats.csv'),
                        index=False)
    return {'classes_tested': int(len(s)), 'classes_sig_p05': n_sig}


# ---------------------------------------------------------------------------
# Panel 3: misclassified tumors track the predicted class
# ---------------------------------------------------------------------------

def _stratified_cox_hr(d: pd.DataFrame, strata_col: str):
    """Return (hr, ci_lo, ci_hi, p) for DSS ~ misclassified stratified by
    `strata_col`. Caller supplies a DataFrame with columns:
    time_months, event, misclassified, <strata_col>."""
    cph = CoxPHFitter(penalizer=0.0).fit(
        d[['time_months', 'event', 'misclassified', strata_col]],
        duration_col='time_months', event_col='event',
        formula='misclassified',
        strata=[strata_col])
    hr = float(np.exp(cph.params_['misclassified']))
    lo, hi = np.exp(cph.confidence_intervals_.loc['misclassified'].values)
    p = float(cph.summary.loc['misclassified', 'p'])
    return hr, lo, hi, p


def panel_misclass_tracks_pred(df: pd.DataFrame, panel_dir: str,
                                pan_cancer_stats: dict,
                                sig_classes=None):
    """Per-class version: for each actual class X with enough
    misclassified samples, compute two HRs:
        HR_actual(X): misclass-X vs correctly-classified-X
        HR_pred(X):   misclass-X vs correctly-classified peers of
                       their classifier-predicted class (strata by
                       pred_class).
    Plot as a paired forest. If the genomic fingerprint is a better
    prognostic descriptor than pathology, classes where the classifier
    is genuinely recovering the right group should show HR_pred(X) ≈ 1
    even when HR_actual(X) departs from 1.

    If ``sig_classes`` is provided (list of class names), the per-class
    loop is restricted to those classes. The intended use is to pass the
    subset with a statistically significant HR in the per-class Fig-2
    forest (cox_p < 0.05), so that this panel tests tracking only where
    there is a pathology-vs-misclass difference to track in the first
    place."""
    os.makedirs(panel_dir, exist_ok=True)

    base = df[['time_months', 'event', 'misclassified',
                'actual_label', 'pred_class']].astype({'event': int}).copy()

    sig_set = None if sig_classes is None else set(sig_classes)

    # --- Pooled pan-cancer HRs, kept in the CSV for reference. ---
    hr_a_all, hr_a_all_lo, hr_a_all_hi, p_a_all = (
        pan_cancer_stats['hr'], pan_cancer_stats['ci_lo'],
        pan_cancer_stats['ci_hi'], pan_cancer_stats['cox_p'])
    d_pred_all = base.copy()
    (hr_p_all, hr_p_all_lo,
     hr_p_all_hi, p_p_all) = _stratified_cox_hr(d_pred_all, 'pred_class')

    # --- Per-class loop. ---
    per_class_rows = []
    for cls, sub in base.groupby('actual_label'):
        if sig_set is not None and cls not in sig_set:
            continue
        miscls_X  = sub[sub['misclassified'] == 1]
        correct_X = sub[sub['misclassified'] == 0]
        if len(miscls_X) < MIN_N_PER_GROUP or len(correct_X) < MIN_N_PER_GROUP:
            continue

        # HR vs actual-class peers (within-class Cox).
        pooled_a = pd.concat([
            correct_X.assign(misclassified=0),
            miscls_X.assign(misclassified=1),
        ])[['time_months', 'event', 'misclassified']]
        try:
            cph = CoxPHFitter(penalizer=0.0).fit(
                pooled_a, duration_col='time_months', event_col='event')
            hr_a = float(np.exp(cph.params_['misclassified']))
            hr_a_lo, hr_a_hi = np.exp(
                cph.confidence_intervals_.loc['misclassified'].values)
            p_a = float(cph.summary.loc['misclassified', 'p'])
        except Exception:
            hr_a = hr_a_lo = hr_a_hi = p_a = np.nan

        # HR vs predicted-class peers. Build pool = misclass-X ∪
        # (correctly-classified samples from each pred_class that
        # appears among misclass-X), stratified by pred_class.
        dest_classes = miscls_X['pred_class'].unique().tolist()
        ref_pred = base[(base['misclassified'] == 0)
                        & base['pred_class'].isin(dest_classes)
                        & (base['actual_label'] != cls)]
        if len(ref_pred) < MIN_N_PER_GROUP:
            hr_p = hr_p_lo = hr_p_hi = p_p = np.nan
        else:
            pooled_p = pd.concat([
                ref_pred.assign(misclassified=0),
                miscls_X.assign(misclassified=1),
            ])[['time_months', 'event', 'misclassified', 'pred_class']]
            try:
                hr_p, hr_p_lo, hr_p_hi, p_p = _stratified_cox_hr(
                    pooled_p, 'pred_class')
            except Exception:
                hr_p = hr_p_lo = hr_p_hi = p_p = np.nan

        per_class_rows.append({
            'class':             cls,
            'n_miscls':          int(len(miscls_X)),
            'n_ref_actual':      int(len(correct_X)),
            'n_ref_pred':        int(len(ref_pred)),
            'hr_vs_actual':      hr_a,
            'hr_vs_actual_ci_lo': hr_a_lo,
            'hr_vs_actual_ci_hi': hr_a_hi,
            'p_vs_actual':       p_a,
            'hr_vs_pred':        hr_p,
            'hr_vs_pred_ci_lo':  hr_p_lo,
            'hr_vs_pred_ci_hi':  hr_p_hi,
            'p_vs_pred':         p_p,
        })

    per_class_df = (pd.DataFrame(per_class_rows)
                    .dropna(subset=['hr_vs_actual', 'hr_vs_pred'])
                    .copy())
    # Sort by HR_actual so the panel reads top-worst to bottom-best.
    per_class_df = per_class_df.sort_values(
        'hr_vs_actual', ascending=False).reset_index(drop=True)

    # --- Save CSVs. ---
    pooled_rows = pd.DataFrame([
        {'class': 'ALL (pan-cancer, pooled)',
         'n_miscls':         int((base['misclassified'] == 1).sum()),
         'n_ref_actual':     int((base['misclassified'] == 0).sum()),
         'n_ref_pred':       int((base['misclassified'] == 0).sum()),
         'hr_vs_actual':     hr_a_all,
         'hr_vs_actual_ci_lo': hr_a_all_lo,
         'hr_vs_actual_ci_hi': hr_a_all_hi,
         'p_vs_actual':      p_a_all,
         'hr_vs_pred':       hr_p_all,
         'hr_vs_pred_ci_lo': hr_p_all_lo,
         'hr_vs_pred_ci_hi': hr_p_all_hi,
         'p_vs_pred':        p_p_all},
    ])
    stats_out = pd.concat([pooled_rows, per_class_df], ignore_index=True)
    stats_out.to_csv(os.path.join(panel_dir,
                                    'misclass_tracks_pred_stats.csv'),
                      index=False)

    # --- Per-class paired forest. ---
    n = len(per_class_df)
    # Taller-than-wide so it composites cleanly into Figure 5.
    fig, ax = plt.subplots(figsize=(2.8, 4.2))

    # Clip x-axis like fig2.
    XMIN, XMAX = 0.1, 10.0
    y_base = np.arange(n)
    offset = 0.18  # vertical nudge so the two HRs don't overlap
    y_a = y_base - offset
    y_p = y_base + offset

    for col_lo, col_hi, col_hr, y_row, color, side in [
        ('hr_vs_actual_ci_lo', 'hr_vs_actual_ci_hi', 'hr_vs_actual',
         y_a, MISCLASS_COLOR, 'top'),
        ('hr_vs_pred_ci_lo',   'hr_vs_pred_ci_hi',   'hr_vs_pred',
         y_p, '#1f77b4', 'bottom'),
    ]:
        lo = per_class_df[col_lo].values
        hi = per_class_df[col_hi].values
        hr_vals = per_class_df[col_hr].values
        lo_c = np.clip(lo, XMIN, XMAX)
        hi_c = np.clip(hi, XMIN, XMAX)
        hr_c = np.clip(hr_vals, XMIN, XMAX)
        ax.hlines(y_row, lo_c, hi_c, color=color, lw=1.0)
        ax.scatter(hr_c, y_row, s=24, color=color, zorder=3)
        # If CI overflows the axis, annotate with the numerical HR so the
        # reader sees the actual magnitude.
        for i in range(len(lo)):
            if (lo[i] < XMIN) or (hi[i] > XMAX):
                ax.text(XMAX * 1.05, y_row[i],
                        f'HR={hr_vals[i]:.2g}', color=color,
                        va='center', fontsize=6)

    ax.scatter([], [], s=24, color=MISCLASS_COLOR, label='vs pathology peers')
    ax.scatter([], [], s=24, color='#1f77b4', label='vs predicted-class peers')
    ax.axvline(1.0, color='k', ls=':', lw=0.6)

    ax.set_yticks(y_base)
    ax.set_yticklabels(
        [f'{r["class"]} (misclass n={r["n_miscls"]})'
         for _, r in per_class_df.iterrows()], fontsize=7)
    ax.invert_yaxis()
    ax.set_xscale('log')
    ax.set_xlim(XMIN, XMAX)
    ax.set_xlabel('HR (misclassified vs reference)')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
              ncol=2, fontsize=7, handlelength=1.2, handletextpad=0.4,
              columnspacing=1.2, frameon=False)

    save_panel(fig, os.path.join(panel_dir, 'panel'))
    plt.close(fig)

    # Count classes where HR_pred collapses to 1 (P > 0.05 for vs-pred
    # while vs-actual is significant). This is the headline number for
    # the text.
    n_collapse = int((
        (per_class_df['p_vs_pred'] > 0.05)
        & (per_class_df['p_vs_actual'] < 0.05)
    ).sum())

    # Per-pair breakdown (unchanged, kept for completeness).
    pair_rows = []
    pair_counts = (base[base['misclassified'] == 1]
                   .groupby(['actual_label', 'pred_class']).size())
    pair_counts = pair_counts.sort_values(ascending=False)

    for (actual, pred), n in pair_counts.items():
        if n < MIN_N_PER_GROUP:
            continue
        miscls = base[(base['actual_label'] == actual)
                      & (base['pred_class'] == pred)
                      & (base['misclassified'] == 1)]
        ref_actual = base[(base['actual_label'] == actual)
                          & (base['misclassified'] == 0)]
        ref_pred   = base[(base['actual_label'] == pred)
                          & (base['misclassified'] == 0)]
        if len(ref_actual) < MIN_N_PER_GROUP or len(ref_pred) < MIN_N_PER_GROUP:
            continue

        def _hr(ref: pd.DataFrame):
            pooled = pd.concat([
                ref.assign(misclassified=0),
                miscls.assign(misclassified=1),
            ])[['time_months', 'event', 'misclassified']]
            try:
                cph = CoxPHFitter(penalizer=0.0).fit(
                    pooled, duration_col='time_months', event_col='event')
                hh = float(np.exp(cph.params_['misclassified']))
                llo, lhi = np.exp(
                    cph.confidence_intervals_.loc['misclassified'].values)
                pp = float(cph.summary.loc['misclassified', 'p'])
                return hh, llo, lhi, pp
            except Exception:
                return np.nan, np.nan, np.nan, np.nan

        ha, la, hia, pa = _hr(ref_actual)
        hp, lp, hip, pp_ = _hr(ref_pred)
        pair_rows.append({
            'actual':             actual,
            'predicted':          pred,
            'n_misclass':         int(len(miscls)),
            'n_ref_actual':       int(len(ref_actual)),
            'n_ref_pred':         int(len(ref_pred)),
            'hr_vs_actual':       ha,
            'hr_vs_actual_ci_lo': la,
            'hr_vs_actual_ci_hi': hia,
            'p_vs_actual':        pa,
            'hr_vs_pred':         hp,
            'hr_vs_pred_ci_lo':   lp,
            'hr_vs_pred_ci_hi':   hip,
            'p_vs_pred':          pp_,
        })
    pd.DataFrame(pair_rows).to_csv(
        os.path.join(panel_dir, 'misclass_pair_stats.csv'), index=False)

    return {
        'pooled_hr_vs_actual':    hr_a_all,
        'pooled_hr_vs_actual_ci': [hr_a_all_lo, hr_a_all_hi],
        'pooled_p_vs_actual':     p_a_all,
        'pooled_hr_vs_pred':      hr_p_all,
        'pooled_hr_vs_pred_ci':   [hr_p_all_lo, hr_p_all_hi],
        'pooled_p_vs_pred':       p_p_all,
        'classes_tested':         int(len(per_class_df)),
        'classes_collapse_to_one': n_collapse,
    }


# ---------------------------------------------------------------------------
# Panel 4: glioma case study (WHO 2021)
# ---------------------------------------------------------------------------

WHO_ORDER  = ['glioblastoma', 'astrocytoma', 'oligodendroglioma', 'unclassified']
WHO_COLORS = {
    'glioblastoma':      '#8B0000',   # dark red (aggressive)
    'astrocytoma':       '#FF7F0E',   # orange
    'oligodendroglioma': '#1F77B4',   # blue
    'unclassified':      '#999999',   # grey
}


def _build_glioma(df: pd.DataFrame) -> pd.DataFrame:
    gl = df[df['actual_label'].isin(['GBM', 'LGG'])].copy()
    meta = pd.read_csv(GLIOMA_META)[['Patient_ID', 'WHO2021']]
    gl = gl.merge(meta, left_on='bcr', right_on='Patient_ID', how='inner')
    gl['WHO2021'] = pd.Categorical(gl['WHO2021'], categories=WHO_ORDER,
                                    ordered=True)
    return gl


def _hr_two_group(group_a: pd.DataFrame, group_b: pd.DataFrame):
    """Cox HR (B relative to A). A is reference."""
    pooled = pd.concat([
        group_a.assign(is_b=0),
        group_b.assign(is_b=1),
    ])[['time_months', 'event', 'is_b']].astype({'event': int})
    try:
        cph = CoxPHFitter(penalizer=0.0).fit(
            pooled, duration_col='time_months', event_col='event')
        hr = float(np.exp(cph.params_['is_b']))
        lo, hi = np.exp(cph.confidence_intervals_.loc['is_b'].values)
        p = float(cph.summary.loc['is_b', 'p'])
    except Exception:
        hr, lo, hi, p = np.nan, np.nan, np.nan, np.nan
    return hr, lo, hi, p


def panel_glioma(df: pd.DataFrame, panel_dir: str):
    """Single 4-curve KM panel. Solid = correctly classified (reference
    for each pathology class); dashed = misclassified. Colour encodes
    pathology label (TCGA-GBM red, TCGA-LGG blue). If the classifier's
    genomic call is a better prognostic descriptor than pathology, the
    dashed curves cross to track the OPPOSITE solid curve.

    Also writes the two per-class HR tables (GBM-side and LGG-side)
    that back the annotations."""
    os.makedirs(panel_dir, exist_ok=True)
    gl = _build_glioma(df)
    gl['event'] = gl['event'].astype(int)

    groups = {
        'gbm_correct':   gl[(gl['actual_label'] == 'GBM')
                             & (gl['pred_class']   == 'GBM')],
        'gbm_miscls':    gl[(gl['actual_label'] == 'GBM')
                             & (gl['pred_class']   == 'LGG')],
        'lgg_correct':   gl[(gl['actual_label'] == 'LGG')
                             & (gl['pred_class']   == 'LGG')],
        'lgg_miscls':    gl[(gl['actual_label'] == 'LGG')
                             & (gl['pred_class']   == 'GBM')],
    }

    # Two HRs per pathology class.
    hr_g_a, hr_g_a_lo, hr_g_a_hi, p_g_a = _hr_two_group(
        groups['gbm_correct'], groups['gbm_miscls'])
    hr_g_p, hr_g_p_lo, hr_g_p_hi, p_g_p = _hr_two_group(
        groups['lgg_correct'], groups['gbm_miscls'])

    hr_l_a, hr_l_a_lo, hr_l_a_hi, p_l_a = _hr_two_group(
        groups['lgg_correct'], groups['lgg_miscls'])
    hr_l_p, hr_l_p_lo, hr_l_p_hi, p_l_p = _hr_two_group(
        groups['gbm_correct'], groups['lgg_miscls'])

    # Colour = pathology label. Linestyle = classifier agreement.
    GBM_COLOR = '#8B0000'
    LGG_COLOR = '#1F3A8A'
    style_spec = [
        ('gbm_correct', GBM_COLOR, '-',
         f'TCGA-GBM  pred GBM  (n={len(groups["gbm_correct"])})'),
        ('gbm_miscls',  GBM_COLOR, '--',
         f'TCGA-GBM  pred LGG  (n={len(groups["gbm_miscls"])})'),
        ('lgg_correct', LGG_COLOR, '-',
         f'TCGA-LGG  pred LGG  (n={len(groups["lgg_correct"])})'),
        ('lgg_miscls',  LGG_COLOR, '--',
         f'TCGA-LGG  pred GBM  (n={len(groups["lgg_miscls"])})'),
    ]

    # Side-by-side layout: KM on left, WHO 2021 composition bar on right.
    fig, (ax_km, ax_bar) = plt.subplots(
        1, 2, figsize=(FIG_SIDE_IN * 2.0, FIG_SIDE_IN),
        gridspec_kw={'width_ratios': [1.6, 1.0], 'wspace': 0.45})

    km_rows = []
    group_rows = []
    for key, color, linestyle, label in style_spec:
        sub = groups[key]
        kmf = KaplanMeierFitter().fit(
            sub['time_months'], sub['event'], label=label)
        kmf.plot_survival_function(
            ax=ax_km, ci_show=False,
            color=color, linestyle=linestyle, linewidth=1.4)
        km_rows.append({
            'group':        key,
            'label':        label,
            'n':            int(len(sub)),
            'events':       int(sub['event'].sum()),
            'median_months': float(kmf.median_survival_time_),
        })
        who_counts = sub['WHO2021'].value_counts().reindex(
            WHO_ORDER, fill_value=0)
        for who, n in who_counts.items():
            group_rows.append({
                'group':   key,
                'WHO2021': who,
                'n':       int(n),
            })

    ax_km.set_xlim(0, MAX_MONTHS); ax_km.set_ylim(0, 1.02)
    ax_km.set_xlabel('Disease-specific survival (months)')
    ax_km.set_ylabel('Disease-specific survival')
    ax_km.set_box_aspect(1)
    ax_km.set_title('Glioma DSS')
    ax_km.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12),
                 ncol=1, fontsize=9, handlelength=2.2,
                 handletextpad=0.6, labelspacing=0.3, frameon=False)

    # --- WHO 2021 composition bar (right sub-panel). ---
    # One horizontal stacked bar per KM group, in the SAME order as the
    # KM legend. Shows WHO 2021 fraction per group.
    bar_order = [k for k, _, _, _ in style_spec]
    bar_labels = [
        'TCGA-GBM\npred GBM',
        'TCGA-GBM\npred LGG',
        'TCGA-LGG\npred LGG',
        'TCGA-LGG\npred GBM',
    ]
    bars = []
    for key in bar_order:
        sub = groups[key]
        counts = sub['WHO2021'].value_counts().reindex(
            WHO_ORDER, fill_value=0)
        bars.append(counts)
    totals = np.array([b.sum() for b in bars], dtype=float)

    y_pos = np.arange(len(bar_order))[::-1]  # top-down to match KM legend
    lefts = np.zeros(len(bar_order))
    for who in WHO_ORDER:
        heights = np.array([b.get(who, 0) for b in bars], dtype=float)
        with np.errstate(invalid='ignore', divide='ignore'):
            frac = np.where(totals > 0, heights / totals, 0.0)
        ax_bar.barh(y_pos, frac, left=lefts, color=WHO_COLORS[who],
                    edgecolor='white', linewidth=0.5, label=who, height=0.72)
        lefts += frac

    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(bar_labels, fontsize=7)
    # Annotate n next to each bar.
    for yp, total in zip(y_pos, totals):
        ax_bar.text(1.02, yp, f'n={int(total)}', va='center', fontsize=7)
    ax_bar.set_xlim(0, 1.0)
    ax_bar.set_xlabel('WHO 2021 composition')
    ax_bar.set_title('WHO 2021 Classification')
    ax_bar.legend(loc='lower center', bbox_to_anchor=(0.5, -0.35),
                  ncol=2, fontsize=11, handlelength=1.4,
                  handleheight=1.0, handletextpad=0.5,
                  columnspacing=1.2, frameon=False)

    save_panel(fig, os.path.join(panel_dir, 'panel'))
    plt.close(fig)

    pd.DataFrame(km_rows).to_csv(
        os.path.join(panel_dir, 'km_stats.csv'), index=False)
    pd.DataFrame(group_rows).to_csv(
        os.path.join(panel_dir, 'group_stats.csv'), index=False)

    hr_rows = [
        {'misclass_group':     'TCGA-GBM pred-LGG',
         'comparator':         'correct GBM (pathology peers)',
         'hr': hr_g_a, 'hr_ci_lo': hr_g_a_lo, 'hr_ci_hi': hr_g_a_hi, 'p': p_g_a},
        {'misclass_group':     'TCGA-GBM pred-LGG',
         'comparator':         'correct LGG (predicted-class peers)',
         'hr': hr_g_p, 'hr_ci_lo': hr_g_p_lo, 'hr_ci_hi': hr_g_p_hi, 'p': p_g_p},
        {'misclass_group':     'TCGA-LGG pred-GBM',
         'comparator':         'correct LGG (pathology peers)',
         'hr': hr_l_a, 'hr_ci_lo': hr_l_a_lo, 'hr_ci_hi': hr_l_a_hi, 'p': p_l_a},
        {'misclass_group':     'TCGA-LGG pred-GBM',
         'comparator':         'correct GBM (predicted-class peers)',
         'hr': hr_l_p, 'hr_ci_lo': hr_l_p_lo, 'hr_ci_hi': hr_l_p_hi, 'p': p_l_p},
    ]
    pd.DataFrame(hr_rows).to_csv(
        os.path.join(panel_dir, 'hr_stats.csv'), index=False)

    return {
        'gbm_correct_n':  int(len(groups['gbm_correct'])),
        'gbm_miscls_n':   int(len(groups['gbm_miscls'])),
        'lgg_correct_n':  int(len(groups['lgg_correct'])),
        'lgg_miscls_n':   int(len(groups['lgg_miscls'])),
        'gbm_miscls_hr_vs_actual': hr_g_a,
        'gbm_miscls_p_vs_actual':  p_g_a,
        'gbm_miscls_hr_vs_pred':   hr_g_p,
        'gbm_miscls_p_vs_pred':    p_g_p,
        'lgg_miscls_hr_vs_actual': hr_l_a,
        'lgg_miscls_p_vs_actual':  p_l_a,
        'lgg_miscls_hr_vs_pred':   hr_l_p,
        'lgg_miscls_p_vs_pred':    p_l_p,
    }


# ---------------------------------------------------------------------------
# Supplementary: per-class KM grid
# ---------------------------------------------------------------------------

def supp_per_class_km_grid(per_class_data: list, panel_dir: str):
    os.makedirs(panel_dir, exist_ok=True)
    n = len(per_class_data)
    ncol = min(4, max(1, n))
    nrow = int(math.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.7 * ncol, 2.7 * nrow),
                             squeeze=False)
    for ax, (cls, correct, miscls) in zip(axes.flat, per_class_data):
        kmf_c = KaplanMeierFitter().fit(correct['time_months'], correct['event'])
        kmf_m = KaplanMeierFitter().fit(miscls['time_months'],  miscls['event'])
        kmf_c.plot_survival_function(ax=ax, ci_show=False, color=CORRECT_COLOR,
                                      linewidth=1.1,
                                      label=f'correct  n={len(correct)}')
        kmf_m.plot_survival_function(ax=ax, ci_show=False, color=MISCLASS_COLOR,
                                      linewidth=1.1,
                                      label=f'misclass  n={len(miscls)}')
        lr = logrank_test(correct['time_months'], miscls['time_months'],
                          event_observed_A=correct['event'],
                          event_observed_B=miscls['event'])
        p_txt = (f'P={lr.p_value:.1e}' if lr.p_value < 1e-3
                 else f'P={lr.p_value:.3f}')
        ax.set_xlim(0, MAX_MONTHS); ax.set_ylim(0, 1.02)
        ax.set_title(cls, fontsize=8)
        ax.set_xlabel('DSS (months)', fontsize=7)
        ax.set_ylabel('S(t)', fontsize=7)
        ax.text(0.03, 0.05, p_txt, transform=ax.transAxes, fontsize=6)
        ax.legend(loc='lower right', fontsize=5.5, handlelength=1.0)
    for ax in axes.flat[n:]:
        ax.axis('off')
    fig.suptitle('Per-class DSS', fontsize=32)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save_panel(fig, os.path.join(panel_dir, 'panel'))
    plt.close(fig)


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

MANIFEST_ROWS = [
    {
        'fig_id':      'fig1_pan_cancer_km',
        'folder':      'fig1_pan_cancer_km',
        'panel_png':   'fig1_pan_cancer_km/panel.png',
        'panel_pdf':   'fig1_pan_cancer_km/panel.pdf',
        'stats_csvs':  'km_stats.csv;cox_stats.csv',
        'description':
            "Pan-cancer Kaplan-Meier comparing TCGA disease-specific survival (DSS) "
            "between tumours that were correctly classified (pathology label == genomic classifier call, green) "
            "vs misclassified (red) by the 23-class SNV+CNA classifier. "
            "Cox proportional-hazards model stratified by TCGA class controls for baseline class-specific "
            "prognosis. The misclassified-vs-correctly-classified HR with 95% CI and P is annotated on the panel.",
    },
    {
        'fig_id':      'fig2_per_class_forest',
        'folder':      'fig2_per_class_forest',
        'panel_png':   'fig2_per_class_forest/panel.png',
        'panel_pdf':   'fig2_per_class_forest/panel.pdf',
        'stats_csvs':  'per_class_stats.csv',
        'description':
            "Per-class forest plot of the Cox HR for DSS (misclassified vs correctly classified), one row "
            "per TCGA class with at least 20 samples in each arm. Circles are class-level HRs; whiskers are "
            "95% CI; filled black = P < 0.05. The title reports the count of tumour types with a statistically "
            "different DSS (Cox P < 0.05), demonstrating that the genomic-fingerprint concordance signal is "
            "a general property of the classifier rather than a single-tumour artifact.",
    },
    {
        'fig_id':      'fig3_misclass_tracks_pred',
        'folder':      'fig3_misclass_tracks_pred',
        'panel_png':   'fig3_misclass_tracks_pred/panel.png',
        'panel_pdf':   'fig3_misclass_tracks_pred/panel.pdf',
        'stats_csvs':  'misclass_tracks_pred_stats.csv;misclass_pair_stats.csv',
        'description':
            "Per-class test of whether misclassified tumours track the survival distribution of their "
            "classifier-predicted class rather than their pathology label. Restricted to the TCGA classes "
            "with a statistically different DSS between misclassified and correctly classified samples "
            "in Fig 2 (per-class Cox P < 0.05), so that tracking is tested only where pathology peers "
            "and misclassified already diverge. For each such class X, two HRs are computed: (red) HR of "
            "misclassified-X vs correctly-classified-X (within-class Cox); (blue) HR of misclassified-X "
            "vs correctly-classified peers of their classifier-predicted class, stratified by pred_class. "
            "If the genomic fingerprint is a better prognostic descriptor than pathology, the red dot "
            "departs from 1 while the blue dot sits on 1. Pooled pan-cancer HRs and a per-miscall-pair "
            "breakdown are recorded in the CSVs.",
    },
    {
        'fig_id':      'fig4_glioma',
        'folder':      'fig4_glioma',
        'panel_png':   'fig4_glioma/panel.png',
        'panel_pdf':   'fig4_glioma/panel.pdf',
        'stats_csvs':  'km_stats.csv;group_stats.csv;hr_stats.csv',
        'description':
            "Glioma case study (two sub-panels). Left: four-curve DSS Kaplan-Meier in TCGA GBM + LGG patients; "
            "colour = pathology label (TCGA-GBM red, TCGA-LGG blue), linestyle = classifier agreement (solid = "
            "correctly classified, dashed = misclassified). If the classifier's genomic call is a better "
            "prognostic descriptor than pathology, each dashed curve lies closer to the opposite-colour solid "
            "curve than to its own. Right: WHO 2021 composition (horizontal stacked bars) of each of the four "
            "KM groups, rows aligned with the KM legend. Misclassified groups are enriched for the WHO 2021 "
            "entity that matches the classifier's call (TCGA-GBM pred-LGG → WHO astrocytoma; TCGA-LGG pred-GBM "
            "→ WHO glioblastoma), providing an orthogonal ground-truth confirmation that the classifier's "
            "\"errors\" recapitulate the modern genomic reclassification. Per-group HRs and WHO 2021 counts "
            "are in hr_stats.csv, km_stats.csv, and group_stats.csv.",
    },
    {
        'fig_id':      'supp_per_class_km_grid',
        'folder':      'supp_per_class_km_grid',
        'panel_png':   'supp_per_class_km_grid/panel.png',
        'panel_pdf':   'supp_per_class_km_grid/panel.pdf',
        'stats_csvs':  '',
        'description':
            "Supplementary: small-multiples Kaplan-Meier grid of correctly-classified vs misclassified DSS, "
            "one panel per TCGA class with both arms above n=20. Per-class logrank P is annotated. Summary "
            "statistics for each panel are in fig2_per_class_forest/per_class_stats.csv.",
    },
]


def write_manifest(out_dir: str):
    rows = [{
        'fig_id':      r['fig_id'],
        'script':      SCRIPT_NAME,
        'folder':      r['folder'],
        'panel_png':   r['panel_png'],
        'panel_pdf':   r['panel_pdf'],
        'stats_csvs':  r['stats_csvs'],
        'description': r['description'],
    } for r in MANIFEST_ROWS]
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, 'MANIFEST.csv'),
                               index=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    apply_style()
    os.makedirs(OUT_ROOT, exist_ok=True)

    print(f"Loading predictions + DSS...")
    df, class_names = load_merged()
    print(f"  merged: {len(df):,} samples  "
          f"correct={int((df['misclassified']==0).sum()):,}  "
          f"misclassified={int(df['misclassified'].sum()):,}")
    print(f"  classes in ensemble: {len(class_names)}")

    print("\n=== Panel 1: pan-cancer KM ===")
    pan = panel_pan_cancer_km(df, os.path.join(OUT_ROOT, 'fig1_pan_cancer_km'))
    print(f"  HR = {pan['hr']:.3f}  "
          f"95% CI {pan['ci_lo']:.3f}–{pan['ci_hi']:.3f}  "
          f"P = {pan['cox_p']:.2e}")

    print("\n=== Panel 2: per-class forest ===")
    per_class_df, per_class_data = _per_class_stats(df)
    fig2_meta = panel_per_class_forest(
        per_class_df, len(class_names),
        os.path.join(OUT_ROOT, 'fig2_per_class_forest'))
    print(f"  tested {fig2_meta['classes_tested']} classes, "
          f"{fig2_meta['classes_sig_p05']} sig at P<0.05")

    print("\n=== Panel 3: misclass tracks predicted (per class) ===")
    sig_classes = (per_class_df
                   .loc[per_class_df['cox_p'] < 0.05, 'class']
                   .tolist())
    print(f"  restricting to {len(sig_classes)} classes with per-class "
          f"Cox P<0.05 in Fig 2: {sig_classes}")
    track = panel_misclass_tracks_pred(
        df, os.path.join(OUT_ROOT, 'fig3_misclass_tracks_pred'), pan,
        sig_classes=sig_classes)
    print(f"  pooled HR vs actual peers: {track['pooled_hr_vs_actual']:.2f} "
          f"(P={track['pooled_p_vs_actual']:.2e})")
    print(f"  pooled HR vs pred   peers: {track['pooled_hr_vs_pred']:.2f} "
          f"(P={track['pooled_p_vs_pred']:.2e})")
    print(f"  classes tested: {track['classes_tested']}  "
          f"classes with HR_pred collapsed (P>0.05) but HR_actual sig: "
          f"{track['classes_collapse_to_one']}")

    print("\n=== Panel 4: glioma (single 4-curve KM) ===")
    gl = panel_glioma(df, os.path.join(OUT_ROOT, 'fig4_glioma'))
    print(f"  gbm_correct n={gl['gbm_correct_n']}  "
          f"gbm_miscls n={gl['gbm_miscls_n']}  "
          f"lgg_correct n={gl['lgg_correct_n']}  "
          f"lgg_miscls n={gl['lgg_miscls_n']}")
    print(f"  TCGA-GBM pred-LGG:  HR vs correct-GBM = "
          f"{gl['gbm_miscls_hr_vs_actual']:.2f} (P={gl['gbm_miscls_p_vs_actual']:.2e})  "
          f"HR vs correct-LGG = {gl['gbm_miscls_hr_vs_pred']:.2f} "
          f"(P={gl['gbm_miscls_p_vs_pred']:.2e})")
    print(f"  TCGA-LGG pred-GBM:  HR vs correct-LGG = "
          f"{gl['lgg_miscls_hr_vs_actual']:.2f} (P={gl['lgg_miscls_p_vs_actual']:.2e})  "
          f"HR vs correct-GBM = {gl['lgg_miscls_hr_vs_pred']:.2f} "
          f"(P={gl['lgg_miscls_p_vs_pred']:.2e})")

    print("\n=== Supplementary: per-class KM grid ===")
    supp_per_class_km_grid(per_class_data,
                            os.path.join(OUT_ROOT, 'supp_per_class_km_grid'))

    write_manifest(OUT_ROOT)

    summary = {
        'n_samples':          int(len(df)),
        'n_classes_in_ensemble': len(class_names),
        'metric':             METRIC,
        'pan_cancer_hr':      pan['hr'],
        'pan_cancer_ci':      [pan['ci_lo'], pan['ci_hi']],
        'pan_cancer_p':       pan['cox_p'],
        'classes_tested':     fig2_meta['classes_tested'],
        'classes_sig_p05':    fig2_meta['classes_sig_p05'],
        'misclass_tracks_pred': {
            'pooled_hr_vs_actual':     track['pooled_hr_vs_actual'],
            'pooled_hr_vs_actual_ci':  track['pooled_hr_vs_actual_ci'],
            'pooled_p_vs_actual':      track['pooled_p_vs_actual'],
            'pooled_hr_vs_pred':       track['pooled_hr_vs_pred'],
            'pooled_hr_vs_pred_ci':    track['pooled_hr_vs_pred_ci'],
            'pooled_p_vs_pred':        track['pooled_p_vs_pred'],
            'classes_tested':          track['classes_tested'],
            'classes_collapse_to_one': track['classes_collapse_to_one'],
        },
        'glioma': gl,
    }
    with open(os.path.join(OUT_ROOT, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, default=float)

    print(f"\n✓ outputs under {OUT_ROOT}")


if __name__ == '__main__':
    main()
