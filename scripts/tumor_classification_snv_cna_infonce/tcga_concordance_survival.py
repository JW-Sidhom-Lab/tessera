"""
TCGA concordance-survival analysis.

For TCGA samples, does classifier-discordance (the OOF prediction
disagrees with the true tumor type) predict worse Disease-Specific
Survival (DSS)?

Hypothesis: misclassified tumors may have drifted genomically from
the canonical profile for their cancer type; if so, that drift should
track with worse disease trajectory even after controlling for cancer
type via stratification.

Inputs
------
  models_macro/snv_cna_infonce_ensemble_results.pkl
      - predictions['y_test']            true labels (len 9286)
      - predictions['y_pred_ensemble']   OOF predicted labels
      - predictions['y_prob_ensemble']   OOF softmax (n × 23)
      - sample_info['common_sample_ids'] TCGA barcodes
      - confusion_matrix['class_names']  23-class name array
  ../data/tcga/clinical.csv
      - bcr_patient_barcode, type, and curated survival columns:
          DSS_cr  / DSS.time.cr    (disease-specific survival)
          DFI.cr  / DFI.time.cr
          PFI.cr  / PFI.time.cr
          PFS     / PFS.time
        Events > 1 (ambiguous cause codes) are treated as censored.

Outputs (under tcga_concordance_<metric>/):
  pan_cancer_km.{png,pdf}
  pan_cancer_stats.csv
  per_class_stats.csv
  per_class_km_grid.{png,pdf}
  per_class_forest.{png,pdf}
  confidence_stratified.{png,pdf}
  confidence_stratified_stats.csv
"""

import math
import os
import pickle
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test

warnings.filterwarnings('ignore')


HERE = os.path.dirname(os.path.abspath(__file__))

# ============================================================================
# Configuration (edit or set env vars to override)
# ============================================================================
METRIC = os.environ.get("METRIC", "DSS")          # DSS | DFI | PFI | PFS
RESULTS_PKL = os.environ.get(
    "RESULTS_PKL",
    os.path.join(HERE, "models_macro", "snv_cna_infonce_ensemble_results.pkl"),
)
CLINICAL_CSV = os.environ.get(
    "CLINICAL_CSV",
    os.path.join(HERE, "..", "..", "data", "TCGA_PanCan", "clinical.csv"),
)
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "")     # blank -> tcga_concordance_<METRIC>
MIN_TOP1_PROB = float(os.environ.get("MIN_TOP1_PROB", "0.0"))
MIN_TRUE_CLASS_PROB = float(os.environ.get("MIN_TRUE_CLASS_PROB", "0.0"))

# Endpoint → (event column, time column). Curated TCGA clinical fields.
ENDPOINT_COLUMNS = {
    'DSS': ('DSS_cr', 'DSS.time.cr'),
    'DFI': ('DFI.cr', 'DFI.time.cr'),
    'PFI': ('PFI.cr', 'PFI.time.cr'),
    'PFS': ('PFS',    'PFS.time'),
}

MIN_N_PER_GROUP = 20
MAX_YEARS       = 10.0   # x-axis cap in KM, in years

CONC_COLOR = '#2ca02c'
DISC_COLOR = '#d62728'


# --- plot style (mirror pdac/crc convention) ---------------------------------

def apply_style():
    import matplotlib as mpl
    mpl.rcParams.update({
        'font.family':        ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size':          8,
        'axes.titlesize':     9,
        'axes.labelsize':     8,
        'xtick.labelsize':    7,
        'ytick.labelsize':    7,
        'legend.fontsize':    7,
        'axes.linewidth':     0.5,
        'axes.spines.top':    False,
        'axes.spines.right':  False,
        'legend.frameon':     False,
        'savefig.dpi':        600,
        'savefig.bbox':       'tight',
        'pdf.fonttype':       42,
    })


def save_panel(fig, path_no_ext):
    fig.savefig(f'{path_no_ext}.png', dpi=600)
    fig.savefig(f'{path_no_ext}.pdf')


# --- data loading ------------------------------------------------------------

def _load_predictions():
    with open(RESULTS_PKL, 'rb') as f:
        r = pickle.load(f)
    sample_ids  = np.array(r['sample_info']['common_sample_ids'])
    y_true      = r['predictions']['y_test']
    y_pred      = r['predictions']['y_pred_ensemble']
    y_prob      = r['predictions']['y_prob_ensemble']
    class_names = list(r['confusion_matrix']['class_names'])

    # Labels are integer-encoded. Map back to class names.
    y_true_names = np.array([class_names[i] for i in y_true])
    y_pred_names = np.array([class_names[i] for i in y_pred])

    df = pd.DataFrame({
        'SAMPLE_ID':            sample_ids,
        'actual_label':         y_true_names,
        'oof_pred_class':       y_pred_names,
        'oof_prob_top1':        y_prob[np.arange(len(y_prob)),
                                        y_prob.argmax(axis=1)],
    })
    for i, c in enumerate(class_names):
        df[f'prob_{c}'] = y_prob[:, i]
    return df, class_names


def _load_clinical(metric: str):
    event_col, time_col = ENDPOINT_COLUMNS[metric]
    cdr = pd.read_csv(CLINICAL_CSV)
    keep = ['bcr_patient_barcode', 'type', time_col, event_col]
    out = cdr[keep].copy()
    out = out.rename(columns={
        'bcr_patient_barcode': 'bcr',
        time_col:              'time_days',
        event_col:             'event',
    })
    out['time_days'] = pd.to_numeric(out['time_days'], errors='coerce')
    out['event']     = pd.to_numeric(out['event'],     errors='coerce')
    # Ambiguous event codes (e.g. DSS_cr == 2) treated as censored.
    n_amb = int((out['event'] > 1).sum())
    if n_amb:
        print(f"  {n_amb:,} rows with event > 1 coerced to censored (event=0)")
    out.loc[out['event'] > 1, 'event'] = 0
    return out


# --- analysis ----------------------------------------------------------------

def _pan_cancer(df: pd.DataFrame, out_no_ext):
    d = df[['time_years', 'event', 'classifier_concordant',
            'actual_label']].astype({'event': int}).copy()
    cph = CoxPHFitter(penalizer=0.0).fit(
        d, duration_col='time_years', event_col='event',
        formula='classifier_concordant',
        strata=['actual_label'])
    hr    = float(np.exp(cph.params_['classifier_concordant']))
    ci_lo, ci_hi = np.exp(cph.confidence_intervals_.loc[
        'classifier_concordant'].values)
    cox_p = float(cph.summary.loc['classifier_concordant', 'p'])

    fig, ax = plt.subplots(figsize=(3.6, 3.6))
    stats_rows = []
    for label, color, mask in [
        ('concordant', CONC_COLOR, d['classifier_concordant'] == 1),
        ('discordant', DISC_COLOR, d['classifier_concordant'] == 0),
    ]:
        sub = d[mask]
        kmf = KaplanMeierFitter().fit(
            sub['time_years'], sub['event'],
            label=f'{label}  n={len(sub)}  ev={int(sub["event"].sum())}')
        kmf.plot_survival_function(ax=ax, ci_show=True, ci_alpha=0.12,
                                    color=color, linewidth=1.3)
        stats_rows.append({
            'group':            label,
            'n':                int(len(sub)),
            'events':           int(sub['event'].sum()),
            'median_years':     float(kmf.median_survival_time_),
        })

    lr = logrank_test(
        d[d['classifier_concordant'] == 1]['time_years'],
        d[d['classifier_concordant'] == 0]['time_years'],
        event_observed_A=d[d['classifier_concordant'] == 1]['event'],
        event_observed_B=d[d['classifier_concordant'] == 0]['event'])

    ax.set_xlim(0, MAX_YEARS); ax.set_ylim(0, 1.02)
    ax.set_xlabel('Time (years)')
    ax.set_ylabel('Survival probability')
    ax.set_box_aspect(1)
    hr_txt = (f'stratified Cox (by class)\n'
              f'HR (disc vs conc) = {(1/hr):.2f}\n'
              f'  95% CI {1/ci_hi:.2f}–{1/ci_lo:.2f}\n'
              f'  P = {cox_p:.1e}' if cox_p < 1e-3
              else f'  P = {cox_p:.3f}')
    ax.text(0.03, 0.18, hr_txt, transform=ax.transAxes, fontsize=7,
            ha='left', va='bottom')
    p_txt = (f'pooled logrank P = {lr.p_value:.1e}'
             if lr.p_value < 1e-3 else f'pooled logrank P = {lr.p_value:.3f}')
    ax.text(0.03, 0.06, p_txt, transform=ax.transAxes, fontsize=7,
            ha='left', va='bottom')
    ax.legend(loc='upper right', fontsize=7)
    save_panel(fig, out_no_ext)
    plt.close(fig)

    return pd.DataFrame({
        'hr_disc_vs_conc':      [1 / hr],
        'hr_ci_lo':             [1 / ci_hi],
        'hr_ci_hi':             [1 / ci_lo],
        'cox_p':                [cox_p],
        'pooled_logrank_p':     [float(lr.p_value)],
        'n_concordant':         [int((d['classifier_concordant'] == 1).sum())],
        'events_concordant':    [int(d[d['classifier_concordant'] == 1]['event'].sum())],
        'median_yr_concordant': [stats_rows[0]['median_years']],
        'n_discordant':         [int((d['classifier_concordant'] == 0).sum())],
        'events_discordant':    [int(d[d['classifier_concordant'] == 0]['event'].sum())],
        'median_yr_discordant': [stats_rows[1]['median_years']],
    })


def _per_class(df: pd.DataFrame):
    rows = []
    per_class_data = []
    for cls, sub in df.groupby('actual_label'):
        conc = sub[sub['classifier_concordant'] == 1]
        disc = sub[sub['classifier_concordant'] == 0]
        if len(conc) < MIN_N_PER_GROUP or len(disc) < MIN_N_PER_GROUP:
            continue
        pooled = pd.concat([
            conc.assign(disc=0),
            disc.assign(disc=1),
        ])[['time_years', 'event', 'disc']].astype({'event': int})
        try:
            cph = CoxPHFitter(penalizer=0.0).fit(
                pooled, duration_col='time_years', event_col='event')
            hr = float(np.exp(cph.params_['disc']))
            ci_lo, ci_hi = np.exp(cph.confidence_intervals_.loc['disc'].values)
            cox_p = float(cph.summary.loc['disc', 'p'])
        except Exception:
            hr, ci_lo, ci_hi, cox_p = np.nan, np.nan, np.nan, np.nan
        lr = logrank_test(conc['time_years'], disc['time_years'],
                          event_observed_A=conc['event'],
                          event_observed_B=disc['event'])
        kmf_c = KaplanMeierFitter().fit(conc['time_years'], conc['event'])
        kmf_d = KaplanMeierFitter().fit(disc['time_years'], disc['event'])
        rows.append({
            'class':                cls,
            'n_concordant':         int(len(conc)),
            'events_concordant':    int(conc['event'].sum()),
            'median_yr_concordant': float(kmf_c.median_survival_time_),
            'n_discordant':         int(len(disc)),
            'events_discordant':    int(disc['event'].sum()),
            'median_yr_discordant': float(kmf_d.median_survival_time_),
            'logrank_p':            float(lr.p_value),
            'hr_disc_vs_conc':      hr,
            'hr_ci_lo':             ci_lo,
            'hr_ci_hi':             ci_hi,
            'cox_p':                cox_p,
        })
        per_class_data.append((cls, conc, disc))
    return pd.DataFrame(rows).sort_values('class').reset_index(drop=True), per_class_data


def _plot_per_class_grid(per_class_data, out_no_ext):
    n = len(per_class_data)
    ncol = min(4, max(1, n))
    nrow = int(math.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.7 * ncol, 2.7 * nrow),
                             squeeze=False)
    for ax, (cls, conc, disc) in zip(axes.flat, per_class_data):
        kmf_c = KaplanMeierFitter().fit(conc['time_years'], conc['event'])
        kmf_d = KaplanMeierFitter().fit(disc['time_years'], disc['event'])
        kmf_c.plot_survival_function(ax=ax, ci_show=False, color=CONC_COLOR,
                                      linewidth=1.1, label=f'conc  n={len(conc)}')
        kmf_d.plot_survival_function(ax=ax, ci_show=False, color=DISC_COLOR,
                                      linewidth=1.1, label=f'disc  n={len(disc)}')
        lr = logrank_test(conc['time_years'], disc['time_years'],
                          event_observed_A=conc['event'],
                          event_observed_B=disc['event'])
        p_txt = (f'P={lr.p_value:.1e}' if lr.p_value < 1e-3
                 else f'P={lr.p_value:.3f}')
        ax.set_xlim(0, MAX_YEARS); ax.set_ylim(0, 1.02)
        ax.set_title(cls, fontsize=8)
        ax.set_xlabel('Time (y)', fontsize=7)
        ax.set_ylabel('S(t)', fontsize=7)
        ax.text(0.03, 0.05, p_txt, transform=ax.transAxes, fontsize=6)
        ax.legend(loc='upper right', fontsize=5.5, handlelength=1.0)
    for ax in axes.flat[n:]:
        ax.axis('off')
    fig.suptitle('TCGA by class: concordant vs discordant', fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save_panel(fig, out_no_ext)
    plt.close(fig)


def _plot_per_class_forest(summary: pd.DataFrame, out_no_ext):
    s = summary.dropna(subset=['hr_disc_vs_conc']).sort_values('hr_disc_vs_conc')
    y = np.arange(len(s))
    fig, ax = plt.subplots(figsize=(4.5, max(3.0, 0.25 * len(s))))
    ax.hlines(y, s['hr_ci_lo'], s['hr_ci_hi'], color='#444', lw=0.8)
    ax.scatter(s['hr_disc_vs_conc'], y, s=20, color='#333', zorder=3)
    ax.axvline(1.0, color='k', ls=':', lw=0.6)
    for i, (cls, nc, nd, p) in enumerate(zip(
            s['class'], s['n_concordant'], s['n_discordant'], s['logrank_p'])):
        stars = '***' if p < 0.001 else ('**' if p < 0.01
                                         else ('*' if p < 0.05 else ''))
        ax.text(max(s['hr_ci_hi'].iloc[i] * 1.05, 1.02), y[i],
                f'{cls}  conc={nc}  disc={nd} {stars}',
                va='center', fontsize=7)
    ax.set_yticks([])
    ax.set_xscale('log')
    ax.set_xlabel('HR (discordant vs concordant)  —  >1 = worse when misclassified')
    ax.set_title('TCGA per-class concordance effect')
    save_panel(fig, out_no_ext)
    plt.close(fig)


def _confidence_strata(df: pd.DataFrame, out_no_ext):
    d = df.copy()
    d['top1_bin'] = pd.qcut(d['oof_prob_top1'], q=4,
                            labels=['Q1 (low)', 'Q2', 'Q3', 'Q4 (high)'])
    rows = []
    for (qb, conc), sub in d.groupby(['top1_bin', 'classifier_concordant'],
                                      observed=True):
        if len(sub) < 10:
            continue
        kmf = KaplanMeierFitter().fit(sub['time_years'],
                                       sub['event'].astype(int))
        rows.append({
            'top1_bin':          str(qb),
            'concordant':        int(conc),
            'n':                 int(len(sub)),
            'events':            int(sub['event'].astype(int).sum()),
            'median_years':      float(kmf.median_survival_time_),
        })
    stats = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    for concord, color, label in [
        (1, CONC_COLOR, 'concordant'),
        (0, DISC_COLOR, 'discordant'),
    ]:
        s = stats[stats['concordant'] == concord]
        ax.plot(s['top1_bin'], s['median_years'],
                marker='o', color=color, linewidth=1.5, label=label)
        for _, r in s.iterrows():
            ax.text(r['top1_bin'], r['median_years'] + 0.3,
                    f'n={r["n"]}', fontsize=6, ha='center')
    ax.set_xlabel('Classifier top-1 confidence quartile')
    ax.set_ylabel('Median survival (years)')
    ax.set_title('TCGA concordance effect vs classifier confidence')
    ax.legend(fontsize=7)
    save_panel(fig, out_no_ext)
    plt.close(fig)
    return stats


def main():
    if METRIC not in ENDPOINT_COLUMNS:
        raise ValueError(f"METRIC={METRIC!r} not one of {list(ENDPOINT_COLUMNS)}")
    out_dir = OUTPUT_DIR or os.path.join(HERE, f"tcga_concordance_{METRIC}")
    if MIN_TOP1_PROB > 0 or MIN_TRUE_CLASS_PROB > 0:
        parts = []
        if MIN_TOP1_PROB > 0:
            parts.append(f"top1{MIN_TOP1_PROB:.2f}".replace(".", ""))
        if MIN_TRUE_CLASS_PROB > 0:
            parts.append(f"true{MIN_TRUE_CLASS_PROB:.2f}".replace(".", ""))
        out_dir = f"{out_dir}_" + "_".join(parts)
    os.makedirs(out_dir, exist_ok=True)

    apply_style()

    print(f"Loading predictions from {RESULTS_PKL}")
    preds, class_names = _load_predictions()
    print(f"  predictions: {len(preds):,} samples, {len(class_names)} classes")

    print(f"\nLoading TCGA clinical from {CLINICAL_CSV}")
    cdr = _load_clinical(METRIC)
    print(f"  clinical rows: {len(cdr):,}  metric: {METRIC}")

    # TCGA sample IDs in the classifier are short-form (15 chars,
    # TCGA-XX-YYYY-ZZ). CDR keys are bcr_patient_barcode (first 12).
    preds['bcr'] = preds['SAMPLE_ID'].str[:12]
    df = preds.merge(cdr[['bcr', 'time_days', 'event']],
                     on='bcr', how='inner')
    df = df.dropna(subset=['time_days', 'event'])
    df = df[df['time_days'] >= 0].copy()
    df['time_years'] = df['time_days'] / 365.25
    df['event'] = df['event'].astype(int)
    df['classifier_concordant'] = (df['oof_pred_class']
                                    == df['actual_label']).astype(int)
    df['true_class_prob'] = df.apply(
        lambda r: r[f'prob_{r["actual_label"]}']
        if f'prob_{r["actual_label"]}' in df.columns else np.nan,
        axis=1)

    n_before = len(df)
    if MIN_TOP1_PROB > 0.0:
        df = df[df['oof_prob_top1'] >= MIN_TOP1_PROB].copy()
        print(f"  --min-top1-prob {MIN_TOP1_PROB:.2f}: "
              f"{n_before:,} → {len(df):,}")
    if MIN_TRUE_CLASS_PROB > 0.0:
        n_before2 = len(df)
        keep = ((df['classifier_concordant'] == 0)
                | (df['true_class_prob'] >= MIN_TRUE_CLASS_PROB))
        df = df[keep].copy()
        print(f"  --min-true-class-prob {MIN_TRUE_CLASS_PROB:.2f}: "
              f"{n_before2:,} → {len(df):,}")

    print(f"  merged + filtered: {len(df):,}   "
          f"concordant={int(df['classifier_concordant'].sum()):,}  "
          f"discordant={int((1 - df['classifier_concordant']).sum()):,}")

    pan = _pan_cancer(df, os.path.join(out_dir, 'pan_cancer_km'))
    pan.to_csv(os.path.join(out_dir, 'pan_cancer_stats.csv'), index=False)
    print(f"\n=== Pan-cancer (stratified by class), metric={METRIC} ===")
    print(pan.to_string(index=False))

    per_class, per_class_data = _per_class(df)
    per_class.to_csv(os.path.join(out_dir, 'per_class_stats.csv'), index=False)
    print(f"\n=== Per-class ({len(per_class)} classes with n >= {MIN_N_PER_GROUP} both) ===")
    print(per_class[['class', 'n_concordant', 'n_discordant',
                     'median_yr_concordant', 'median_yr_discordant',
                     'hr_disc_vs_conc', 'hr_ci_lo', 'hr_ci_hi',
                     'logrank_p']].to_string(index=False))

    if per_class_data:
        _plot_per_class_grid(per_class_data,
                              os.path.join(out_dir, 'per_class_km_grid'))
        _plot_per_class_forest(per_class,
                                os.path.join(out_dir, 'per_class_forest'))

    conf = _confidence_strata(df,
                               os.path.join(out_dir, 'confidence_stratified'))
    conf.to_csv(os.path.join(out_dir, 'confidence_stratified_stats.csv'),
                index=False)
    print(f"\n=== Confidence strata ===")
    print(conf.to_string(index=False))
    print(f"\n✓ {out_dir}")


if __name__ == '__main__':
    main()
