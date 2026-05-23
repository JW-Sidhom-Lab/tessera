"""
Exact-replication of Abraham 2021 FOLFOXai RF ensemble.

Architecture (faithful to the paper's "DEAN" design):
  5 distinct RandomForestClassifier configurations × 1,000 trees each.
  Ensemble-mean P(IB) across the 5 configs = per-patient `p_IB`.
  3% "no-call" buffer around 0.5:
      no_call = 1 if 0.47 < p_IB < 0.53

Label: IB (1) vs DB (0) on the FOLFOX arm only (see 1_build_features.py).
FOLFIRI patients are scored out-of-sample for reverse-direction validation.

Inputs
------
  causal_inference_results/features/patient_features.csv

Outputs
-------
  causal_inference_results/predictions/folfoxai_predictions.csv
    columns: PATIENT_ID, REGIMEN, T, event_cens, ib_db,
             p_IB_cfg_0..4, p_IB, no_call, predicted_class
  causal_inference_results/predictions/rf_training_log.csv

Usage:
    python3 2_train_rf_ensemble.py
"""

import os
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings('ignore')


_THIS_DIR     = os.path.dirname(os.path.abspath(__file__))
FEATURES_CSV  = os.path.join(_THIS_DIR,
                             'causal_inference_results', 'features',
                             'patient_features.csv')
PRED_DIR      = os.path.join(_THIS_DIR,
                             'causal_inference_results', 'predictions')
PRED_CSV      = os.path.join(PRED_DIR, 'folfoxai_predictions.csv')
LOG_CSV       = os.path.join(PRED_DIR, 'rf_training_log.csv')

NO_CALL_BAND  = 0.03           # 3% buffer around 0.5 (paper-exact)

# Set QUICK=True for a 2-fold × 2-config × 200-tree smoke test.
QUICK = False

if QUICK:
    N_FOLDS      = 2
    N_ESTIMATORS = 200
    RF_CONFIGS   = [
        dict(max_features='sqrt', min_samples_leaf=10, random_state=0),
        dict(max_features=0.3,    min_samples_leaf=20, random_state=1),
    ]
else:
    N_FOLDS      = 5
    N_ESTIMATORS = 1000
    RF_CONFIGS   = [
        dict(max_features='sqrt', min_samples_leaf=5,  random_state=0),
        dict(max_features='log2', min_samples_leaf=10, random_state=1),
        dict(max_features=0.3,    min_samples_leaf=10, random_state=2),
        dict(max_features=0.5,    min_samples_leaf=20, random_state=3),
        dict(max_features='sqrt', min_samples_leaf=20, random_state=4),
    ]
N_JOBS = -1


def fit_rf(cfg, x_train, y_train):
    rf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_features=cfg['max_features'],
        min_samples_leaf=cfg['min_samples_leaf'],
        class_weight='balanced',                       # paper doesn't say; IB:DB is 60:40, mild
        random_state=cfg['random_state'],
        n_jobs=N_JOBS,
    )
    rf.fit(x_train, y_train)
    return rf


def main():
    os.makedirs(PRED_DIR, exist_ok=True)

    df = pd.read_csv(FEATURES_CSV)
    feat_cols = [c for c in df.columns
                 if c.endswith('_copies') or c.endswith('_mutations')]
    print(f'patients={len(df)}  features={len(feat_cols)}')

    fox_all = df[df['REGIMEN'] == 'FOLFOX'].reset_index(drop=True)
    fir     = df[df['REGIMEN'] == 'FOLFIRI'].reset_index(drop=True)

    # training set = FOLFOX patients with IB/DB defined (not NaN)
    fox = fox_all[fox_all['ib_db'].notna()].reset_index(drop=True)
    print(f'FOLFOX: total={len(fox_all)}  train(IB/DB)={len(fox)}  '
          f'FOLFIRI (OOS only)={len(fir)}')
    print(f'  IB (1)={int((fox.ib_db == 1).sum())}   '
          f'DB (0)={int((fox.ib_db == 0).sum())}')

    # RandomForestClassifier is invariant to monotonic feature transformations:
    # tree splits are threshold-based, so any standardisation only shifts the
    # threshold and produces an identical partition. Feeding raw feature values
    # avoids an unnecessary preprocessing step and keeps the pipeline closer to
    # Abraham et al. 2021, which trains the RF on raw panel features.
    fox_X = fox[feat_cols].values
    fir_X = fir[feat_cols].values
    fox_y = fox['ib_db'].astype(int).values

    # --- FOLFOX 5-fold OOF ---
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    fox_oof = np.zeros((len(fox), len(RF_CONFIGS)))
    log_rows = []
    for fold, (tr, te) in enumerate(skf.split(fox_X, fox_y)):
        print(f'--- fold {fold + 1}/{N_FOLDS}  n_tr={len(tr)}  n_te={len(te)} ---')
        for k, cfg in enumerate(RF_CONFIGS):
            t0 = time.time()
            rf = fit_rf(cfg, fox_X[tr], fox_y[tr])
            p  = rf.predict_proba(fox_X[te])[:, 1]
            fox_oof[te, k] = p
            try:
                auc = float(roc_auc_score(fox_y[te], p))
            except ValueError:
                auc = float('nan')
            dt = time.time() - t0
            print(f'    cfg{k} ({cfg["max_features"]}, '
                  f'leaf={cfg["min_samples_leaf"]})  '
                  f'AUC={auc:.3f}  {dt:.1f}s')
            log_rows.append({
                'fold': fold, 'cfg': k, 'n_train': len(tr), 'n_test': len(te),
                'max_features': str(cfg['max_features']),
                'min_samples_leaf': cfg['min_samples_leaf'],
                'auc': auc, 'fit_seconds': dt,
            })

    # --- FOLFIRI out-of-sample: fit each config on all FOLFOX IB/DB ---
    print('--- FOLFIRI out-of-sample scoring ---')
    fir_oos = np.zeros((len(fir), len(RF_CONFIGS)))
    for k, cfg in enumerate(RF_CONFIGS):
        t0 = time.time()
        rf = fit_rf(cfg, fox_X, fox_y)
        fir_oos[:, k] = rf.predict_proba(fir_X)[:, 1]
        dt = time.time() - t0
        print(f'  cfg{k}  {dt:.1f}s')
        log_rows.append({
            'fold': -1, 'cfg': k, 'n_train': len(fox), 'n_test': len(fir),
            'max_features': str(cfg['max_features']),
            'min_samples_leaf': cfg['min_samples_leaf'],
            'auc': float('nan'), 'fit_seconds': dt,
        })

    # --- Assemble output (FOLFOX has IB/DB training patients + unlabelled short-censored) ---
    out_fox = fox[['PATIENT_ID', 'REGIMEN', 'OS_MONTHS',
                   'OS_EVENT', 'ib_db']].copy()
    for k in range(len(RF_CONFIGS)):
        out_fox[f'p_IB_cfg_{k}'] = fox_oof[:, k]
    out_fox['p_IB'] = fox_oof.mean(axis=1)

    out_fir = fir[['PATIENT_ID', 'REGIMEN', 'OS_MONTHS',
                   'OS_EVENT', 'ib_db']].copy()
    for k in range(len(RF_CONFIGS)):
        out_fir[f'p_IB_cfg_{k}'] = fir_oos[:, k]
    out_fir['p_IB'] = fir_oos.mean(axis=1)

    out = pd.concat([out_fox, out_fir], ignore_index=True)
    out = out.rename(columns={'OS_MONTHS': 'T', 'OS_EVENT': 'event_cens'})
    out['no_call'] = (
        (out['p_IB'] > 0.5 - NO_CALL_BAND) & (out['p_IB'] < 0.5 + NO_CALL_BAND)
    ).astype(int)
    out['predicted_class'] = np.where(out['no_call'] == 1, np.nan,
                                      (out['p_IB'] >= 0.5).astype(float))

    out.to_csv(PRED_CSV, index=False)
    pd.DataFrame(log_rows).to_csv(LOG_CSV, index=False)

    # Summary
    p_IB_fox = out.loc[out['REGIMEN'] == 'FOLFOX', 'p_IB']
    p_IB_fir = out.loc[out['REGIMEN'] == 'FOLFIRI', 'p_IB']
    auc_fox = float(roc_auc_score(fox['ib_db'].astype(int), fox_oof.mean(axis=1)))
    print('=' * 72)
    print(f'Ensemble FOLFOX OOF AUC = {auc_fox:.3f}')
    print(f'p_IB summary — FOLFOX OOF: mean={p_IB_fox.mean():.3f} '
          f'std={p_IB_fox.std():.3f}    '
          f'FOLFIRI OOS: mean={p_IB_fir.mean():.3f} '
          f'std={p_IB_fir.std():.3f}')
    print(f'No-call count (3% buffer): FOLFOX {int(out.loc[out.REGIMEN=="FOLFOX","no_call"].sum())} / {len(p_IB_fox)}   '
          f'FOLFIRI {int(out.loc[out.REGIMEN=="FOLFIRI","no_call"].sum())} / {len(p_IB_fir)}')
    print(f'wrote {PRED_CSV}')
    print(f'wrote {LOG_CSV}')


if __name__ == '__main__':
    main()
