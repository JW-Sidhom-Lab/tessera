"""Glioma cohort config -- WHO 2021 primary classes (Leiria 2025; see
``data/prognostic/glioma/build_glioma_metadata.py``)."""

CLINICAL_DATA_PATH = '../../data/prognostic/glioma/glioma_clinical_metadata.csv'
CATEGORICAL_VAR_COL = 'WHO2021'
CATEGORIES_INCLUDE = ['astrocytoma', 'glioblastoma', 'oligodendroglioma']

SURV_TIME_COL = 'DSS.time'
SURV_EVENT_COL = 'DSS'
TIME_LIMIT = 1825  # 5 years

N_GMM_CLUSTERS = 3
BASE_POWER = 1.0
POLYNOMIAL_DEGREE = 1
INCLUDE_INTERACTION_TERMS = True
N_RISK_GROUPS = 3
