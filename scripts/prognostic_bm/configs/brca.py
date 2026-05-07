"""BRCA (breast) cohort config -- research-reconstructed OncotypeDX RS
tertiles (Howard 2023 / DLRS; see
``data/prognostic/brca/build_brca_metadata.py``)."""

CLINICAL_DATA_PATH = '../../data/prognostic/brca/brca_clinical_metadata.csv'
CATEGORICAL_VAR_COL = 'Subtype_Oncotype'
CATEGORIES_INCLUDE = ['Low', 'Intermediate', 'High']

SURV_TIME_COL = 'DSS.time'
SURV_EVENT_COL = 'DSS'
TIME_LIMIT = 1825  # 5 years
BASE_POWER = 1.0
POLYNOMIAL_DEGREE = 1
INCLUDE_INTERACTION_TERMS = True
N_RISK_GROUPS = 3
