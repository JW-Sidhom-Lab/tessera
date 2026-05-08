"""UCEC (endometrial) cohort config -- TCGA PanCancer molecular subtypes
(Kandoth 2013 / Hoadley 2018; see ``data/prognostic/ucec/build_ucec_metadata.py``).

Endometrial carcinoma is analysed by unsupervised projection only: the
polynomial-Cox risk score and joint-Cox steps are skipped because the
downstream comparators (OncotypeDX-RS, Decipher) do not apply.
"""

CLINICAL_DATA_PATH = '../../data/prognostic/ucec/ucec_clinical_metadata.csv'
CATEGORICAL_VAR_COL = 'Subtype'
CATEGORIES_INCLUDE = [
    'UCEC_CN_HIGH',
    'UCEC_CN_LOW',
    'UCEC_MSI',
    'Copy-number low (Endometriod)',
]

SURV_TIME_COL = 'DSS.time'
SURV_EVENT_COL = 'DSS'
TIME_LIMIT = 1825  # 5 years
BASE_POWER = 1.0
POLYNOMIAL_DEGREE = 1
INCLUDE_INTERACTION_TERMS = True
N_RISK_GROUPS = 3
