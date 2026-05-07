"""PRAD (prostate adenocarcinoma) cohort config.

Subtype = within-cohort Decipher tertile from the published curatedPCaData
TCGA scores (Laajala 2023, Bioconductor EH8024). See
``data/prognostic/prad/build_prad_metadata.py``.

Endpoint = PFI.1 (Liu 2018-curated progression-free interval); DSS is
underpowered in TCGA-PRAD (~5 disease-specific death events).
"""

CLINICAL_DATA_PATH = '../../data/prognostic/prad/prad_clinical_metadata.csv'
CATEGORICAL_VAR_COL = 'Subtype_Decipher'
CATEGORIES_INCLUDE = ['Low', 'Intermediate', 'High']

SURV_TIME_COL = 'PFI.time.1'
SURV_EVENT_COL = 'PFI.1'
TIME_LIMIT = 1825  # 5 years (days)
BASE_POWER = 1.0
POLYNOMIAL_DEGREE = 1
INCLUDE_INTERACTION_TERMS = True
N_RISK_GROUPS = 3
