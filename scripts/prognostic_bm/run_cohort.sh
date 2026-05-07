#!/bin/bash
# Deterministic launcher for cohort_analysis.py.
#
# Sets the env vars that control non-deterministic threading and Python
# hash randomization BEFORE the Python interpreter starts (some libs
# only honor these at process-launch time, not via os.environ from
# inside the script).
#
# Usage:
#   ./run_cohort.sh                  # uses COHORT/FEATURES_SOURCE set in cohort_analysis.py
#   COHORT=glioma ./run_cohort.sh    # override via env
#
# Required to make UMAP + polynomial Cox + GMM bit-reproducible across
# repeated runs with the same input data.

set -e
cd "$(dirname "$0")"

export PYTHONHASHSEED=0
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMBA_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
# numba threading layer: 'workqueue' is the most deterministic.
export NUMBA_THREADING_LAYER=workqueue
# BLAS implementations sometimes accumulate FP differently across runs;
# fixing the algorithm choice helps stabilize SVD / matmul reductions.
export BLIS_NUM_THREADS=1

# Allow caller to override COHORT / FEATURES_SOURCE without editing the
# script. Default: leave whatever the script currently has hard-coded.
if [ -n "$COHORT" ]; then
  echo "[run_cohort.sh] Setting COHORT='$COHORT' in cohort_analysis.py"
  python3 -c "
import re
with open('cohort_analysis.py') as f: s = f.read()
s = re.sub(r\"^COHORT = '[a-z_]+'\", \"COHORT = '$COHORT'\", s, count=1, flags=re.MULTILINE)
with open('cohort_analysis.py','w') as f: f.write(s)
"
fi

exec python3 cohort_analysis.py "$@"
