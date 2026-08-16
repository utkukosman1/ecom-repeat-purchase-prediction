"""Project-wide constants.

Every parameter that more than one stage needs lives here and is imported. No notebook
redefines a value, so there is exactly one place to change a decision and exactly one
place to read what the current decision is.

The temporal and cost constants below are the Stage 0 decisions. Changing any of them
invalidates work downstream, so read `docs/problem_definition.md` before editing.
"""

from datetime import datetime, timedelta
from pathlib import Path

# --------------------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_DIR = DATA_DIR / "output"
MODELS_DIR = PROJECT_ROOT / "models"
DOCS_DIR = PROJECT_ROOT / "docs"

# Named artifacts, so no stage spells a filename twice.
RAW_XLSX = RAW_DIR / "online_retail_II.xlsx"
RAW_PARQUET = RAW_DIR / "transactions_raw.parquet"
CLEAN_PARQUET = PROCESSED_DIR / "transactions_clean.parquet"
FEATURES_PARQUET = PROCESSED_DIR / "customer_features.parquet"
TRAIN_PARQUET = PROCESSED_DIR / "train.parquet"
TEST_PARQUET = PROCESSED_DIR / "test.parquet"
MODEL_SELECTION_JSON = MODELS_DIR / "model_selection.json"
FINAL_MODEL_PATH = MODELS_DIR / "final_model.joblib"
MODEL_METADATA_PATH = MODELS_DIR / "model_metadata.json"

# --------------------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------------------

RANDOM_SEED = 42

# --------------------------------------------------------------------------------------
# Temporal design (Stage 0, section 5 of docs/problem_definition.md)
#
#   OBSERVATION WINDOW (features)              PREDICTION WINDOW (label)
#   2009-12-01 -------------------> 2011-09-10 -------------------> 2011-12-09
#                                       ^
#                                    CUTOFF
#
# Features use transactions strictly BEFORE the cutoff. The label uses transactions in
# [CUTOFF_DATE, PREDICTION_END). Recency is always measured against CUTOFF_DATE and never
# against max(invoice_date), which would silently import the future.
#
# The cutoff is chosen so that CUTOFF_DATE + HORIZON_DAYS lands on the last day present in
# the dataset, which keeps the prediction window fully populated. Stage 2 verifies the
# actual max(invoice_date); if transactions exist on 2011-12-09 itself, the exclusive
# midnight bound here drops that final partial day and the cutoff shifts by one.
#
# These are datetimes rather than dates because `invoice_date` is a timestamp column;
# comparing like to like avoids dtype friction in Polars. Midnight is the boundary.
# --------------------------------------------------------------------------------------

OBSERVATION_START = datetime(2009, 12, 1)
CUTOFF_DATE = datetime(2011, 9, 10)
HORIZON_DAYS = 90
PREDICTION_END = CUTOFF_DATE + timedelta(days=HORIZON_DAYS)

# Lookback windows used by the momentum features in Stage 5, all anchored to CUTOFF_DATE.
LOOKBACK_WINDOWS_DAYS = (30, 90, 180, 365)

# --------------------------------------------------------------------------------------
# Validation (Stage 0, section 6)
# --------------------------------------------------------------------------------------

TEST_SIZE = 0.2
CV_FOLDS = 5

# --------------------------------------------------------------------------------------
# Decision layer cost model (Stage 0, section 3)
#
# Named by business meaning, not by confusion-matrix cell, because the polarity inverts:
# the model predicts P(repeat), but the campaign acts on the NEGATIVE class. So a missed
# churner is a false POSITIVE in standard label terms, and a wasted offer is a false
# negative. Naming these COST_FP / COST_FN would invite exactly the wrong wiring.
#
# Only the ratio matters. We normalize the recoverable value of a churner to 1.0 and
# assume the discount costs 15 percent of what it can recover.
# --------------------------------------------------------------------------------------

COST_MISSED_CHURNER = 1.0  # did not repeat, we predicted repeat, no offer sent
COST_WASTED_OFFER = 0.15  # would have repeated, we predicted otherwise, offer sent

# Minimum acceptable performance, fixed before modeling (Stage 0, section 3).
MIN_ACCEPTABLE_ROC_AUC = 0.75
MIN_COST_REDUCTION_VS_TRIVIAL = 0.20

# --------------------------------------------------------------------------------------
# Feature schema
#
# COUNTRY_VOCAB and FEATURE_COLUMNS are pinned here in Stage 5, once EDA has shown which
# countries clear a minimum customer count. They are deliberately NOT defined yet: an
# early import should fail loudly rather than quietly encode against an empty or None
# vocabulary.
# --------------------------------------------------------------------------------------

# COUNTRY_VOCAB: tuple[str, ...]     <- set in Stage 5
# FEATURE_COLUMNS: tuple[str, ...]   <- set in Stage 5
